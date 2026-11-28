"""Batch-index loader for paired SHUTrack training.

Yields whole batches via the supplied dataset's ``batch`` method
instead of one item at a time.  An optional producer thread overlaps the CPU
gather with GPU compute; the sampler remains the owner of the epoch schedule
and the yielded batch order is exactly the sampler order.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

_SENTINEL = object()


class _BatchSource(Protocol):
    def batch(self, indices: Any) -> Any: ...


class BatchIndexLoader:
    def __init__(
        self,
        dataset: _BatchSource,
        sampler: Iterable[Any],
        *,
        prefetch: int = 2,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.prefetch = max(0, int(prefetch))
        self._thread: threading.Thread | None = None

    @staticmethod
    def _put(sink: queue.Queue, item: object, stop: threading.Event) -> bool:
        """Enqueue an item, honouring the stop signal even when the queue is full."""

        while not stop.is_set():
            try:
                sink.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self, sink: queue.Queue, stop: threading.Event) -> None:
        try:
            for indices in self.sampler:
                if stop.is_set():
                    return
                batch = self.dataset.batch(indices)
                if not self._put(sink, batch, stop):
                    return
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            if not stop.is_set():
                self._put(sink, error, stop)
        finally:
            if not stop.is_set():
                self._put(sink, _SENTINEL, stop)

    def __iter__(self) -> Iterator[Any]:
        if self.prefetch <= 0:
            for indices in self.sampler:
                yield self.dataset.batch(indices)
            return
        sink: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        thread = threading.Thread(
            target=self._produce, args=(sink, stop), daemon=True
        )
        self._thread = thread
        thread.start()
        try:
            while True:
                item = sink.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            # Blocking join: every producer wait honours `stop`, so this
            # returns after the gather has actually left the mmap-backed
            # dataset and the caller may safely close it.
            thread.join()
            self._thread = None


__all__ = ["BatchIndexLoader"]
