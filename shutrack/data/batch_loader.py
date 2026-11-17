"""Batch-index loader for the packed SHUTrack dataset.

Yields whole batches via ``PackedShutrackDataset.batch`` (vectorised gather)
instead of one item at a time.  An optional producer thread overlaps the CPU
gather with GPU compute; the sampler remains the owner of the epoch schedule
and the yielded batch order is exactly the sampler order.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator

from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.data.sampling import SequenceBatchSampler, ShardBatchSampler

_SENTINEL = object()


class BatchIndexLoader:
    def __init__(
        self,
        dataset: PackedShutrackDataset,
        sampler: SequenceBatchSampler | ShardBatchSampler,
        *,
        prefetch: int = 2,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.prefetch = max(0, int(prefetch))

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.sampler)

    def _produce(
        self, sink: queue.Queue, stop: threading.Event
    ) -> None:
        try:
            for indices in self.sampler:
                if stop.is_set():
                    return
                batch = self.dataset.batch(indices)
                while not stop.is_set():
                    try:
                        sink.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            if not stop.is_set():
                sink.put(error)
        finally:
            if not stop.is_set():
                sink.put(_SENTINEL)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.prefetch <= 0:
            for indices in self.sampler:
                yield self.dataset.batch(indices)
            return
        sink: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        thread = threading.Thread(
            target=self._produce, args=(sink, stop), daemon=True
        )
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
            thread.join(timeout=5.0)


__all__ = ["BatchIndexLoader"]
