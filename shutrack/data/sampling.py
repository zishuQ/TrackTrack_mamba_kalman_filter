"""Sequence-organized batch sampling for bounded packed-data caching.

Global random shuffling with a small per-sequence LRU reloads a ``data.pt``
file whenever consecutive samples jump between sequences.  This sampler keeps
batches inside one sequence, shuffles the sequence order once per epoch and
shuffles local sample order inside each sequence.  Every sample is visited
exactly once per epoch and each sequence file is loaded at most once per
epoch (the LRU only has to hold one sequence at a time).
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class SequenceBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        boundaries: Sequence[int],
        batch_size: int,
        *,
        total_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.boundaries = [int(value) for value in boundaries]
        if not self.boundaries or any(
            right <= left
            for left, right in zip([0] + self.boundaries[:-1], self.boundaries)
        ):
            raise ValueError("boundaries must be strictly increasing positive ints")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.total_samples = (
            int(total_samples) if total_samples is not None else self.boundaries[-1]
        )
        self.epoch = 0

    @property
    def starts(self) -> list[int]:
        return [0] + self.boundaries[:-1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for start, end in zip(self.starts, self.boundaries):
            end = min(end, self.total_samples)
            if start < end:
                ranges.append((start, end))
        return ranges

    def __iter__(self) -> Iterator[list[int]]:
        ranges = self._ranges()
        if not ranges:
            return
        rng = np.random.RandomState(self.seed + self.epoch)
        for range_index in rng.permutation(len(ranges)):
            start, end = ranges[int(range_index)]
            local = rng.permutation(end - start)
            for offset in range(0, end - start, self.batch_size):
                yield [
                    start + int(index)
                    for index in local[offset : offset + self.batch_size]
                ]

    def __len__(self) -> int:
        return sum(
            math.ceil((end - start) / self.batch_size)
            for start, end in self._ranges()
        )


__all__ = ["SequenceBatchSampler"]
