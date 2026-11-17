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
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from shutrack.data.packed_dataset import PackedShutrackDataset


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


class ShardBatchSampler(Sampler[list[int]]):
    """Use AgentGuard's shard ranges and quotas with sequence-local I/O."""

    def __init__(
        self, dataset: PackedShutrackDataset, config: Mapping[str, Any]
    ) -> None:
        from agentguard.training.train_iwg_rg_cma import build_memory_shard_schedule

        self.schedule = build_memory_shard_schedule(dataset, dict(config))
        self.batch_size = int(config["batch_size"])
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.seed = int(config["seed"])
        self.epoch = 1
        self.steps_per_epoch = [
            math.ceil(int(phase["samples"]) / self.batch_size)
            for phase in self.schedule["phases"]
            for _ in range(int(phase["epochs"]))
        ]
        if any(steps == 0 for steps in self.steps_per_epoch):
            raise ValueError("a training shard contains no samples")

    @property
    def phase(self) -> dict[str, Any]:
        return self.schedule["phases"][
            (self.epoch - 1) // self.schedule["epochs_per_shard"]
        ]

    def set_epoch(self, epoch: int) -> None:
        if not 1 <= epoch <= len(self.steps_per_epoch):
            raise ValueError(f"epoch {epoch} is outside the training schedule")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch[self.epoch - 1]

    def __iter__(self) -> Iterator[list[int]]:
        phase = self.phase
        ranges = [item for item in phase["ranges"] if item["samples"] > 0]
        rng = np.random.RandomState(self.seed + self.epoch)
        sequence_order = rng.permutation(len(ranges))
        if self.schedule["sequence_sampling_replacement"]:
            from agentguard.training.train_iwg_rg_cma import SequenceSqrtSampler

            phase_seed = (
                self.seed
                + (phase["cycle"] - 1) * self.schedule["memory_shards"]
                + phase["shard"] - 1
            )
            source = SequenceSqrtSampler(
                ranges,
                phase["sequence_sampling_target_counts"],
                num_samples=phase["samples"],
                seed=phase_seed,
            )
            source.set_epoch(self.epoch)
            indices = np.fromiter(source, dtype=np.int64, count=phase["samples"])
            ends = np.cumsum([item["samples"] for item in ranges])
            starts = np.r_[0, ends[:-1]]
            groups = np.searchsorted(ends, indices, side="right")
            # Retain the sampled rows and repeats, grouping by sequence for mmap reuse.
            ordered = [
                indices[groups == i] - starts[i] + ranges[i]["global_start"]
                for i in sequence_order
            ]
        else:
            ordered = [
                rng.permutation(ranges[i]["samples"]) + ranges[i]["global_start"]
                for i in sequence_order
            ]
        indices = np.concatenate(ordered)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size].tolist()


__all__ = ["SequenceBatchSampler", "ShardBatchSampler"]
