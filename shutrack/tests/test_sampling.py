"""Sequence-organized sampling and bounded sequence-load tests."""

from __future__ import annotations

import bisect
from pathlib import Path

import pytest

from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.data.sampling import SequenceBatchSampler


def _dataset(path: Path) -> PackedShutrackDataset:
    return PackedShutrackDataset(path)


def _batch_sequence(
    boundaries: tuple[int, ...], batch: list[int]
) -> int:
    positions = {bisect.bisect_right(boundaries, index) for index in batch}
    assert len(positions) == 1
    return positions.pop()


def test_sampler_covers_every_sample_once(tiny_packed_dir: Path):
    dataset = _dataset(tiny_packed_dir)
    try:
        sampler = SequenceBatchSampler(
            dataset.sample_boundaries,
            2,
            total_samples=len(dataset),
            seed=42,
        )
        assert len(sampler) == 3  # seqA: 2 batches, seqB: 1 batch
        visited: list[int] = []
        for batch in sampler:
            assert 1 <= len(batch) <= 2
            _batch_sequence(dataset.sample_boundaries, batch)
            visited.extend(batch)
        assert sorted(visited) == list(range(len(dataset)))
    finally:
        dataset.close()


def test_one_sequence_load_per_epoch(tiny_packed_dir: Path):
    dataset = _dataset(tiny_packed_dir)
    try:
        sampler = SequenceBatchSampler(
            dataset.sample_boundaries,
            1,
            total_samples=len(dataset),
            seed=42,
        )
        for epoch in (1, 2):
            sampler.set_epoch(epoch)
            loads_before = dataset.sequence_load_count
            for batch in sampler:
                for index in batch:
                    dataset[index]
            loads = dataset.sequence_load_count - loads_before
            assert loads <= len(dataset.sequences)
            if epoch == 1:
                assert loads == len(dataset.sequences)
    finally:
        dataset.close()


def test_max_samples_is_respected_by_sampler(tiny_packed_dir: Path):
    dataset = _dataset(tiny_packed_dir)
    try:
        sampler = SequenceBatchSampler(
            dataset.sample_boundaries,
            2,
            total_samples=2,
            seed=7,
        )
        visited = [index for batch in sampler for index in batch]
        assert sorted(visited) == [0, 1]
    finally:
        dataset.close()
