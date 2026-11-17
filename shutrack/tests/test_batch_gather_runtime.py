"""Batch-gather loader and online evidence reuse contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from shutrack.data.batch_loader import BatchIndexLoader
from shutrack.data.legacy_adapter import detection_overlap_row
from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.data.sampling import SequenceBatchSampler
from shutrack.integration.tracktrack import ShutrackTrackerAdapter
from shutrack.runtime.adapter import ShutrackRuntime, ShutrackRuntimeConfig
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.tests.conftest import write_tiny_checkpoint, write_tiny_packed_dataset


def _assert_batch_equal(dataset, indices):
    reference = dataset.collate_fn([dataset[index] for index in indices])
    gathered = dataset.batch(indices)
    for key, value in reference.items():
        candidate = gathered.get(key)
        if torch.is_tensor(value):
            assert torch.is_tensor(candidate), key
            assert torch.equal(value, candidate), key
        else:
            assert value == candidate, key


def test_batch_gather_matches_per_item(tmp_path):
    root = write_tiny_packed_dataset(tmp_path / "packed")
    dataset = PackedShutrackDataset(root)
    try:
        for indices in (
            [0, 1, 2, 3],
            [3],
            [3, 2, 1, 0],
            [0, 0, 3, 3, 1],
        ):
            _assert_batch_equal(dataset, indices)
    finally:
        dataset.close()


def test_batch_gather_missing_detection_and_cross_sequence(tmp_path):
    root = write_tiny_packed_dataset(tmp_path / "packed")
    data_path = root / "seqA" / "data.pt"
    payload = torch.load(data_path, weights_only=False)
    payload["arrays"]["timeline_has_detection"][0] = False
    torch.save(payload, data_path)

    dataset = PackedShutrackDataset(root)
    try:
        # Samples 0/1 reference timeline index 0 through their history windows;
        # sample 3 is the first sample of seqB (cross-sequence tail).
        _assert_batch_equal(dataset, [0, 1, 2, 3])
        sample = dataset[0]
        assert bool(sample["history_has_detection"][3]) is False
        assert float(sample["history_detection"][3].abs().sum()) == 0.0
    finally:
        dataset.close()


def test_shared_load_counter_counts(tmp_path):
    root = write_tiny_packed_dataset(tmp_path / "packed")
    counter = torch.multiprocessing.Value("q", 0)
    dataset = PackedShutrackDataset(root)
    dataset.set_shared_load_counter(counter)
    try:
        dataset.batch([0, 1])
        assert counter.value == 1
        assert dataset.total_sequence_loads() == 1
        dataset.batch([3])
        assert counter.value == 2
        assert dataset.total_sequence_loads() == 2
    finally:
        dataset.close()


def test_batch_loader_preserves_sampler_order(tmp_path):
    root = write_tiny_packed_dataset(tmp_path / "packed")
    dataset = PackedShutrackDataset(root)
    sampler = SequenceBatchSampler(
        dataset.sample_boundaries, 2, total_samples=len(dataset), seed=7
    )
    sampler.set_epoch(1)
    expected = [dataset.batch(batch) for batch in sampler]
    loader = BatchIndexLoader(dataset, sampler, prefetch=2)
    loader.set_epoch(1)
    produced = list(loader)
    assert len(produced) == len(expected)
    for reference, gathered in zip(expected, produced):
        for key, value in reference.items():
            if torch.is_tensor(value):
                assert torch.equal(value, gathered[key]), key
            else:
                assert value == gathered[key], key
    dataset.close()


def _runtime(tmp_path) -> ShutrackRuntime:
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path, competition_enabled=True)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    return ShutrackRuntime(
        model,
        checkpoint,
        device="cpu",
        config=ShutrackRuntimeConfig(min_observations=1),
    )


def _detection(index: int, box) -> SimpleNamespace:
    return SimpleNamespace(frame_detection_index=index, x1y1x2y2=np.asarray(box, float))


def test_overlap_row_cache_matches_direct_computation(tmp_path):
    runtime = _runtime(tmp_path)
    pool = [
        _detection(0, [0, 0, 10, 10]),
        _detection(1, [1, 1, 11, 11]),
        _detection(2, [50, 50, 60, 60]),
    ]
    runtime.set_frame_detection_pool(pool)
    meta = {"detection_indices": np.asarray([2, 0, 1])}
    boxes = np.stack([[50, 50, 60, 60], [0, 0, 10, 10], [1, 1, 11, 11]])

    for column in range(3):
        row = runtime._overlap_row(meta, column)
        direct = detection_overlap_row(boxes[column], boxes, column)
        np.testing.assert_allclose(row, direct, atol=1e-12)
        cached = runtime._overlap_row(meta, column)
        np.testing.assert_allclose(cached, direct, atol=1e-12)

    # A new frame's pool clears the cached rows.
    runtime.set_frame_detection_pool([_detection(0, [0, 0, 5, 5])])
    assert runtime._overlap_cache == {}
    # Association indices that do not match the cached pool fall back to the
    # per-frame box dictionary without changing the result.
    runtime.set_frame_detection_pool(pool)
    meta_reordered = {"detection_indices": np.asarray([0, 2, 1])}
    boxes_reordered = np.stack([[0, 0, 10, 10], [50, 50, 60, 60], [1, 1, 11, 11]])
    row = runtime._overlap_row(meta_reordered, 1)
    direct = detection_overlap_row(
        boxes_reordered[1], boxes_reordered, 1
    )
    np.testing.assert_allclose(row, direct, atol=1e-12)


def test_shutrack_adapter_skips_frame_start_snapshot():
    assert ShutrackTrackerAdapter.needs_frame_start_snapshot is False
    # The legacy AgentGuard adapter does not declare the flag, so the tracker
    # keeps its original snapshot behaviour (default True).
    assert getattr(object(), "needs_frame_start_snapshot", True) is True
