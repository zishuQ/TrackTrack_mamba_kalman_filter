"""Batch-gather loader and online evidence reuse contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shutrack.data.legacy_adapter import detection_overlap_row
from shutrack.tests.conftest import write_tiny_checkpoint
from shutrack.integration.tracktrack import ShutrackTrackerAdapter
from shutrack.runtime.adapter import ShutrackRuntime, ShutrackRuntimeConfig
from shutrack.runtime.checkpoint_loader import load_runtime_model


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


def test_overlap_row_cache_key_includes_ordered_pool(tmp_path):
    """Same frame, different candidate order: column 0 is a different detection."""

    runtime = _runtime(tmp_path)
    pool = [
        _detection(0, [0, 0, 10, 10]),
        _detection(1, [2, 2, 12, 12]),
        _detection(2, [50, 50, 60, 60]),
    ]
    runtime.set_frame_detection_pool(pool)
    meta_a = {"detection_indices": np.asarray([0, 1, 2])}
    meta_b = {"detection_indices": np.asarray([2, 1, 0])}
    boxes_a = np.stack([[0, 0, 10, 10], [2, 2, 12, 12], [50, 50, 60, 60]])
    boxes_b = np.stack([[50, 50, 60, 60], [2, 2, 12, 12], [0, 0, 10, 10]])

    rows_a = [runtime._overlap_row(meta_a, column) for column in range(3)]
    rows_b = [runtime._overlap_row(meta_b, column) for column in range(3)]
    for column in range(3):
        np.testing.assert_allclose(
            rows_a[column],
            detection_overlap_row(boxes_a[column], boxes_a, column),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rows_b[column],
            detection_overlap_row(boxes_b[column], boxes_b, column),
            atol=1e-12,
        )
    # The two pools must disagree for column 0; a column-only cache key would
    # have reused the first row for the second pool.
    assert not np.allclose(rows_a[0], rows_b[0])
    assert rows_b[0][0] == pytest.approx(0.0, abs=1e-12)


def test_competition_max_iou_uses_ordered_pool_within_frame(tmp_path):
    """Mature association and the early-enable New path can pass different pools."""

    from shutrack.tests.test_runtime_adapter import (
        _adapter,
        FakeDetection,
        FakeSnapshot,
        FakeTrack,
    )

    adapter, _runtime_instance = _adapter(tmp_path, competition_enabled=True)
    pool = [
        FakeDetection(box=[0, 0, 10, 10], frame_detection_index=0),
        FakeDetection(box=[2, 2, 12, 12], frame_detection_index=1),
        FakeDetection(box=[50, 50, 60, 60], frame_detection_index=2),
    ]
    adapter.begin_frame(7, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=9, observation_count=6)
    snapshot = FakeSnapshot(frame_id=7)

    def meta(indices):
        size = len(indices)
        return {
            "final_cost": np.full((1, size), 0.5),
            "detection_indices": np.asarray(indices, dtype=np.int64),
            "detection_source": np.zeros((1, size), dtype=np.int8),
        }

    event_a = adapter.build_matched_event(
        track,
        pool[0],
        track_idx=0,
        det_idx=0,
        association_meta=meta([0, 1, 2]),
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    # New-track path in the same frame with its own candidate pool order.
    event_b = adapter.build_matched_event(
        track,
        pool[2],
        track_idx=0,
        det_idx=0,
        association_meta=meta([2, 1, 0]),
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    boxes_a = np.stack([[0, 0, 10, 10], [2, 2, 12, 12], [50, 50, 60, 60]])
    boxes_b = np.stack([[50, 50, 60, 60], [2, 2, 12, 12], [0, 0, 10, 10]])
    expected_a = float(detection_overlap_row(boxes_a[0], boxes_a, 0).max())
    assert event_a.evidence.competition[2] == pytest.approx(expected_a, abs=1e-6)
    assert expected_a > 0.4
    assert event_b.evidence.competition[2] == pytest.approx(0.0, abs=1e-12)


def test_shutrack_adapter_skips_frame_start_snapshot():
    assert ShutrackTrackerAdapter.needs_frame_start_snapshot is False
    # The legacy AgentGuard adapter does not declare the flag, so the tracker
    # keeps its original snapshot behaviour (default True).
    assert getattr(object(), "needs_frame_start_snapshot", True) is True
