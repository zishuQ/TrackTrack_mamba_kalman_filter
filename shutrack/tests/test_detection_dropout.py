"""Detection dropout must keep boxes and stable detection IDs aligned."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shutrack.integration.tracktrack import ShutrackTrackerAdapter
from shutrack.runtime.adapter import ShutrackRuntime, ShutrackRuntimeConfig
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.tests.conftest import (
    FakeDetection,
    FakeSnapshot,
    FakeTrack,
    write_tiny_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_DIR = REPO_ROOT / "3. Tracker"
if str(TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKER_DIR))

import run as tracker_run  # noqa: E402


def _seed_for_mask(mask: list[bool], *, rate: float = 0.5) -> int:
    for seed in range(10000):
        np.random.seed(seed)
        if (np.random.random(len(mask)) >= rate).tolist() == mask:
            return seed
    raise AssertionError(f"no seed produced mask {mask}")


def _detection_rows() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0, 0.9, 0.0, 1.0, 1.0],
            [0.0, 0.0, 10.0, 10.0, 0.8, 0.0, 1.0, 1.0],
            [100.0, 100.0, 110.0, 110.0, 0.7, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_dropout_filters_rows_and_target_indices_with_same_mask():
    rows = _detection_rows()
    indices = np.array([11, 12, 13], dtype=np.int64)
    seed = _seed_for_mask([True, False, True])
    np.random.seed(seed)
    filtered_rows, filtered_indices, keep = (
        tracker_run.inject_noise_with_detection_indices(
            rows,
            indices,
            drop_rate=0.5,
        )
    )
    assert keep.tolist() == [True, False, True]
    assert filtered_indices.tolist() == [11, 13]
    np.testing.assert_allclose(filtered_rows, rows[[0, 2]])
    # The kept IDs still describe their own boxes.
    np.testing.assert_allclose(filtered_rows[0], rows[0])
    np.testing.assert_allclose(filtered_rows[1], rows[2])


def test_dropout_zero_keeps_rows_and_indices_unchanged():
    rows = _detection_rows()
    indices = np.array([11, 12, 13], dtype=np.int64)
    filtered_rows, filtered_indices, keep = (
        tracker_run.inject_noise_with_detection_indices(
            rows,
            indices,
            drop_rate=0.0,
        )
    )
    assert keep.tolist() == [True, True, True]
    assert filtered_indices.tolist() == [11, 12, 13]
    np.testing.assert_allclose(filtered_rows, rows)


def test_competition_max_iou_uses_aligned_filtered_boxes(tmp_path):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path, competition_enabled=True)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    runtime = ShutrackRuntime(
        model,
        checkpoint,
        device="cpu",
        config=ShutrackRuntimeConfig(min_observations=1),
    )
    adapter = ShutrackTrackerAdapter(
        SimpleNamespace(img_w=640, img_h=480, shutrack_early_enable=False),
        "tiny",
        runtime=runtime,
    )
    association_meta = {
        "raw_cost": np.array([[0.2, 0.9]], dtype=np.float64),
        "final_cost": np.array([[0.2, 0.9]], dtype=np.float64),
        "iou_similarity": np.ones((1, 2)),
        "detection_indices": np.array([11, 13], dtype=np.int64),
        "detection_source": np.zeros((1, 2), dtype=np.int8),
    }
    snapshot = FakeSnapshot(frame_id=2)
    track = FakeTrack(track_id=1, observation_count=6)

    # Correct alignment: accepted ID 11 has box A, remaining ID 13 has the
    # far box C -> max IoU is 0.
    aligned_pool = [
        FakeDetection(box=[0, 0, 10, 10], frame_detection_index=11),
        FakeDetection(box=[100, 100, 110, 110], frame_detection_index=13),
    ]
    adapter.begin_frame(2, 640, 480, detection_pool=aligned_pool)
    aligned_event = adapter.build_matched_event(
        track,
        aligned_pool[0],
        track_idx=0,
        det_idx=0,
        association_meta=association_meta,
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    assert aligned_event.evidence.competition[2] == 0.0
    assert aligned_event.evidence.competition[0] == np.float32(0.7)

    # Control: if ID 13 were paired with the overlapping middle box B, the
    # same computation would report IoU 1.0.  This proves the assertion above
    # is sensitive to box/ID misalignment, not just array lengths.
    misaligned_pool = [
        FakeDetection(box=[0, 0, 10, 10], frame_detection_index=11),
        FakeDetection(box=[0, 0, 10, 10], frame_detection_index=13),
    ]
    adapter.begin_frame(3, 640, 480, detection_pool=misaligned_pool)
    misaligned_event = adapter.build_matched_event(
        track,
        misaligned_pool[0],
        track_idx=0,
        det_idx=0,
        association_meta=association_meta,
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    assert misaligned_event.evidence.competition[2] == np.float32(1.0)
