"""Test that compute_motion_benefit produces non-zero differences for events with GT.

When an event has a detection and future oracle data exists, the write branch
(detection applied) and skip branch (detection ignored) should produce
different future losses, resulting in a non-zero motion benefit.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import DetectionObservation, TrackStateSnapshot
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.motion import compute_motion_benefit


# ---------------------------------------------------------------------------
#  Helper: build a RolloutContext from old-style (event, oracle_data)
# ---------------------------------------------------------------------------


def _build_rollout_context(
    event: TrackEvent,
    oracle_data: dict,
) -> RolloutContext:
    """Convert old-style (event, oracle_data) arguments to a RolloutContext.

    The *oracle_data* dict may contain:
        future_gt_boxes         — list of (4,) boxes or None (default [None]*5)
        future_oracle_detections — list of (4,) raw boxes or None
        future_oracle_features   — list of (1, D) features or None
        future_oracle_scores     — list of float scores
        future_warp_matrices     — list of (2, 3) warp matrices
        current_gt_box           — (4,) current GT box

    Raw oracle boxes are converted to ``DetectionObservation`` objects with
    the corresponding feature and score (or defaults).
    """
    # Use explicit future_gt_boxes if provided; otherwise fall back to
    # future_oracle_detections (old-style) as the GT boxes for loss computation.
    future_gt = oracle_data.get("future_gt_boxes")
    if future_gt is None:
        future_gt = oracle_data.get("future_oracle_detections", [None] * 5)

    raw_dets = oracle_data.get("future_oracle_detections", [None] * 5)
    raw_feats = oracle_data.get("future_oracle_features", [None] * 5)
    raw_scores = oracle_data.get("future_oracle_scores", [0.0] * 5)
    future_warps = oracle_data.get("future_warp_matrices", [np.eye(2, 3)] * 5)
    current_gt = oracle_data.get("current_gt_box")

    if current_gt is None:
        current_gt = np.array([0, 0, 0, 0], dtype=np.float64)

    # Convert raw boxes → DetectionObservation objects
    oracle_dets: List[Optional[DetectionObservation]] = []
    for i in range(len(raw_dets)):
        box = raw_dets[i]
        if box is None:
            oracle_dets.append(None)
        else:
            feat = raw_feats[i] if i < len(raw_feats) and raw_feats[i] is not None else np.zeros((1, 64), dtype=np.float64)
            score = raw_scores[i] if i < len(raw_scores) else 0.0
            oracle_dets.append(
                DetectionObservation(
                    detection_index=i,
                    box=box,
                    score=score,
                    feature=feat,
                    source=0,
                    class_id=1,
                )
            )

    return RolloutContext(
        frame_id=event.frame_id,
        target_gt_id=getattr(event, "target_gt_id", -1),
        pre_update_state=event.pre_update_state,
        current_candidate=event.detection if event.has_detection else None,
        current_gt_box=current_gt,
        future_gt_boxes=future_gt,
        future_oracle_detections=oracle_dets,
        future_warp_matrices=future_warps,
        identity_prototype=None,
        appearance_alpha=0.95,
    )


@pytest.fixture
def motion_kf() -> NSAKalmanFilter:
    return NSAKalmanFilter()


def test_motion_benefit_nonzero_with_gt(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """Motion benefit should be non-zero when detection and oracle data exist."""
    ctx = _build_rollout_context(mock_event, mock_future_oracle_data)
    B_m, write_losses, skip_losses, valid_mask = compute_motion_benefit(
        ctx, motion_kf, future_frames=5,
    )
    # At least one branch should differ from the other
    assert not np.allclose(sum(write_losses), sum(skip_losses)), \
        "Write and skip branch losses should differ"
    # Benefit should be non-zero
    assert B_m != pytest.approx(0.0, abs=1e-10), \
        "Motion benefit should be non-zero with detection and oracle GT"
    assert valid_mask.dtype == bool


def test_motion_benefit_shapes(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """Output shapes should be consistent with future_frames."""
    future_frames = 3
    ctx = _build_rollout_context(mock_event, mock_future_oracle_data)
    B_m, write_losses, skip_losses, valid_mask = compute_motion_benefit(
        ctx, motion_kf, future_frames=future_frames,
    )
    assert len(write_losses) == future_frames
    assert len(skip_losses) == future_frames
    assert len(valid_mask) == future_frames
    assert valid_mask.dtype == bool
    assert isinstance(B_m, float)


def test_motion_benefit_losses_nonnegative(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """All per-frame losses should be non-negative."""
    ctx = _build_rollout_context(mock_event, mock_future_oracle_data)
    _, write_losses, skip_losses, _ = compute_motion_benefit(
        ctx, motion_kf, future_frames=5,
    )
    for wl in write_losses:
        assert wl >= 0.0, f"Write loss {wl} should be >= 0"
    for sl in skip_losses:
        assert sl >= 0.0, f"Skip loss {sl} should be >= 0"


def test_motion_benefit_no_detection(
    mock_event_no_detection: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """Without detection, write and skip branches should be identical -> benefit ~0."""
    ctx = _build_rollout_context(mock_event_no_detection, mock_future_oracle_data)
    B_m, write_losses, skip_losses, _ = compute_motion_benefit(
        ctx, motion_kf, future_frames=3,
    )
    assert abs(B_m) < 1e-10
    assert write_losses == pytest.approx(skip_losses)


def test_motion_benefit_no_oracle(
    mock_event: TrackEvent,
    motion_kf: NSAKalmanFilter,
):
    """Without oracle data, all losses should be zero."""
    empty_oracle: Dict[str, Any] = {
        "future_oracle_detections": [],
        "future_warp_matrices": [],
    }
    ctx = _build_rollout_context(mock_event, empty_oracle)
    B_m, write_losses, skip_losses, valid_mask = compute_motion_benefit(
        ctx, motion_kf, future_frames=5,
    )
    assert B_m == 0.0
    assert all(wl == 0.0 for wl in write_losses)
    assert all(sl == 0.0 for sl in skip_losses)
    assert len(valid_mask) == 0


def test_motion_benefit_with_oracle_mismatch(
    mock_event: TrackEvent,
    motion_kf: NSAKalmanFilter,
):
    """When oracle detection diverges from current detection, write branch
    should have different loss than skip branch."""
    # Oracle data where box moves far away from current detection
    oracle_data: Dict[str, Any] = {
        "future_oracle_detections": [
            np.array([500.0, 500.0, 600.0, 600.0], dtype=np.float64),
            np.array([520.0, 520.0, 620.0, 620.0], dtype=np.float64),
        ],
        "future_warp_matrices": [
            np.eye(2, 3, dtype=np.float64),
            np.eye(2, 3, dtype=np.float64),
        ],
    }
    ctx = _build_rollout_context(mock_event, oracle_data)
    B_m, write_losses, skip_losses, _ = compute_motion_benefit(
        ctx, motion_kf, future_frames=2,
    )
    assert abs(B_m) > 1e-6, "Benefit should be non-zero with mismatched oracle"


def test_motion_benefit_valid_mask(
    mock_event: TrackEvent,
    mock_future_oracle_data: dict,
    motion_kf: NSAKalmanFilter,
):
    """Valid mask should be a boolean array matching future_frames."""
    ctx = _build_rollout_context(mock_event, mock_future_oracle_data)
    _, _, _, valid_mask = compute_motion_benefit(
        ctx, motion_kf, future_frames=3,
    )
    assert valid_mask.shape == (3,)
    assert valid_mask.dtype == bool


def test_motion_benefit_truncates_to_available_data(
    mock_event: TrackEvent,
    motion_kf: NSAKalmanFilter,
):
    """If oracle has fewer frames than future_frames, should use min()."""
    oracle_data: Dict[str, Any] = {
        "future_oracle_detections": [
            np.array([80.0, 80.0, 130.0, 130.0], dtype=np.float64),
        ],
        "future_warp_matrices": [
            np.eye(2, 3, dtype=np.float64),
        ],
    }
    ctx = _build_rollout_context(mock_event, oracle_data)
    B_m, write_losses, skip_losses, valid_mask = compute_motion_benefit(
        ctx, motion_kf, future_frames=10,
    )
    assert len(write_losses) == 1
    assert len(skip_losses) == 1
    assert len(valid_mask) == 1


def test_motion_benefit_pre_state_none_raises(motion_kf):
    """Event with no pre_update_state should raise ValueError."""
    ev = TrackEvent(
        event_id="no_state",
        dataset="test", sequence="test", frame_id=0, track_id=0,
        image_width=640, image_height=480, has_detection=False,
    )
    ctx = _build_rollout_context(ev, {})
    with pytest.raises(ValueError, match="no pre_update_state"):
        compute_motion_benefit(ctx, motion_kf)
