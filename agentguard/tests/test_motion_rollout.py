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
from agentguard.contracts.states import TrackStateSnapshot
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.motion import compute_motion_benefit


@pytest.fixture
def motion_kf() -> NSAKalmanFilter:
    return NSAKalmanFilter()


def test_motion_benefit_nonzero_with_gt(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """Motion benefit should be non-zero when detection and oracle data exist."""
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event, mock_future_oracle_data, motion_kf, future_frames=5,
    )
    # At least one branch should differ from the other
    assert not np.allclose(sum(write_losses), sum(skip_losses)), \
        "Write and skip branch losses should differ"
    # Benefit should be non-zero
    assert B_m != pytest.approx(0.0, abs=1e-10), \
        "Motion benefit should be non-zero with detection and oracle GT"


def test_motion_benefit_shapes(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """Output shapes should be consistent with future_frames."""
    future_frames = 3
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event, mock_future_oracle_data, motion_kf, future_frames=future_frames,
    )
    assert len(write_losses) == future_frames
    assert len(skip_losses) == future_frames
    assert len(write_boxes) == future_frames
    assert len(skip_boxes) == future_frames
    assert isinstance(B_m, float)


def test_motion_benefit_losses_nonnegative(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    motion_kf: NSAKalmanFilter,
):
    """All per-frame losses should be non-negative."""
    _, write_losses, skip_losses, _, _ = compute_motion_benefit(
        mock_event, mock_future_oracle_data, motion_kf, future_frames=5,
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
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event_no_detection, mock_future_oracle_data, motion_kf, future_frames=3,
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
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event, empty_oracle, motion_kf, future_frames=5,
    )
    assert B_m == 0.0
    assert all(wl == 0.0 for wl in write_losses)
    assert all(sl == 0.0 for sl in skip_losses)


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
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event, oracle_data, motion_kf, future_frames=2,
    )
    assert abs(B_m) > 1e-6, "Benefit should be non-zero with mismatched oracle"


def test_motion_benefit_oracle_boxes_shape(
    mock_event: TrackEvent,
    mock_future_oracle_data: dict,
    motion_kf: NSAKalmanFilter,
):
    """Oracle boxes should be returned as (4,) arrays."""
    _, _, _, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event, mock_future_oracle_data, motion_kf, future_frames=3,
    )
    for wb in write_boxes:
        assert wb.shape == (4,), f"Expected (4,) got {wb.shape}"
    for sb in skip_boxes:
        assert sb.shape == (4,), f"Expected (4,) got {sb.shape}"


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
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        mock_event, oracle_data, motion_kf, future_frames=10,
    )
    assert len(write_losses) == 1
    assert len(skip_losses) == 1


def test_motion_benefit_pre_state_none_raises(motion_kf):
    """Event with no pre_update_state should raise ValueError."""
    ev = TrackEvent(
        event_id="no_state",
        dataset="test", sequence="test", frame_id=0, track_id=0,
        image_width=640, image_height=480, has_detection=False,
    )
    with pytest.raises(ValueError, match="no pre_update_state"):
        compute_motion_benefit(ev, {}, motion_kf)
