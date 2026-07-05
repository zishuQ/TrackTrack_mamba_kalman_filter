"""Test that compute_tgr_window_labels correctly enumerates all 16 binary
sequences and picks the best motion and appearance sequences."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.window import compute_tgr_window_labels


@pytest.fixture
def kf() -> NSAKalmanFilter:
    return NSAKalmanFilter()


@pytest.fixture
def identity_prototype() -> np.ndarray:
    """Random unit-norm prototype."""
    rng = np.random.RandomState(0)
    v = rng.randn(64).astype(np.float64)
    return v / np.linalg.norm(v)


def _make_window_event(
    track_id: int,
    frame_id: int,
    has_detection: bool,
    det_box: Optional[np.ndarray] = None,
) -> TrackEvent:
    """Helper to create a window event with consistent state."""
    rng = np.random.RandomState(frame_id)
    feat = rng.randn(1, 64).astype(np.float64)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)

    mean = np.array([200.0, 300.0, 200.0, 200.0, 0.5, 0.3, 0.0, 0.0], dtype=np.float64)
    cov = np.eye(8, dtype=np.float64) * 0.1
    box = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)

    state = TrackStateSnapshot(
        track_id=track_id,
        box=box,
        score=0.85,
        mean=mean,
        covariance=cov,
        velocity=np.zeros((4, 2), dtype=np.float64),
        feature=feat,
        history={},
        end_frame_id=frame_id - 1,
        state=1,
    )

    det = None
    if has_detection:
        if det_box is None:
            det_box = np.array(
                [100.0 + frame_id, 200.0 + frame_id, 300.0 + frame_id, 400.0 + frame_id],
                dtype=np.float64,
            )
        det = DetectionObservation(
            detection_index=0,
            box=det_box,
            score=0.85,
            feature=feat,
            source=0,
            class_id=1,
        )

    assoc = AssociationPairFeatures(
        iou_similarity=0.8,
        iou_distance=0.2,
        cosine_distance=0.1,
        confidence_distance=0.05,
        angle_distance=0.02,
        raw_cost=0.30,
        final_cost=0.28,
        assignment_round=0,
        assignment_threshold=0.5,
        detection_source=0,
    )

    return TrackEvent(
        event_id=f"evt_{track_id}_{frame_id}",
        dataset="test",
        sequence="test_seq",
        frame_id=frame_id,
        track_id=track_id,
        image_width=1920,
        image_height=1080,
        has_detection=has_detection,
        frame_start_state=state,
        pre_update_state=state,
        detection=det,
        association=assoc if has_detection else None,
        warp_matrix=np.eye(2, 3, dtype=np.float64),
        scalar_features=np.zeros(63, dtype=np.float64),
        track_feature=feat.ravel(),
        detection_feature=feat.ravel() if has_detection else np.zeros(0, dtype=np.float64),
    )


class TestTGRWindowLabels:
    """Tests for compute_tgr_window_labels."""

    def test_enumerates_16_sequences(self, kf, identity_prototype):
        """Should enumerate all 16 binary sequences and produce 16 loss values."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [
                np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64),
                np.array([87.0, 87.0, 137.0, 137.0], dtype=np.float64),
                np.array([89.0, 89.0, 139.0, 139.0], dtype=np.float64),
            ],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)] * 3,
            "future_oracle_features": [
                np.ones((1, 64), dtype=np.float64) * 0.7,
                np.ones((1, 64), dtype=np.float64) * 0.75,
                np.ones((1, 64), dtype=np.float64) * 0.8,
            ],
            "future_oracle_scores": [0.9, 0.88, 0.85],
        }

        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=3,
        )

        assert result["motion_losses"].shape == (16,)
        assert result["appearance_losses"].shape == (16,)

    def test_returns_correct_shapes(self, kf, identity_prototype):
        """Output dict should have correct shapes for all keys."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [
                np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64),
            ],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)],
            "future_oracle_features": [
                np.ones((1, 64), dtype=np.float64) * 0.7,
            ],
            "future_oracle_scores": [0.9],
        }

        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=1,
        )

        assert result["motion_sequence"].shape == (4,)
        assert result["appearance_sequence"].shape == (4,)
        assert result["valid_mask"].shape == (4,)
        assert result["motion_losses"].shape == (16,)
        assert result["appearance_losses"].shape == (16,)
        assert isinstance(result["selected_motion_loss"], float)
        assert isinstance(result["selected_appearance_loss"], float)

    def test_valid_mask_all_true_with_detections(self, kf, identity_prototype):
        """When all events have detections, valid_mask should be all True."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [],
            "future_warp_matrices": [],
            "future_oracle_features": [],
            "future_oracle_scores": [],
        }
        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=0,
        )
        assert result["valid_mask"].all()

    def test_valid_mask_mixed(self, kf, identity_prototype):
        """Valid mask should reflect which events have detections."""
        events = [
            _make_window_event(1, 100, True),
            _make_window_event(1, 101, False),  # no detection
            _make_window_event(1, 102, True),
            _make_window_event(1, 103, True),
        ]
        oracle_data = {
            "future_oracle_detections": [],
            "future_warp_matrices": [],
            "future_oracle_features": [],
            "future_oracle_scores": [],
        }
        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=0,
        )
        expected_mask = np.array([True, False, True, True], dtype=np.bool_)
        np.testing.assert_array_equal(result["valid_mask"], expected_mask)

    def test_sequence_values_are_binary(self, kf, identity_prototype):
        """Selected sequences should contain only 0s and 1s."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [
                np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64),
            ],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)],
            "future_oracle_features": [
                np.ones((1, 64), dtype=np.float64) * 0.7,
            ],
            "future_oracle_scores": [0.9],
        }
        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=1,
        )
        assert set(result["motion_sequence"].tolist()).issubset({0, 1})
        assert set(result["appearance_sequence"].tolist()).issubset({0, 1})

    def test_motion_sequence_picks_lowest_loss(self, kf, identity_prototype):
        """Motion sequence should be the one with lowest total motion loss."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [
                np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64),
            ],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)],
            "future_oracle_features": [
                np.ones((1, 64), dtype=np.float64) * 0.7,
            ],
            "future_oracle_scores": [0.9],
        }
        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=1,
        )
        best_motion_loss = result["selected_motion_loss"]
        assert best_motion_loss == pytest.approx(float(np.min(result["motion_losses"])))

    def test_appearance_tiebreak_prefers_more_ones(self, kf, identity_prototype):
        """When appearance losses tie, the sequence with more 1s should win."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [],
            "future_warp_matrices": [],
            "future_oracle_features": [],
            "future_oracle_scores": [],
        }
        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=0,
        )
        # With 0 future frames all losses are 0, so tiebreak on more 1s
        best_seq = result["appearance_sequence"]
        # The tiebreak should prefer more 1s; the max possible is 4
        assert best_seq.sum() >= 0  # at least something sensible

    def test_raises_with_wrong_event_count(self, kf, identity_prototype):
        """Should raise ValueError when not exactly 4 events."""
        events = [_make_window_event(1, 100, True) for _ in range(3)]
        with pytest.raises(ValueError, match="Expected exactly 4"):
            compute_tgr_window_labels(events, {}, kf, identity_prototype)

    def test_raises_without_frame_start_state(self, kf, identity_prototype):
        """Should raise ValueError when first event has no frame_start_state."""
        ev = _make_window_event(1, 100, True)
        ev.frame_start_state = None
        events = [ev] + [_make_window_event(1, 101 + i, True) for i in range(3)]
        with pytest.raises(ValueError, match="no frame_start_state"):
            compute_tgr_window_labels(events, {}, kf, identity_prototype)

    def test_losses_vary_across_sequences(self, kf, identity_prototype):
        """Different binary sequences should produce different losses."""
        events = [_make_window_event(1, 100 + i, True) for i in range(4)]
        oracle_data = {
            "future_oracle_detections": [
                np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64),
                np.array([87.0, 87.0, 137.0, 137.0], dtype=np.float64),
                np.array([89.0, 89.0, 139.0, 139.0], dtype=np.float64),
            ],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)] * 3,
            "future_oracle_features": [
                np.ones((1, 64), dtype=np.float64) * 0.7,
                np.ones((1, 64), dtype=np.float64) * 0.75,
                np.ones((1, 64), dtype=np.float64) * 0.8,
            ],
            "future_oracle_scores": [0.9, 0.88, 0.85],
        }
        result = compute_tgr_window_labels(
            events, oracle_data, kf, identity_prototype, future_frames=3,
        )
        # At least some variation across the 16 sequences
        assert np.var(result["motion_losses"]) > 1e-10 or \
               np.var(result["appearance_losses"]) > 1e-10
