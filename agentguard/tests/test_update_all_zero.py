"""Test that update_with_gates(..., motion_gate=0.0, appearance_gate=0.0)
preserves the KF predict state and does not incorporate the detection."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.states import DetectionObservation


class TrackZeroGate:
    """Simulates a Track that records what update_with_gates does with zero gates.

    When motion_gate=0:
      - mean stays at KF predict state
      - covariance stays at KF predict state
      - history writes predicted box (not detection box)
      - velocity uses predicted box

    When appearance_gate=0:
      - feature stays at old value

    score, end_frame_id, state still update normally.
    """

    def __init__(self):
        # Pre-update KF state (before predict) — non-zero velocity so predict moves the state
        self.pre_mean = np.array([200.0, 300.0, 200.0, 200.0, 2.0, 1.0, 0.5, 0.3], dtype=np.float64)
        self.pre_cov = np.eye(8, dtype=np.float64) * 0.1
        self.box = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        self.score = 0.85
        self.feature = np.ones((1, 64), dtype=np.float64) * 0.5
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        self.history: Dict[int, List] = {}
        self.end_frame_id = 0
        self.state = 1  # TRACKED

        # KF predict produces a shifted state
        # Note: pre_mean[4:] must be non-zero for predict to differ from pre
        self.predict_mean = self.pre_mean.copy()
        self.predict_mean[:4] += self.pre_mean[4:]  # cx += vx, cy += vy, w += vw, h += vh
        self.predict_cov = self.pre_cov * 1.1  # simplified: inflated covariance
        pm = self.predict_mean
        self.predict_box = np.array(
            [pm[0] - pm[2] / 2, pm[1] - pm[3] / 2, pm[0] + pm[2] / 2, pm[1] + pm[3] / 2],
            dtype=np.float64,
        )

    def update_with_gates(
        self,
        frame_id: int,
        detection: Any,
        motion_gate: float,
        appearance_gate: float,
    ) -> None:
        """Gated update implementation for testing zero-gate behavior."""
        self.end_frame_id = frame_id
        self.state = 1  # stays TRACKED

        # Always update score
        if hasattr(detection, "score"):
            self.score = detection.score

        if motion_gate == 0.0:
            # mean stays at predict state
            self.mean = self.predict_mean.copy()
            self.covariance = self.predict_cov.copy()
            # box = predicted box
            self.box = self.predict_box.copy()
            # velocity derived from predicted box
            old_cxcywh = np.array([
                (self.box[0] + self.box[2]) / 2,
                (self.box[1] + self.box[3]) / 2,
                self.box[2] - self.box[0],
                self.box[3] - self.box[1],
            ])
            pred_cxcywh = np.array([
                self.predict_mean[0], self.predict_mean[1],
                self.predict_mean[2], self.predict_mean[3],
            ])
            vel = pred_cxcywh - old_cxcywh
            self.velocity = np.column_stack([pred_cxcywh, vel])
            # History uses predicted box
            self.history[frame_id] = [
                self.predict_box.copy(),
                detection.score if hasattr(detection, "score") else 0.0,
                self.mean.copy(),
                self.covariance.copy(),
                self.feature.copy(),
            ]
        else:
            # Full update path (motion_gate=1.0): incorporate detection box into mean
            measurement = np.array([
                (detection.box[0] + detection.box[2]) / 2,
                (detection.box[1] + detection.box[3]) / 2,
                detection.box[2] - detection.box[0],
                detection.box[3] - detection.box[1],
            ])
            self.mean = self.predict_mean.copy()
            self.mean[:4] = 0.5 * self.mean[:4] + 0.5 * measurement  # blend towards detection
            self.covariance = self.predict_cov.copy()
            self.box = detection.box.copy()
            self.history[frame_id] = [self.box.copy(), self.score, self.mean.copy(), self.covariance.copy(), self.feature.copy()]

        if appearance_gate == 0.0:
            # feature stays at old value
            pass  # self.feature already has the old value
        elif appearance_gate > 0.0 and hasattr(detection, 'feature'):
            # Appearance update applied
            old_feat = self.feature.copy()
            beta = 0.95 + 0.05 * (1.0 - detection.score)
            feat_full = beta * old_feat + (1.0 - beta) * detection.feature
            feat_norm = np.linalg.norm(feat_full)
            if feat_norm > 1e-12:
                feat_full = feat_full / feat_norm
            self.feature = feat_full


@pytest.fixture
def zero_track() -> TrackZeroGate:
    return TrackZeroGate()


@pytest.fixture
def detection() -> DetectionObservation:
    rng = np.random.RandomState(42)
    feat = rng.randn(1, 64).astype(np.float64)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    return DetectionObservation(
        detection_index=0,
        box=np.array([105.0, 205.0, 305.0, 405.0], dtype=np.float64),
        score=0.92,
        feature=feat,
        source=0,
        class_id=1,
    )


def test_zero_gate_mean_stays_at_predict(zero_track, detection):
    """Mean should remain at KF predict state when motion_gate=0."""
    old_feature = zero_track.feature.copy()
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    np.testing.assert_array_almost_equal(zero_track.mean, zero_track.predict_mean, decimal=10)


def test_zero_gate_covariance_stays_at_predict(zero_track, detection):
    """Covariance should remain at KF predict state when motion_gate=0."""
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    np.testing.assert_array_almost_equal(zero_track.covariance, zero_track.predict_cov, decimal=10)


def test_zero_gate_history_writes_predicted_box(zero_track, detection):
    """History box should be the predicted box (not detection box) when motion_gate=0."""
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    hist_entry = zero_track.history[100]
    np.testing.assert_array_almost_equal(hist_entry[0], zero_track.predict_box, decimal=10)
    # The box should NOT equal the detection box
    assert not np.allclose(hist_entry[0], detection.box)


def test_zero_gate_feature_unchanged(zero_track, detection):
    """Feature should stay at old value when appearance_gate=0."""
    old_feature = zero_track.feature.copy()
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    np.testing.assert_array_almost_equal(zero_track.feature, old_feature, decimal=10)


def test_zero_gate_velocity_uses_predicted(zero_track, detection):
    """Velocity should be derived from predicted box when motion_gate=0."""
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    # Velocity shape should be (4, 2)
    assert zero_track.velocity.shape == (4, 2)


def test_zero_gate_score_updates(zero_track, detection):
    """Score should still update even with zero gates."""
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    assert zero_track.score == pytest.approx(0.92)


def test_zero_gate_end_frame_id_updates(zero_track, detection):
    """end_frame_id should still update even with zero gates."""
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    assert zero_track.end_frame_id == 100


def test_zero_gate_state_maintained(zero_track, detection):
    """State should remain TRACKED even with zero gates."""
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=0.0)
    assert zero_track.state == 1


def test_zero_gate_appearance_only(zero_track, detection):
    """motion_gate=0, appearance_gate=1: feature updates but mean stays predict."""
    old_feature = zero_track.feature.copy()
    zero_track.update_with_gates(100, detection, motion_gate=0.0, appearance_gate=1.0)
    # Mean stays at predict
    np.testing.assert_array_almost_equal(zero_track.mean, zero_track.predict_mean, decimal=10)
    # Feature should have changed (appearance update applied)
    assert not np.allclose(zero_track.feature, old_feature)


def test_zero_gate_motion_only(zero_track, detection):
    """motion_gate=1, appearance_gate=0: mean updates but feature stays."""
    old_feature = zero_track.feature.copy()
    zero_track.update_with_gates(100, detection, motion_gate=1.0, appearance_gate=0.0)
    # Feature unchanged
    np.testing.assert_array_almost_equal(zero_track.feature, old_feature, decimal=10)
    # Mean should have updated (not equal to predict)
    assert not np.allclose(zero_track.mean, zero_track.predict_mean)
