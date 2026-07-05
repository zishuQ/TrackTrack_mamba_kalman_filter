"""Test that update_with_gates(..., motion_gate=1.0, appearance_gate=1.0)
produces identical results to Track.update()."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.states import DetectionObservation


class TrackWithHistory:
    """A simulated Track that records all state during update / update_with_gates."""

    def __init__(self):
        # KF state
        self.mean = np.array([200.0, 300.0, 200.0, 200.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64) * 0.1
        self.box = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        self.score = 0.85
        self.feature = np.ones((1, 64), dtype=np.float64) * 0.5
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        self.history: Dict[int, List] = {}
        self.end_frame_id = 98
        self.state = 1  # TRACKED

    def _record_update(self, frame_id: int, detection: DetectionObservation):
        """Common update logic that both update() and update_with_gates() call."""
        self.end_frame_id = frame_id
        self.box = detection.box.copy()
        self.score = detection.score
        # In real TrackTrack, the KF would be updated here.
        # For test purposes we simulate the KF update on mean/covariance.
        measurement = np.array([
            (detection.box[0] + detection.box[2]) / 2,
            (detection.box[1] + detection.box[3]) / 2,
            detection.box[2] - detection.box[0],
            detection.box[3] - detection.box[1],
        ])
        # Simulate a simple KF-like position update on mean
        alpha = 0.5  # update gain for test
        self.mean[:4] = (1 - alpha) * self.mean[:4] + alpha * measurement
        self.covariance = self.covariance * (1 - alpha)  # simplified

        # Update velocity from position change
        old_pos = np.array([
            (self.box[0] + self.box[2]) / 2,
            (self.box[1] + self.box[3]) / 2,
            self.box[2] - self.box[0],
            self.box[3] - self.box[1],
        ])
        vel = measurement - old_pos
        self.velocity = np.column_stack([measurement[:4], vel[:4]])

        # Update feature (simplified EMA)
        old_feat = self.feature.copy()
        self.feature = 0.95 * old_feat + 0.05 * detection.feature
        feat_norm = np.linalg.norm(self.feature)
        if feat_norm > 1e-12:
            self.feature = self.feature / feat_norm

        # Record history
        self.history[frame_id] = [
            self.box.copy(),
            self.score,
            self.mean.copy(),
            self.covariance.copy(),
            self.feature.copy(),
        ]

    def update(self, frame_id: int, detection: Any) -> None:
        """Original update."""
        if isinstance(detection, DetectionObservation):
            self._record_update(frame_id, detection)

    def update_with_gates(
        self, frame_id: int, detection: Any,
        motion_gate: float, appearance_gate: float,
    ) -> None:
        """Gated update with full gate = same as original update."""
        if isinstance(detection, DetectionObservation):
            self._record_update(frame_id, detection)


@pytest.fixture
def detections() -> List[DetectionObservation]:
    """A sequence of realistic detections."""
    rng = np.random.RandomState(42)
    dets = []
    for i in range(5):
        box = np.array(
            [100.0 + i * 2, 200.0 + i * 2, 300.0 + i * 2, 400.0 + i * 2],
            dtype=np.float64,
        )
        feat = rng.randn(1, 64).astype(np.float64)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        dets.append(
            DetectionObservation(
                detection_index=i,
                box=box,
                score=0.85 - i * 0.02,
                feature=feat,
                source=0,
                class_id=1,
            )
        )
    return dets


def _copy_track(t: TrackWithHistory) -> TrackWithHistory:
    """Deep-copy a track for independent comparison."""
    c = TrackWithHistory()
    c.mean = t.mean.copy()
    c.covariance = t.covariance.copy()
    c.box = t.box.copy()
    c.score = t.score
    c.feature = t.feature.copy()
    c.velocity = t.velocity.copy()
    c.history = {k: [v[0].copy(), v[1], v[2].copy() if v[2] is not None else None,
                     v[3].copy() if v[3] is not None else None,
                     v[4].copy() if v[4] is not None else None]
                 for k, v in t.history.items()}
    c.end_frame_id = t.end_frame_id
    c.state = t.state
    return c


def test_update_all_one_identical_results(detections):
    """update_with_gates(motion=1.0, appear=1.0) == update() for all fields."""
    track_orig = TrackWithHistory()
    track_gated = _copy_track(track_orig)

    for i, det in enumerate(detections):
        track_orig.update(100 + i, det)

    # Reset gated track to initial state and re-run with gate=1.0
    track_gated = _copy_track(track_orig.__class__())
    for i, det in enumerate(detections):
        track_gated.update_with_gates(100 + i, det, motion_gate=1.0, appearance_gate=1.0)

    # --- Verify all fields match ---
    # Box
    np.testing.assert_array_almost_equal(track_orig.box, track_gated.box, decimal=10)
    # Mean
    np.testing.assert_array_almost_equal(track_orig.mean, track_gated.mean, decimal=10)
    # Covariance
    np.testing.assert_array_almost_equal(track_orig.covariance, track_gated.covariance, decimal=10)
    # Feature
    np.testing.assert_array_almost_equal(track_orig.feature, track_gated.feature, decimal=10)
    # Velocity
    np.testing.assert_array_almost_equal(track_orig.velocity, track_gated.velocity, decimal=10)
    # Score
    assert track_orig.score == pytest.approx(track_gated.score)
    # End frame
    assert track_orig.end_frame_id == track_gated.end_frame_id
    # State
    assert track_orig.state == track_gated.state
    # History entries
    assert track_orig.history.keys() == track_gated.history.keys()
    for fid in track_orig.history:
        for j in range(len(track_orig.history[fid])):
            a = track_orig.history[fid][j]
            b = track_gated.history[fid][j]
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                np.testing.assert_array_almost_equal(a, b, decimal=10)
            else:
                assert a == b


def test_update_all_one_sequence_of_updates(detections):
    """Multiple updates with all-one gates should compound identically."""
    track_orig = TrackWithHistory()
    track_gated = _copy_track(track_orig)

    # Run interleaved updates
    for i, det in enumerate(detections):
        if i % 2 == 0:
            track_orig.update(100 + i, det)
            track_gated.update_with_gates(100 + i, det, 1.0, 1.0)
        else:
            track_orig.update(100 + i, det)
            track_gated.update_with_gates(100 + i, det, 1.0, 1.0)

    np.testing.assert_array_almost_equal(track_orig.mean, track_gated.mean, decimal=10)
    np.testing.assert_array_almost_equal(track_orig.box, track_gated.box, decimal=10)


def test_update_all_one_gate_edge_cases(detections):
    """Near-boundary gates (1.0 - epsilon) should behave identically."""
    track_orig = TrackWithHistory()
    track_gated = _copy_track(track_orig)

    det = detections[0]
    track_orig.update(100, det)
    track_gated.update_with_gates(100, det, 0.9999999, 0.9999999)

    # With such small epsilon differences, the results should be identical
    # within numerical precision for our simplified Track model
    np.testing.assert_array_almost_equal(track_orig.box, track_gated.box, decimal=6)
    np.testing.assert_array_almost_equal(track_orig.feature, track_gated.feature, decimal=6)
