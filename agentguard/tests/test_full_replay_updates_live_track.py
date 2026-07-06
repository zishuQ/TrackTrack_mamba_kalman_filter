"""Tests that full replay correctly updates live track state."""

import numpy as np
import pytest

from agentguard.contracts.states import (
    TrackStateSnapshot,
    DetectionObservation,
)
from agentguard.runtime.replay import ReplayPlan, ReplayStep


def test_full_replay_updates_live_track_state():
    """Verify that after full replay the live track's state reflects both
    matched and unmatched steps."""
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    detection = DetectionObservation(
        detection_index=0,
        box=np.array([100, 100, 200, 200], dtype=np.float64),
        score=0.9,
        feature=np.ones((1, 64), dtype=np.float64) / 8.0,
        source=0,
        class_id=1,
    )

    snapshot = TrackStateSnapshot(
        track_id=1,
        box=np.array([50, 50, 150, 150], dtype=np.float64),
        score=0.8,
        mean=np.array([100, 100, 100, 100, 0, 0, 0, 0], dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64),
        velocity=np.zeros((4, 2), dtype=np.float64),
        feature=np.ones((1, 64), dtype=np.float64) / 8.0,
        history={},
        end_frame_id=5,
        state=1,
    )

    steps = [
        ReplayStep(6, np.eye(2, 3), True, detection, 1.0, 1.0),
        ReplayStep(7, np.eye(2, 3), True, detection, 0.0, 0.0),
        ReplayStep(8, np.eye(2, 3), False, None, 0.0, 0.0),
    ]
    plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=steps)

    class SpyTrack:
        def __init__(self):
            self.calls = []
            self.mean = None
            self.covariance = None
            self.box = np.zeros(4)
            self.score = 0.0
            self.feat = np.ones((1, 64), dtype=np.float64) / 8.0
            self.history = {}
            self.end_frame_id = 0
            self.state = 1
            self.track_id = 1
            self.velocity = np.zeros((4, 2))
            self.kalman_filter = _DummyKF()

        def predict(self):
            self.calls.append("predict")

        def update_with_gates(self, fid, det, mg, ag):
            self.calls.append(("update_with_gates", fid, mg, ag))
            self.end_frame_id = fid
            self.state = 1

        def mark_lost(self):
            self.calls.append("mark_lost")
            self.state = 2

    class _DummyKF:
        def predict(self, m, c):
            return m, c

        def update(self, m, c, meas, score):
            return m, c

        def initiate(self, cxcywh):
            return np.zeros(8), np.eye(8)

    spy = SpyTrack()
    backend = TrackTrackReplayBackend()
    backend.replay_into_live_track(spy, plan)

    assert len(spy.calls) == 6
    assert spy.state == 2
    # After step 3 (unmatched): end_frame_id unchanged from step 2's value (7)
    assert spy.end_frame_id == 7


def test_all_one_replay_preserves_original_tracktrack_behavior():
    """Gate = 1.0 replay should be equivalent to standard Track.update()."""
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    detection = DetectionObservation(
        detection_index=0,
        box=np.array([100, 100, 200, 200], dtype=np.float64),
        score=0.9,
        feature=np.ones((1, 64), dtype=np.float64) / 8.0,
        source=0,
        class_id=1,
    )

    snapshot = TrackStateSnapshot(
        track_id=1,
        box=np.array([100, 100, 200, 200], dtype=np.float64),
        score=0.9,
        mean=np.array([150, 150, 100, 100, 0, 0, 0, 0], dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64) * 0.1,
        velocity=np.zeros((4, 2)),
        feature=np.ones((1, 64), dtype=np.float64) / 8.0,
        history={},
        end_frame_id=10,
        state=1,
    )

    plan = ReplayPlan(
        track_id=1,
        checkpoint=snapshot,
        steps=[ReplayStep(11, np.eye(2, 3), True, detection, 1.0, 1.0)],
    )

    class SpyTrack:
        def __init__(self):
            self.calls = []
            self.mean = np.array([150, 150, 100, 100, 0, 0, 0, 0], dtype=np.float64)
            self.covariance = np.eye(8, dtype=np.float64) * 0.1
            self.box = np.array([100, 100, 200, 200], dtype=np.float64)
            self.score = 0.9
            self.feat = np.ones((1, 64), dtype=np.float64) / 8.0
            self.history = {}
            self.end_frame_id = 10
            self.state = 1
            self.track_id = 1
            self.velocity = np.zeros((4, 2))
            self.kalman_filter = _DummyKF()

        def predict(self): pass

        def update_with_gates(self, fid, det, mg, ag):
            self.calls.append(("update_with_gates", mg, ag))
            self.end_frame_id = fid

    class _DummyKF:
        def predict(self, m, c): return m, c
        def update(self, m, c, meas, score): return m, c
        def initiate(self, cxcywh): return np.zeros(8), np.eye(8)

    spy = SpyTrack()
    backend = TrackTrackReplayBackend()
    backend.replay_into_live_track(spy, plan)

    assert spy.calls[0] == ("update_with_gates", 1.0, 1.0)
    assert spy.end_frame_id == 11
