"""Tests that replay_backend calls real Track methods directly, not ReplayEngine."""

import numpy as np
import pytest

from agentguard.contracts.states import (
    TrackStateSnapshot,
    DetectionObservation,
)
from agentguard.runtime.replay import ReplayPlan, ReplayStep


def _make_minimal_snapshot(track_id=1):
    return TrackStateSnapshot(
        track_id=track_id,
        box=np.array([100, 200, 300, 400], dtype=np.float64),
        score=0.9,
        mean=np.array([200, 300, 200, 200, 1, 0.5, 0, 0], dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64) * 0.1,
        velocity=np.zeros((4, 2), dtype=np.float64),
        feature=np.random.RandomState(42).randn(1, 64).astype(np.float64),
        history={},
        end_frame_id=10,
        state=1,
    )


def _make_detection():
    return DetectionObservation(
        detection_index=0,
        box=np.array([105, 205, 310, 410], dtype=np.float64),
        score=0.85,
        feature=np.random.RandomState(7).randn(1, 64).astype(np.float64),
        source=0,
        class_id=1,
    )


class SpyTrack:
    """A Track-like object that records which methods were called."""

    def __init__(self):
        self.calls = []
        self.mean = None
        self.covariance = None
        self.feat = np.zeros((1, 64))
        self.box = np.zeros(4)
        self.score = 0.0
        self.history = {}
        self.end_frame_id = 0
        self.state = 1
        self.track_id = 1
        self.velocity = np.zeros((4, 2))
        self.kalman_filter = None

    def predict(self):
        self.calls.append("predict")

    def update_with_gates(self, frame_id, detection, motion_gate, appearance_gate):
        self.calls.append(
            ("update_with_gates", frame_id, motion_gate, appearance_gate)
        )
        self.end_frame_id = frame_id

    def mark_lost(self):
        self.calls.append("mark_lost")
        self.state = 2

    def restore_state(self, snapshot):
        self.calls.append("restore_state")
        self.end_frame_id = snapshot.end_frame_id
        self.state = snapshot.state


def test_replay_backend_calls_track_methods_directly():
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    backend = TrackTrackReplayBackend()

    snapshot = _make_minimal_snapshot()
    detection = _make_detection()

    steps = [
        ReplayStep(
            frame_id=15,
            warp_matrix=np.eye(2, 3, dtype=np.float64),
            has_detection=True,
            detection=detection,
            motion_gate=0.8,
            appearance_gate=0.6,
        ),
        ReplayStep(
            frame_id=16,
            warp_matrix=np.eye(2, 3, dtype=np.float64),
            has_detection=False,
            detection=None,
            motion_gate=0.0,
            appearance_gate=0.0,
        ),
    ]
    plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=steps)

    spy = SpyTrack()
    backend.replay_into_live_track(spy, plan)

    assert "predict" in spy.calls
    assert spy.calls.count("predict") == 2

    update_calls = [c for c in spy.calls if isinstance(c, tuple) and c[0] == "update_with_gates"]
    assert len(update_calls) == 1
    assert update_calls[0][1] == 15
    assert update_calls[0][2] == 0.8
    assert update_calls[0][3] == 0.6

    assert spy.calls.count("mark_lost") == 1
    assert spy.state == 2


def test_replay_backend_unmatched_preserves_end_frame_id():
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    backend = TrackTrackReplayBackend()

    snapshot = _make_minimal_snapshot()
    steps = [
        ReplayStep(
            frame_id=20,
            warp_matrix=np.eye(2, 3, dtype=np.float64),
            has_detection=False,
            detection=None,
            motion_gate=0.0,
            appearance_gate=0.0,
        ),
    ]
    plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=steps)

    spy = SpyTrack()
    backend.replay_into_live_track(spy, plan)

    assert spy.state == 2
    assert spy.end_frame_id == 20


def test_replay_backend_does_not_import_replay_engine():
    import integrations.agentguard.replay_backend as rb
    import inspect
    source = inspect.getsource(rb.TrackTrackReplayBackend.replay_into_live_track)
    assert "ReplayEngine" not in source
    assert "replay_event" not in source
