"""Test that unmatched replay preserves end_frame_id."""

import numpy as np

from agentguard.contracts.states import TrackStateSnapshot
from agentguard.runtime.replay import ReplayPlan, ReplayStep


def test_unmatched_replay_preserves_end_frame_id():
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    snapshot = TrackStateSnapshot(
        track_id=1,
        box=np.array([100, 100, 200, 200], dtype=np.float64),
        score=0.9,
        mean=np.array([150, 150, 100, 100, 0, 0, 0, 0], dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64),
        velocity=np.zeros((4, 2)),
        feature=np.ones((1, 64), dtype=np.float64) / 8.0,
        history={},
        end_frame_id=5,
        state=1,
    )

    plan = ReplayPlan(
        track_id=1,
        checkpoint=snapshot,
        steps=[
            ReplayStep(10, np.eye(2, 3), False, None, 0.0, 0.0),
            ReplayStep(11, np.eye(2, 3), False, None, 0.0, 0.0),
        ],
    )

    class SpyTrack:
        def __init__(self):
            self.calls = []
            self.state = 1
            self.end_frame_id = 0
            self.mean = None
            self.covariance = None
            self.box = np.zeros(4)
            self.score = 0.0
            self.feat = np.ones((1, 64)) / 8.0
            self.history = {}
            self.track_id = 1
            self.velocity = np.zeros((4, 2))
            self.kalman_filter = _DummyKF()

        def predict(self): pass

        def mark_lost(self):
            self.state = 2
            self.calls.append("mark_lost")

    class _DummyKF:
        def predict(self, m, c): return m, c
        def update(self, m, c, meas, score): return m, c
        def initiate(self, cxcywh): return np.zeros(8), np.eye(8)

    spy = SpyTrack()
    backend = TrackTrackReplayBackend()
    backend.replay_into_live_track(spy, plan)

    assert spy.state == 2
    assert spy.end_frame_id == 11


def test_no_detection_frame_enters_tgr_window():
    from agentguard.runtime.buffers import WindowBuffer
    from agentguard.contracts.events import TrackEvent

    buf = WindowBuffer(window_size=4)

    evt = TrackEvent(
        event_id="evt_1", dataset="d", sequence="s",
        frame_id=1, track_id=1,
        image_width=1920, image_height=1080,
        has_detection=False,
    )
    buf.push(evt)
    assert not buf.is_full

    for i in range(2, 5):
        buf.push(TrackEvent(
            event_id=f"evt_{i}", dataset="d", sequence="s",
            frame_id=i, track_id=1,
            image_width=1920, image_height=1080,
            has_detection=False,
        ))

    assert buf.is_full
    window = buf.get_window()
    assert len(window) == 4
    assert all(not e.has_detection for e in window)
