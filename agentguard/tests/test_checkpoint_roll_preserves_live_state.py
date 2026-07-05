"""Tests for checkpoint roll behavior around full replay."""

import numpy as np
import pytest

from agentguard.contracts.states import TrackStateSnapshot
from agentguard.runtime.manager import AgentGuardRuntime
from agentguard.runtime.replay import ReplayPlan, ReplayStep


def test_runtime_does_not_use_replay_engine_in_finalize():
    """The AgentGuardRuntime should not reference ReplayEngine."""
    import inspect
    source = inspect.getsource(AgentGuardRuntime)
    assert "ReplayEngine" not in source
    assert "self.replay" not in source


def test_finalize_first_stage_returns_plans_only():
    """finalize_first_stage should return dict[int, ReplayPlan] for full mode."""
    runtime = AgentGuardRuntime({"mode": "full"}, iwg_model=None, tgr_model=None)
    # Mode is full, but model is None → should raise
    with pytest.raises(RuntimeError):
        runtime.finalize_first_stage()


def test_off_mode_returns_empty_dict():
    runtime = AgentGuardRuntime({"mode": "off"})
    result = runtime.finalize_first_stage()
    assert result == {}


def test_iwg_mode_returns_empty_dict():
    runtime = AgentGuardRuntime({"mode": "iwg"})
    result = runtime.finalize_first_stage()
    assert result == {}


def test_checkpoint_manager_basic_ops():
    from agentguard.runtime.checkpoint import CheckpointManager
    from agentguard.contracts.states import TrackStateSnapshot

    cm = CheckpointManager()
    snap = TrackStateSnapshot(
        track_id=1,
        box=np.zeros(4),
        score=0.5,
        mean=None,
        covariance=None,
        velocity=np.zeros((4, 2)),
        feature=np.zeros((1, 64)),
        history={},
        end_frame_id=5,
        state=1,
    )

    cm.save_checkpoint(1, "evt_1", snap)
    assert cm.get_checkpoint(1, "evt_1") is not None
    assert cm.get_checkpoint(1, "evt_2") is None

    cm.remove_checkpoint(1, "evt_1")
    assert cm.get_checkpoint(1, "evt_1") is None

    cm.cleanup_track(1)
    assert len(cm.checkpoints) == 0


def test_checkpoint_roll_preserves_live_state():
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    class FakeRuntime:
        def __init__(self):
            self.pending_checkpoint_rolls = {
                7: {
                    "oldest_event_id": "evt-1",
                    "next_event_id": "evt-2",
                    "oldest_step": ReplayStep(
                        frame_id=9,
                        warp_matrix=np.eye(2, 3, dtype=np.float64),
                        has_detection=False,
                        detection=None,
                        motion_gate=0.0,
                        appearance_gate=0.0,
                    ),
                    "checkpoint": TrackStateSnapshot(
                        track_id=7,
                        box=np.array([0.0, 0.0, 10.0, 10.0]),
                        score=0.5,
                        mean=None,
                        covariance=None,
                        velocity=np.zeros((4, 2)),
                        feature=np.zeros((1, 4)),
                        history={},
                        end_frame_id=3,
                        state=1,
                    ),
                }
            }
            self.checkpoints = __import__("agentguard.runtime.checkpoint", fromlist=["CheckpointManager"]).CheckpointManager()
            self.window_buffers = {7: __import__("agentguard.runtime.buffers", fromlist=["WindowBuffer"]).WindowBuffer(window_size=4)}
            for idx in range(4):
                self.window_buffers[7].push(type("Evt", (), {"event_id": f"evt-{idx+1}"})())

    class Args:
        dataset = "MOT17"

    class LiveTrack:
        def __init__(self):
            self.track_id = 7
            self.box = np.array([5.0, 5.0, 15.0, 15.0])
            self.score = 0.9
            self.mean = None
            self.covariance = None
            self.velocity = np.ones((4, 2))
            self.feat = np.ones((1, 4))
            self.history = {8: [self.box.copy(), self.score, None, None, self.feat.copy()]}
            self.end_frame_id = 8
            self.state = 1

        def restore_state(self, snapshot):
            self.track_id = snapshot.track_id
            self.box = snapshot.box.copy()
            self.score = snapshot.score
            self.mean = snapshot.mean
            self.covariance = snapshot.covariance
            self.velocity = snapshot.velocity.copy()
            self.feat = snapshot.feature.copy()
            self.history = dict(snapshot.history)
            self.end_frame_id = snapshot.end_frame_id
            self.state = snapshot.state

    class FakeBackend:
        def replay_into_live_track(self, live_track, plan):
            live_track.box = np.array([99.0, 99.0, 199.0, 199.0])
            live_track.end_frame_id = 99
            live_track.state = 2

    adapter = AgentGuardTrackerAdapter(Args(), "seq", agentguard_runtime=FakeRuntime())
    track = LiveTrack()
    original_box = track.box.copy()
    original_end_frame_id = track.end_frame_id
    original_state = track.state

    adapter._roll_checkpoint_with_live_track(FakeBackend(), track, 7)

    assert np.array_equal(track.box, original_box)
    assert track.end_frame_id == original_end_frame_id
    assert track.state == original_state
    assert adapter.runtime.checkpoints.get_checkpoint(7, "evt-2") is not None
    assert adapter.runtime.checkpoints.get_checkpoint(7, "evt-1") is None
    assert len(adapter.runtime.window_buffers[7].events) == 3
