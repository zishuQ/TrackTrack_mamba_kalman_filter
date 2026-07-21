"""Tests for agentguard.runtime."""
from __future__ import annotations

import sys
from typing import Any, Dict, List

import pytest
import numpy as np

# ── Ensure ``src`` is on the path ──────────────────────────────────────────
sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.contracts.states import (
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.runtime import AgentGuardRuntime, CheckpointManager, EventBuffer, WindowBuffer
from agentguard.runtime.buffers import EventBuffer, WindowBuffer
from agentguard.runtime.checkpoint import CheckpointManager
from agentguard.runtime.replay import ReplayEngine
from agentguard.runtime.statistics import RuntimeStatistics


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_event(
    event_id: str,
    track_id: int = 0,
    frame_id: int = 0,
    has_detection: bool = True,
    reid_dim: int = 128,
) -> TrackEvent:
    return TrackEvent(
        event_id=event_id,
        dataset="test",
        sequence="test_seq",
        frame_id=frame_id,
        track_id=track_id,
        image_width=640,
        image_height=480,
        has_detection=has_detection,
        detection=(
            DetectionObservation(
                detection_index=0,
                box=np.array([10, 20, 110, 220], dtype=np.float64),
                score=0.9,
                feature=np.random.randn(1, reid_dim).astype(np.float64),
                source=0,
                class_id=1,
            )
            if has_detection
            else None
        ),
        warp_matrix=np.eye(2, 3, dtype=np.float64),
        scalar_features=np.zeros(63, dtype=np.float64),
        track_feature=np.random.randn(reid_dim).astype(np.float64),
        detection_feature=(
            np.random.randn(reid_dim).astype(np.float64)
            if has_detection
            else np.zeros(0, dtype=np.float64)
        ),
    )


def _make_snapshot(track_id: int = 0, frame_id: int = 0) -> TrackStateSnapshot:
    return TrackStateSnapshot(
        track_id=track_id,
        box=np.array([10, 20, 110, 220], dtype=np.float64),
        score=0.9,
        mean=np.array([60.0, 120.0, 100.0, 200.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64),
        velocity=np.zeros((4, 2), dtype=np.float64),
        feature=np.random.randn(1, 128).astype(np.float64),
        history={frame_id: [np.array([10, 20, 110, 220]), 0.9, None, None, np.zeros((1, 128))]},
        end_frame_id=frame_id,
        state=1,  # Tracked
    )


class FakeTrack:
    """Minimal track stub that mimics the TrackTrack Track interface."""
    def __init__(self, track_id: int = 0):
        self.track_id = track_id
        self.state = 1  # Tracked
        self.history = {i: [np.array([10, 20, 110, 220]), 0.9, None, None, np.zeros((1, 128))]
                        for i in range(6)}

    def update_with_gates(self, frame_id, detection, motion_gate, appearance_gate):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  EventBuffer
# ══════════════════════════════════════════════════════════════════════════════

def test_event_buffer_push():
    buf = EventBuffer(max_len=6)
    assert len(buf.events) == 0

    for i in range(6):
        buf.push(_make_event(f"e{i}", frame_id=i))
    assert len(buf.events) == 6

    # Overflow trims oldest
    buf.push(_make_event("e7", frame_id=7))
    assert len(buf.events) == 6
    assert buf.events[0].event_id == "e1"
    assert buf.events[-1].event_id == "e7"


def test_event_buffer_get_sequence_empty():
    buf = EventBuffer(max_len=6)
    seq = buf.get_sequence()
    assert len(seq) == 6
    assert all(e is None for e in seq)


def test_event_buffer_get_sequence_partial():
    buf = EventBuffer(max_len=6)
    buf.push(_make_event("e0", frame_id=0))
    buf.push(_make_event("e1", frame_id=1))

    seq = buf.get_sequence()
    assert len(seq) == 6
    assert seq[0] is None  # padded
    assert seq[1] is None
    assert seq[2] is None
    assert seq[3] is None
    assert seq[4] is not None and seq[4].event_id == "e0"
    assert seq[5] is not None and seq[5].event_id == "e1"


def test_event_buffer_clear():
    buf = EventBuffer(max_len=6)
    buf.push(_make_event("e0"))
    buf.clear()
    assert len(buf.events) == 0
    assert all(e is None for e in buf.get_sequence())


# ══════════════════════════════════════════════════════════════════════════════
#  WindowBuffer
# ══════════════════════════════════════════════════════════════════════════════

def test_window_buffer_push():
    buf = WindowBuffer(window_size=4)
    assert not buf.is_full

    for i in range(4):
        buf.push(_make_event(f"e{i}", frame_id=i))
    assert buf.is_full
    assert len(buf.events) == 4


def test_window_buffer_get_window():
    buf = WindowBuffer(window_size=4)
    for i in range(6):
        buf.push(_make_event(f"e{i}", frame_id=i))

    window = buf.get_window()
    assert len(window) == 4
    assert [e.event_id for e in window] == ["e2", "e3", "e4", "e5"]


def test_window_buffer_pop_oldest():
    buf = WindowBuffer(window_size=4)
    buf.push(_make_event("e0"))
    buf.push(_make_event("e1"))

    oldest = buf.pop_oldest()
    assert oldest.event_id == "e0"
    assert len(buf.events) == 1
    assert buf.events[0].event_id == "e1"

    # Empty pop returns None
    buf.pop_oldest()
    assert buf.pop_oldest() is None


# ══════════════════════════════════════════════════════════════════════════════
#  CheckpointManager
# ══════════════════════════════════════════════════════════════════════════════

def test_checkpoint_save_get_remove():
    mgr = CheckpointManager()
    snap = _make_snapshot()

    mgr.save_checkpoint(0, "evt1", snap)
    assert mgr.get_checkpoint(0, "evt1") is snap

    mgr.remove_checkpoint(0, "evt1")
    assert mgr.get_checkpoint(0, "evt1") is None


def test_checkpoint_cleanup_track():
    mgr = CheckpointManager()
    mgr.save_checkpoint(0, "evt1", _make_snapshot())
    mgr.save_checkpoint(0, "evt2", _make_snapshot())

    mgr.cleanup_track(0)
    assert 0 not in mgr.checkpoints
    assert mgr.get_checkpoint(0, "evt1") is None


def test_checkpoint_get_nonexistent():
    mgr = CheckpointManager()
    assert mgr.get_checkpoint(0, "ghost") is None


# ══════════════════════════════════════════════════════════════════════════════
#  ReplayEngine
# ══════════════════════════════════════════════════════════════════════════════

def test_replay_event_with_detection():
    """replay_event with has_detection should apply warp + predict + gated update."""
    engine = ReplayEngine()
    snap = _make_snapshot()
    event = _make_event("test_replay", has_detection=True)

    result = engine.replay_event(snap, event, revised_gate=np.array([1.0, 1.0]))

    assert isinstance(result, TrackStateSnapshot)
    assert result.track_id == snap.track_id
    assert result.end_frame_id == event.frame_id
    # With gate=[1,1] the KF update is fully applied
    assert result.mean is not None
    assert result.covariance is not None
    assert result.history.get(event.frame_id) is not None


def test_replay_event_no_detection():
    """replay_event without detection should mark state as LOST (2)."""
    engine = ReplayEngine()
    snap = _make_snapshot()
    event = _make_event("test_lost", has_detection=False)

    result = engine.replay_event(snap, event, revised_gate=None)
    assert result.state == 2  # LOST
    assert result.end_frame_id == event.frame_id


def test_replay_window():
    """replay_window should process all events sequentially."""
    engine = ReplayEngine()
    snap = _make_snapshot()

    events = [
        _make_event("r0", frame_id=1, has_detection=True),
        _make_event("r1", frame_id=2, has_detection=True),
        _make_event("r2", frame_id=3, has_detection=True),
        _make_event("r3", frame_id=4, has_detection=True),
    ]
    gates = np.ones((4, 2), dtype=np.float64)

    result = engine.replay_window(snap, events, gates)
    assert isinstance(result, TrackStateSnapshot)
    assert result.end_frame_id == 4
    # All 4 events should be in history
    for evt in events:
        assert evt.frame_id in result.history


# ══════════════════════════════════════════════════════════════════════════════
#  AgentGuardRuntime
# ══════════════════════════════════════════════════════════════════════════════

def test_runtime_off_mode():
    """In 'off' mode, process_matched_event and process_unmatched_event are no-ops."""
    config = {"mode": "off"}
    runtime = AgentGuardRuntime(config)
    track = FakeTrack()
    event = _make_event("off_test")

    runtime.process_matched_event(track, None, event, GateDecision(1.0, 1.0, np.ones(5) / 5, 0.5))
    assert len(runtime.event_buffers) == 0  # nothing stored

    runtime.process_unmatched_event(track, event)
    assert len(runtime.event_buffers) == 0


def test_runtime_iwg_mode():
    """In 'iwg' mode, events are pushed to buffers."""
    config = {"mode": "iwg"}
    runtime = AgentGuardRuntime(config)
    runtime.init_feature_builder(reid_dim=128)
    track = FakeTrack(track_id=42)

    event = _make_event("iwg_test", track_id=42)
    runtime.process_matched_event(
        track, None, event,
        GateDecision(0.8, 0.6, np.ones(5) / 5, 0.5),
    )

    assert 42 in runtime.event_buffers
    assert 42 in runtime.window_buffers
    assert runtime.event_buffers[42].events[0].event_id == "iwg_test"
    assert event.iwg_gate is not None
    np.testing.assert_allclose(event.iwg_gate, [0.8, 0.6])


def test_runtime_unmatched_event():
    """Unmatched events get zero gate and are pushed to buffers."""
    config = {"mode": "iwg"}
    runtime = AgentGuardRuntime(config)
    track = FakeTrack(track_id=7)
    event = _make_event("unmatch_test", track_id=7, has_detection=False)

    runtime.process_unmatched_event(track, event)
    assert 7 in runtime.event_buffers
    np.testing.assert_allclose(event.iwg_gate, [0.0, 0.0])


def test_runtime_cleanup_track():
    """cleanup_track should remove all per-track state."""
    config = {"mode": "iwg"}
    runtime = AgentGuardRuntime(config)
    track = FakeTrack(track_id=99)
    event = _make_event("clean_test", track_id=99)

    runtime.process_matched_event(
        track, None, event,
        GateDecision(1.0, 1.0, np.ones(5) / 5, 0.5),
    )
    runtime.checkpoints.save_checkpoint(99, event.event_id, _make_snapshot())

    runtime.cleanup_track(99)
    assert 99 not in runtime.event_buffers
    assert 99 not in runtime.window_buffers
    assert 99 not in runtime.checkpoints.checkpoints


def test_runtime_iwg_inference_no_model():
    """run_iwg_inference should return fallback gates when model is None in off mode."""
    config = {"mode": "off"}
    runtime = AgentGuardRuntime(config)
    runtime.init_feature_builder(reid_dim=128)
    # No iwg model set → fallback (allowed only in off mode)
    result = runtime.run_iwg_inference([_make_event("f0")])
    assert result["gate"][0] == 1.0
    assert result["gate"][1] == 1.0
    assert "policy_probs" in result
    assert "risk" in result
    assert "cue" in result
    assert "iwg_gate_residual" in result


def test_runtime_tgr_inference_no_model():
    """_run_tgr_inference should raise RuntimeError when TGR is None."""
    config = {"mode": "full"}
    runtime = AgentGuardRuntime(config)
    runtime.init_feature_builder(reid_dim=128)
    with pytest.raises(RuntimeError, match="TGR model is not loaded"):
        runtime._run_tgr_inference([
            _make_event("t0"), _make_event("t1"),
            _make_event("t2"), _make_event("t3"),
        ])


def _make_mock_tgr(reid_dim=128):
    """Create a minimal TGR model stub that returns zeros."""
    import torch
    import torch.nn as nn
    from agentguard.models.tgr import TGR
    tgr = TGR(reid_dim=reid_dim)
    tgr.eval()
    return tgr


def test_runtime_finalize_first_stage_full_mode_no_full_windows():
    """finalize_first_stage does nothing when no windows are full."""
    import torch
    config = {"mode": "full"}
    tgr = _make_mock_tgr(reid_dim=128)
    runtime = AgentGuardRuntime(config, tgr_model=tgr)
    runtime.init_feature_builder(reid_dim=128)
    result = runtime.finalize_first_stage([], [])
    assert result == {}
    # No crash = success


def test_runtime_finalize_first_stage_partial_window():
    """finalize_first_stage skips windows that aren't full."""
    config = {"mode": "full"}
    tgr = _make_mock_tgr(reid_dim=128)
    runtime = AgentGuardRuntime(config, tgr_model=tgr)
    runtime.init_feature_builder(reid_dim=128)
    track = FakeTrack(track_id=1)

    # Push only 3 events — window size is 4
    for i in range(3):
        evt = _make_event(f"p{i}", track_id=1, frame_id=i)
        runtime.process_matched_event(
            track, None, evt,
            GateDecision(1.0, 1.0, np.ones(5) / 5, 0.5),
        )
        runtime.checkpoints.save_checkpoint(1, evt.event_id, _make_snapshot())

    result = runtime.finalize_first_stage([], [])
    assert result == {}
    # Window not full → nothing popped
    assert len(runtime.window_buffers[1].events) == 3


def test_runtime_finalize_first_stage_full_window():
    """With a full window and checkpoint, finalize_first_stage runs TGR replay."""
    config = {"mode": "full"}
    tgr = _make_mock_tgr(reid_dim=128)
    runtime = AgentGuardRuntime(config, tgr_model=tgr)
    runtime.init_feature_builder(reid_dim=128)
    track = FakeTrack(track_id=2)

    # Push 4 events + checkpoints to fill the window
    for i in range(4):
        evt = _make_event(f"w{i}", track_id=2, frame_id=i)
        runtime.process_matched_event(
            track, None, evt,
            GateDecision(0.8, 0.6, np.ones(5) / 5, 0.5),
        )
        runtime.checkpoints.save_checkpoint(2, evt.event_id, _make_snapshot())

    # Sanity: window is full
    assert runtime.window_buffers[2].is_full
    assert len(runtime.window_buffers[2].events) == 4

    plans = runtime.finalize_first_stage([], [])

    # Runtime only builds plans. Checkpoint roll and window pop happen
    # after replay inside the adapter/backend path.
    assert len(runtime.window_buffers[2].events) == 4
    assert runtime.checkpoints.get_checkpoint(2, "w0") is not None
    assert 2 in runtime.pending_checkpoint_rolls
    assert plans[2].oldest_event_id == "w0"


# ══════════════════════════════════════════════════════════════════════════════
#  RuntimeStatistics
# ══════════════════════════════════════════════════════════════════════════════

def test_statistics_defaults():
    stats = RuntimeStatistics()
    s = stats.summary()
    assert s["total_events"] == 0
    assert s["iwg_calls"] == 0
    assert s["tgr_calls"] == 0
    assert s["replay_events"] == 0
    assert s["avg_motion_gate"] == 0.0
    assert s["avg_appearance_gate"] == 0.0


def test_statistics_recording():
    stats = RuntimeStatistics()
    stats.record_event()
    stats.record_iwg(np.array([0.8, 0.6]))
    stats.record_tgr()
    stats.record_replay()

    s = stats.summary()
    assert s["total_events"] == 1
    assert s["iwg_calls"] == 1
    assert s["tgr_calls"] == 1
    assert s["replay_events"] == 1
    assert s["avg_motion_gate"] == 0.8
    assert s["avg_appearance_gate"] == 0.6


def test_statistics_multiple_gates():
    stats = RuntimeStatistics()
    stats.record_iwg(np.array([0.5, 0.3]))
    stats.record_iwg(np.array([0.7, 0.9]))

    s = stats.summary()
    assert s["avg_motion_gate"] == 0.6
    assert s["avg_appearance_gate"] == 0.6


def test_statistics_rg_cma_correction_diagnostics():
    stats = RuntimeStatistics()
    stats.record_iwg(
        np.array([0.35, 0.65]),
        base_gate=np.array([0.4, 0.6]),
        final_gate=np.array([0.35, 0.65]),
        correction=np.array([-0.05, 0.05]),
        correction_bound=0.05,
    )
    summary = stats.summary()
    assert summary["avg_iwg_base_motion_gate"] == 0.4
    assert summary["avg_iwg_final_motion_gate"] == 0.35
    assert np.isclose(summary["avg_iwg_base_final_abs_diff_motion"], 0.05)
    assert np.isclose(summary["avg_iwg_correction_abs_appearance"], 0.05)
    assert summary["iwg_correction_nonzero_rate_motion"] == 1.0
    assert summary["iwg_correction_saturation_rate_appearance"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                failures += 1
    if failures:
        print(f"\n❌ {failures} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed")
