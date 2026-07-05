"""Test that AgentGuard in 'off' mode produces identical results to baseline.

When AgentGuardRuntime is in 'off' mode:
- process_matched_event should be a no-op (no buffers created, no gates applied).
- process_unmatched_event should be a no-op.
- finalize_first_stage should do nothing.
This means that the tracker continues with its original TrackTrack logic unchanged.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.contracts.states import DetectionObservation, TrackStateSnapshot
from agentguard.runtime import AgentGuardRuntime


class BaselineTrack:
    """Simulates a track running without AgentGuard (original TrackTrack)."""

    def __init__(self, track_id: int = 0):
        self.track_id = track_id
        self.state = 1  # TRACKED
        self.box = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        self.score = 0.85
        self.end_frame_id = 0
        self.history: Dict[int, List] = {
            i: [self.box.copy(), self.score, None, None, np.zeros((1, 64))]
            for i in range(6)
        }
        self.update_calls: List[int] = []  # track which frame IDs were updated

    def update(self, frame_id: int, detection: Any) -> None:
        """Original update method - records the call."""
        self.end_frame_id = frame_id
        self.update_calls.append(frame_id)
        if hasattr(detection, "box"):
            self.box = detection.box.copy()
        if hasattr(detection, "score"):
            self.score = detection.score

    def update_with_gates(
        self, frame_id: int, detection: Any,
        motion_gate: float, appearance_gate: float,
    ) -> None:
        """Gated update - should not be called in 'off' mode."""
        self.update_calls.append(frame_id)

    def mark_lost(self) -> None:
        self.state = 2


@pytest.fixture
def off_config() -> Dict[str, Any]:
    return {"mode": "off"}


@pytest.fixture
def detection() -> DetectionObservation:
    return DetectionObservation(
        detection_index=0,
        box=np.array([110.0, 210.0, 310.0, 410.0], dtype=np.float64),
        score=0.9,
        feature=np.zeros((1, 64), dtype=np.float64),
        source=0,
        class_id=1,
    )


@pytest.fixture
def event(detection) -> TrackEvent:
    state = TrackStateSnapshot(
        track_id=0,
        box=np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64),
        score=0.85,
        mean=np.zeros(8),
        covariance=np.eye(8),
        velocity=np.zeros((4, 2)),
        feature=np.zeros((1, 64)),
        history={},
        end_frame_id=0,
        state=1,
    )
    return TrackEvent(
        event_id="off_test",
        dataset="test",
        sequence="test_seq",
        frame_id=100,
        track_id=0,
        image_width=640,
        image_height=480,
        has_detection=True,
        frame_start_state=state,
        pre_update_state=state,
        detection=detection,
        warp_matrix=np.eye(2, 3),
    )


class TestOnlineOffEquivalence:
    """AgentGuard in 'off' mode should leave the track completely unchanged."""

    def test_off_mode_process_matched_no_side_effects(self, off_config, detection):
        """process_matched_event should not create buffers or modify track."""
        track = BaselineTrack(track_id=1)
        initial_box = track.box.copy()
        initial_score = track.score
        initial_end_frame = track.end_frame_id

        runtime = AgentGuardRuntime(off_config)
        runtime.process_matched_event(
            track, detection, event := TrackEvent(
                event_id="off_test",
                dataset="test", sequence="seq1", frame_id=100, track_id=1,
                image_width=640, image_height=480, has_detection=True,
                detection=detection,
            ),
            GateDecision(1.0, 1.0, np.ones(5) / 5, 0.5),
        )

        # Track should be completely unchanged
        np.testing.assert_array_equal(track.box, initial_box)
        assert track.score == initial_score
        assert track.end_frame_id == initial_end_frame
        # No buffers should be created
        assert len(runtime.event_buffers) == 0
        assert len(runtime.window_buffers) == 0

    def test_off_mode_process_unmatched_no_side_effects(self, off_config):
        """process_unmatched_event should not create buffers."""
        track = BaselineTrack(track_id=2)
        runtime = AgentGuardRuntime(off_config)

        event = TrackEvent(
            event_id="off_unmatched",
            dataset="test", sequence="seq1", frame_id=100, track_id=2,
            image_width=640, image_height=480, has_detection=False,
        )
        runtime.process_unmatched_event(track, event)

        assert len(runtime.event_buffers) == 0
        assert len(runtime.window_buffers) == 0

    def test_off_mode_stats_events_still_recorded(self, off_config):
        """In 'off' mode, event counts are still recorded (but no IWG/TGR gates)."""
        track = BaselineTrack(track_id=3)
        runtime = AgentGuardRuntime(off_config)

        event = TrackEvent(
            event_id="off_stats",
            dataset="test", sequence="seq1", frame_id=100, track_id=3,
            image_width=640, image_height=480, has_detection=True,
        )

        runtime.process_matched_event(
            track, None, event,
            GateDecision(1.0, 1.0, np.ones(5) / 5, 0.5),
        )
        stats = runtime.stats.summary()
        # Events are always counted (stats.record_event called before mode check),
        # but IWG/TGR specific stats should be zero
        assert stats["total_events"] == 1
        assert stats["iwg_calls"] == 0
        assert stats["tgr_calls"] == 0
        assert stats["replay_events"] == 0

    def test_off_mode_finalize_first_stage_noop(self, off_config):
        """finalize_first_stage should do nothing in 'off' mode."""
        runtime = AgentGuardRuntime(off_config)
        # Should not raise
        runtime.finalize_first_stage([], [])
        # No windows created
        assert len(runtime.window_buffers) == 0

    def test_off_mode_iwg_inference_no_model(self, off_config):
        """run_iwg_inference should still return fallback gates even in off mode."""
        runtime = AgentGuardRuntime(off_config)
        result = runtime.run_iwg_inference([])
        assert result["gate"][0] == 1.0
        assert result["gate"][1] == 1.0

    def test_off_mode_track_unchanged_across_multiple_calls(self, off_config, detection):
        """Multiple process_matched_event calls in off mode should not affect track."""
        track = BaselineTrack(track_id=4)
        runtime = AgentGuardRuntime(off_config)

        for frame_id in [100, 101, 102, 103, 104]:
            evt = TrackEvent(
                event_id=f"off_{frame_id}",
                dataset="test", sequence="seq1", frame_id=frame_id, track_id=4,
                image_width=640, image_height=480, has_detection=True,
                detection=detection,
            )
            runtime.process_matched_event(
                track, detection, evt,
                GateDecision(0.8, 0.6, np.ones(5) / 5, 0.5),
            )

        # Track should still be completely unchanged
        np.testing.assert_array_equal(
            track.box, np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        )
        assert track.score == 0.85
        assert track.end_frame_id == 0  # unchanged

    def test_off_mode_vs_baseline_identical(self, off_config, detection):
        """Track running with 'off' mode should behave identically to no-runtime baseline.

        In 'off' mode, the runtime does NOT call track.update() or
        track.update_with_gates(), so the track stays in its initial state.
        A baseline without any runtime also leaves the track unchanged.
        """
        # Track without any runtime intervention
        baseline_track = BaselineTrack(track_id=5)
        baseline_box = baseline_track.box.copy()
        baseline_score = baseline_track.score

        # Track with AgentGuard 'off' mode
        off_track = BaselineTrack(track_id=5)
        off_box = off_track.box.copy()
        off_score = off_track.score

        runtime = AgentGuardRuntime(off_config)
        evt = TrackEvent(
            event_id="off_cmp",
            dataset="test", sequence="seq1", frame_id=100, track_id=5,
            image_width=640, image_height=480, has_detection=True,
            detection=detection,
        )
        runtime.process_matched_event(
            off_track, detection, evt,
            GateDecision(0.5, 0.5, np.ones(5) / 5, 0.3),
        )

        # Both should be identical to the original (no update applied)
        np.testing.assert_array_equal(off_track.box, baseline_box)
        assert off_track.score == baseline_score

    def test_off_mode_no_checkpoints_created(self, off_config):
        """In 'off' mode, no checkpoints should be saved."""
        runtime = AgentGuardRuntime(off_config)
        assert len(runtime.checkpoints.checkpoints) == 0

        # Attempting to save a checkpoint should still work (runtime doesn't prevent it)
        snap = TrackStateSnapshot(
            track_id=0,
            box=np.zeros(4), score=0.0, mean=None, covariance=None,
            velocity=np.zeros((4, 2)), feature=np.zeros((1, 64)),
            history={}, end_frame_id=0, state=1,
        )
        runtime.checkpoints.save_checkpoint(1, "evt", snap)
        # The checkpoint exists because the runtime doesn't disable it,
        # but the key point is that process_matched/unmatched don't create them.
        assert runtime.checkpoints.get_checkpoint(1, "evt") is snap

    def test_off_mode_track_update_not_called(self, off_config, detection):
        """Track.update() should NOT be called by process_matched_event in off mode."""
        track = BaselineTrack(track_id=6)
        initial_calls = len(track.update_calls)

        runtime = AgentGuardRuntime(off_config)
        evt = TrackEvent(
            event_id="off_nocall",
            dataset="test", sequence="seq1", frame_id=100, track_id=6,
            image_width=640, image_height=480, has_detection=True,
            detection=detection,
        )
        runtime.process_matched_event(
            track, detection, evt,
            GateDecision(1.0, 1.0, np.ones(5) / 5, 1.0),
        )

        # Track.update() should NOT have been called
        assert len(track.update_calls) == initial_calls
