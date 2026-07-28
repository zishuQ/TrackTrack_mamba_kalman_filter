"""Offline/capture invariants for the current online integration."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.runtime import AgentGuardRuntime


class _BaselineTrack:
    def __init__(self):
        self.update_calls = []

    def update(self, frame_id, detection):
        self.update_calls.append((frame_id, detection))


class _CompactSink:
    compact = True

    def __init__(self):
        self.frames = []

    def on_frame(self, record, events):
        self.frames.append((record, events))


def _event(frame_id: int, track_id: int = 1, matched: bool = True) -> TrackEvent:
    return TrackEvent(
        event_id=f"event-{frame_id}",
        dataset="test",
        sequence="seq",
        frame_id=frame_id,
        track_id=track_id,
        image_width=640,
        image_height=480,
        has_detection=matched,
        scalar_features=np.zeros(63, dtype=np.float32),
        track_feature=np.zeros(4, dtype=np.float32),
        detection_feature=(
            np.zeros(4, dtype=np.float32) if matched else np.zeros(0, dtype=np.float32)
        ),
    )


def test_off_mode_leaves_baseline_update_path_untouched():
    runtime = AgentGuardRuntime({"mode": "off"})
    track = _BaselineTrack()
    detection = object()
    track.update(1, detection)

    assert track.update_calls == [(1, detection)]
    assert runtime.event_buffers == {}
    assert runtime.rg_cma_model is None


def test_capture_flushes_compact_events_without_model_loading():
    sink = _CompactSink()
    runtime = AgentGuardRuntime({"mode": "capture"})
    runtime.event_sink = sink
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    adapter = AgentGuardTrackerAdapter(
        SimpleNamespace(dataset="test", agentguard_mode="capture"),
        "seq",
        agentguard_runtime=runtime,
    )
    adapter.begin_frame(1, 640, 480)
    adapter.record_event(1, _event(1), GateDecision(1.0, 1.0, np.ones(5) / 5.0))
    adapter.record_unmatched_event(1, _event(1, matched=False))
    adapter.end_frame()

    assert runtime.rg_cma_model is None
    assert runtime.event_buffers == {}
    assert len(sink.frames) == 1
    assert len(sink.frames[0][1]) == 2
