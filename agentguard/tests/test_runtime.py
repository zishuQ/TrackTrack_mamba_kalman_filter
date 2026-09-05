"""Tests for the current AgentGuard Runtime contract."""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.runtime import AgentGuardRuntime, EventBuffer, RuntimeStatistics


def _make_event(event_id: str, frame_id: int = 0) -> TrackEvent:
    return TrackEvent(
        event_id=event_id,
        dataset="test",
        sequence="test_seq",
        frame_id=frame_id,
        track_id=0,
        image_width=640,
        image_height=480,
        has_detection=True,
        scalar_features=np.zeros(63, dtype=np.float32),
        track_feature=np.zeros(16, dtype=np.float32),
        detection_feature=np.zeros(16, dtype=np.float32),
    )


def test_event_buffer_keeps_six_events_and_left_padding():
    buffer = EventBuffer(max_len=6)
    assert buffer.get_sequence() == [None] * 6
    for index in range(7):
        buffer.push(_make_event(f"event-{index}", frame_id=index))
    assert len(buffer.events) == 6
    assert buffer.events[0].event_id == "event-1"
    sequence = buffer.get_sequence()
    assert [event.event_id for event in sequence] == [
        "event-1",
        "event-2",
        "event-3",
        "event-4",
        "event-5",
        "event-6",
    ]


def test_off_and_capture_do_not_load_a_model():
    for mode in ("off", "capture"):
        runtime = AgentGuardRuntime({"mode": mode})
        assert runtime.rg_cma_model is None
        assert runtime.event_buffers == {}
        assert runtime.event_sink is None
        assert not hasattr(runtime, "window_buffers")
        assert not hasattr(runtime, "checkpoints")


def test_legacy_runtime_modes_are_rejected():
    for mode in ("iwg", "full", "joint", "iwg-attn"):
        with pytest.raises(ValueError, match="unsupported AgentGuard mode"):
            AgentGuardRuntime({"mode": mode})


def test_runtime_cleanup_track_only_clears_event_context():
    runtime = AgentGuardRuntime({"mode": "capture"})
    runtime.get_or_create_event_buffer(7).push(_make_event("event-7"))
    runtime.cleanup_track(7)
    assert 7 not in runtime.event_buffers


def test_rg_cma_config_uses_current_keys():
    runtime = AgentGuardRuntime(
        {
            "mode": "capture",
            "rg_cma_output": "base",
            "rg_cma_alpha": 0.5,
            "rg_cma_max_gap": 12,
            "fallback_threshold": 0.6,
        }
    )
    assert runtime.rg_cma_output == "base"
    assert runtime.rg_cma_alpha == 0.5
    assert runtime.rg_cma_max_gap == 12
    assert runtime.fallback_threshold == 0.6

    with pytest.raises(ValueError, match="rg_cma_alpha"):
        AgentGuardRuntime({"mode": "capture", "rg_cma_alpha": -0.1})

    with pytest.raises(ValueError, match="at most one"):
        AgentGuardRuntime(
            {
                "mode": "capture",
                "disable_kf_gate": True,
                "disable_ema_gate": True,
            }
        )

    with pytest.raises(ValueError, match="fallback_threshold"):
        AgentGuardRuntime({"mode": "capture", "fallback_threshold": 1.1})


def test_statistics_record_rg_cma_matched_and_unmatched():
    stats = RuntimeStatistics()
    stats.record_event()
    stats.record_rg_cma(
        np.array([0.35, 0.65]),
        np.array([0.4, 0.6]),
        np.array([0.35, 0.65]),
        np.array([-0.05, 0.05]),
        matched=True,
        correction_bound=0.05,
        policy_confidence=0.8,
        fallback_applied=True,
    )
    stats.record_rg_cma(
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        matched=False,
        correction_bound=0.05,
        policy_confidence=0.2,
    )

    summary = stats.summary()
    assert summary["total_events"] == 1
    assert summary["rg_cma_inference_count"] == 2
    assert summary["matched_events"] == 1
    assert summary["unmatched_events"] == 1
    assert np.isclose(summary["avg_rg_cma_base_motion_gate"], 0.2)
    assert np.isclose(summary["avg_rg_cma_correction_abs_appearance"], 0.025)
    assert summary["rg_cma_correction_saturation_rate_appearance"] == 0.5
    assert np.isclose(summary["avg_policy_confidence"], 0.5)
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate"] == 0.5
    assert summary["fallback_rate_matched"] == 1.0
