"""Online runtime/adapter contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shutrack.integration.tracktrack import ShutrackTrackerAdapter
from shutrack.runtime.adapter import ShutrackRuntime, ShutrackRuntimeConfig
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.tests.conftest import (
    FakeDetection,
    FakeSnapshot,
    FakeTrack,
    write_tiny_checkpoint,
)


def _association_meta(num_tracks: int = 2, num_detections: int = 3) -> dict:
    raw = np.array(
        [[0.9, 0.2, 0.8], [0.1, 0.7, 0.6]], dtype=np.float64
    )[:num_tracks, :num_detections]
    return {
        "raw_cost": raw,
        "final_cost": raw.copy(),
        "iou_similarity": np.ones_like(raw),
        "iou_distance": np.zeros_like(raw),
        "cosine_distance": np.zeros_like(raw),
        "confidence_distance": np.zeros_like(raw),
        "angle_distance": np.zeros_like(raw),
        "assignment_round": np.zeros_like(raw, dtype=np.int16),
        "assignment_threshold": np.full_like(raw, 0.5),
        "detection_source": np.zeros_like(raw, dtype=np.int8),
    }


def _adapter(tmp_path, min_observations: int = 1):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    runtime = ShutrackRuntime(
        model,
        checkpoint,
        device="cpu",
        config=ShutrackRuntimeConfig(min_observations=min_observations),
    )
    args = SimpleNamespace(img_w=640, img_h=480)
    return ShutrackTrackerAdapter(args, "tiny", runtime=runtime), runtime


def test_matched_event_produces_gates_and_delegates_update(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    adapter.begin_frame(5, 640, 480)
    track = FakeTrack(track_id=1, observation_count=3)
    detection = FakeDetection()
    snapshot = FakeSnapshot(frame_id=5)
    event = adapter.build_matched_event(
        track,
        detection,
        track_idx=0,
        det_idx=1,
        association_meta=_association_meta(),
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    decision = adapter.get_iwg_decisions_batch([(track.track_id, event)])[0]
    assert 0.0 <= decision.motion_gate <= 1.0
    assert 0.0 <= decision.appearance_gate <= 1.0
    adapter.apply_gate(track, detection, decision)
    assert len(track.calls) == 1
    frame_id, applied_detection, motion_gate, appearance_gate = track.calls[0]
    assert frame_id == 5
    assert applied_detection is detection
    assert motion_gate == pytest.approx(decision.motion_gate)
    assert appearance_gate == pytest.approx(decision.appearance_gate)
    adapter.record_event(track.track_id, event, decision)
    assert len(runtime.event_buffers[track.track_id]) == 1
    assert runtime.stats.matched_events == 1
    assert runtime.stats.model_inferences == 1


def test_unmatched_event_is_zero_gated_without_model_call(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    adapter.begin_frame(2, 640, 480)
    track = FakeTrack(track_id=2, observation_count=2)
    snapshot = FakeSnapshot(frame_id=2)
    event = adapter.build_unmatched_event(track, np.eye(2, 3), snapshot, snapshot)
    assert event.has_detection is False
    adapter.record_unmatched_event(track.track_id, event)
    assert runtime.stats.unmatched_events == 1
    assert runtime.stats.no_detection_events == 1
    assert runtime.stats.model_inferences == 0
    assert len(runtime.event_buffers[track.track_id]) == 1


def test_cold_start_enters_model_with_empty_history(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    adapter.begin_frame(1, 640, 480)
    track = FakeTrack(track_id=3, observation_count=1)
    snapshot = FakeSnapshot(frame_id=1)
    event = adapter.build_matched_event(
        track,
        FakeDetection(),
        0,
        0,
        _association_meta(),
        np.eye(2, 3),
        snapshot,
        snapshot,
    )
    decision = adapter.get_iwg_decisions_batch([(track.track_id, event)])[0]
    assert np.isfinite([decision.motion_gate, decision.appearance_gate]).all()
    # Cold start is applied, not bypassed.
    assert runtime.stats.model_inferences == 1


def test_gap_resets_per_track_history(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    track = FakeTrack(track_id=4, observation_count=3)
    adapter.begin_frame(1, 640, 480)
    snapshot = FakeSnapshot(frame_id=1)
    first = adapter.build_matched_event(
        track, FakeDetection(), 0, 0, _association_meta(), np.eye(2, 3), snapshot, snapshot
    )
    adapter.record_event(track.track_id, first, adapter.get_iwg_decisions_batch([(track.track_id, first)])[0])
    assert len(runtime.event_buffers[track.track_id]) == 1
    adapter.begin_frame(50, 640, 480)
    snapshot2 = FakeSnapshot(frame_id=50)
    second = adapter.build_matched_event(
        track, FakeDetection(), 0, 0, _association_meta(), np.eye(2, 3), snapshot2, snapshot2
    )
    adapter.record_event(track.track_id, second, adapter.get_iwg_decisions_batch([(track.track_id, second)])[0])
    assert len(runtime.event_buffers[track.track_id]) == 1


def test_track_buffers_do_not_mix(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    adapter.begin_frame(1, 640, 480)
    snapshot = FakeSnapshot(frame_id=1)
    for track_id in (10, 11):
        track = FakeTrack(track_id=track_id, observation_count=3)
        event = adapter.build_matched_event(
            track, FakeDetection(), 0, 0, _association_meta(), np.eye(2, 3), snapshot, snapshot
        )
        adapter.record_event(track.track_id, event, adapter.get_iwg_decisions_batch([(track.track_id, event)])[0])
    assert len(runtime.event_buffers[10]) == 1
    assert len(runtime.event_buffers[11]) == 1
    assert runtime.event_buffers[10][0].track_id == 10
    assert runtime.event_buffers[11][0].track_id == 11
    adapter.remove_track(10)
    assert 10 not in runtime.event_buffers
    assert 11 in runtime.event_buffers


def test_is_mature_track_respects_state_and_min_observations(tmp_path):
    adapter, runtime = _adapter(tmp_path, min_observations=4)
    assert not runtime.is_mature_track(FakeTrack(state=0, observation_count=10))
    assert not runtime.is_mature_track(FakeTrack(state=1, observation_count=3))
    assert runtime.is_mature_track(FakeTrack(state=1, observation_count=4))
    assert runtime.is_mature_track(FakeTrack(state=2, observation_count=4))


def test_disable_channel_gate_forces_one(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    runtime.config = ShutrackRuntimeConfig(disable_kf_gate=True)
    adapter.begin_frame(1, 640, 480)
    track = FakeTrack(track_id=5)
    snapshot = FakeSnapshot(frame_id=1)
    event = adapter.build_matched_event(
        track, FakeDetection(), 0, 0, _association_meta(), np.eye(2, 3), snapshot, snapshot
    )
    decision = adapter.get_iwg_decisions_batch([(track.track_id, event)])[0]
    assert decision.motion_gate == 1.0
    assert 0.0 <= decision.appearance_gate <= 1.0
