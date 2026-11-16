"""Online runtime/adapter contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
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
        "detection_indices": np.arange(num_detections, dtype=np.int64),
    }


def _adapter(
    tmp_path,
    min_observations: int = 1,
    competition_enabled: bool = True,
    early_enable: bool = False,
):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path, competition_enabled=competition_enabled)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    runtime = ShutrackRuntime(
        model,
        checkpoint,
        device="cpu",
        config=ShutrackRuntimeConfig(min_observations=min_observations),
    )
    args = SimpleNamespace(
        img_w=640,
        img_h=480,
        shutrack_early_enable=early_enable,
    )
    return ShutrackTrackerAdapter(args, "tiny", runtime=runtime), runtime


def _pool(count: int = 3) -> list:
    boxes = [
        [180, 180, 220, 240],
        [180, 180, 220, 240],
        [300, 300, 340, 360],
    ]
    return [
        FakeDetection(box=boxes[i % len(boxes)], frame_detection_index=i)
        for i in range(count)
    ]


def test_matched_event_produces_gates_and_delegates_update(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    pool = _pool()
    adapter.begin_frame(5, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=1, observation_count=3)
    detection = pool[1]
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


def test_competition_token_uses_recorded_candidate_pool(tmp_path):
    adapter, runtime = _adapter(tmp_path, competition_enabled=True)
    pool = _pool()
    adapter.begin_frame(2, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=2, observation_count=6)
    snapshot = FakeSnapshot(frame_id=2)
    event = adapter.build_matched_event(
        track,
        pool[1],
        track_idx=0,
        det_idx=1,
        association_meta=_association_meta(),
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    competition = event.evidence.competition
    # Row diff: second-best 0.8 - selected 0.2 = 0.6.
    # Column diff: second-best 0.7 - selected 0.2 = 0.5.
    # Self-excluded max IoU: detection 1 overlaps pool entry 0.
    assert competition[0] == pytest.approx(0.6, abs=1e-6)
    assert competition[1] == pytest.approx(0.5, abs=1e-6)
    assert competition[2] == pytest.approx(1.0, abs=1e-6)


def test_single_candidate_competition_uses_old_boundary(tmp_path):
    adapter, runtime = _adapter(tmp_path, competition_enabled=True)
    pool = [FakeDetection(frame_detection_index=0)]
    adapter.begin_frame(2, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=3, observation_count=6)
    snapshot = FakeSnapshot(frame_id=2)
    meta = {
        "raw_cost": np.array([[0.4]]),
        "final_cost": np.array([[0.4]]),
        "iou_similarity": np.ones((1, 1)),
        "detection_indices": np.array([0], dtype=np.int64),
        "detection_source": np.zeros((1, 1), dtype=np.int8),
    }
    event = adapter.build_matched_event(
        track,
        pool[0],
        track_idx=0,
        det_idx=0,
        association_meta=meta,
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    competition = event.evidence.competition
    assert competition[0] == pytest.approx(0.6, abs=1e-6)
    assert competition[1] == pytest.approx(0.6, abs=1e-6)
    assert competition[2] == pytest.approx(0.0, abs=1e-6)


def test_competition_disabled_checkpoint_zeros_the_token(tmp_path):
    adapter, runtime = _adapter(tmp_path, competition_enabled=False)
    pool = _pool()
    adapter.begin_frame(2, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=4, observation_count=6)
    snapshot = FakeSnapshot(frame_id=2)
    event = adapter.build_matched_event(
        track,
        pool[1],
        track_idx=0,
        det_idx=1,
        association_meta=_association_meta(),
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    assert event.evidence.competition.tolist() == [0.0, 0.0, 0.0]


def test_unmatched_event_is_zero_gated_without_model_call(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    pool = _pool()
    adapter.begin_frame(2, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=5, observation_count=2)
    snapshot = FakeSnapshot(frame_id=2)
    event = adapter.build_unmatched_event(track, np.eye(2, 3), snapshot, snapshot)
    assert event.has_detection is False
    adapter.record_unmatched_event(track.track_id, event)
    assert runtime.stats.unmatched_events == 1
    assert runtime.stats.no_detection_events == 1
    assert runtime.stats.model_inferences == 0
    assert len(runtime.event_buffers[track.track_id]) == 1


def test_reset_happens_once_and_segment_ids_match(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    pool = _pool()
    track = FakeTrack(track_id=6, observation_count=1)
    seen_masks: list[torch.Tensor] = []
    original_forward = runtime.model.forward

    def spy(*args, **kwargs):
        seen_masks.append(kwargs["history_mask"].detach().clone())
        return original_forward(*args, **kwargs)

    runtime.model.forward = spy

    adapter.begin_frame(1, 640, 480, detection_pool=pool)
    first = adapter.build_matched_event(
        track, pool[0], 0, 0, _association_meta(), np.eye(2, 3),
        FakeSnapshot(frame_id=1), FakeSnapshot(frame_id=1),
    )
    adapter.record_event(track.track_id, first, adapter.get_iwg_decisions_batch([(track.track_id, first)])[0])

    adapter.begin_frame(50, 640, 480, detection_pool=pool)
    snap50 = FakeSnapshot(frame_id=50)
    event50 = adapter.build_matched_event(
        track, pool[0], 0, 0, _association_meta(), np.eye(2, 3), snap50, snap50
    )
    decision50 = adapter.get_iwg_decisions_batch([(track.track_id, event50)])[0]
    adapter.record_event(track.track_id, event50, decision50)
    assert runtime.stats.buffer_resets == 1
    assert seen_masks[-1].tolist() == [[True] * 5]
    assert event50.evidence.segment_id == 1
    assert all(event.segment_id == event50.evidence.segment_id for event in runtime.event_buffers[track.track_id])

    adapter.begin_frame(51, 640, 480, detection_pool=pool)
    snap51 = FakeSnapshot(frame_id=51)
    event51 = adapter.build_matched_event(
        track, pool[0], 0, 0, _association_meta(), np.eye(2, 3), snap51, snap51
    )
    decision51 = adapter.get_iwg_decisions_batch([(track.track_id, event51)])[0]
    adapter.record_event(track.track_id, event51, decision51)
    assert runtime.stats.buffer_resets == 1  # same event never resets twice
    assert seen_masks[-1].tolist() == [[True, True, True, True, False]]
    assert event51.evidence.segment_id == event50.evidence.segment_id
    window = runtime.window_for(track.track_id, event51.evidence)
    np.testing.assert_allclose(window.history_evidence[-1], event50.evidence.evidence)
    assert runtime.event_buffers[track.track_id][-1].segment_id == event51.evidence.segment_id


def test_track_buffers_do_not_mix(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    pool = _pool()
    adapter.begin_frame(1, 640, 480, detection_pool=pool)
    snapshot = FakeSnapshot(frame_id=1)
    for track_id in (10, 11):
        track = FakeTrack(track_id=track_id, observation_count=3)
        event = adapter.build_matched_event(
            track, pool[0], 0, 0, _association_meta(), np.eye(2, 3), snapshot, snapshot
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


def test_default_min_observations_matches_legacy_maturity():
    assert ShutrackRuntimeConfig().min_observations == 6


def test_disable_channel_gate_forces_one(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    runtime.config = ShutrackRuntimeConfig(disable_kf_gate=True)
    adapter.begin_frame(1, 640, 480, detection_pool=_pool())
    track = FakeTrack(track_id=12)
    snapshot = FakeSnapshot(frame_id=1)
    event = adapter.build_matched_event(
        track, FakeDetection(), 0, 0, _association_meta(), np.eye(2, 3), snapshot, snapshot
    )
    decision = adapter.get_iwg_decisions_batch([(track.track_id, event)])[0]
    assert decision.motion_gate == 1.0
    assert 0.0 <= decision.appearance_gate <= 1.0


def test_recovered_track_counter(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    pool = _pool()
    adapter.begin_frame(1, 640, 480, detection_pool=pool)
    track = FakeTrack(track_id=13, observation_count=3)
    snapshot = FakeSnapshot(frame_id=1)
    unmatched = adapter.build_unmatched_event(track, np.eye(2, 3), snapshot, snapshot)
    adapter.record_unmatched_event(track.track_id, unmatched)
    adapter.begin_frame(2, 640, 480, detection_pool=pool)
    snapshot2 = FakeSnapshot(frame_id=2)
    matched = adapter.build_matched_event(
        track, pool[0], 0, 0, _association_meta(), np.eye(2, 3), snapshot2, snapshot2
    )
    adapter.get_iwg_decisions_batch([(track.track_id, matched)])
    assert runtime.stats.recovered_tracks == 1
