from __future__ import annotations

import copy

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.models.iwg_tsrm import IWGTSRM
from agentguard.runtime.manager import AgentGuardRuntime


def _event(track_id: int, frame_id: int, *, matched: bool = True) -> TrackEvent:
    generator = np.random.default_rng(track_id * 100 + frame_id)
    return TrackEvent(
        event_id=f"event-{track_id}-{frame_id}",
        dataset="test",
        sequence="seq",
        frame_id=frame_id,
        track_id=track_id,
        image_width=640,
        image_height=480,
        has_detection=matched,
        scalar_features=generator.normal(size=63).astype(np.float32),
        track_feature=generator.normal(size=16).astype(np.float32),
        detection_feature=(
            generator.normal(size=16).astype(np.float32)
            if matched
            else np.zeros(16, dtype=np.float32)
        ),
    )


def _runtime(model: IWGTSRM, joint_output: str = "final") -> AgentGuardRuntime:
    runtime = AgentGuardRuntime(
        {
            "mode": "joint",
            "joint_output": joint_output,
            "joint_window_size": 16,
            "joint_max_frame_gap": 30,
        },
        device="cpu",
        joint_model=model,
    )
    runtime.init_feature_builder(reid_dim=16)
    return runtime


def _sequence(event: TrackEvent) -> list[TrackEvent | None]:
    return [None, None, None, None, None, event]


def test_joint_runtime_single_batch_parity_and_zero_init():
    torch.manual_seed(7)
    template = IWGTSRM(reid_dim=16).eval()
    batch_runtime = _runtime(copy.deepcopy(template))
    single_runtime = _runtime(copy.deepcopy(template))
    events = [_event(1, 10), _event(2, 10)]

    batch = batch_runtime.run_joint_batch_inference(
        [1, 2],
        [_sequence(event) for event in events],
        frame_ids=[10, 10],
        has_detection=[True, True],
    )
    singles = [
        single_runtime.run_joint_inference(
            event.track_id,
            _sequence(event),
            frame_id=event.frame_id,
            has_detection=True,
        )
        for event in events
    ]
    for batched, single in zip(batch, singles):
        for key in (
            "gate",
            "base_gate",
            "final_gate",
            "temporal_gate_correction",
            "policy_probs",
            "cue",
            "risk",
        ):
            np.testing.assert_allclose(batched[key], single[key], atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(batched["final_gate"], batched["base_gate"], atol=1e-7)


def test_joint_runtime_unmatched_sentinel_and_base_output():
    runtime = _runtime(IWGTSRM(reid_dim=16).eval(), joint_output="base")
    event = _event(3, 20, matched=False)
    result = runtime.run_joint_inference(
        3, _sequence(event), frame_id=20, has_detection=False
    )
    np.testing.assert_array_equal(result["gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["base_gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["final_gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["policy_probs"], [0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(result["cue"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result["risk"], [0.0, 0.0, 1.0, 0.0])


def test_joint_runtime_temporal_buffer_resets_on_gap_and_cleanup():
    runtime = _runtime(IWGTSRM(reid_dim=16).eval())
    for frame_id in (1, 2):
        event = _event(4, frame_id)
        runtime.run_joint_inference(4, _sequence(event), frame_id=frame_id, has_detection=True)
    assert len(runtime.temporal_buffers[4].tokens) == 2

    event = _event(4, 100)
    runtime.run_joint_inference(4, _sequence(event), frame_id=100, has_detection=True)
    assert len(runtime.temporal_buffers[4].tokens) == 1
    runtime.cleanup_track(4)
    assert 4 not in runtime.temporal_buffers


def test_joint_mode_never_builds_replay_or_tgr_windows():
    runtime = _runtime(IWGTSRM(reid_dim=16).eval())
    event = _event(5, 1)
    runtime.run_joint_inference(5, _sequence(event), frame_id=1, has_detection=True)
    assert runtime.tgr is None
    assert runtime.window_buffers == {}
    assert runtime.checkpoints.checkpoints == {}
    assert runtime.finalize_first_stage() == {}
    summary = runtime.stats.summary()
    assert summary["tgr_calls"] == 0
    assert summary["replay_events"] == 0
