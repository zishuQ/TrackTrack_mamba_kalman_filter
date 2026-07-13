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


def test_joint_gap_clears_iwg_history_before_current_encoding():
    torch.manual_seed(23)
    template = IWGTSRM(reid_dim=16).eval()
    stale_runtime = _runtime(copy.deepcopy(template))
    fresh_runtime = _runtime(copy.deepcopy(template))
    prior = _event(6, 1)
    stale_runtime.run_joint_inference(
        6, _sequence(prior), frame_id=1, has_detection=True
    )
    stale_runtime.get_or_create_event_buffer(6).push(prior)

    current = _event(6, 100)
    stale = stale_runtime.run_joint_inference(
        6,
        [None, None, None, None, prior, current],
        frame_id=100,
        has_detection=True,
    )
    fresh = fresh_runtime.run_joint_inference(
        6, _sequence(current), frame_id=100, has_detection=True
    )
    assert stale_runtime.event_buffers[6].events == []
    assert len(stale_runtime.temporal_buffers[6].tokens) == 1
    for key in ("base_gate", "final_gate", "event_embedding", "policy_probs", "cue", "risk"):
        np.testing.assert_allclose(stale[key], fresh[key], atol=1e-6, rtol=0.0)


def test_offline_warmup_window_matches_online_streaming_tokens_and_final_gate():
    # Single-threaded CPU kernels make the 21-event batched path and the
    # streaming six-event path use reproducible accumulation order.
    torch.set_num_threads(1)
    torch.manual_seed(29)
    model = IWGTSRM(reid_dim=16).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.tsrm.delta_head[-1].weight, std=0.02)
        torch.nn.init.normal_(model.tsrm.delta_head[-1].bias, std=0.02)
    runtime = _runtime(copy.deepcopy(model))
    events = [_event(9, frame_id, matched=frame_id % 7 != 0) for frame_id in range(1, 22)]

    runtime_results = []
    for event in events:
        event_buffer = runtime.get_or_create_event_buffer(9)
        sequence = event_buffer.get_sequence()[-5:] + [event]
        result = runtime.run_joint_inference(
            9,
            sequence,
            frame_id=event.frame_id,
            has_detection=event.has_detection,
        )
        runtime_results.append(result)
        event_buffer.push(event)

    builder = runtime.feature_builder
    iwg_inputs = builder.build_iwg_input(events)
    formal_length = 16
    offline_batch = {
        "iwg_track_feats": iwg_inputs["track_feats"],
        "iwg_det_feats": iwg_inputs["det_feats"],
        "iwg_scalar_feats": iwg_inputs["scalar_feats"],
        "iwg_padding_mask": iwg_inputs["mask"],
        "track_feats": iwg_inputs["track_feats"][:, -formal_length:],
        "det_feats": iwg_inputs["det_feats"][:, -formal_length:],
        "scalar_feats": iwg_inputs["scalar_feats"][:, -formal_length:],
        "padding_mask": iwg_inputs["mask"][:, -formal_length:],
        "has_detection_mask": torch.tensor(
            [[event.has_detection for event in events[-formal_length:]]],
            dtype=torch.bool,
        ),
        "reset_mask": torch.tensor(
            [[True] + [False] * (formal_length - 1)], dtype=torch.bool
        ),
    }
    with torch.no_grad():
        offline = model(offline_batch)

    online_formal = runtime_results[-formal_length:]
    comparisons = {
        "base_gate": np.stack([item["base_gate"] for item in online_formal]),
        "event_embedding": np.stack(
            [item["event_embedding"] for item in online_formal]
        ),
        "policy_probs": np.stack([item["policy_probs"] for item in online_formal]),
        "cue": np.stack([item["cue"] for item in online_formal]),
        "risk": np.stack([item["risk"] for item in online_formal]),
    }
    for key, online in comparisons.items():
        np.testing.assert_allclose(
            offline[key][0].cpu().numpy(), online, atol=1e-6, rtol=0.0
        )
    np.testing.assert_allclose(
        offline["final_gate"][0, -1].cpu().numpy(),
        runtime_results[-1]["final_gate"],
        atol=1e-6,
        rtol=0.0,
    )


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
