from __future__ import annotations

import copy

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.models.iwg_rg_cma import IWGRGCMA
from agentguard.runtime.manager import AgentGuardRuntime


def _event(track_id: int, frame_id: int, *, matched: bool = True) -> TrackEvent:
    generator = np.random.default_rng(track_id * 1000 + frame_id)
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


def _runtime(model: IWGRGCMA, output: str = "final") -> AgentGuardRuntime:
    runtime = AgentGuardRuntime(
        {
            "mode": "iwg-attn",
            "iwg_attn_output": output,
            "iwg_attn_max_frame_gap": 30,
        },
        device="cpu",
        iwg_attn_model=model,
    )
    runtime.init_feature_builder(reid_dim=16)
    return runtime


def _sequence(event: TrackEvent) -> list[TrackEvent | None]:
    return [None, None, None, None, None, event]


def test_iwg_attn_single_batch_parity_unmatched_and_no_second_buffer():
    torch.manual_seed(29)
    template = IWGRGCMA(16).eval()
    batch_runtime = _runtime(copy.deepcopy(template))
    single_runtime = _runtime(copy.deepcopy(template))
    events = [_event(1, 10), _event(2, 10)]
    batched = batch_runtime.run_iwg_attn_batch_inference(
        [1, 2],
        [_sequence(event) for event in events],
        frame_ids=[10, 10],
        has_detection=[True, True],
    )
    singles = [
        single_runtime.run_iwg_attn_inference(
            event.track_id,
            _sequence(event),
            frame_id=event.frame_id,
            has_detection=True,
        )
        for event in events
    ]
    for batch_item, single_item in zip(batched, singles):
        for key in (
            "gate",
            "base_gate",
            "refined_gate",
            "gate_correction",
            "policy_probs",
            "cue",
            "risk",
            "appearance_token",
            "motion_token",
        ):
            np.testing.assert_allclose(
                batch_item[key], single_item[key], atol=1e-6, rtol=0.0
            )
    unmatched = _event(3, 10, matched=False)
    result = batch_runtime.run_iwg_attn_inference(
        3, _sequence(unmatched), frame_id=10, has_detection=False
    )
    np.testing.assert_array_equal(result["gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["base_gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["refined_gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["policy_probs"], [0.0, 0.0, 0.0, 1.0, 0.0])
    assert batch_runtime.temporal_buffers == {}
    assert batch_runtime.window_buffers == {}
    assert batch_runtime.checkpoints.checkpoints == {}


def test_offline_six_event_outputs_match_online_streaming_and_gap_reset():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = IWGRGCMA(16).eval()
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.02)
    runtime = _runtime(copy.deepcopy(model))
    events = [_event(9, frame, matched=frame % 4 != 0) for frame in range(1, 13)]
    online = []
    for event in events:
        event_buffer = runtime.get_or_create_event_buffer(9)
        sequence = event_buffer.get_sequence()[-5:] + [event]
        online.append(
            runtime.run_iwg_attn_inference(
                9,
                sequence,
                frame_id=event.frame_id,
                has_detection=event.has_detection,
            )
        )
        event_buffer.push(event)

    inputs = runtime.feature_builder.build_iwg_input(events)
    detection = torch.tensor([[event.has_detection for event in events]])
    reset = torch.zeros_like(inputs["mask"])
    reset[:, 0] = True
    with torch.no_grad():
        offline = model.forward_sequence(
            inputs["track_feats"],
            inputs["det_feats"],
            inputs["scalar_feats"],
            inputs["mask"],
            detection,
            reset,
        )
    for key in (
        "appearance_token",
        "motion_token",
        "base_gate",
        "refined_gate",
    ):
        expected = np.stack([item[key] for item in online])
        np.testing.assert_allclose(
            offline[key][0].cpu().numpy(), expected, atol=1e-6, rtol=0.0
        )

    stale = _event(9, 100)
    runtime.run_iwg_attn_inference(
        9,
        runtime.event_buffers[9].get_sequence()[-5:] + [stale],
        frame_id=100,
        has_detection=True,
    )
    assert runtime.event_buffers[9].events == []
