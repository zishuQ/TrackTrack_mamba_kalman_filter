from __future__ import annotations

import copy
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "../3. Tracker"))

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


def _runtime(
    model: IWGRGCMA,
    output: str = "final",
    alpha: float = 1.0,
    *,
    disable_kf_gate: bool = False,
    disable_ema_gate: bool = False,
    fallback_threshold: float = 0.0,
    return_attention_diagnostics: bool = False,
    context_size: int | None = None,
) -> AgentGuardRuntime:
    config = {
        "mode": "iwg-rg-cma",
        "rg_cma_output": output,
        "rg_cma_alpha": alpha,
        "rg_cma_max_gap": 30,
        "disable_kf_gate": disable_kf_gate,
        "disable_ema_gate": disable_ema_gate,
        "fallback_threshold": fallback_threshold,
        "return_attention_diagnostics": return_attention_diagnostics,
    }
    if context_size is not None:
        config["iwg_context_size"] = context_size
    runtime = AgentGuardRuntime(
        config,
        device="cpu",
        rg_cma_model=model,
    )
    runtime.init_feature_builder(reid_dim=16)
    return runtime


def _sequence(event: TrackEvent, context_size: int = 6) -> list[TrackEvent | None]:
    return [None] * (context_size - 1) + [event]


def test_iwg_rg_cma_single_batch_parity_unmatched_and_no_second_buffer():
    torch.manual_seed(29)
    template = IWGRGCMA(16).eval()
    batch_runtime = _runtime(copy.deepcopy(template))
    single_runtime = _runtime(copy.deepcopy(template))
    events = [_event(1, 10), _event(2, 10)]
    batched = batch_runtime.run_iwg_rg_cma_batch_inference(
        [1, 2],
        [_sequence(event) for event in events],
        frame_ids=[10, 10],
        has_detection=[True, True],
    )
    singles = [
        single_runtime.run_iwg_rg_cma_inference(
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
            "final_gate",
            "gate_correction",
            "policy_probs",
            "cue",
        ):
                np.testing.assert_allclose(
                    batch_item[key], single_item[key], atol=2e-6, rtol=0.0
                )
    unmatched = _event(3, 10, matched=False)
    result = batch_runtime.run_iwg_rg_cma_inference(
        3, _sequence(unmatched), frame_id=10, has_detection=False
    )
    np.testing.assert_array_equal(result["gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["base_gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["final_gate"], [0.0, 0.0])
    np.testing.assert_array_equal(result["policy_probs"], [0.0, 0.0, 0.0, 1.0, 0.0])
    assert batch_runtime.event_buffers == {}


def test_runtime_alpha_scales_only_the_cma_correction():
    torch.manual_seed(30)
    model = IWGRGCMA(16).eval()
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.05)
            torch.nn.init.normal_(head[-1].bias, std=0.05)
    event = _event(4, 10)
    standard = _runtime(copy.deepcopy(model), alpha=1.0).run_iwg_rg_cma_inference(
        4, _sequence(event), frame_id=10, has_detection=True
    )
    base_only = _runtime(copy.deepcopy(model), alpha=0.0).run_iwg_rg_cma_inference(
        4, _sequence(event), frame_id=10, has_detection=True
    )
    half = _runtime(copy.deepcopy(model), alpha=0.5).run_iwg_rg_cma_inference(
        4, _sequence(event), frame_id=10, has_detection=True
    )

    np.testing.assert_allclose(base_only["final_gate"], standard["base_gate"])
    np.testing.assert_allclose(base_only["gate_correction"], 0.0)
    np.testing.assert_allclose(
        half["gate_correction"], 0.5 * standard["gate_correction"], atol=1e-7
    )
    np.testing.assert_allclose(
        half["final_gate"],
        np.clip(
            standard["base_gate"] + 0.5 * standard["gate_correction"],
            0.0,
            1.0,
        ),
        atol=1e-7,
    )


def test_gate_channel_ablations_force_only_the_disabled_channel_to_full_write():
    torch.manual_seed(32)
    model = IWGRGCMA(16).eval()
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.05)
            torch.nn.init.normal_(head[-1].bias, std=0.05)

    event = _event(5, 10)
    standard = _runtime(copy.deepcopy(model)).run_iwg_rg_cma_inference(
        5, _sequence(event), frame_id=10, has_detection=True
    )
    without_ema = _runtime(
        copy.deepcopy(model), disable_ema_gate=True
    ).run_iwg_rg_cma_inference(5, _sequence(event), frame_id=10, has_detection=True)
    without_kf = _runtime(
        copy.deepcopy(model), disable_kf_gate=True
    ).run_iwg_rg_cma_inference(5, _sequence(event), frame_id=10, has_detection=True)

    np.testing.assert_allclose(without_ema["gate"][0], standard["gate"][0])
    np.testing.assert_allclose(without_kf["gate"][1], standard["gate"][1])
    np.testing.assert_equal(without_ema["gate"][1], 1.0)
    np.testing.assert_equal(without_kf["gate"][0], 1.0)
    np.testing.assert_allclose(without_ema["final_gate"], standard["final_gate"])
    np.testing.assert_allclose(without_kf["final_gate"], standard["final_gate"])


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
            runtime.run_iwg_rg_cma_inference(
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
        "base_gate",
        "final_gate",
    ):
        expected = np.stack([item[key] for item in online])
        np.testing.assert_allclose(
            offline[key][0].cpu().numpy(), expected, atol=1e-6, rtol=0.0
        )

    stale = _event(9, 100)
    runtime.run_iwg_rg_cma_inference(
        9,
        runtime.event_buffers[9].get_sequence()[-5:] + [stale],
        frame_id=100,
        has_detection=True,
    )
    assert runtime.event_buffers[9].events == []


def test_adapter_context_size_one_does_not_reuse_history_as_current_window():
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    torch.manual_seed(33)
    runtime = AgentGuardRuntime(
        {
            "mode": "iwg-rg-cma",
            "iwg_context_size": 1,
            "rg_cma_max_gap": 30,
        },
        rg_cma_model=IWGRGCMA(16, context_size=1).eval(),
        device="cpu",
    )
    runtime.init_feature_builder(reid_dim=16)
    adapter = AgentGuardTrackerAdapter(
        SimpleNamespace(dataset="test", agentguard_mode="iwg-rg-cma"),
        "seq",
        agentguard_runtime=runtime,
    )

    first = _event(12, 1)
    second = _event(12, 2)
    adapter.get_iwg_decisions_batch([(12, first)])
    runtime.get_or_create_event_buffer(12).push(first)

    decisions = adapter.get_iwg_decisions_batch([(12, second)])

    assert len(decisions) == 1


def test_runtime_can_restore_attention_diagnostics():
    torch.manual_seed(34)
    model = IWGRGCMA(16).eval()
    captured = {}
    original = model.forward

    def wrapped(*args, **kwargs):
        captured["return_diagnostics"] = kwargs.get("return_diagnostics")
        captured["endpoint_rank"] = args[4].ndim if len(args) > 4 else None
        return original(*args, **kwargs)

    model.forward = wrapped
    runtime = _runtime(model, return_attention_diagnostics=True)
    runtime.run_iwg_rg_cma_inference(
        4, _sequence(_event(4, 10)), frame_id=10, has_detection=True
    )
    assert captured["return_diagnostics"] is True
    assert captured["endpoint_rank"] == 1


@pytest.mark.parametrize("output, alpha", [("base", 1.0), ("final", 0.0), ("final", 0.5), ("final", 1.0), ("final", 2.0)])
def test_runtime_fast_batch_preserves_order_outputs_and_event_statistics(
    output, alpha
):
    torch.manual_seed(35)
    template = IWGRGCMA(16).eval()
    with torch.no_grad():
        for head in (template.motion_correction_head, template.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.05)
            torch.nn.init.normal_(head[-1].bias, std=0.05)
    batch_runtime = _runtime(copy.deepcopy(template), output=output, alpha=alpha)
    single_runtime = _runtime(copy.deepcopy(template), output=output, alpha=alpha)
    matched = [_event(1, 10), _event(2, 10)]
    unmatched = _event(3, 10, matched=False)
    batched = batch_runtime.run_iwg_rg_cma_batch_inference(
        [1, 3, 2],
        [_sequence(matched[0]), _sequence(unmatched), _sequence(matched[1])],
        frame_ids=[10, 10, 10],
        has_detection=[True, False, True],
    )
    singles = [
        single_runtime.run_iwg_rg_cma_inference(
            1, _sequence(matched[0]), frame_id=10, has_detection=True
        ),
        single_runtime.run_iwg_rg_cma_inference(
            3, _sequence(unmatched), frame_id=10, has_detection=False
        ),
        single_runtime.run_iwg_rg_cma_inference(
            2, _sequence(matched[1]), frame_id=10, has_detection=True
        ),
    ]
    for batch_item, single_item in zip(batched, singles):
        for key in ("gate", "base_gate", "final_gate", "gate_correction", "policy_probs", "cue"):
            np.testing.assert_allclose(
                batch_item[key], single_item[key], atol=2e-6, rtol=0.0
            )
    assert batch_runtime.stats.summary()["matched_events"] == 2
    assert batch_runtime.stats.summary()["unmatched_events"] == 1
    assert (
        batch_runtime.stats.summary()["rg_cma_inference_count"]
        == single_runtime.stats.summary()["rg_cma_inference_count"]
    )


def test_unmatched_skips_model_but_resets_history_and_preserves_reacquisition():
    torch.manual_seed(36)
    model = IWGRGCMA(16).eval()
    calls = {"n": 0}
    original = model.forward

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.forward = wrapped
    runtime = _runtime(model)
    runtime.get_or_create_event_buffer(9).push(_event(9, 1))
    unmatched = _event(9, 100, matched=False)
    skipped = runtime.run_iwg_rg_cma_inference(
        9, _sequence(unmatched), frame_id=100, has_detection=False
    )
    np.testing.assert_array_equal(skipped["gate"], [0.0, 0.0])
    np.testing.assert_array_equal(skipped["policy_probs"], [0.0, 0.0, 0.0, 1.0, 0.0])
    assert calls["n"] == 0
    assert runtime.event_buffers[9].events == []

    matched = _event(9, 101)
    recovered = runtime.run_iwg_rg_cma_inference(
        9, _sequence(matched), frame_id=101, has_detection=True
    )
    assert calls["n"] == 1
    assert recovered["gate"].shape == (2,)


def test_unmatched_fast_path_still_rejects_all_padding():
    runtime = _runtime(IWGRGCMA(16).eval())
    with pytest.raises(ValueError, match="current event"):
        runtime.run_iwg_rg_cma_inference(
            1, [None] * 6, frame_id=1, has_detection=False
        )
    with pytest.raises(ValueError, match="current event"):
        runtime.run_iwg_rg_cma_inference(
            2, [None] * 6, frame_id=1, has_detection=True
        )


def test_runtime_uses_endpoint_detection_mask_and_preserves_override_semantics():
    torch.manual_seed(37)
    model = IWGRGCMA(16).eval()
    captured = []
    original = model.forward

    def wrapped(*args, **kwargs):
        captured.append(args[4] if len(args) > 4 else kwargs.get("has_detection_mask"))
        return original(*args, **kwargs)

    model.forward = wrapped
    runtime = _runtime(model)
    missing_event = _event(8, 10, matched=False)
    overridden = runtime.run_iwg_rg_cma_inference(
        8, _sequence(missing_event), frame_id=10, has_detection=True
    )
    assert captured[0].ndim == 1
    assert bool(captured[0].all())
    present = runtime.run_iwg_rg_cma_inference(
        8, _sequence(_event(8, 11)), frame_id=11, has_detection=True
    )
    np.testing.assert_equal(overridden["gate"].shape, present["gate"].shape)
    skipped = runtime.run_iwg_rg_cma_inference(
        8, _sequence(_event(8, 12)), frame_id=12, has_detection=False
    )
    np.testing.assert_array_equal(skipped["gate"], [0.0, 0.0])
    assert len(captured) == 2


@pytest.mark.parametrize("output", ["base", "final"])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("ablation", [None, "kf", "ema"])
@pytest.mark.parametrize("context_size", [6, 8])
def test_fast_path_preserves_new_ablation_fallback_and_confidence_statistics(
    output, alpha, ablation, context_size
):
    torch.manual_seed(38)
    model = IWGRGCMA(16, context_size=context_size).eval()
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.05)
            torch.nn.init.normal_(head[-1].bias, std=0.05)
    runtime = _runtime(
        model,
        output=output,
        alpha=alpha,
        disable_kf_gate=ablation == "kf",
        disable_ema_gate=ablation == "ema",
        fallback_threshold=1.0,
        context_size=context_size,
    )
    event = _event(5, 10)
    unmatched = _event(6, 10, matched=False)
    results = runtime.run_iwg_rg_cma_batch_inference(
        [5, 6],
        [_sequence(event, context_size), _sequence(unmatched, context_size)],
        frame_ids=[10, 10],
        has_detection=[True, False],
    )
    if ablation == "kf":
        np.testing.assert_equal(results[0]["gate"][0], 1.0)
    if ablation == "ema":
        np.testing.assert_equal(results[0]["gate"][1], 1.0)
    np.testing.assert_array_equal(results[1]["gate"], [0.0, 0.0])
    summary = runtime.stats.summary()
    assert summary["matched_events"] == 1
    assert summary["unmatched_events"] == 1
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate_matched"] == 1.0


@pytest.mark.parametrize("ablation", [None, "kf", "ema"])
@pytest.mark.parametrize("context_size", [6, 8])
def test_all_unmatched_fast_path_keeps_new_fallback_statistics(
    ablation, context_size
):
    torch.manual_seed(39)
    model = IWGRGCMA(16, context_size=context_size).eval()
    calls = {"n": 0}
    original = model.forward

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.forward = wrapped
    runtime = _runtime(
        model,
        disable_kf_gate=ablation == "kf",
        disable_ema_gate=ablation == "ema",
        fallback_threshold=1.0,
        context_size=context_size,
    )
    events = [_event(index, 10, matched=False) for index in (1, 2)]
    results = runtime.run_iwg_rg_cma_batch_inference(
        [1, 2],
        [_sequence(event, context_size) for event in events],
        frame_ids=[10, 10],
        has_detection=[False, False],
    )
    assert calls["n"] == 0
    for result in results:
        np.testing.assert_array_equal(result["gate"], [0.0, 0.0])
    summary = runtime.stats.summary()
    assert summary["unmatched_events"] == 2
    assert summary["matched_events"] == 0
    assert summary["fallback_count"] == 0
