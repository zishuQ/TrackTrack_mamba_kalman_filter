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


def _model_reference(runtime, sequences, has_detection):
    """Original full-batch calculation, including attention diagnostics."""
    inputs = runtime.feature_builder.build_iwg_batch_input(sequences)
    detection = torch.tensor([
        [event is not None and event.has_detection for event in sequence]
        for sequence in sequences
    ])
    reset = torch.zeros_like(inputs["mask"])
    for index, row in enumerate(inputs["mask"]):
        reset[index, (~row).nonzero()[0, 0]] = True
    with torch.inference_mode():
        outputs = runtime.rg_cma_model(
            inputs["track_feats"],
            inputs["det_feats"],
            inputs["scalar_feats"],
            inputs["mask"],
            detection,
            reset,
        )
    base = outputs["base_gate"].numpy().copy()
    correction = runtime.rg_cma_alpha * outputs["gate_correction"].numpy()
    final = np.clip(base + correction, 0.0, 1.0)
    applied = (base if runtime.rg_cma_output == "base" else final).copy()
    if runtime.disable_kf_gate:
        applied[:, 0] = 1.0
    if runtime.disable_ema_gate:
        applied[:, 1] = 1.0
    for i, matched in enumerate(has_detection):
        if (
            matched
            and runtime.fallback_threshold > 0.0
            and float(outputs["policy_probs"][i].max()) < runtime.fallback_threshold
        ):
            applied[i] = 1.0
    result = []
    for i, matched in enumerate(has_detection):
        if not matched:
            base[i] = final[i] = correction[i] = applied[i] = 0.0
        result.append(
            dict(
                gate=applied[i],
                base_gate=base[i],
                final_gate=final[i],
                gate_correction=correction[i],
                policy_probs=outputs["policy_probs"][i].numpy(),
                cue=outputs["cue"][i].numpy(),
            )
        )
    return result


def test_runtime_can_restore_attention_diagnostics(monkeypatch):
    torch.manual_seed(79)
    runtime = _runtime(IWGRGCMA(16).eval())
    event = _event(1, 10)
    sequence = _sequence(event)
    fast = runtime.run_iwg_rg_cma_inference(
        1, sequence, frame_id=10, has_detection=True
    )
    runtime.return_attention_diagnostics = True
    original_forward = runtime.rg_cma_model.forward
    calls = []

    def recording_forward(*args, **kwargs):
        calls.append(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(runtime.rg_cma_model, "forward", recording_forward)
    diagnostic = runtime.run_iwg_rg_cma_inference(
        1, sequence, frame_id=10, has_detection=True
    )
    assert calls == [{"return_diagnostics": True}]
    for key in fast:
        np.testing.assert_allclose(fast[key], diagnostic[key], atol=2e-6, rtol=0.0)


@pytest.mark.parametrize(
    "output,alpha",
    [("base", 1.0), ("final", 0.0), ("final", 0.5), ("final", 1.0), ("final", 2.0)],
)
def test_runtime_fast_batch_preserves_order_outputs_and_event_statistics(
    output, alpha
):
    torch.manual_seed(82)
    model = IWGRGCMA(16).eval()
    for head in (model.motion_correction_head, model.appearance_correction_head):
        torch.nn.init.normal_(head[-1].weight, std=0.05)
    runtime = _runtime(model, output, alpha)
    events = [_event(i, 12, matched=i % 2 == 0) for i in range(1, 5)]
    sequences = [[None] * 4 + [_event(e.track_id, 11), e] for e in events]
    expected = _model_reference(runtime, sequences, [e.has_detection for e in events])
    calls = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append((args[0].shape[0], kwargs)),
        with_kwargs=True,
    )
    try:
        actual = runtime.run_iwg_rg_cma_batch_inference(
            [e.track_id for e in events],
            sequences,
            frame_ids=[12] * 4,
            has_detection=[e.has_detection for e in events],
        )
    finally:
        handle.remove()
    assert calls == [(2, {"return_diagnostics": False})]
    for a, b in zip(actual, expected):
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=2e-6, rtol=0.0)
    assert runtime.stats.matched_events == 2
    assert runtime.stats.unmatched_events == 2
    assert runtime.stats.rg_cma_inference_count == 4


def test_unmatched_skips_model_but_resets_history_and_preserves_reacquisition(
    monkeypatch,
):
    runtime = _runtime(IWGRGCMA(16).eval())
    buffer = runtime.get_or_create_event_buffer(1)
    buffer.push(_event(1, 1))
    lost = _event(1, 100, matched=False)

    def fail(*args, **kwargs):
        raise AssertionError("unmatched endpoints must not run the model")

    with monkeypatch.context() as patch:
        patch.setattr(runtime.rg_cma_model, "forward", fail)
        patch.setattr(runtime.feature_builder, "build_iwg_batch_input", fail)
        result = runtime.run_iwg_rg_cma_inference(
            1, buffer.get_sequence()[-5:] + [lost], frame_id=100, has_detection=False
        )
    assert buffer.events == []
    assert result["policy_probs"].tolist() == [0, 0, 0, 1, 0]
    buffer.push(lost)
    matched = _event(1, 101)
    sequence = buffer.get_sequence()[-5:] + [matched]
    expected = _model_reference(runtime, [sequence], [True])[0]
    actual = runtime.run_iwg_rg_cma_inference(
        1, sequence, frame_id=101, has_detection=True
    )
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=0.0)
    assert buffer.events == [lost]
    runtime.cleanup_track(1)
    assert 1 not in runtime.event_buffers


def test_unmatched_fast_path_still_rejects_all_padding():
    runtime = _runtime(IWGRGCMA(16).eval())
    with pytest.raises(ValueError, match="only padding"):
        runtime.run_iwg_rg_cma_inference(
            1, [None] * 6, frame_id=10, has_detection=False
        )


def test_runtime_uses_endpoint_detection_mask_and_preserves_override_semantics():
    runtime = _runtime(IWGRGCMA(16).eval())
    sequence = [None] * 4 + [_event(1, 10), None]
    expected = _model_reference(runtime, [sequence], [False])[0]
    actual = runtime.run_iwg_rg_cma_inference(
        1, sequence, frame_id=10, has_detection=False
    )
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=0.0)

    unmatched_event = _event(2, 11, matched=False)
    unmatched_seq = _sequence(unmatched_event)
    skipped = runtime.run_iwg_rg_cma_inference(
        2, unmatched_seq, frame_id=11, has_detection=True
    )
    assert skipped["gate"].tolist() == [0.0, 0.0]
    np.testing.assert_array_equal(skipped["policy_probs"], [0.0, 0.0, 0.0, 1.0, 0.0])

    matched_event = _event(3, 12)
    matched_seq = _sequence(matched_event)
    expected_override = _model_reference(runtime, [matched_seq], [False])[0]
    overridden = runtime.run_iwg_rg_cma_inference(
        3, matched_seq, frame_id=12, has_detection=False
    )
    for key in overridden:
        np.testing.assert_allclose(
            overridden[key], expected_override[key], atol=2e-6, rtol=0.0
        )
    np.testing.assert_array_equal(overridden["gate"], [0.0, 0.0])
    assert not np.allclose(overridden["policy_probs"], [0.0, 0.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("context", [6, 8])
@pytest.mark.parametrize("disabled", [None, "kf", "ema"])
@pytest.mark.parametrize("threshold", [0.0, 1.0])
@pytest.mark.parametrize("output", ["base", "final"])
def test_fast_path_preserves_new_ablation_fallback_and_confidence_statistics(
    context, disabled, threshold, output
):
    torch.manual_seed(219)
    model = IWGRGCMA(16, context_size=context).eval()
    for head in (model.motion_correction_head, model.appearance_correction_head):
        torch.nn.init.normal_(head[-1].weight, std=0.05)
    runtime = AgentGuardRuntime(
        {
            "mode": "iwg-rg-cma",
            "iwg_context_size": context,
            "rg_cma_output": output,
            "rg_cma_alpha": 0.5,
            "disable_kf_gate": disabled == "kf",
            "disable_ema_gate": disabled == "ema",
            "fallback_threshold": threshold,
        },
        rg_cma_model=model,
        device="cpu",
    )
    runtime.init_feature_builder(reid_dim=16)
    events = [
        _event(i, 10, matched=m) for i, m in enumerate([True, False, True, False], 1)
    ]
    flags = [True, False, False, True]
    sequences = [[None] * (context - 1) + [e] for e in events]
    expected = _model_reference(runtime, sequences, flags)
    actual = runtime.run_iwg_rg_cma_batch_inference(
        [e.track_id for e in events],
        sequences,
        frame_ids=[10] * 4,
        has_detection=flags,
    )
    for got, wanted in zip(actual, expected):
        for key in got:
            np.testing.assert_allclose(got[key], wanted[key], atol=2e-6, rtol=0.0)
    confidence = [float(item["policy_probs"].max()) for item in expected]
    expected_fallbacks = int(threshold > 0.0)
    summary = runtime.stats.summary()
    assert summary["fallback_count"] == expected_fallbacks
    assert summary["fallback_rate_matched"] == expected_fallbacks / 2
    assert summary["avg_policy_confidence"] == pytest.approx(
        np.mean(confidence), abs=2e-6
    )
    np.testing.assert_allclose(
        runtime.stats.gate_sum,
        np.sum([e["gate"] for e in expected], axis=0),
        atol=2e-6,
    )
    assert summary["rg_cma_inference_count"] == 4
    assert summary["matched_events"] == summary["unmatched_events"] == 2
    np.testing.assert_array_equal(actual[1]["gate"], [0.0, 0.0])


@pytest.mark.parametrize("ablation", [None, "kf", "ema"])
@pytest.mark.parametrize("context_size", [6, 8])
def test_all_unmatched_fast_path_keeps_new_fallback_statistics(
    ablation, context_size, monkeypatch
):
    runtime = _runtime(
        IWGRGCMA(16, context_size=context_size).eval(),
        disable_kf_gate=ablation == "kf",
        disable_ema_gate=ablation == "ema",
        fallback_threshold=1.0,
        context_size=context_size,
    )

    def fail(*args, **kwargs):
        raise AssertionError("all unmatched must skip feature tensors and model forward")

    monkeypatch.setattr(runtime.rg_cma_model, "forward", fail)
    monkeypatch.setattr(runtime.feature_builder, "build_iwg_batch_input", fail)
    result = runtime.run_iwg_rg_cma_batch_inference(
        [1, 2],
        [
            [None] * (context_size - 1) + [_event(i, 10, matched=False)]
            for i in (1, 2)
        ],
        frame_ids=[10, 10],
        has_detection=[False, False],
    )
    for item in result:
        for key in ("gate", "base_gate", "final_gate", "gate_correction", "cue"):
            assert np.count_nonzero(item[key]) == 0
    summary = runtime.stats.summary()
    assert summary["fallback_count"] == 0
    assert summary["avg_policy_confidence"] == 1.0
    assert summary["unmatched_events"] == 2
    assert summary["matched_events"] == 0


def test_builder_has_detection_is_cpu_bool_and_false_on_padding():
    runtime = _runtime(IWGRGCMA(16).eval())
    matched = _event(1, 12)
    unmatched = _event(2, 12, matched=False)
    sequences = [
        [None, None, None, None, _event(1, 11), matched],
        [None] * 5 + [unmatched],
    ]
    inputs = runtime.feature_builder.build_iwg_batch_input(sequences)
    expected = np.array(
        [
            [event is not None and event.has_detection for event in sequence]
            for sequence in sequences
        ]
    )
    has_detection = inputs["has_detection"]
    assert has_detection.device.type == "cpu"
    assert has_detection.dtype == torch.bool
    np.testing.assert_array_equal(has_detection.numpy(), expected)
    padding = np.array(
        [[event is None for event in sequence] for sequence in sequences]
    )
    np.testing.assert_array_equal(inputs["mask"].numpy(), padding)
    assert not bool(has_detection[0, 0])
    assert bool(has_detection[0, 5])
    assert not bool(has_detection[1, 5])


def test_runtime_moves_complete_detection_mask_not_elementwise_gpu_writes(monkeypatch):
    runtime = _runtime(IWGRGCMA(16).eval())
    sequences = [[None] * 4 + [_event(1, 11), _event(1, 12)]]
    captured = {}
    original_move = AgentGuardRuntime._move_sequence_masks

    def recording_move(padding_cpu, detection_cpu, device):
        captured["padding_device"] = padding_cpu.device.type
        captured["detection_device"] = detection_cpu.device.type
        captured["detection"] = detection_cpu.detach().cpu().clone()
        captured["device"] = str(device)
        return original_move(padding_cpu, detection_cpu, device)

    monkeypatch.setattr(AgentGuardRuntime, "_move_sequence_masks", staticmethod(recording_move))
    runtime.run_iwg_rg_cma_batch_inference(
        [1], sequences, frame_ids=[12], has_detection=[True]
    )
    assert captured["padding_device"] == "cpu"
    assert captured["detection_device"] == "cpu"
    np.testing.assert_array_equal(
        captured["detection"].numpy(),
        [[event is not None and event.has_detection for event in sequences[0]]],
    )


def test_packed_runtime_heads_are_nonoverlapping_views():
    runtime = _runtime(IWGRGCMA(16).eval())
    outputs = {
        "base_gate": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        "refined_gate": torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float32),
        "gate_correction": torch.tensor([[0.01, -0.02], [0.03, -0.04]], dtype=torch.float32),
        "policy_probs": torch.tensor(
            [[0.1, 0.2, 0.3, 0.25, 0.15], [0.05, 0.15, 0.2, 0.4, 0.2]],
            dtype=torch.float32,
        ),
        "cue": torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float32),
    }
    base, refined, correction, policy, cue = AgentGuardRuntime._numpy_runtime_heads(outputs)
    assert base.shape == (2, 2)
    assert refined.shape == (2, 2)
    assert correction.shape == (2, 2)
    assert policy.shape == (2, 5)
    assert cue.shape == (2, 3)
    np.testing.assert_allclose(base, outputs["base_gate"].numpy())
    np.testing.assert_allclose(policy, outputs["policy_probs"].numpy())
    marker = base[0, 0]
    base[0, 0] = 9.0
    assert policy[0, 0] != 9.0
    assert cue[0, 0] != 9.0
    assert refined[0, 0] != 9.0
    assert correction[0, 0] != 9.0
    assert marker != 9.0


def test_left_padding_reset_is_first_valid_event_on_cpu():
    padding = torch.tensor(
        [[True, True, False, False], [False, False, False, True]],
        dtype=torch.bool,
    )
    detection = torch.tensor(
        [[False, False, True, False], [True, False, True, False]],
        dtype=torch.bool,
    )
    moved = AgentGuardRuntime._move_sequence_masks(padding, detection, torch.device("cpu"))
    padding_out, detection_out, reset = moved
    np.testing.assert_array_equal(padding_out.numpy(), padding.numpy())
    np.testing.assert_array_equal(detection_out.numpy(), detection.numpy())
    np.testing.assert_array_equal(
        reset.numpy(),
        np.array([[False, False, True, False], [True, False, False, False]]),
    )


@pytest.mark.parametrize("context_size", [6, 8])
@pytest.mark.parametrize(
    "architecture_variant,correction_bound",
    [
        ("legacy", 0.10),
        ("direct-base", 0.05),
        ("clean-cross-modal", 0.05),
        ("selective-correction", 0.05),
        ("legacy-clean-6x6-cma", 0.05),
    ],
)
def test_packed_transfer_matches_reference_across_supported_models(
    context_size, architecture_variant, correction_bound
):
    if architecture_variant != "legacy" and context_size != 6:
        pytest.skip("structural variants require context_size=6")
    torch.manual_seed(401)
    model = IWGRGCMA(
        16,
        context_size=context_size,
        correction_bound=correction_bound,
        architecture_variant=architecture_variant,
    ).eval()
    runtime = AgentGuardRuntime(
        {
            "mode": "iwg-rg-cma",
            "iwg_context_size": context_size,
            "rg_cma_output": "final",
            "rg_cma_alpha": 0.5,
            "fallback_threshold": 0.0,
        },
        rg_cma_model=model,
        device="cpu",
    )
    runtime.init_feature_builder(reid_dim=16)
    events = [_event(i, 10, matched=i != 2) for i in range(1, 5)]
    sequences = [[None] * (context_size - 2) + [_event(e.track_id, 9), e] for e in events]
    flags = [True, True, False, False]
    expected = _model_reference(runtime, sequences, flags)
    actual = runtime.run_iwg_rg_cma_batch_inference(
        [e.track_id for e in events],
        sequences,
        frame_ids=[10] * 4,
        has_detection=flags,
    )
    for got, wanted in zip(actual, expected):
        for key in got:
            np.testing.assert_allclose(got[key], wanted[key], atol=2e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_runtime_matches_cpu_public_outputs():
    torch.manual_seed(77)
    template = IWGRGCMA(16).eval()
    cpu_runtime = _runtime(copy.deepcopy(template))
    cuda_model = copy.deepcopy(template).to("cuda")
    cuda_runtime = AgentGuardRuntime(
        {
            "mode": "iwg-rg-cma",
            "rg_cma_output": "final",
            "rg_cma_alpha": 1.0,
            "rg_cma_max_gap": 30,
        },
        rg_cma_model=cuda_model,
        device="cuda",
    )
    cuda_runtime.init_feature_builder(reid_dim=16)
    events = [_event(i, 14, matched=i != 3) for i in range(1, 5)]
    sequences = [[None] * 4 + [_event(e.track_id, 13), e] for e in events]
    flags = [True, False, True, True]
    cpu = cpu_runtime.run_iwg_rg_cma_batch_inference(
        [e.track_id for e in events],
        sequences,
        frame_ids=[14] * 4,
        has_detection=flags,
    )
    cuda = cuda_runtime.run_iwg_rg_cma_batch_inference(
        [e.track_id for e in events],
        sequences,
        frame_ids=[14] * 4,
        has_detection=flags,
    )
    for cpu_item, cuda_item in zip(cpu, cuda):
        for key in cpu_item:
            np.testing.assert_allclose(
                cpu_item[key], cuda_item[key], atol=1e-4, rtol=0.0
            )
