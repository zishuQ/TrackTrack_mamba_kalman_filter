from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch

from agentguard.models.iwg_rg_cma import IWGRGCMA
from agentguard.training.loss_iwg_rg_cma import compute_iwg_rg_cma_loss
from agentguard.training.train_iwg_rg_cma import (
    _build_iwg_rg_cma_optimizer,
    _forward,
    _optimizer_group_grad_norms,
    _optimizer_learning_rates,
    _run_training_attempt,
    _scheduler,
    _seed_everything,
)
from agentguard.training.train_stats import (
    EXPENSIVE_COMPONENT_KEYS,
    TrainingEpochStats,
    resolve_diagnostic_interval,
    resolve_epoch_log_interval,
    resolve_log_interval_setting,
    should_sample_diagnostics,
)


TRAINING_COMPONENT_KEYS = (
    "loss",
    "base_loss",
    "base_bce",
    "base_mse",
    "policy_loss",
    "cue_loss",
    "risk_loss",
    "final_loss",
    "final_bce",
    "final_mse",
    "residual_loss",
    "revision_loss",
    "no_harm_loss",
    "correction_abstain_loss",
)
REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "agentguard"


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_batch(
    *,
    batch_size: int = 4,
    length: int = 6,
    reid_dim: int = 16,
    seed: int = 11,
    valid_motion: torch.Tensor | None = None,
    valid_appearance: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    padding = torch.zeros((batch_size, length), dtype=torch.bool)
    detection = ~padding
    reset = torch.zeros_like(padding)
    reset[:, 0] = True
    policy = torch.rand((batch_size, 5), generator=generator)
    policy = policy / policy.sum(dim=-1, keepdim=True)
    if valid_motion is None:
        valid_motion = torch.ones(batch_size, dtype=torch.bool)
    if valid_appearance is None:
        valid_appearance = torch.ones(batch_size, dtype=torch.bool)
    if sample_weight is None:
        sample_weight = torch.ones(batch_size)
    batch = {
        "track_feats": torch.randn(
            batch_size, length, reid_dim, generator=generator
        ),
        "det_feats": torch.randn(
            batch_size, length, reid_dim, generator=generator
        ),
        "scalar_feats": torch.randn(
            batch_size, length, 63, generator=generator
        ),
        "padding_mask": padding,
        "has_detection_mask": detection,
        "reset_mask": reset,
        "safe_gate_target": torch.rand((batch_size, 2), generator=generator),
        "policy_safe_soft_target": policy,
        "cue_target": torch.rand((batch_size, 3), generator=generator),
        "risk_target": torch.rand((batch_size, 4), generator=generator),
        "valid_motion": valid_motion,
        "valid_appearance": valid_appearance,
        "sample_weight": sample_weight,
    }
    device = torch.device(device)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _scenario_batches(device: torch.device | str = "cpu") -> list[dict[str, torch.Tensor]]:
    device = torch.device(device)
    return [
        _make_batch(batch_size=4, seed=11, device=device),
        _make_batch(
            batch_size=4,
            seed=12,
            valid_motion=torch.tensor([True, True, False, True]),
            valid_appearance=torch.tensor([True, False, True, False]),
            device=device,
        ),
        _make_batch(
            batch_size=4,
            seed=13,
            sample_weight=torch.tensor([0.2, 1.5, 0.0, 3.0]),
            device=device,
        ),
        _make_batch(batch_size=1, seed=14, device=device),
        _make_batch(
            batch_size=3,
            seed=15,
            valid_motion=torch.zeros(3, dtype=torch.bool),
            valid_appearance=torch.zeros(3, dtype=torch.bool),
            sample_weight=torch.tensor([1.0, 2.0, 0.5]),
            device=device,
        ),
    ]


def _legacy_accumulate(
    *,
    components: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count: int,
    grad_norm: torch.Tensor | float,
    group_grad_norms: dict[str, torch.Tensor | float],
    totals: dict[str, float],
    motion_positions: torch.Tensor,
    appearance_positions: torch.Tensor,
    cross_matrix: torch.Tensor,
    motion_entropy: float,
    appearance_entropy: float,
    channel_abs_error: list[float],
    channel_correct: list[float],
    channel_weight: list[float],
) -> tuple[float, float]:
    for key, value in components.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
    totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm) * count
    for group_name, group_grad_norm in group_grad_norms.items():
        key = f"{group_name}_grad_norm"
        totals[key] = totals.get(key, 0.0) + float(group_grad_norm) * count
    motion_positions += (
        outputs["motion_attention_weights"]
        .detach()
        .double()
        .mean(dim=1)
        .sum(dim=0)
        .cpu()
    )
    appearance_positions += (
        outputs["appearance_attention_weights"]
        .detach()
        .double()
        .mean(dim=1)
        .sum(dim=0)
        .cpu()
    )
    if "cross_modal_attention_weights" in outputs:
        cross_matrix += (
            outputs["cross_modal_attention_weights"]
            .detach()
            .double()
            .mean(dim=1)
            .sum(dim=0)
            .cpu()
        )
    motion_entropy += float(outputs["motion_attention_entropy"].detach().mean()) * count
    appearance_entropy += (
        float(outputs["appearance_attention_entropy"].detach().mean()) * count
    )
    valid_channels = torch.stack(
        [batch["valid_motion"], batch["valid_appearance"]], dim=-1
    )
    gate_weights = (
        valid_channels.float() * batch["sample_weight"].float().unsqueeze(-1)
    )
    gate_error = (outputs["refined_gate"].detach() - batch["safe_gate_target"]).abs()
    gate_correct = (
        (outputs["refined_gate"].detach() >= 0.5)
        == (batch["safe_gate_target"] >= 0.5)
    ).float()
    for channel in range(2):
        channel_abs_error[channel] += float(
            (gate_error[:, channel] * gate_weights[:, channel]).sum()
        )
        channel_correct[channel] += float(
            (gate_correct[:, channel] * gate_weights[:, channel]).sum()
        )
        channel_weight[channel] += float(gate_weights[:, channel].sum())
    return motion_entropy, appearance_entropy


def _legacy_metrics(
    *,
    totals: dict[str, float],
    samples: int,
    n_batches: int,
    motion_positions: torch.Tensor,
    appearance_positions: torch.Tensor,
    cross_matrix: torch.Tensor,
    motion_entropy: float,
    appearance_entropy: float,
    channel_abs_error: list[float],
    channel_correct: list[float],
    channel_weight: list[float],
) -> dict:
    total_channel_weight = sum(channel_weight)
    metrics = {
        "samples": samples,
        "n_batches": n_batches,
        **{key: value / max(samples, 1) for key, value in totals.items()},
        "gate_accuracy": sum(channel_correct) / max(total_channel_weight, 1.0),
        "motion_mae": channel_abs_error[0] / max(channel_weight[0], 1.0),
        "appearance_mae": channel_abs_error[1] / max(channel_weight[1], 1.0),
        "motion_attention_entropy": motion_entropy / max(samples, 1),
        "appearance_attention_entropy": appearance_entropy / max(samples, 1),
        "motion_history_attention": (
            motion_positions / max(samples, 1)
        ).tolist(),
        "appearance_history_attention": (
            appearance_positions / max(samples, 1)
        ).tolist(),
        "cross_modal_attention": (cross_matrix / max(samples, 1)).tolist(),
    }
    return metrics


def _init_model(device: torch.device | str = "cpu") -> IWGRGCMA:
    _seed_everything(7)
    model = IWGRGCMA(16)
    return model.to(device).train()


def _clone_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _run_training_snapshots(
    batches: list[dict[str, torch.Tensor]],
    *,
    diagnostic_interval: int,
    device: torch.device | str = "cpu",
):
    device = torch.device(device)
    _seed_everything(7)
    model = _init_model(device)
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )
    scheduler = _scheduler(
        optimizer, total_steps=max(len(batches), 1), warmup_steps=1
    )
    snapshots = []
    for batch_index, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        sample = should_sample_diagnostics(batch_index, diagnostic_interval)
        loss, components = compute_iwg_rg_cma_loss(
            outputs,
            batch,
            compute_expensive_diagnostics=sample,
        )
        training_components = {
            key: components[key].detach().clone()
            for key in TRAINING_COMPONENT_KEYS
        }
        loss.backward()
        pre_clip_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        if sample:
            _optimizer_group_grad_norms(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        post_clip_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        optimizer.step()
        scheduler.step()
        snapshots.append(
            {
                "loss": loss.detach().clone(),
                "components": training_components,
                "pre_clip_grads": pre_clip_grads,
                "grad_norm": grad_norm.detach().clone()
                if torch.is_tensor(grad_norm)
                else torch.tensor(float(grad_norm)),
                "post_clip_grads": post_clip_grads,
                "params": _clone_params(model),
                "lrs": _optimizer_learning_rates(optimizer),
            }
        )
    return snapshots


def test_expensive_diagnostics_do_not_change_training_loss_or_grads():
    device = torch.device("cpu")
    batch = _make_batch(
        sample_weight=torch.tensor([0.5, 1.25, 0.0, 2.0]),
        valid_motion=torch.tensor([True, False, True, True]),
        valid_appearance=torch.tensor([True, True, False, True]),
        device=device,
    )
    model = _init_model(device)
    outputs = _forward(model, batch)
    loss_on, components_on = compute_iwg_rg_cma_loss(
        outputs, batch, compute_expensive_diagnostics=True
    )
    loss_off, components_off = compute_iwg_rg_cma_loss(
        outputs, batch, compute_expensive_diagnostics=False
    )
    torch.testing.assert_close(loss_on, loss_off)
    for key in TRAINING_COMPONENT_KEYS:
        torch.testing.assert_close(components_on[key], components_off[key])
        assert components_on[key].requires_grad
    assert "correction_abs_p50" in components_on
    assert "correction_abs_p95" in components_on
    assert "correction_abs_p50" not in components_off
    assert components_on["correction_abs_mean"].grad_fn is None
    assert components_off["correction_abs_mean"].grad_fn is None

    model.zero_grad(set_to_none=True)
    loss_on.backward(retain_graph=True)
    grads_on = _clone_params_grads(model)
    model.zero_grad(set_to_none=True)
    loss_off.backward()
    grads_off = _clone_params_grads(model)
    assert set(grads_on) == set(grads_off)
    for name in grads_on:
        torch.testing.assert_close(grads_on[name], grads_off[name])


def _clone_params_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.parametrize("interval", [0, 1, 3])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_diagnostic_interval_does_not_change_training_computation(interval, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    batches = _scenario_batches(device)
    reference = _run_training_snapshots(
        batches, diagnostic_interval=1, device=device
    )
    candidate = _run_training_snapshots(
        batches, diagnostic_interval=interval, device=device
    )
    assert len(reference) == len(candidate)
    for left, right in zip(reference, candidate):
        torch.testing.assert_close(left["loss"], right["loss"])
        for key in TRAINING_COMPONENT_KEYS:
            torch.testing.assert_close(left["components"][key], right["components"][key])
        assert set(left["pre_clip_grads"]) == set(right["pre_clip_grads"])
        for name in left["pre_clip_grads"]:
            torch.testing.assert_close(
                left["pre_clip_grads"][name], right["pre_clip_grads"][name]
            )
            torch.testing.assert_close(
                left["post_clip_grads"][name], right["post_clip_grads"][name]
            )
        torch.testing.assert_close(left["grad_norm"], right["grad_norm"])
        for name in left["params"]:
            torch.testing.assert_close(left["params"][name], right["params"][name])
        assert left["lrs"] == right["lrs"]


def test_device_stats_match_legacy_when_interval_is_one():
    device = torch.device("cpu")
    batches = _scenario_batches(device)
    model = _init_model(device)
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )
    stats = TrainingEpochStats(
        device=device,
        context_size=6,
        cross_modal_token_count=model.cross_modal_token_count,
        diagnostic_interval=1,
    )
    totals: dict[str, float] = {}
    samples = 0
    motion_positions = torch.zeros(6, dtype=torch.float64)
    appearance_positions = torch.zeros(6, dtype=torch.float64)
    cross_matrix = torch.zeros((3, 3), dtype=torch.float64)
    motion_entropy = 0.0
    appearance_entropy = 0.0
    channel_abs_error = [0.0, 0.0]
    channel_correct = [0.0, 0.0]
    channel_weight = [0.0, 0.0]
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        loss, components = compute_iwg_rg_cma_loss(outputs, batch)
        loss.backward()
        group_grad_norms = _optimizer_group_grad_norms(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(batch["track_feats"].shape[0])
        samples += count
        stats.update(
            components=components,
            outputs=outputs,
            batch=batch,
            count=count,
            grad_norm=grad_norm,
            group_grad_norms=group_grad_norms,
            include_expensive=True,
        )
        motion_entropy, appearance_entropy = _legacy_accumulate(
            components=components,
            outputs=outputs,
            batch=batch,
            count=count,
            grad_norm=grad_norm,
            group_grad_norms=group_grad_norms,
            totals=totals,
            motion_positions=motion_positions,
            appearance_positions=appearance_positions,
            cross_matrix=cross_matrix,
            motion_entropy=motion_entropy,
            appearance_entropy=appearance_entropy,
            channel_abs_error=channel_abs_error,
            channel_correct=channel_correct,
            channel_weight=channel_weight,
        )
    new_metrics = stats.as_metrics()
    old_metrics = _legacy_metrics(
        totals=totals,
        samples=samples,
        n_batches=len(batches),
        motion_positions=motion_positions,
        appearance_positions=appearance_positions,
        cross_matrix=cross_matrix,
        motion_entropy=motion_entropy,
        appearance_entropy=appearance_entropy,
        channel_abs_error=channel_abs_error,
        channel_correct=channel_correct,
        channel_weight=channel_weight,
    )
    assert new_metrics["diagnostics_available"] is True
    assert new_metrics["diagnostic_batches"] == len(batches)
    assert new_metrics["diagnostic_samples"] == samples
    assert new_metrics["correction_quantile_definition"] == "mean_of_batch_quantiles"
    compared_keys = [
        "loss",
        "base_loss",
        "final_loss",
        "residual_loss",
        "correction_abs_mean",
        "correction_abs_p50",
        "correction_abs_p95",
        "correction_sign_agreement",
        "grad_norm",
        "base_iwg_grad_norm",
        "rg_cma_grad_norm",
        "gate_accuracy",
        "motion_mae",
        "appearance_mae",
        "motion_attention_entropy",
        "appearance_attention_entropy",
    ]
    for key in compared_keys:
        assert new_metrics[key] == pytest.approx(old_metrics[key], rel=1e-6, abs=1e-7)
    assert new_metrics["motion_history_attention"] == pytest.approx(
        old_metrics["motion_history_attention"], rel=1e-6, abs=1e-7
    )
    torch.testing.assert_close(
        torch.tensor(new_metrics["cross_modal_attention"]),
        torch.tensor(old_metrics["cross_modal_attention"]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_sampled_diagnostics_match_independent_reference():
    device = torch.device("cpu")
    batches = _scenario_batches(device)
    interval = 2
    model = _init_model(device)
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )
    stats = TrainingEpochStats(
        device=device,
        context_size=6,
        cross_modal_token_count=model.cross_modal_token_count,
        diagnostic_interval=interval,
    )
    ref_totals: dict[str, float] = {}
    ref_samples = 0
    motion_positions = torch.zeros(6, dtype=torch.float64)
    appearance_positions = torch.zeros(6, dtype=torch.float64)
    cross_matrix = torch.zeros((3, 3), dtype=torch.float64)
    motion_entropy = 0.0
    appearance_entropy = 0.0
    dummy_channel = [0.0, 0.0]
    for batch_index, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        sample = should_sample_diagnostics(batch_index, interval)
        loss, components = compute_iwg_rg_cma_loss(
            outputs, batch, compute_expensive_diagnostics=sample
        )
        loss.backward()
        group_grad_norms = (
            _optimizer_group_grad_norms(optimizer) if sample else {}
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(batch["track_feats"].shape[0])
        stats.update(
            components=components,
            outputs=outputs,
            batch=batch,
            count=count,
            grad_norm=grad_norm,
            group_grad_norms=group_grad_norms,
            include_expensive=sample,
        )
        if sample:
            ref_samples += count
            motion_entropy, appearance_entropy = _legacy_accumulate(
                components=components,
                outputs=outputs,
                batch=batch,
                count=count,
                grad_norm=grad_norm,
                group_grad_norms=group_grad_norms,
                totals=ref_totals,
                motion_positions=motion_positions,
                appearance_positions=appearance_positions,
                cross_matrix=cross_matrix,
                motion_entropy=motion_entropy,
                appearance_entropy=appearance_entropy,
                channel_abs_error=dummy_channel,
                channel_correct=dummy_channel,
                channel_weight=dummy_channel,
            )
    metrics = stats.as_metrics()
    assert metrics["diagnostic_interval"] == interval
    assert metrics["diagnostic_batches"] == sum(
        1 for index in range(len(batches)) if should_sample_diagnostics(index, interval)
    )
    assert metrics["diagnostic_samples"] == ref_samples
    for key in ("correction_abs_p50", "correction_abs_p95", "base_iwg_grad_norm"):
        assert metrics[key] == pytest.approx(
            ref_totals[key] / max(ref_samples, 1), rel=1e-6, abs=1e-7
        )
    assert metrics["motion_attention_entropy"] == pytest.approx(
        motion_entropy / max(ref_samples, 1), rel=1e-6, abs=1e-7
    )


def test_unsampled_diagnostics_are_explicit_missing_values():
    device = torch.device("cpu")
    batch = _make_batch(device=device)
    model = _init_model(device)
    stats = TrainingEpochStats(
        device=device,
        context_size=6,
        cross_modal_token_count=model.cross_modal_token_count,
        diagnostic_interval=0,
    )
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = _forward(model, batch)
    loss, components = compute_iwg_rg_cma_loss(
        outputs, batch, compute_expensive_diagnostics=False
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    stats.update(
        components=components,
        outputs=outputs,
        batch=batch,
        count=int(batch["track_feats"].shape[0]),
        grad_norm=grad_norm,
        group_grad_norms={},
        include_expensive=False,
    )
    metrics = stats.as_metrics()
    assert metrics["diagnostics_available"] is False
    assert metrics["diagnostic_batches"] == 0
    assert metrics["diagnostic_samples"] == 0
    for key in (
        *EXPENSIVE_COMPONENT_KEYS,
        "base_iwg_grad_norm",
        "rg_cma_grad_norm",
        "motion_attention_entropy",
        "appearance_attention_entropy",
        "motion_history_attention",
        "appearance_history_attention",
        "cross_modal_attention",
    ):
        assert metrics[key] is None
    assert metrics["loss"] is not None
    assert metrics["correction_abs_mean"] is not None
    assert metrics["grad_norm"] is not None


def test_epoch_stats_reset_between_epochs():
    device = torch.device("cpu")
    first, second = _scenario_batches(device)[:2]
    model = _init_model(device)
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )

    def _run(batch):
        stats = TrainingEpochStats(
            device=device,
            context_size=6,
            cross_modal_token_count=model.cross_modal_token_count,
            diagnostic_interval=1,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        loss, components = compute_iwg_rg_cma_loss(outputs, batch)
        loss.backward()
        group_grad_norms = _optimizer_group_grad_norms(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        stats.update(
            components=components,
            outputs=outputs,
            batch=batch,
            count=int(batch["track_feats"].shape[0]),
            grad_norm=grad_norm,
            group_grad_norms=group_grad_norms,
            include_expensive=True,
        )
        return stats.as_metrics()

    first_metrics = _run(first)
    second_metrics = _run(second)
    assert first_metrics["samples"] == 4
    assert second_metrics["samples"] == 4
    assert first_metrics["loss"] != pytest.approx(second_metrics["loss"])


def test_accumulator_does_not_grow_or_retain_graph():
    device = torch.device("cpu")
    model = _init_model(device)
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )
    stats = TrainingEpochStats(
        device=device,
        context_size=6,
        cross_modal_token_count=model.cross_modal_token_count,
        diagnostic_interval=1,
    )
    fingerprint = None
    for index in range(24):
        batch = _make_batch(batch_size=3 if index == 23 else 4, seed=20 + index)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        loss, components = compute_iwg_rg_cma_loss(outputs, batch)
        loss.backward()
        group_grad_norms = _optimizer_group_grad_norms(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        stats.update(
            components=components,
            outputs=outputs,
            batch=batch,
            count=int(batch["track_feats"].shape[0]),
            grad_norm=grad_norm,
            group_grad_norms=group_grad_norms,
            include_expensive=True,
        )
        current = stats.buffer_fingerprint()
        if fingerprint is None:
            fingerprint = current
        else:
            assert current[0] == fingerprint[0]
            assert current[1] == fingerprint[1]
        assert stats.retains_graph() is False
    assert stats.batches == 24
    assert stats.samples == 4 * 23 + 3


def test_interval_helpers_and_log_cadence():
    assert resolve_diagnostic_interval({}) == 0
    assert resolve_diagnostic_interval({"diagnostic_interval": 1}) == 1
    assert resolve_log_interval_setting({}) == 0
    assert resolve_epoch_log_interval(0, 100) == 20
    assert resolve_epoch_log_interval(7, 100) == 7
    assert should_sample_diagnostics(0, 0) is False
    assert should_sample_diagnostics(0, 1) is True
    assert [should_sample_diagnostics(index, 3) for index in range(6)] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="non-negative"):
        resolve_diagnostic_interval({"diagnostic_interval": -1})


def test_render_script_handles_unsampled_attention(tmp_path):
    renderer = _load_script("render_iwg_rg_cma_training_log.py")
    handle = io.StringIO()
    renderer._render_epoch(
        handle,
        {
            "epoch": 1,
            "loss": 1.0,
            "base_loss": 0.4,
            "final_loss": 0.3,
            "policy_loss": 0.1,
            "cue_loss": 0.1,
            "risk_loss": 0.1,
            "residual_loss": 0.01,
            "revision_loss": 0.0,
            "learning_rate": 1e-4,
            "epoch_time_s": 1.2,
            "motion_history_attention": None,
            "appearance_history_attention": None,
            "motion_attention_entropy": None,
            "appearance_attention_entropy": None,
            "diagnostics_available": False,
            "diagnostic_interval": 0,
            "diagnostic_batches": 0,
        },
        2,
        0.0,
    )
    text = handle.getvalue()
    assert "motion_H=n/a" in text
    assert "diagnostics=unsampled" in text


def test_smoke_training_records_diagnostic_config(tmp_path):
    class TinyDataset(torch.utils.data.Dataset):
        metadata = {
            "format": "train_data",
            "dataset": "MOT17",
            "split": "all",
            "train_sequences": ["A"],
            "timeline_counts": {"A": {"labeled_endpoints": 6}},
            "reid_dim": 16,
            "scalar_dim": 63,
            "event_dim": 128,
            "context_size": 6,
            "max_frame_gap": 30,
            "index_format": "train_data",
        }

        def __init__(self):
            import numpy as np

            self.norm_stats = type(
                "Norm", (), {"mean": np.zeros(63), "std": np.ones(63)}
            )()

        def __len__(self):
            return 6

        def __getitem__(self, index):
            item = _make_batch(batch_size=1, seed=index)
            return {key: value.squeeze(0) for key, value in item.items()}

        def release_cached_pages(self):
            return {"mmap_regions": 0, "files": 0}

    dataset = TinyDataset()
    summary = _run_training_attempt(
        {
            "seed": 42,
            "device": "cpu",
            "epochs": 2,
            "batch_size": 4,
            "num_workers": 0,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "grad_clip": 1.0,
            "amp": False,
            "checkpoint_dir": str(tmp_path),
            "memory_shards": 1,
            "epochs_per_shard": 2,
            "shard_cycles": 1,
            "diagnostic_interval": 0,
            "log_interval": 1,
        },
        dataset=dataset,
        batch_size=4,
    )
    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    checkpoint = torch.load(
        tmp_path / "iwg_rg_cma_last.pt", map_location="cpu", weights_only=False
    )
    assert summary["epochs"] == 2
    assert checkpoint["training_config"]["diagnostic_interval"] == 0
    assert checkpoint["training_config"]["log_interval"] == 1
    assert metrics[0]["diagnostics_available"] is False
    assert metrics[0]["correction_abs_p50"] is None
    assert metrics[0]["motion_history_attention"] is None
    assert metrics[0]["log_interval"] == 1
    assert metrics[1]["diagnostics_available"] is False
    assert metrics[0]["loss"] != pytest.approx(metrics[1]["loss"])


def _cuda_time_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def test_training_stat_performance_and_peak_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _make_batch(batch_size=32, device=device)
    model = _init_model(device)
    optimizer = _build_iwg_rg_cma_optimizer(
        model, base_lr=1e-4, cma_lr=1e-4, weight_decay=1e-4
    )

    def _one_step(include_expensive: bool, legacy: bool, stats: TrainingEpochStats | None):
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        loss, components = compute_iwg_rg_cma_loss(
            outputs, batch, compute_expensive_diagnostics=include_expensive
        )
        loss.backward()
        group_grad_norms = (
            _optimizer_group_grad_norms(optimizer) if include_expensive else {}
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(batch["track_feats"].shape[0])
        if legacy:
            totals: dict[str, float] = {}
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
            totals["grad_norm"] = float(grad_norm) * count
            for name, value in group_grad_norms.items():
                totals[f"{name}_grad_norm"] = float(value) * count
            _ = (
                outputs["motion_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
                .cpu()
            )
            _ = float(outputs["motion_attention_entropy"].detach().mean())
            _ = float(outputs["appearance_attention_entropy"].detach().mean())
        elif stats is not None:
            stats.update(
                components=components,
                outputs=outputs,
                batch=batch,
                count=count,
                grad_norm=grad_norm,
                group_grad_norms=group_grad_norms,
                include_expensive=include_expensive,
            )
        return outputs, components, loss

    def _reset_stats():
        return TrainingEpochStats(
            device=device,
            context_size=6,
            cross_modal_token_count=model.cross_modal_token_count,
            diagnostic_interval=1,
        )

    warmup = 3
    repeats = 8
    if device.type != "cuda":
        for _ in range(2):
            _one_step(True, True, None)
            _one_step(True, False, _reset_stats())
            _one_step(False, False, _reset_stats())
        pytest.skip("CUDA unavailable; GPU performance was not measured")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    legacy_ms = _cuda_time_ms(
        lambda: _one_step(True, True, None), warmup, repeats
    )
    torch.cuda.synchronize(device)
    legacy_peak = int(torch.cuda.max_memory_allocated(device))

    torch.cuda.reset_peak_memory_stats(device)
    stats_every = _reset_stats()
    torch.cuda.synchronize(device)
    every_ms = _cuda_time_ms(
        lambda: _one_step(True, False, stats_every), warmup, repeats
    )
    torch.cuda.synchronize(device)
    every_peak = int(torch.cuda.max_memory_allocated(device))

    torch.cuda.reset_peak_memory_stats(device)
    stats_sparse = TrainingEpochStats(
        device=device,
        context_size=6,
        cross_modal_token_count=model.cross_modal_token_count,
        diagnostic_interval=8,
    )
    counter = {"step": 0}

    def _sparse_step():
        include = should_sample_diagnostics(counter["step"], 8)
        counter["step"] += 1
        _one_step(include, False, stats_sparse)

    torch.cuda.synchronize(device)
    sparse_ms = _cuda_time_ms(_sparse_step, warmup, repeats)
    torch.cuda.synchronize(device)
    sparse_peak = int(torch.cuda.max_memory_allocated(device))

    optimizer.zero_grad(set_to_none=True)
    outputs = _forward(model, batch)
    loss, components = compute_iwg_rg_cma_loss(outputs, batch)
    loss.backward()
    group_grad_norms = _optimizer_group_grad_norms(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    count = int(batch["track_feats"].shape[0])
    isolated_device = TrainingEpochStats(
        device=device,
        context_size=6,
        cross_modal_token_count=model.cross_modal_token_count,
        diagnostic_interval=1,
    )

    def _legacy_stats_only():
        totals: dict[str, float] = {}
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
        _ = float(grad_norm) * count
        for name, value in group_grad_norms.items():
            totals[f"{name}_grad_norm"] = float(value) * count
        _ = (
            outputs["motion_attention_weights"]
            .detach()
            .double()
            .mean(dim=1)
            .sum(dim=0)
            .cpu()
        )
        _ = (
            outputs["appearance_attention_weights"]
            .detach()
            .double()
            .mean(dim=1)
            .sum(dim=0)
            .cpu()
        )
        if "cross_modal_attention_weights" in outputs:
            _ = (
                outputs["cross_modal_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
                .cpu()
            )
        _ = float(outputs["motion_attention_entropy"].detach().mean())
        _ = float(outputs["appearance_attention_entropy"].detach().mean())

    def _device_stats_only(include_expensive: bool):
        isolated_device.update(
            components=components,
            outputs=outputs,
            batch=batch,
            count=count,
            grad_norm=grad_norm,
            group_grad_norms=group_grad_norms if include_expensive else {},
            include_expensive=include_expensive,
        )

    legacy_stats_ms = _cuda_time_ms(_legacy_stats_only, warmup, repeats)
    device_stats_ms = _cuda_time_ms(
        lambda: _device_stats_only(True), warmup, repeats
    )
    sparse_stats_ms = _cuda_time_ms(
        lambda: _device_stats_only(False), warmup, repeats
    )
    report = {
        "device": str(device),
        "includes_dataloader": False,
        "includes_log_write": False,
        "warmup": warmup,
        "repeats": repeats,
        "batch_size": 32,
        "legacy_step_ms": legacy_ms,
        "device_stats_every_batch_ms": every_ms,
        "device_stats_interval_8_ms": sparse_ms,
        "legacy_stats_only_ms": legacy_stats_ms,
        "device_stats_only_every_batch_ms": device_stats_ms,
        "device_stats_only_unsampled_ms": sparse_stats_ms,
        "legacy_peak_bytes": legacy_peak,
        "device_stats_every_batch_peak_bytes": every_peak,
        "device_stats_interval_8_peak_bytes": sparse_peak,
    }
    print("TRAIN_STATS_PERF " + json.dumps(report, sort_keys=True))
    assert stats_every.retains_graph() is False
    assert stats_sparse.retains_graph() is False
    assert every_ms > 0.0
    assert sparse_ms > 0.0
