from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from agentguard.models.iwg_rg_cma import RG_CMA_CORRECTION_BOUND


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=values.dtype)
    return torch.where(
        weights.sum() > 0,
        (values * weights).sum() / weights.sum().clamp(min=1.0),
        values.sum() * 0.0,
    )


def _masked_quantile(
    values: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return torch.quantile(selected, quantile)


def validate_iwg_rg_cma_loss_config(
    *,
    residual_beta: float = 1.0,
    residual_weight: float = 0.5,
    revision_weight: float = 0.01,
    hard_example_gain: float = 0.0,
) -> None:
    values = {
        "residual_beta": float(residual_beta),
        "residual_weight": float(residual_weight),
        "revision_weight": float(revision_weight),
        "hard_example_gain": float(hard_example_gain),
    }
    if not math.isfinite(values["residual_beta"]) or values["residual_beta"] <= 0.0:
        raise ValueError("residual_beta must be finite and positive")
    for key in ("residual_weight", "revision_weight", "hard_example_gain"):
        if not math.isfinite(values[key]) or values[key] < 0.0:
            raise ValueError(f"{key} must be finite and non-negative")


def _probability_gate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction.float()
    target = target.float()
    bce = _weighted_mean(
        F.binary_cross_entropy(
            prediction.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
        ),
        weights,
    )
    mse = _weighted_mean((prediction - target).square(), weights)
    return bce + 0.1 * mse, bce, mse


def compute_iwg_rg_cma_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    correction_bound: float = RG_CMA_CORRECTION_BOUND,
    residual_beta: float = 1.0,
    residual_weight: float = 0.5,
    revision_weight: float = 0.01,
    hard_example_gain: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_iwg_rg_cma_loss_config(
        residual_beta=residual_beta,
        residual_weight=residual_weight,
        revision_weight=revision_weight,
        hard_example_gain=hard_example_gain,
    )
    correction_bound = float(correction_bound)
    if not math.isfinite(correction_bound) or correction_bound <= 0.0:
        raise ValueError("correction_bound must be finite and positive")
    valid_channels = torch.stack(
        [batch["valid_motion"], batch["valid_appearance"]], dim=-1
    )
    sample_weight = batch["sample_weight"].float()
    gate_weights = valid_channels * sample_weight.unsqueeze(-1)
    safe_target = batch["safe_gate_target"].float()

    base_loss, base_bce, base_mse = _probability_gate_loss(
        outputs["base_gate"], safe_target, gate_weights
    )
    both_valid = batch["valid_motion"] & batch["valid_appearance"]
    policy_kl = F.kl_div(
        outputs["policy_probs"].float().clamp(min=1e-8).log(),
        batch["policy_safe_soft_target"].float().clamp(min=1e-8),
        reduction="none",
    ).sum(dim=-1)
    policy_loss = _weighted_mean(policy_kl, both_valid * sample_weight)

    cue_valid = torch.stack(
        [
            batch["valid_motion"],
            batch["valid_appearance"],
            batch["valid_motion"] | batch["valid_appearance"],
        ],
        dim=-1,
    )
    cue_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            outputs["cue_logits"], batch["cue_target"].float(), reduction="none"
        ),
        cue_valid * sample_weight.unsqueeze(-1),
    )
    risk_valid = torch.stack(
        [
            batch["valid_motion"],
            batch["valid_appearance"],
            batch["valid_motion"] & batch["valid_appearance"],
            batch["valid_motion"] & batch["valid_appearance"],
        ],
        dim=-1,
    )
    risk_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            outputs["risk_logits"], batch["risk_target"].float(), reduction="none"
        ),
        risk_valid * sample_weight.unsqueeze(-1),
    )
    final_loss, final_bce, final_mse = _probability_gate_loss(
        outputs["refined_gate"], safe_target, gate_weights
    )
    correction_target = torch.clamp(
        safe_target - outputs["base_gate"].detach(),
        -correction_bound,
        correction_bound,
    )
    correction_abs = outputs["gate_correction"].float().abs()
    correction_target_abs = correction_target.abs()
    hard_weight = 1.0 + float(hard_example_gain) * (
        correction_target_abs.detach() / correction_bound
    )
    residual_loss = _weighted_mean(
        F.smooth_l1_loss(
            outputs["gate_correction"].float(),
            correction_target,
            reduction="none",
            beta=float(residual_beta),
        ),
        gate_weights * hard_weight,
    )
    revision_loss = correction_abs.mean()
    valid_correction_mask = gate_weights > 0
    sign_mask = valid_correction_mask & (correction_target_abs > 1e-6)
    correction_nonzero = (correction_abs > 1e-6).float()
    correction_saturated = (
        correction_abs >= 0.95 * correction_bound
    ).float()
    sign_agreement = _weighted_mean(
        ((outputs["gate_correction"].float() * correction_target) > 0).float(),
        gate_weights * sign_mask.float(),
    )
    total = (
        base_loss
        + 0.1 * policy_loss
        + 0.2 * cue_loss
        + 0.1 * risk_loss
        + final_loss
        + float(residual_weight) * residual_loss
        + float(revision_weight) * revision_loss
    )
    return total, {
        "loss": total,
        "base_loss": base_loss,
        "base_bce": base_bce,
        "base_mse": base_mse,
        "policy_loss": policy_loss,
        "cue_loss": cue_loss,
        "risk_loss": risk_loss,
        "final_loss": final_loss,
        "final_bce": final_bce,
        "final_mse": final_mse,
        "residual_loss": residual_loss,
        "revision_loss": revision_loss,
        "correction_abs_mean": _weighted_mean(correction_abs, gate_weights),
        "correction_abs_p50": _masked_quantile(
            correction_abs, valid_correction_mask, 0.50
        ),
        "correction_abs_p95": _masked_quantile(
            correction_abs, valid_correction_mask, 0.95
        ),
        "correction_target_abs_mean": _weighted_mean(
            correction_target_abs, gate_weights
        ),
        "correction_nonzero_rate": _weighted_mean(
            correction_nonzero, gate_weights
        ),
        "correction_saturation_rate": _weighted_mean(
            correction_saturated, gate_weights
        ),
        "correction_sign_agreement": sign_agreement,
    }
