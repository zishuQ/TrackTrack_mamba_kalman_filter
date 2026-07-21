from __future__ import annotations

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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        -float(correction_bound),
        float(correction_bound),
    )
    residual_loss = _weighted_mean(
        F.smooth_l1_loss(
            outputs["gate_correction"].float(),
            correction_target,
            reduction="none",
        ),
        gate_weights,
    )
    revision_loss = outputs["gate_correction"].float().abs().mean()
    total = (
        base_loss
        + 0.1 * policy_loss
        + 0.2 * cue_loss
        + 0.1 * risk_loss
        + final_loss
        + 0.5 * residual_loss
        + 0.01 * revision_loss
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
    }
