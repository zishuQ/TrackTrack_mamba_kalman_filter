from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from agentguard.models.iwg_rg_cma import RG_CMA_CORRECTION_BOUND


SAFE_SELECTIVE_CONFIDENCE_START = 0.50
SAFE_SELECTIVE_CONFIDENCE_FULL = 0.75
SAFE_SELECTIVE_DECISIVENESS_START = 0.20
SAFE_SELECTIVE_DECISIVENESS_FULL = 0.50


def _piecewise_linear_ramp(
    value: torch.Tensor,
    *,
    start: float,
    full: float,
) -> torch.Tensor:
    return ((value - float(start)) / (float(full) - float(start))).clamp(
        0.0, 1.0
    )


def _safe_selective_correction_target(
    safe_target: torch.Tensor,
    oracle_target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    correction_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the CMA-only safe-to-oracle residual and its unlock strength."""
    decisiveness = (2.0 * oracle_target - 1.0).abs()
    unlock = (
        _piecewise_linear_ramp(
            confidence,
            start=SAFE_SELECTIVE_CONFIDENCE_START,
            full=SAFE_SELECTIVE_CONFIDENCE_FULL,
        )
        * _piecewise_linear_ramp(
            decisiveness,
            start=SAFE_SELECTIVE_DECISIVENESS_START,
            full=SAFE_SELECTIVE_DECISIVENESS_FULL,
        )
    ).detach()
    correction_target = torch.clamp(
        unlock * (oracle_target - safe_target),
        -float(correction_bound),
        float(correction_bound),
    )
    return correction_target, unlock


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
    no_harm_weight: float = 0.0,
) -> None:
    values = {
        "residual_beta": float(residual_beta),
        "residual_weight": float(residual_weight),
        "revision_weight": float(revision_weight),
        "hard_example_gain": float(hard_example_gain),
        "no_harm_weight": float(no_harm_weight),
    }
    if not math.isfinite(values["residual_beta"]) or values["residual_beta"] <= 0.0:
        raise ValueError("residual_beta must be finite and positive")
    for key in (
        "residual_weight",
        "revision_weight",
        "hard_example_gain",
        "no_harm_weight",
    ):
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


def compute_iwg_without_cma_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the base IWG while removing every CMA objective.

    The final gate is the base gate for this architecture. Auxiliary policy,
    CUE, and risk heads retain their normal supervision; all correction
    diagnostics are explicit zeros so downstream training reports keep one
    stable schema without pretending that CMA was evaluated.
    """
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
    policy_loss = policy_loss * outputs.get(
        "policy_loss_weight", policy_loss.new_tensor(1.0)
    )

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

    # This is deliberately one base-gate loss. Adding a second copy for the
    # identical final gate would change the optimization protocol of the
    # ablation rather than isolate the removed module.
    total = base_loss + 0.1 * policy_loss + 0.2 * cue_loss + 0.1 * risk_loss
    correction = torch.zeros_like(outputs["base_gate"])
    correction_target = torch.zeros_like(correction)
    correction_scale = torch.ones_like(correction)
    valid_correction_mask = gate_weights > 0
    correction_improvement = torch.zeros_like(correction)
    zero = total * 0.0
    return total, {
        "loss": total,
        "base_loss": base_loss,
        "base_bce": base_bce,
        "base_mse": base_mse,
        "policy_loss": policy_loss,
        "cue_loss": cue_loss,
        "risk_loss": risk_loss,
        "final_loss": base_loss,
        "final_bce": base_bce,
        "final_mse": base_mse,
        "residual_loss": zero,
        "revision_loss": zero,
        "no_harm_loss": zero,
        "correction_improvement_mean": zero,
        "correction_help_rate": zero,
        "correction_harm_rate": zero,
        "correction_scale_mean": _weighted_mean(correction_scale, gate_weights),
        "correction_target_confidence_mean": zero,
        "correction_unlock_mean": zero,
        "correction_unlock_rate": zero,
        "correction_abstain_loss": zero,
        "motion_correction_scale_mean": _weighted_mean(
            correction_scale[:, 0], gate_weights[:, 0]
        ),
        "appearance_correction_scale_mean": _weighted_mean(
            correction_scale[:, 1], gate_weights[:, 1]
        ),
        "correction_abs_mean": _weighted_mean(correction, gate_weights),
        "correction_abs_p50": _masked_quantile(
            correction, valid_correction_mask, 0.50
        ),
        "correction_abs_p95": _masked_quantile(
            correction, valid_correction_mask, 0.95
        ),
        "correction_target_abs_mean": _weighted_mean(
            correction_target, gate_weights
        ),
        "correction_nonzero_rate": zero,
        "correction_active_rate": zero,
        "correction_target_nonzero_rate": zero,
        "correction_saturation_rate": zero,
        "correction_sign_agreement": zero,
    }


def compute_iwg_rg_cma_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    correction_bound: float = RG_CMA_CORRECTION_BOUND,
    residual_beta: float = 1.0,
    residual_weight: float = 0.5,
    revision_weight: float = 0.01,
    hard_example_gain: float = 0.0,
    no_harm_weight: float = 0.0,
    residual_target_mode: str = "safe",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_iwg_rg_cma_loss_config(
        residual_beta=residual_beta,
        residual_weight=residual_weight,
        revision_weight=revision_weight,
        hard_example_gain=hard_example_gain,
        no_harm_weight=no_harm_weight,
    )
    correction_bound = float(correction_bound)
    if not math.isfinite(correction_bound) or correction_bound <= 0.0:
        raise ValueError("correction_bound must be finite and positive")
    residual_target_mode = str(residual_target_mode).strip().lower()
    if residual_target_mode not in {"safe", "oracle-confidence", "safe-selective"}:
        raise ValueError(
            "residual_target_mode must be 'safe', 'oracle-confidence', or "
            "'safe-selective', got "
            f"{residual_target_mode!r}"
        )
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
    policy_loss = policy_loss * outputs.get(
        "policy_loss_weight", policy_loss.new_tensor(1.0)
    )

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
    base_detached = outputs["base_gate"].detach().float()
    unlock = torch.ones_like(gate_weights)
    if residual_target_mode in {"oracle-confidence", "safe-selective"}:
        required = ("oracle_gate_target", "gate_confidence")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(
                "oracle residual targets are missing from the batch: "
                f"{missing}"
            )
        oracle_target = batch["oracle_gate_target"].float().clamp(0.0, 1.0)
        confidence = batch["gate_confidence"].float().clamp(0.0, 1.0)
        if residual_target_mode == "oracle-confidence":
            correction_target = torch.clamp(
                confidence * (oracle_target - base_detached),
                -correction_bound,
                correction_bound,
            )
            refined_target = torch.clamp(
                base_detached + correction_target,
                0.0,
                1.0,
            )
            diagnostic_weights = gate_weights * confidence
            # Weight samples by both label confidence and the available correction
            # opportunity. Easy/no-op endpoints should not dominate CMA updates.
            opportunity_weight = (
                0.25
                + confidence
                + 2.0 * confidence * (oracle_target - base_detached).abs().detach()
            )
            residual_weights = gate_weights * opportunity_weight
        else:
            # The safe target already contains one confidence-weighted blend
            # toward the oracle. Piecewise ramps provide a real zero region,
            # so uncertain or indecisive labels explicitly supervise abstention.
            correction_target, unlock = _safe_selective_correction_target(
                safe_target,
                oracle_target,
                confidence,
                correction_bound=correction_bound,
            )
            diagnostic_weights = gate_weights
            # CMA learns only the extra safe-to-oracle residual. Base errors
            # remain the responsibility of the independently supervised IWG.
            refined_target = torch.clamp(
                base_detached + correction_target,
                0.0,
                1.0,
            )
            opportunity_weight = 0.10 + unlock
            residual_weights = gate_weights * opportunity_weight
    else:
        confidence = torch.ones_like(gate_weights)
        refined_target = safe_target
        diagnostic_weights = gate_weights
        correction_target = torch.clamp(
            safe_target - base_detached,
            -correction_bound,
            correction_bound,
        )
        residual_weights = gate_weights
    final_loss, final_bce, final_mse = _probability_gate_loss(
        outputs["refined_gate"], refined_target, gate_weights
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
        residual_weights * hard_weight,
    )
    if residual_target_mode == "oracle-confidence":
        # Abstain only where the label is uncertain. High-confidence residuals
        # are not penalized for being nonzero.
        revision_loss = _weighted_mean(
            correction_abs,
            gate_weights * (1.0 - confidence),
        )
    elif residual_target_mode == "safe-selective":
        # Partial unlock is already encoded in correction_target. Penalizing it
        # again by (1 - unlock) shifts the optimum toward zero, so the explicit
        # abstention term is restricted to the exact-zero region.
        revision_loss = _weighted_mean(
            correction_abs,
            gate_weights * (unlock <= 0.0).to(dtype=gate_weights.dtype),
        )
    else:
        revision_loss = correction_abs.mean()
    correction_error = (
        outputs["gate_correction"].float() - correction_target
    ).abs()
    abstain_error = correction_target_abs
    no_harm_loss = _weighted_mean(
        F.relu(correction_error - abstain_error), diagnostic_weights
    )
    correction_improvement = abstain_error - correction_error
    correction_help = (correction_improvement > 1e-6).float()
    correction_harm = (correction_improvement < -1e-6).float()
    correction_scale = outputs.get(
        "correction_scale", torch.ones_like(outputs["gate_correction"])
    ).float()
    valid_correction_mask = gate_weights > 0
    sign_mask = valid_correction_mask & (correction_target_abs > 1e-6)
    correction_nonzero = (correction_abs > 1e-6).float()
    correction_active = (
        correction_abs >= 0.10 * correction_bound
    ).float()
    correction_target_nonzero = (correction_target_abs > 1e-6).float()
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
        + float(no_harm_weight) * no_harm_loss
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
        "no_harm_loss": no_harm_loss,
        "correction_improvement_mean": _weighted_mean(
            correction_improvement, gate_weights
        ),
        "correction_help_rate": _weighted_mean(correction_help, gate_weights),
        "correction_harm_rate": _weighted_mean(correction_harm, gate_weights),
        "correction_scale_mean": _weighted_mean(correction_scale, gate_weights),
        "correction_target_confidence_mean": _weighted_mean(
            confidence, gate_weights
        ),
        "correction_unlock_mean": _weighted_mean(unlock, gate_weights),
        "correction_unlock_rate": _weighted_mean(
            (unlock >= 0.5).float(), gate_weights
        ),
        "correction_abstain_loss": revision_loss,
        "motion_correction_scale_mean": _weighted_mean(
            correction_scale[:, 0], gate_weights[:, 0]
        ),
        "appearance_correction_scale_mean": _weighted_mean(
            correction_scale[:, 1], gate_weights[:, 1]
        ),
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
        "correction_active_rate": _weighted_mean(
            correction_active, gate_weights
        ),
        "correction_target_nonzero_rate": _weighted_mean(
            correction_target_nonzero, gate_weights
        ),
        "correction_saturation_rate": _weighted_mean(
            correction_saturated, gate_weights
        ),
        "correction_sign_agreement": sign_agreement,
    }
