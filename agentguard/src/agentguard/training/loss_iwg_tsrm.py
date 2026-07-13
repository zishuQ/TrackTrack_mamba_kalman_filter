from __future__ import annotations

import torch
import torch.nn.functional as F

from agentguard.models.tsrm import DYN_SCALAR_INDICES


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=values.dtype)
    numerator = (values * weights).sum()
    denominator = weights.sum()
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=1.0),
        values.sum() * 0.0,
    )


def compute_iwg_tsrm_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    training_mode: str = "joint",
    delta_max: float = 0.2,
    lambda_final: float = 1.0,
    lambda_residual: float = 0.5,
    lambda_dynamics: float = 0.1,
    lambda_revision: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if training_mode not in {"base", "joint"}:
        raise ValueError("training_mode must be 'base' or 'joint'")
    label_mask = batch["label_mask"]
    valid_motion = batch["valid_motion"]
    valid_appearance = batch["valid_appearance"]
    sample_weight = batch["sample_weight"]
    valid_channels = torch.stack([valid_motion, valid_appearance], dim=-1)
    gate_weights = (
        label_mask.unsqueeze(-1)
        * valid_channels
        * sample_weight.unsqueeze(-1)
    )

    # Gate outputs are probabilities produced by policy mixing, so their BCE
    # cannot be replaced with BCEWithLogits. Compute probability-space losses
    # explicitly in FP32 outside autocast-sensitive kernels.
    base_gate = outputs["base_gate"].float()
    base_target = batch["base_gate_target"].float()
    base_bce = _weighted_mean(
        F.binary_cross_entropy(
            base_gate.clamp(1e-6, 1.0 - 1e-6), base_target, reduction="none"
        ),
        gate_weights,
    )
    base_mse = _weighted_mean((base_gate - base_target).square(), gate_weights)
    base_gate_loss = base_bce + 0.1 * base_mse

    both_valid = label_mask & valid_motion & valid_appearance
    policy_weights = both_valid * sample_weight
    policy_kl = F.kl_div(
        outputs["policy_probs"].float().clamp(min=1e-8).log(),
        batch["policy_soft_target"].float().clamp(min=1e-8),
        reduction="none",
    ).sum(dim=-1)
    policy_loss = _weighted_mean(policy_kl, policy_weights)

    cue_valid = torch.stack(
        [valid_motion, valid_appearance, valid_motion | valid_appearance], dim=-1
    ) & label_mask.unsqueeze(-1)
    cue_weights = cue_valid * sample_weight.unsqueeze(-1)
    cue_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            outputs["cue_logits"], batch["cue_target"], reduction="none"
        ),
        cue_weights,
    )

    risk_valid = torch.stack(
        [valid_motion, valid_appearance, valid_motion & valid_appearance, valid_motion & valid_appearance],
        dim=-1,
    ) & label_mask.unsqueeze(-1)
    risk_weights = risk_valid * sample_weight.unsqueeze(-1)
    risk_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            outputs["risk_logits"], batch["risk_target"], reduction="none"
        ),
        risk_weights,
    )

    total = base_gate_loss + 0.1 * policy_loss + 0.2 * cue_loss + 0.1 * risk_loss
    zero = total * 0.0
    components = {
        "base_gate_loss": base_gate_loss,
        "base_gate_bce": base_bce,
        "base_gate_mse": base_mse,
        "policy_loss": policy_loss,
        "cue_loss": cue_loss,
        "risk_loss": risk_loss,
        "final_gate_loss": zero,
        "final_gate_bce": zero,
        "final_gate_mse": zero,
        "residual_loss": zero,
        "dynamics_loss": zero,
        "revision_loss": zero,
    }
    if training_mode == "joint":
        if "temporal_endpoint_mask" not in batch:
            raise KeyError("joint loss requires temporal_endpoint_mask")
        endpoint_mask = batch["temporal_endpoint_mask"].bool()
        if endpoint_mask.shape != label_mask.shape:
            raise ValueError(
                "temporal_endpoint_mask must match label_mask shape: "
                f"{tuple(endpoint_mask.shape)} != {tuple(label_mask.shape)}"
            )
        temporal_gate_weights = gate_weights * endpoint_mask.unsqueeze(-1)
        final_gate = outputs["final_gate"].float()
        final_target = batch["final_gate_target"].float()
        final_bce = _weighted_mean(
            F.binary_cross_entropy(
                final_gate.clamp(1e-6, 1.0 - 1e-6), final_target, reduction="none"
            ),
            temporal_gate_weights,
        )
        final_mse = _weighted_mean(
            (final_gate - final_target).square(), temporal_gate_weights
        )
        final_loss = final_bce + 0.1 * final_mse

        correction_target = torch.clamp(
            final_target - base_gate.detach(), -float(delta_max), float(delta_max)
        )
        residual_loss = _weighted_mean(
            F.smooth_l1_loss(
                outputs["temporal_gate_correction"].float(),
                correction_target,
                reduction="none",
            ),
            temporal_gate_weights,
        )

        padding = batch["padding_mask"]
        if float(lambda_dynamics) > 0.0:
            scalar = batch["scalar_feats"].float()
            indices = torch.as_tensor(DYN_SCALAR_INDICES, device=scalar.device)
            true_next_delta = scalar[:, 1:, indices] - scalar[:, :-1, indices]
            predicted = outputs["predicted_next_scalar_delta"][:, :-1].float()
            reset = batch["reset_mask"]
            dynamics_valid = ~padding[:, :-1] & ~padding[:, 1:] & ~reset[:, 1:]
            dynamics_per_position = F.smooth_l1_loss(
                predicted, true_next_delta, reduction="none"
            ).mean(dim=-1)
            dynamics_loss = _weighted_mean(dynamics_per_position, dynamics_valid)
        else:
            dynamics_loss = zero

        revision_valid = (
            endpoint_mask & ~padding & batch["has_detection_mask"]
        )
        revision_per_position = outputs["temporal_gate_correction"].float().abs().mean(dim=-1)
        revision_loss = _weighted_mean(revision_per_position, revision_valid)
        total = (
            total
            + float(lambda_final) * final_loss
            + float(lambda_residual) * residual_loss
            + float(lambda_dynamics) * dynamics_loss
            + float(lambda_revision) * revision_loss
        )
        components.update(
            {
                "final_gate_loss": final_loss,
                "final_gate_bce": final_bce,
                "final_gate_mse": final_mse,
                "residual_loss": residual_loss,
                "dynamics_loss": dynamics_loss,
                "revision_loss": revision_loss,
            }
        )
    components["loss"] = total
    return total, components
