"""SHUTrack v2 gate loss: weighted BCE + 0.1 * MSE only."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def gate_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize by channel validity and sample weight.

    This is the only supervised objective in the new path.  There is no policy
    KL, cue/risk BCE, base/final duplication, correction or abstention term.
    """

    logits = outputs["gate_logits"].float()
    target = batch["safe_gate_target"].float()
    valid = batch["valid_channels"].bool()
    sample_weight = batch["sample_weight"].float().unsqueeze(-1)
    weights = valid.float() * sample_weight
    if logits.shape != target.shape or logits.shape != weights.shape:
        raise ValueError(
            "gate logits, targets and validity weights must share a shape, got "
            f"{tuple(logits.shape)}, {tuple(target.shape)}, {tuple(weights.shape)}"
        )

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mse = (torch.sigmoid(logits) - target).square()
    elementwise = bce + 0.1 * mse
    denominator = weights.sum()
    safe_denominator = torch.where(
        denominator > 0,
        denominator,
        torch.ones_like(denominator),
    )
    normalized = (elementwise * weights).sum() / safe_denominator
    zero = logits.sum() * 0.0
    total = torch.where(denominator > 0, normalized, zero)

    with torch.no_grad():
        bce_value = torch.where(
            denominator > 0,
            (bce * weights).sum() / safe_denominator,
            zero.detach(),
        )
        mse_value = torch.where(
            denominator > 0,
            (mse * weights).sum() / safe_denominator,
            zero.detach(),
        )
        gates = torch.sigmoid(logits)
        motion_valid = weights[:, 0].sum()
        appearance_valid = weights[:, 1].sum()
        motion_mean = torch.where(
            motion_valid > 0,
            (gates[:, 0] * weights[:, 0]).sum()
            / torch.where(motion_valid > 0, motion_valid, torch.ones_like(motion_valid)),
            zero.detach(),
        )
        appearance_mean = torch.where(
            appearance_valid > 0,
            (gates[:, 1] * weights[:, 1]).sum()
            / torch.where(
                appearance_valid > 0,
                appearance_valid,
                torch.ones_like(appearance_valid),
            ),
            zero.detach(),
        )
    components = {
        "loss": total,
        "bce": bce_value,
        "mse": mse_value,
        "weight_sum": denominator.detach(),
        "motion_gate_mean": motion_mean,
        "appearance_gate_mean": appearance_mean,
    }
    return total, components


__all__ = ["gate_loss"]
