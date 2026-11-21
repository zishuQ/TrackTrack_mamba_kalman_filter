"""SHUTrack v2 gate loss: weighted BCE + 0.1 * MSE.

Default behaviour is unchanged: one weighted mean over both gate channels.
When ``motion_low_weight > 1`` (motion-difficult experiment), motion samples
with ``safe_gate_target[:, 0] < 0.5`` get their motion-channel weight
multiplied by that factor; motion and appearance channels are then each
normalized by their own effective weight sum and merged with equal weight.
Appearance weights and samples are otherwise untouched.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def gate_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    motion_low_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize by channel validity and sample weight.

    This is the only supervised objective in the new path.  There is no policy
    KL, cue/risk BCE, base/final duplication, correction or abstention term.
    """

    factor = float(motion_low_weight)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("motion_low_weight must be finite and >= 1.0")

    logits = outputs["gate_logits"].float()
    target = batch["safe_gate_target"].float()
    valid = batch["valid_channels"].bool()
    sample_weight = batch["sample_weight"].float().unsqueeze(-1)
    base_weights = valid.float() * sample_weight
    if logits.shape != target.shape or logits.shape != base_weights.shape:
        raise ValueError(
            "gate logits, targets and validity weights must share a shape, got "
            f"{tuple(logits.shape)}, {tuple(target.shape)}, "
            f"{tuple(base_weights.shape)}"
        )

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mse = (torch.sigmoid(logits) - target).square()
    elementwise = bce + 0.1 * mse
    zero = logits.sum() * 0.0
    low_mask = (target[:, 0] < 0.5) & valid[:, 0]

    def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = weights.sum()
        safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        mean = torch.where(denominator > 0, (values * weights).sum() / safe, zero)
        return mean, denominator

    if factor == 1.0:
        total, denominator = weighted_mean(elementwise, base_weights)
        bce_value, _ = weighted_mean(bce, base_weights)
        mse_value, _ = weighted_mean(mse, base_weights)
    else:
        motion_weights = base_weights[:, 0] * torch.where(
            low_mask, torch.full_like(base_weights[:, 0], factor), torch.ones_like(base_weights[:, 0])
        )
        appearance_weights = base_weights[:, 1]
        motion_total, motion_den = weighted_mean(elementwise[:, 0], motion_weights)
        appearance_total, appearance_den = weighted_mean(
            elementwise[:, 1], appearance_weights
        )
        motion_bce, _ = weighted_mean(bce[:, 0], motion_weights)
        appearance_bce, _ = weighted_mean(bce[:, 1], appearance_weights)
        motion_mse, _ = weighted_mean(mse[:, 0], motion_weights)
        appearance_mse, _ = weighted_mean(mse[:, 1], appearance_weights)

        def merge(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            first_ok = motion_den > 0
            second_ok = appearance_den > 0
            return torch.where(
                first_ok & second_ok,
                0.5 * (first + second),
                torch.where(first_ok, first, torch.where(second_ok, second, zero)),
            )

        total = merge(motion_total, appearance_total)
        bce_value = merge(motion_bce, appearance_bce)
        mse_value = merge(motion_mse, appearance_mse)
        denominator = motion_den + appearance_den

    with torch.no_grad():
        gates = torch.sigmoid(logits)
        motion_valid_weights = base_weights[:, 0]
        reweight = torch.where(
            low_mask, torch.full_like(motion_valid_weights, factor),
            torch.ones_like(motion_valid_weights),
        )
        weighted_motion = motion_valid_weights * reweight
        weighted_motion_sum = weighted_motion.sum()
        low_count = low_mask.sum()
        high_mask = valid[:, 0] & ~low_mask
        high_count = high_mask.sum()
        low_weight_share = torch.where(
            weighted_motion_sum > 0,
            (weighted_motion * low_mask.float()).sum()
            / torch.where(
                weighted_motion_sum > 0,
                weighted_motion_sum,
                torch.ones_like(weighted_motion_sum),
            ),
            zero.detach(),
        )
        low_gate_mean = torch.where(
            low_count > 0,
            (gates[:, 0] * low_mask.float()).sum()
            / torch.where(low_count > 0, low_count.float(), torch.ones_like(low_count.float())),
            zero.detach(),
        )
        high_gate_mean = torch.where(
            high_count > 0,
            (gates[:, 0] * high_mask.float()).sum()
            / torch.where(high_count > 0, high_count.float(), torch.ones_like(high_count.float())),
            zero.detach(),
        )
        low_recall = torch.where(
            low_count > 0,
            ((gates[:, 0] < 0.5) & low_mask).float().sum()
            / torch.where(low_count > 0, low_count.float(), torch.ones_like(low_count.float())),
            zero.detach(),
        )
        false_suppression = torch.where(
            high_count > 0,
            ((gates[:, 0] < 0.5) & high_mask).float().sum()
            / torch.where(high_count > 0, high_count.float(), torch.ones_like(high_count.float())),
            zero.detach(),
        )
        motion_valid = motion_valid_weights.sum()
        appearance_valid = base_weights[:, 1].sum()
        motion_mean = torch.where(
            motion_valid > 0,
            (gates[:, 0] * motion_valid_weights).sum()
            / torch.where(motion_valid > 0, motion_valid, torch.ones_like(motion_valid)),
            zero.detach(),
        )
        appearance_mean = torch.where(
            appearance_valid > 0,
            (gates[:, 1] * base_weights[:, 1]).sum()
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
        "motion_low_weight_factor": torch.tensor(factor, device=logits.device),
        "motion_low_count": low_count.float().detach(),
        "motion_high_count": high_count.float().detach(),
        "motion_low_weight_share": low_weight_share,
        "motion_low_gate_mean": low_gate_mean,
        "motion_high_gate_mean": high_gate_mean,
        "motion_low_recall": low_recall,
        "motion_false_suppression": false_suppression,
    }
    return total, components


__all__ = [
    "gate_loss",
    "motion_box_diou_loss",
    "appearance_soft_loss",
    "appearance_rank_loss",
    "motion_speed_l1_loss",
    "motion_speed4_l1_loss",
    "gated_appearance_template",
]


def appearance_soft_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft BCE + 0.1 MSE on the appearance channel only (motion-DIoU group)."""
    logits = outputs["gate_logits"].float()[:, 1]
    target = batch["appearance_target"].float()
    valid = batch["appearance_valid"].bool()
    weight = batch["sample_weight"].float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mse = (torch.sigmoid(logits) - target).square()
    weights = valid.float() * weight
    denominator = weights.sum().clamp_min(1.0)
    loss = ((bce + 0.1 * mse) * weights).sum() / denominator
    components = {
        "loss": loss,
        "appearance_bce": ((bce * weights).sum() / denominator).detach(),
        "appearance_mse": ((mse * weights).sum() / denominator).detach(),
        "appearance_valid_ratio": valid.float().mean().detach(),
    }
    return loss, components


def _diou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """DIoU for pixel xyxy boxes, matching the CLEAR-side definition."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp_min(0.0)
    inter_h = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp_min(0.0)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1).clamp_min(0.0) * (ay2 - ay1).clamp_min(0.0)
    area_b = (bx2 - bx1).clamp_min(0.0) * (by2 - by1).clamp_min(0.0)
    union = (area_a + area_b - inter).clamp_min(1e-9)
    iou = inter / union
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    cx1 = torch.minimum(ax1, bx1)
    cy1 = torch.minimum(ay1, by1)
    cx2 = torch.maximum(ax2, bx2)
    cy2 = torch.maximum(ay2, by2)
    c_diag2 = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2).clamp_min(1e-9)
    center_dist2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    return iou - center_dist2 / c_diag2


def motion_box_diou_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    weight: float = 1.0,
    harm_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the motion gate by interpolated next-frame box DIoU.

    ``pred_box_g0`` / ``pred_box_g1`` are the two counterfactual t+1 boxes
    produced by the real NSA write-back, GMC warp and KF predict.  The model's
    motion gate interpolates them linearly in box space (verified against the
    online path), and the loss is ``1 - DIoU`` against the same identity's
    t+1 GT box.  Events without a valid GT box are masked out.

    Optional harm_weight reweights only events where the fixed g=1 endpoint
    has worse GT DIoU loss than g=0 (tolerance 1e-6). It changes importance,
    never assigns a target gate. Default 1 preserves the original path.
    """
    if not math.isfinite(float(harm_weight)) or float(harm_weight) < 1.0:
        raise ValueError("harm_weight must be finite and >= 1")
    gate = torch.sigmoid(outputs["gate_logits"].float())[:, 0]
    box0 = batch["pred_box_g0"].float()
    box1 = batch["pred_box_g1"].float()
    target = batch["gt_box_t1"].float()
    valid = batch["motion_gt_valid"].bool()
    if not (box0.shape == box1.shape == target.shape == (gate.shape[0], 4)):
        raise ValueError(
            "motion boxes and GT must be (N, 4), got "
            f"{tuple(box0.shape)}, {tuple(box1.shape)}, {tuple(target.shape)}"
        )
    box = box0 + gate.unsqueeze(-1) * (box1 - box0)
    diou = _diou_xyxy(box, target)
    elementwise = 1.0 - diou
    mask = valid.float()
    denominator = mask.sum().clamp_min(1.0)
    loss = (elementwise * mask).sum() / denominator
    components = {
        "loss": loss * float(weight),
        "diou_loss": loss.detach(),
        "valid_ratio": mask.mean().detach(),
        "diou_at_gate_mean": ((elementwise * mask).sum() / denominator).detach(),
    }
    if float(harm_weight) != 1.0:
        with torch.no_grad():
            loss0 = 1.0 - _diou_xyxy(box0, target)
            loss1 = 1.0 - _diou_xyxy(box1, target)
            harmful = valid & (loss1 > loss0 + 1e-6)
            effective_weights = mask * (1.0 + (float(harm_weight) - 1.0) * harmful.float())
            effective_sum = effective_weights.sum().clamp_min(1.0)
        weighted_loss = (elementwise * effective_weights).sum() / effective_sum
        components.update({
            "loss": weighted_loss * float(weight),
            "diou_loss": weighted_loss.detach(),
            "unweighted_diou_loss": loss.detach(),
            "harm_fraction": (harmful.float().sum() / denominator).detach(),
            "harm_weight_share": (
                (effective_weights * harmful.float()).sum() / effective_sum
            ).detach(),
        })
    return components["loss"], components


def motion_speed4_l1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    center_weight: float = 50.0,
    size_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """4-D KF velocity L1 (centre vx/vy + size vw/vh).

    ``kf_velocity4_g0`` / ``kf_velocity4_g1`` hold the t+1 KF velocities at
    the two gate endpoints, normalised by image width/height per component;
    the predicted velocity is linear in the gate (verified against the numpy
    online path).  ``gt_velocity4`` is the adjacent-frame GT label in the same
    normalisation.  With ``size_weight=0`` the loss is numerically identical
    to :func:`motion_speed_l1_loss` on the centre two dimensions.
    """
    gate = torch.sigmoid(outputs["gate_logits"].float())[:, 0]
    v0 = batch["kf_velocity4_g0"].float()
    v1 = batch["kf_velocity4_g1"].float()
    target = batch["gt_velocity4"].float()
    valid = batch["size_adj_valid"].bool()
    if not (v0.shape == v1.shape == target.shape == (gate.shape[0], 4)):
        raise ValueError(
            "4-D KF velocities and GT must be (N, 4), got "
            f"{tuple(v0.shape)}, {tuple(v1.shape)}, {tuple(target.shape)}"
        )
    velocity = v0 + gate.unsqueeze(-1) * (v1 - v0)
    error = (velocity - target).abs()
    center_error = error[:, :2].sum(dim=-1)
    size_error = error[:, 2:].sum(dim=-1)
    mask = valid.float()
    denominator = mask.sum().clamp_min(1.0)
    center_loss = (center_error * mask).sum() / denominator
    size_loss = (size_error * mask).sum() / denominator
    loss = float(center_weight) * center_loss + float(size_weight) * size_loss
    components = {
        "loss": loss,
        "speed_center_l1": center_loss.detach(),
        "speed_size_l1": size_loss.detach(),
        "speed4_l1": (center_loss + size_loss).detach(),
        "valid_ratio": mask.mean().detach(),
        "center_at_g0": (
            ((v0 - target)[:, :2].abs().sum(dim=-1) * mask).sum() / denominator
        ).detach(),
        "center_at_g1": (
            ((v1 - target)[:, :2].abs().sum(dim=-1) * mask).sum() / denominator
        ).detach(),
        "size_at_g0": (
            ((v0 - target)[:, 2:].abs().sum(dim=-1) * mask).sum() / denominator
        ).detach(),
        "size_at_g1": (
            ((v1 - target)[:, 2:].abs().sum(dim=-1) * mask).sum() / denominator
        ).detach(),
    }
    return loss, components


def gated_appearance_template(
    template0: torch.Tensor,
    detection: torch.Tensor,
    score: torch.Tensor,
    gate: torch.Tensor,
    *,
    alpha: float = 0.95,
) -> torch.Tensor:
    """Differentiable mirror of ``Track.update_with_gates`` appearance update."""
    template0 = template0 / template0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    detection = detection / detection.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    beta = float(alpha) + (1.0 - float(alpha)) * (1.0 - score)
    full = beta.unsqueeze(-1) * template0 + (1.0 - beta).unsqueeze(-1) * detection
    blended = (1.0 - gate).unsqueeze(-1) * template0 + gate.unsqueeze(-1) * full
    return blended / blended.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def appearance_column_loss(
    outputs, batch, *, alpha=0.95, tau=0.1, hinge=False, components=None,
    tensor_statistics=False,
):
    """Train the previous write against the strongest eligible rival track.

    ``hinge=False`` keeps the original softplus penalty; ``hinge=True`` uses
    ``relu`` so events whose correct track already costs no more than the
    rival contribute neither loss nor gradient.
    """
    gate = torch.sigmoid(outputs["gate_logits"].float()[:, 1])
    updated = gated_appearance_template(batch["current_track"], batch["current_detection"],
                                        batch["column_score"], gate, alpha=alpha)
    # Match cos_distance and the clipping applied by iterative_assignment.
    distance = (1 - (updated * batch["column_positive"]).sum(-1)).clamp(0, 1)
    cost = (batch["column_offset"] + 0.5 * distance).clamp(0, 1)
    margin = (cost - batch["column_rival"]) / tau
    penalty = F.relu(margin) if hinge else F.softplus(margin)
    valid = batch["column_valid"].float()
    loss = (penalty * valid).sum() / max(1, gate.numel())
    if components is not None:
        valid_count = valid.sum().detach()
        active_count = ((margin > 0) & (valid > 0)).sum().detach()
        # Training defers these statistics until epoch end to avoid CUDA syncs.
        components["valid"] = valid_count if tensor_statistics else int(valid_count)
        components["active"] = active_count if tensor_statistics else int(active_count)
        components["margin"] = margin.detach()
        components["valid_mask"] = valid.detach() > 0
    return loss


def appearance_rank_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    alpha: float = 0.95,
    tau: float = 0.1,
    rank_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Identity ranking loss on the gate-updated track template.

    The appearance gate interpolates between the unchanged template and the
    full EMA write exactly like the online ``Track.update_with_gates`` path
    (including renormalisation).  On events with a reliable future same-GT
    positive and a same-frame competitor, the loss is
    ``softplus((d_pos - d_neg) / tau)`` where both distances are cosine
    distances from the updated template.  Events without reliable competitor
    evidence fall back to the original appearance soft label (BCE + 0.1 MSE),
    matching the joint group's fallback rule.
    """
    logits = outputs["gate_logits"].float()[:, 1]
    gate = torch.sigmoid(logits)
    valid = batch["rank_valid"].bool()
    appearance_valid = batch["appearance_valid"].bool()

    def unit(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    positive = unit(batch["rank_positive"].float())
    competitor = unit(batch["rank_competitor"].float())
    score = batch["rank_score"].float()
    updated = gated_appearance_template(
        batch["rank_template0"].float(),
        batch["rank_detection"].float(),
        score,
        gate,
        alpha=alpha,
    )
    d_positive = 1.0 - (updated * positive).sum(dim=-1)
    d_competitor = 1.0 - (updated * competitor).sum(dim=-1)
    ranking = F.softplus((d_positive - d_competitor) / float(tau))

    target = batch["appearance_target"].float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mse = (torch.sigmoid(logits) - target).square()
    fallback = bce + 0.1 * mse

    rank_mask = valid & appearance_valid
    fallback_mask = appearance_valid & ~valid
    denominator = appearance_valid.float().sum().clamp_min(1.0)
    rank_term = (ranking * rank_mask.float()).sum() / denominator
    fallback_term = (fallback * fallback_mask.float()).sum() / denominator
    loss = float(rank_weight) * rank_term + fallback_term
    with torch.no_grad():
        denominator_rank = rank_mask.float().sum().clamp_min(1.0)
        denominator_fallback = fallback_mask.float().sum().clamp_min(1.0)
    components = {
        "loss": loss,
        "rank_loss": ((float(rank_weight) * rank_term).detach()),
        "rank_mean": ((ranking * rank_mask.float()).sum() / denominator_rank).detach(),
        "fallback_loss": fallback_term.detach(),
        "fallback_mean": (
            (fallback * fallback_mask.float()).sum() / denominator_fallback
        ).detach(),
        "rank_valid_ratio": rank_mask.float().mean().detach(),
        "fallback_ratio": fallback_mask.float().mean().detach(),
        "d_positive_mean": ((d_positive * rank_mask.float()).sum() / denominator_rank).detach(),
        "d_competitor_mean": (
            (d_competitor * rank_mask.float()).sum() / denominator_rank
        ).detach(),
    }
    return loss, components


def motion_speed_l1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L1 between the gate-updated KF centre velocity and the GT displacement.

    ``kf_velocity_g0`` / ``kf_velocity_g1`` are the t+1 KF centre velocities
    (already normalised by image width/height) at the two gate endpoints; the
    prediction is linear in the interpolated KF mean, verified against the
    numpy online path.  The GT target is the same-identity centre displacement
    between t and t+1 after the online GMC warp, normalised the same way.
    """
    gate = torch.sigmoid(outputs["gate_logits"].float())[:, 0]
    v0 = batch["kf_velocity_g0"].float()
    v1 = batch["kf_velocity_g1"].float()
    target = batch["gt_velocity"].float()
    valid = batch["speed_valid"].bool()
    velocity = v0 + gate.unsqueeze(-1) * (v1 - v0)
    elementwise = (velocity - target).abs().sum(dim=-1)
    mask = valid.float()
    denominator = mask.sum().clamp_min(1.0)
    loss = (elementwise * mask).sum() / denominator
    components = {
        "loss": loss * float(weight),
        "speed_l1": loss.detach(),
        "valid_ratio": mask.mean().detach(),
        "speed_at_g0": (
            ((v0 - target).abs().sum(dim=-1) * mask).sum() / denominator
        ).detach(),
        "speed_at_g1": (
            ((v1 - target).abs().sum(dim=-1) * mask).sum() / denominator
        ).detach(),
        "speed_at_gate": ((elementwise * mask).sum() / denominator).detach(),
    }
    return components["loss"], components
