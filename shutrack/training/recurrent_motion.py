"""Differentiable NSA-KF rollout pieces for continuous-window training.

The math mirrors ``agentguard.motion.nsa_numpy.NSAKalmanFilter`` and the
TrackTrack gate application in ``3. Tracker/trackers/track.py``:

* state ``[cx, cy, w, h, vx, vy, vw, vh]``, measurement ``[cx, cy, w, h]``
* motion gate interpolates mean linearly and covariance between the prior and
  the full update; covariance is symmetrised
* GMC warp is applied before the constant-velocity prediction
* evidence is the 13-dim schema from ``shutrack.schema``; the full-write
  replay checkpoints store raw scalars; the trainer applies its fitted scalar
  normalisation after reconstruction and before the model
"""
from __future__ import annotations

from functools import lru_cache

import torch

STD_POS = 1.0 / 20.0
STD_VEL = 1.0 / 160.0


@lru_cache(maxsize=8)
def _motion_mat(device, dtype):
    motion = torch.eye(8, device=device, dtype=dtype)
    for index in range(4):
        motion[index, 4 + index] = 1.0
    return motion


def _motion_cov(mean, device, dtype):
    scale = torch.stack([mean[..., 2], mean[..., 3], mean[..., 2], mean[..., 3],
                         mean[..., 2], mean[..., 3], mean[..., 2], mean[..., 3]], dim=-1)
    weights = scale.new_tensor([STD_POS]*4 + [STD_VEL]*4)
    return torch.diag_embed((scale * weights).square())


def kf_predict(mean: torch.Tensor, covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One constant-velocity prediction step (batched)."""
    device, dtype = mean.device, mean.dtype
    motion = _motion_mat(device, dtype)
    pred_mean = mean @ motion.T
    pred_cov = motion @ covariance @ motion.T + _motion_cov(pred_mean, device, dtype)
    return pred_mean, pred_cov


def kf_update(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    measurement: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full NSA update (batched). ``confidence`` has shape ``[B]``."""
    device, dtype = mean.device, mean.dtype
    projected_mean = mean[..., :4]
    scale = mean[..., [2, 3, 2, 3]] * STD_POS
    innovation = torch.diag_embed(scale.square() * (1.0-confidence).unsqueeze(-1))
    projected_cov = covariance[..., :4, :4] + innovation
    cross = covariance[..., :, :4]
    # solve() performs a host-side error check on every CUDA call. All input
    # covariances are verified by replay tests; solve_ex keeps this batch async.
    solution, _ = torch.linalg.solve_ex(projected_cov, cross.transpose(1, 2),
                                        check_errors=False)
    gain = solution.transpose(1, 2)
    residual = measurement - projected_mean
    new_mean = mean + (gain @ residual.unsqueeze(-1)).squeeze(-1)
    new_cov = covariance - gain @ projected_cov @ gain.transpose(1, 2)
    new_cov = 0.5 * (new_cov + new_cov.transpose(1, 2))
    return new_mean, new_cov


def kf_update_gated(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    measurement: torch.Tensor,
    confidence: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate-interpolated update matching ``Track.update_with_gates``."""
    full_mean, full_cov = kf_update(mean, covariance, measurement, confidence)
    weight = gate.reshape(-1, 1)
    new_mean = mean + weight * (full_mean - mean)
    new_cov = (1.0 - weight).unsqueeze(-1) * covariance + weight.unsqueeze(-1) * full_cov
    new_cov = 0.5 * (new_cov + new_cov.transpose(1, 2))
    return new_mean, new_cov


def warp_predict(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    warp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the 2x3 GMC warp then predict (batched)."""
    device, dtype = mean.device, mean.dtype
    rotation = warp[:, :, :2]
    rotation8 = torch.kron(torch.eye(4, device=device, dtype=dtype), rotation)
    translation = warp[:, :, 2]
    moved = (rotation8 @ mean.unsqueeze(-1)).squeeze(-1)
    moved = moved.clone()
    moved[:, :2] += translation
    moved_cov = rotation8 @ covariance @ rotation8.transpose(1, 2)
    return kf_predict(moved, moved_cov)


def evidence_from_state(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    measurement: torch.Tensor,
    score: torch.Tensor,
    gap: torch.Tensor,
    has_detection: torch.Tensor,
    image_width: float,
    image_height: float,
) -> torch.Tensor:
    """Rebuild the 13-dim evidence vector from a KF state and observation.

    All inputs are batched; ``measurement`` is ``[B, 4]`` in cxcywh pixels,
    ``gap`` is frames since the last observation, ``has_detection`` is bool.
    """
    pred_cx, pred_cy, pred_w, pred_h = (mean[..., i] for i in range(4))
    pred_w_safe = pred_w.clamp(min=1.0)
    pred_h_safe = pred_h.clamp(min=1.0)
    # measurement is already cxcywh (pilot stores detection.cxcywh)
    det_cx = measurement[..., 0]
    det_cy = measurement[..., 1]
    det_w = measurement[..., 2]
    det_h = measurement[..., 3]
    present = has_detection.to(mean.dtype)
    columns = [
        (det_cx - pred_cx) / pred_w_safe,
        (det_cy - pred_cy) / pred_h_safe,
        torch.log(det_w.clamp(min=1.0) / pred_w_safe),
        torch.log(det_h.clamp(min=1.0) / pred_h_safe),
        mean[..., 4] / pred_w_safe,
        mean[..., 5] / pred_h_safe,
        mean[..., 6] / pred_w_safe,
        mean[..., 7] / pred_h_safe,
        torch.log((covariance[..., 0, 0] / (image_width ** 2) + 1e-6).clamp(min=1e-12)),
        torch.log((covariance[..., 1, 1] / (image_height ** 2) + 1e-6).clamp(min=1e-12)),
        score * present,
        (gap.clamp(min=0, max=30).to(mean.dtype) / 30.0),
        present,
    ]
    for index in range(4):
        columns[index] = columns[index] * present
    return torch.stack(columns, dim=-1)


def ema_update(
    old_feature: torch.Tensor,
    new_feature: torch.Tensor,
    score: torch.Tensor,
    gate: torch.Tensor,
    alpha: float = 0.95,
) -> torch.Tensor:
    """Gate-interpolated appearance EMA with L2 renormalisation."""
    beta = alpha + (1.0 - alpha) * (1.0 - score)
    full = beta.unsqueeze(-1) * old_feature + (1.0 - beta).unsqueeze(-1) * new_feature
    weight = gate.unsqueeze(-1)
    mixed = (1.0 - weight) * old_feature + weight * full
    norm = torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)
    return mixed / (norm + 1e-12)


__all__ = [
    "kf_predict", "kf_update", "kf_update_gated", "warp_predict",
    "evidence_from_state", "ema_update",
]

