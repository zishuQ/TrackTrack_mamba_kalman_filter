from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import DetectionObservation
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.losses import appearance_loss


def compute_appearance_benefit(
    ctx: RolloutContext,
    future_frames: int = 5,
) -> Tuple[float, List[float], List[float], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Compute appearance benefit with proper GT/oracle separation.

    Two branches are simulated from the event's pre-update state feature:

    * **Write branch** — ``appearance_gate=1``: the current candidate detection
      feature is written into the track feature via an EMA update **before**
      the future rollout.
    * **Skip branch**  — ``appearance_gate=0``: the track feature stays
      unchanged.

    Both branches then continue with EMA updates using oracle detection
    features for *future_frames*. Loss is computed against the identity
    prototype only when a GT box is present (valid mask).

    Parameters
    ----------
    ctx : RolloutContext
        Context containing pre-update state, oracle detection features/scores,
        identity prototype, and future GT box availability.
    future_frames : int
        Number of future frames to simulate (default 5).

    Returns
    -------
    B_a : float
        Appearance benefit (= L_skip - L_write). Negative means write is better.
    write_losses : list of float
        Per-frame appearance losses for the write branch.
    skip_losses : list of float
        Per-frame appearance losses for the skip branch.
    write_features : list of ndarray (D,)
        Features after each future frame for the write branch.
    skip_features : list of ndarray (D,)
        Features after each future frame for the skip branch.
    valid_mask : ndarray of bool, shape (future_frames,)
        Whether each future frame had a GT box (used for loss validity).
    """
    pre_state = ctx.pre_update_state
    if pre_state is None:
        raise ValueError(
            "Context has no pre_update_state — cannot compute appearance benefit."
        )

    identity_prototype = ctx.identity_prototype
    if identity_prototype is None:
        raise ValueError("Context has no identity_prototype — cannot compute appearance benefit.")

    old_feat = pre_state.feature.ravel().copy()
    oracle_dets: List[Optional[DetectionObservation]] = ctx.future_oracle_detections
    gt_boxes: List[Optional[np.ndarray]] = ctx.future_gt_boxes
    alpha = ctx.appearance_alpha

    num_frames = min(future_frames, len(oracle_dets), len(gt_boxes))
    oracle_dets = oracle_dets[:num_frames]
    gt_boxes = gt_boxes[:num_frames]

    # Valid mask: only meaningful when GT box is present
    valid_mask = np.array([g is not None for g in gt_boxes], dtype=bool)

    # --- Write branch initial state ---
    candidate = ctx.current_candidate
    if candidate is not None:
        det_feat = candidate.feature.ravel()
        det_score = candidate.score
        write_feat = ema_update(old_feat, det_feat, det_score, alpha)
    else:
        write_feat = old_feat.copy()

    # --- Skip branch initial state ---
    skip_feat = old_feat.copy()

    write_losses: List[float] = []
    skip_losses: List[float] = []
    write_features: List[np.ndarray] = []
    skip_features: List[np.ndarray] = []

    for i in range(num_frames):
        oracle_det = oracle_dets[i]

        # --- Write branch ---
        if oracle_det is not None:
            w_feat = ema_update(write_feat, oracle_det.feature.ravel(), oracle_det.score, alpha)
        else:
            w_feat = write_feat.copy()
        w_loss = appearance_loss(w_feat, identity_prototype)
        write_losses.append(w_loss)
        write_features.append(w_feat)
        write_feat = w_feat

        # --- Skip branch ---
        if oracle_det is not None:
            s_feat = ema_update(skip_feat, oracle_det.feature.ravel(), oracle_det.score, alpha)
        else:
            s_feat = skip_feat.copy()
        s_loss = appearance_loss(s_feat, identity_prototype)
        skip_losses.append(s_loss)
        skip_features.append(s_feat)
        skip_feat = s_feat

    # Only sum over valid frames
    B_a = 0.0
    for i in range(num_frames):
        if valid_mask[i]:
            B_a += skip_losses[i] - write_losses[i]

    return B_a, write_losses, skip_losses, write_features, skip_features, valid_mask


def ema_update(
    old_feat: np.ndarray,
    new_feat: np.ndarray,
    score: float,
    alpha: float = 0.95,
) -> np.ndarray:
    """EMA update matching the TrackTrack formula.

    .. math::

        \\beta &= \\alpha + (1 - \\alpha) \\cdot (1 - \\text{score})  \\\\
        \\text{feat} &= \\beta \\cdot \\text{old\\_feat} + (1 - \\beta) \\cdot \\text{new\\_feat}  \\\\
        \\text{feat} &= \\text{feat} / \\|\\text{feat}\\|_2

    Parameters
    ----------
    old_feat : ndarray, shape ``(D,)``
        Previous track feature.
    new_feat : ndarray, shape ``(D,)``
        New detection feature to blend in.
    score : float
        Detection score in ``[0, 1]``.
    alpha : float
        Base EMA decay (default 0.95).

    Returns
    -------
    ndarray, shape ``(D,)``
        L2-normalised updated feature.
    """
    beta = alpha + (1.0 - alpha) * (1.0 - score)
    feat = beta * old_feat + (1.0 - beta) * new_feat
    norm = np.linalg.norm(feat)
    if norm > 1e-12:
        feat = feat / norm
    return feat
