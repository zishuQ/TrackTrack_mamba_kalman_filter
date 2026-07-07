from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import TrackStateSnapshot
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.losses import motion_frame_loss


def compute_motion_benefit(
    ctx: RolloutContext,
    motion_model: NSAKalmanFilter,
    future_frames: int = 5,
    include_current: bool = False,
) -> Tuple[float, List[float], List[float], np.ndarray]:
    """Compute motion benefit with proper GT/oracle separation.

    Two branches are simulated from the event's pre-update state:

    * **Write branch**  — ``motion_gate=1``: the current candidate detection
      box is written into the Kalman filter state via an update step **before**
      the future rollout.
    * **Skip branch**   — ``motion_gate=0``: the detection is ignored and the
      pre-update state is used as-is.

    Both branches then run through *future_frames* of oracle-guided rollout:

    1. Apply the frame's warp matrix.
    2. Kalman-filter predict.
    3. Loss is computed against ``future_gt_boxes[t]`` (the GT box).
    4. KF update uses ``future_oracle_detections[t]`` with its score for
       confidence (NOT the GT box).

    Parameters
    ----------
    ctx : RolloutContext
        Context containing pre-update state, current candidate, GT boxes,
        oracle detections, and warp matrices.
    motion_model : NSAKalmanFilter
        Kalman filter instance used for predict / update.
    future_frames : int
        Number of future frames to simulate (default 5).

    Returns
    -------
    B_m : float
        Motion benefit (= L_skip - L_write). Negative means write branch is better.
    write_losses : list of float
        Per-frame losses for the write branch.
    skip_losses : list of float
        Per-frame losses for the skip branch.
    valid_mask : ndarray of bool, shape (future_frames,)
        Whether each future frame had a GT box available for loss computation.
    """
    pre_state = ctx.pre_update_state
    if pre_state is None:
        raise ValueError("Context has no pre_update_state — cannot compute motion benefit.")

    oracle_dets: List[Optional[DetectionObservation]] = ctx.future_oracle_detections
    gt_boxes: List[Optional[np.ndarray]] = ctx.future_gt_boxes
    warp_matrices: List[np.ndarray] = ctx.future_warp_matrices

    num_frames = min(future_frames, len(oracle_dets), len(warp_matrices), len(gt_boxes))
    oracle_dets = oracle_dets[:num_frames]
    gt_boxes = gt_boxes[:num_frames]
    warp_matrices = warp_matrices[:num_frames]

    # Valid mask: only compute loss when GT box is available
    valid_mask = np.array([g is not None for g in gt_boxes], dtype=bool)

    # --- Write branch initial state ----------------------------------------
    candidate = ctx.current_candidate
    if candidate is not None:
        write_mean, write_cov = _apply_detection_update(
            pre_state, candidate.box, candidate.score, motion_model
        )
    else:
        write_mean = pre_state.mean.copy() if pre_state.mean is not None else None
        write_cov = pre_state.covariance.copy() if pre_state.covariance is not None else None

    # --- Skip branch initial state -----------------------------------------
    skip_mean = pre_state.mean.copy() if pre_state.mean is not None else None
    skip_cov = pre_state.covariance.copy() if pre_state.covariance is not None else None

    write_losses: List[float] = []
    skip_losses: List[float] = []
    valid_values: List[bool] = []

    if include_current:
        current_valid = ctx.current_gt_box is not None
        valid_values.append(bool(current_valid))
        if current_valid and write_mean is not None and skip_mean is not None:
            write_box = motion_model.mean_to_bbox(write_mean)
            skip_box = motion_model.mean_to_bbox(skip_mean)
            write_losses.append(motion_frame_loss(ctx.current_gt_box, write_box))
            skip_losses.append(motion_frame_loss(ctx.current_gt_box, skip_box))
        else:
            write_losses.append(0.0)
            skip_losses.append(0.0)

    for i in range(num_frames):
        gt_box = gt_boxes[i]
        oracle_det = oracle_dets[i]
        warp = warp_matrices[i]

        # --- Write branch step ---
        w_loss, w_mean, w_cov = _rollout_step(
            write_mean, write_cov, warp, gt_box, oracle_det, motion_model
        )
        write_mean, write_cov = w_mean, w_cov
        write_losses.append(w_loss)

        # --- Skip branch step ---
        s_loss, s_mean, s_cov = _rollout_step(
            skip_mean, skip_cov, warp, gt_box, oracle_det, motion_model
        )
        skip_mean, skip_cov = s_mean, s_cov
        skip_losses.append(s_loss)
        valid_values.append(bool(valid_mask[i]))

    # Only sum over valid frames
    B_m = 0.0
    for i, valid in enumerate(valid_values if include_current else valid_mask.tolist()):
        if valid:
            B_m += skip_losses[i] - write_losses[i]

    if include_current:
        valid_mask = np.asarray(valid_values, dtype=bool)
    return B_m, write_losses, skip_losses, valid_mask


def compute_motion_rollout(
    ctx: RolloutContext,
    motion_model: NSAKalmanFilter,
    future_frames: int = 5,
    include_current: bool = False,
) -> Dict[str, Any]:
    """Full motion rollout returning detailed results for labeling.

    Parameters
    ----------
    ctx : RolloutContext
    motion_model : NSAKalmanFilter
    future_frames : int

    Returns
    -------
    dict
        Keys: ``benefit``, ``write_losses``, ``skip_losses``, ``valid_mask``.
    """
    B_m, write_losses, skip_losses, valid_mask = compute_motion_benefit(
        ctx, motion_model, future_frames, include_current=include_current
    )
    return {
        "benefit": B_m,
        "write_losses": write_losses,
        "skip_losses": skip_losses,
        "valid_mask": valid_mask.tolist(),
    }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _apply_detection_update(
    state: TrackStateSnapshot,
    det_box: np.ndarray,
    det_score: float,
    motion_model: NSAKalmanFilter,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a KF update with the detection box on top of *state*.

    Parameters
    ----------
    state : TrackStateSnapshot
    det_box : ndarray (4,) x1y1x2y2
    det_score : float
    motion_model : NSAKalmanFilter

    Returns
    -------
    mean : ndarray (8,)
    covariance : ndarray (8, 8)
    """
    if state.mean is None or state.covariance is None:
        raise ValueError("Cannot apply detection update — state has no KF mean/covariance.")
    measurement = motion_model.bbox_to_measurement(det_box)
    new_mean, new_cov = motion_model.update(
        state.mean.copy(),
        state.covariance.copy(),
        measurement,
        det_score,
    )
    return new_mean, new_cov


def _rollout_step(
    mean: Optional[np.ndarray],
    cov: Optional[np.ndarray],
    warp_matrix: np.ndarray,
    gt_box: Optional[np.ndarray],
    oracle_det: Optional[DetectionObservation],
    motion_model: NSAKalmanFilter,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run a single rollout step (warp -> predict -> [optional update]).

    Loss is computed against *gt_box* (ground truth). The KF update uses
    *oracle_det* with its own score for confidence.

    Parameters
    ----------
    mean : ndarray (8,) or None
    cov : ndarray (8, 8) or None
    warp_matrix : ndarray (2, 3)
    gt_box : ndarray (4,) or None — used for loss only
    oracle_det : DetectionObservation or None — used for KF update
    motion_model : NSAKalmanFilter

    Returns
    -------
    loss : float
        Motion loss against gt_box (0 if gt_box is None).
    new_mean : ndarray (8,) or None
    new_cov : ndarray (8, 8) or None
    """
    if mean is None or cov is None:
        return 0.0, None, None

    # 1. Apply warp
    w_mean, w_cov = motion_model.apply_warp(mean, cov, warp_matrix)

    # 2. KF predict
    p_mean, p_cov = motion_model.predict(w_mean, w_cov)
    pred_box = motion_model.mean_to_bbox(p_mean)

    # 3. Compute loss against GT box (not oracle)
    if gt_box is not None:
        loss = motion_frame_loss(gt_box, pred_box)
    else:
        loss = 0.0

    # 4. Update with oracle detection (if available) — uses its own score
    if oracle_det is not None:
        measurement = motion_model.bbox_to_measurement(oracle_det.box)
        new_mean, new_cov = motion_model.update(p_mean, p_cov, measurement, oracle_det.score)
    else:
        new_mean, new_cov = p_mean, p_cov

    return loss, new_mean, new_cov
