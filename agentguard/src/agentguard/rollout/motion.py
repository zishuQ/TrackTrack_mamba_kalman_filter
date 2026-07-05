from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import TrackStateSnapshot
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.losses import motion_frame_loss


def compute_motion_benefit(
    event: TrackEvent,
    future_oracle_data: Dict[str, Any],
    motion_model: NSAKalmanFilter,
    future_frames: int = 5,
) -> Tuple[float, List[float], List[float], List[np.ndarray], List[np.ndarray]]:
    """Compute motion benefit for a single event.

    Two branches are simulated from the event's ``pre_update_state``:

    * **Write branch**  — ``motion_gate=1``: the detection box is written
      into the Kalman filter state via an update step **before** the future
      rollout.
    * **Skip branch**   — ``motion_gate=0``: the detection is ignored and the
      pre-update state is used as-is.

    Both branches then run through *future_frames* of oracle-guided rollout:

    1. Apply the frame's warp matrix.
    2. Kalman-filter predict.
    3. If an oracle detection exists: compute per-frame motion loss against
       the oracle box, then do a full KF update (appearance unchanged).
    4. If no oracle: predict only (no loss, no update).

    Per-frame loss
        ``L = 1 - IoU(b_oracle, b_pred) + 0.25 * L1_normalised(b_oracle, b_pred)``

    Benefit
        ``B_m = L_skip - L_write``

        If *negative* the write branch is better (we gain from writing).

    Parameters
    ----------
    event : TrackEvent
        The event to evaluate.
    future_oracle_data : dict
        Must contain keys ``future_oracle_detections`` (list of ``(4,)``
        ndarray or ``None``, length >= *future_frames*) and
        ``future_warp_matrices`` (list of ``(2, 3)`` ndarray).
    motion_model : NSAKalmanFilter
        Kalman filter instance used for predict / update.
    future_frames : int
        Number of future frames to simulate (default 5).

    Returns
    -------
    B_m : float
        Motion benefit (= L_skip - L_write).
    write_losses : list of float
        Per-frame losses for the write branch.
    skip_losses : list of float
        Per-frame losses for the skip branch.
    write_boxes : list of ndarray (4,)
        Predicted boxes per frame for the write branch.
    skip_boxes : list of ndarray (4,)
        Predicted boxes per frame for the skip branch.
    """
    pre_state = event.pre_update_state
    if pre_state is None:
        raise ValueError("Event has no pre_update_state — cannot compute motion benefit.")

    oracle_boxes: List[Optional[np.ndarray]] = future_oracle_data.get(
        "future_oracle_detections", []
    )
    warp_matrices: List[np.ndarray] = future_oracle_data.get(
        "future_warp_matrices", []
    )

    num_frames = min(future_frames, len(oracle_boxes), len(warp_matrices))
    oracle_boxes = oracle_boxes[:num_frames]
    warp_matrices = warp_matrices[:num_frames]

    # --- Write branch initial state ----------------------------------------
    if event.has_detection and event.detection is not None:
        write_mean, write_cov = _apply_detection_update(
            pre_state, event.detection.box, event.detection.score, motion_model
        )
    else:
        # No detection to write — start identical to pre_state
        write_mean = pre_state.mean.copy() if pre_state.mean is not None else None
        write_cov = pre_state.covariance.copy() if pre_state.covariance is not None else None

    # --- Skip branch initial state -----------------------------------------
    skip_mean = pre_state.mean.copy() if pre_state.mean is not None else None
    skip_cov = pre_state.covariance.copy() if pre_state.covariance is not None else None

    write_losses: List[float] = []
    skip_losses: List[float] = []
    write_boxes: List[np.ndarray] = []
    skip_boxes: List[np.ndarray] = []

    for i in range(num_frames):
        oracle_box = oracle_boxes[i]
        warp = warp_matrices[i]

        # --- Write branch step ---
        w_loss, w_box, w_mean, w_cov = _rollout_step(
            write_mean, write_cov, warp, oracle_box, motion_model
        )
        write_mean, write_cov = w_mean, w_cov
        write_losses.append(w_loss)
        write_boxes.append(w_box)

        # --- Skip branch step ---
        s_loss, s_box, s_mean, s_cov = _rollout_step(
            skip_mean, skip_cov, warp, oracle_box, motion_model
        )
        skip_mean, skip_cov = s_mean, s_cov
        skip_losses.append(s_loss)
        skip_boxes.append(s_box)

    B_m = sum(skip_losses) - sum(write_losses)
    return B_m, write_losses, skip_losses, write_boxes, skip_boxes


def compute_motion_rollout(
    event: TrackEvent,
    future_oracle_data: Dict[str, Any],
    motion_model: NSAKalmanFilter,
    future_frames: int = 5,
) -> Dict[str, Any]:
    """Full motion rollout returning detailed results for labeling.

    Parameters
    ----------
    event : TrackEvent
    future_oracle_data : dict
    motion_model : NSAKalmanFilter
    future_frames : int

    Returns
    -------
    dict
        Keys: ``benefit``, ``write_losses``, ``skip_losses``,
        ``write_boxes``, ``skip_boxes``.
    """
    B_m, write_losses, skip_losses, write_boxes, skip_boxes = compute_motion_benefit(
        event, future_oracle_data, motion_model, future_frames
    )
    return {
        "benefit": B_m,
        "write_losses": write_losses,
        "skip_losses": skip_losses,
        "write_boxes": write_boxes,
        "skip_boxes": skip_boxes,
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
    oracle_box: Optional[np.ndarray],
    motion_model: NSAKalmanFilter,
) -> Tuple[float, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run a single rollout step (warp → predict → [optional update]).

    Parameters
    ----------
    mean : ndarray (8,) or None
    cov : ndarray (8, 8) or None
    warp_matrix : ndarray (2, 3)
    oracle_box : ndarray (4,) or None
    motion_model : NSAKalmanFilter

    Returns
    -------
    loss : float
        Motion loss (0 if no oracle available).
    pred_box : ndarray (4,)
        Predicted box after warp + predict.
    new_mean : ndarray (8,) or None
        State mean after optional oracle update (None if *mean* was None).
    new_cov : ndarray (8, 8) or None
        State covariance after optional oracle update.
    """
    if mean is None or cov is None:
        return 0.0, np.zeros(4, dtype=np.float64), None, None

    # 1. Apply warp
    w_mean, w_cov = motion_model.apply_warp(mean, cov, warp_matrix)

    # 2. KF predict
    p_mean, p_cov = motion_model.predict(w_mean, w_cov)
    pred_box = motion_model.mean_to_bbox(p_mean)

    # 3. Compute loss against oracle (if available)
    if oracle_box is not None:
        loss = motion_frame_loss(oracle_box, pred_box)
    else:
        loss = 0.0

    # 4. Update with oracle (if available) to carry state forward
    if oracle_box is not None:
        measurement = motion_model.bbox_to_measurement(oracle_box)
        # Use a default score of 0.95 for oracle detections
        new_mean, new_cov = motion_model.update(p_mean, p_cov, measurement, 0.95)
    else:
        new_mean, new_cov = p_mean, p_cov

    return loss, pred_box, new_mean, new_cov
