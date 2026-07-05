from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.appearance import ema_update
from agentguard.rollout.losses import appearance_loss, motion_frame_loss


def compute_tgr_window_labels(
    window_events: List[TrackEvent],
    future_oracle_data: Dict[str, Any],
    motion_model: NSAKalmanFilter,
    identity_prototype: np.ndarray,
    future_frames: int = 3,
) -> Dict[str, Any]:
    """Compute TGR window labels for a 4-event window.

    Enumerates all 16 binary sequences of length 4.  For each sequence:

    1. **Replay** in-window events from the window checkpoint (the first
       event's ``frame_start_state``).

       - Apply warp matrix
       - KF predict
       - If the event has a detection **and** ``sequence[i] == 1``:
         KF update with the detection box + EMA update with the detection
         feature.
       - If ``sequence[i] == 0`` or the event has no detection: predict only
         (no update).  No-detection positions carry ``valid_mask=0``.

    2. **Continue** with oracle detections for ``future_frames`` after the
       window.

       - Apply warp, KF predict, compute motion loss.
       - EMA update with oracle feature (if available), compute appearance
         loss.

    3. Accumulate total motion and appearance loss over the future frames.

    **Selection**:

    * The **motion sequence** is the binary sequence with the lowest total
      motion loss.
    * The **appearance sequence** is the binary sequence with the lowest total
      appearance loss.  If multiple sequences are within ``1e-6`` of the
      minimum, the one with more ``1`` bits is preferred.

    Parameters
    ----------
    window_events : list of TrackEvent
        Exactly 4 events, ordered oldest first.  These are the four events
        in a TGR window.
    future_oracle_data : dict
        Must contain keys:

        * ``future_oracle_detections`` — list of ``(4,)`` ndarray or
          ``None``, length >= *future_frames*.
        * ``future_warp_matrices`` — list of ``(2, 3)`` ndarray.
        * ``future_oracle_features`` — list of ``(1, D)`` ndarray or
          ``None``.
        * ``future_oracle_scores`` — list of float.
    motion_model : NSAKalmanFilter
    identity_prototype : ndarray, shape ``(D,)``
    future_frames : int
        Number of oracle-guided frames after the window (default 3).

    Returns
    -------
    dict
        Keys:

        * ``motion_sequence`` — ``(4,)`` np.int64, best motion gate sequence.
        * ``appearance_sequence`` — ``(4,)`` np.int64, best appearance gate
          sequence.
        * ``valid_mask`` — ``(4,)`` np.bool_, ``True`` where event has
          detection.
        * ``motion_losses`` — ``(16,)`` total motion loss per sequence.
        * ``appearance_losses`` — ``(16,)`` total appearance loss per
          sequence.
        * ``selected_motion_loss`` — float, the minimum motion loss.
        * ``selected_appearance_loss`` — float, the minimum appearance loss.
    """
    if len(window_events) != 4:
        raise ValueError(
            f"Expected exactly 4 window events, got {len(window_events)}."
        )

    seq_len = 4
    num_sequences = 1 << seq_len  # 16

    # Pre-compute per-event data so we don't repeatedly dig into objects.
    has_det: List[bool] = []
    det_boxes: List[Optional[np.ndarray]] = []
    det_scores: List[float] = []
    det_features: List[Optional[np.ndarray]] = []
    warp_mats: List[np.ndarray] = []

    for evt in window_events:
        has_det.append(evt.has_detection)
        if evt.has_detection and evt.detection is not None:
            det_boxes.append(evt.detection.box.copy())
            det_scores.append(evt.detection.score)
            det_features.append(evt.detection.feature.ravel().copy())
        else:
            det_boxes.append(None)
            det_scores.append(0.0)
            det_features.append(None)
        warp_mats.append(evt.warp_matrix.copy())

    valid_mask = np.array(has_det, dtype=np.bool_)

    # Window checkpoint — state before the first event in the window
    checkpoint = window_events[0].frame_start_state
    if checkpoint is None:
        raise ValueError(
            "First window event has no frame_start_state — cannot replay window."
        )

    initial_mean = checkpoint.mean.copy() if checkpoint.mean is not None else None
    initial_cov = checkpoint.covariance.copy() if checkpoint.covariance is not None else None
    initial_feat = checkpoint.feature.ravel().copy()

    # Oracle data for after-window rollout
    oracle_boxes: List[Optional[np.ndarray]] = future_oracle_data.get(
        "future_oracle_detections", []
    )
    oracle_warp: List[np.ndarray] = future_oracle_data.get(
        "future_warp_matrices", []
    )
    oracle_features: List[Optional[np.ndarray]] = future_oracle_data.get(
        "future_oracle_features", []
    )
    oracle_scores: List[float] = future_oracle_data.get("future_oracle_scores", [])

    num_future = min(
        future_frames,
        len(oracle_boxes),
        len(oracle_warp),
        len(oracle_features),
        len(oracle_scores),
    )
    oracle_boxes = oracle_boxes[:num_future]
    oracle_warp = oracle_warp[:num_future]
    oracle_features = oracle_features[:num_future]
    oracle_scores = oracle_scores[:num_future]

    total_motion_losses = np.zeros(num_sequences, dtype=np.float64)
    total_appearance_losses = np.zeros(num_sequences, dtype=np.float64)

    for seq_idx in range(num_sequences):
        # Decode bits: bit 0 = event 0 (oldest), bit 3 = event 3 (newest)
        gates = np.array(
            [(seq_idx >> (seq_len - 1 - i)) & 1 for i in range(seq_len)],
            dtype=np.int64,
        )

        # --- Replay through window ---
        w_mean = initial_mean.copy() if initial_mean is not None else None
        w_cov = initial_cov.copy() if initial_cov is not None else None
        w_feat = initial_feat.copy()

        for i in range(seq_len):
            if w_mean is None or w_cov is None:
                continue

            wm = warp_mats[i]
            # Apply warp
            w_mean, w_cov = motion_model.apply_warp(w_mean, w_cov, wm)
            # KF predict
            w_mean, w_cov = motion_model.predict(w_mean, w_cov)

            # Update if detection exists AND gate == 1
            if has_det[i] and det_boxes[i] is not None and gates[i] == 1:
                measurement = motion_model.bbox_to_measurement(det_boxes[i])
                w_mean, w_cov = motion_model.update(
                    w_mean, w_cov, measurement, det_scores[i]
                )
                w_feat = ema_update(
                    w_feat, det_features[i], det_scores[i], alpha=0.95
                )
            # else: predict-only (no update)

        # --- Oracle-guided rollout after window ---
        motion_future_loss = 0.0
        appearance_future_loss = 0.0

        for j in range(num_future):
            if w_mean is None or w_cov is None:
                continue

            warp = oracle_warp[j]
            # Apply warp
            w_mean, w_cov = motion_model.apply_warp(w_mean, w_cov, warp)
            # KF predict
            w_mean, w_cov = motion_model.predict(w_mean, w_cov)
            pred_box = motion_model.mean_to_bbox(w_mean)

            oracle_box = oracle_boxes[j]
            if oracle_box is not None:
                motion_future_loss += motion_frame_loss(oracle_box, pred_box)
                measurement = motion_model.bbox_to_measurement(oracle_box)
                w_mean, w_cov = motion_model.update(
                    w_mean, w_cov, measurement, 0.95
                )

            oracle_feat = oracle_features[j]
            oracle_score = oracle_scores[j]
            if oracle_feat is not None:
                w_feat = ema_update(w_feat, oracle_feat.ravel(), oracle_score, alpha=0.95)

            appearance_future_loss += appearance_loss(w_feat, identity_prototype)

        total_motion_losses[seq_idx] = motion_future_loss
        total_appearance_losses[seq_idx] = appearance_future_loss

    # --- Select best sequences ---
    # Motion: lowest total motion loss
    best_motion_idx = int(np.argmin(total_motion_losses))
    motion_sequence = np.array(
        [(best_motion_idx >> (seq_len - 1 - i)) & 1 for i in range(seq_len)],
        dtype=np.int64,
    )
    selected_motion_loss = float(total_motion_losses[best_motion_idx])

    # Appearance: lowest total appearance loss, tie-break by more 1s
    min_app_loss = float(np.min(total_appearance_losses))
    candidates = np.where(
        np.abs(total_appearance_losses - min_app_loss) <= 1e-6
    )[0]
    if len(candidates) > 1:
        # Prefer sequence with more 1s
        popcounts = np.array(
            [bin(idx).count("1") for idx in candidates], dtype=np.int64
        )
        best_app_idx = int(candidates[np.argmax(popcounts)])
    else:
        best_app_idx = int(candidates[0])

    appearance_sequence = np.array(
        [(best_app_idx >> (seq_len - 1 - i)) & 1 for i in range(seq_len)],
        dtype=np.int64,
    )
    selected_appearance_loss = float(total_appearance_losses[best_app_idx])

    return {
        "motion_sequence": motion_sequence,
        "appearance_sequence": appearance_sequence,
        "valid_mask": valid_mask,
        "motion_losses": total_motion_losses,
        "appearance_losses": total_appearance_losses,
        "selected_motion_loss": selected_motion_loss,
        "selected_appearance_loss": selected_appearance_loss,
    }


def generate_window_augmentations(
    window_events: List[TrackEvent],
    candidate_builder: Any,
) -> List[List[TrackEvent]]:
    """Generate up to 4 augmented versions of a 4-event window.

    Augmentations are produced by replacing events at certain positions with
    their **B** candidate (wrong-identity hard candidate):

    1. Original window (no replacement).
    2. Position 2 (third event, 0-indexed) replaced with B, if available.
    3. Position 3 (fourth event, 0-indexed) replaced with B, if available.
    4. Both positions 2 and 3 replaced with B, if available.

    Parameters
    ----------
    window_events : list of TrackEvent
        Exactly 4 events.
    candidate_builder
        An object with a ``build_candidates`` method (see
        :class:`~agentguard.data.candidate_builder.CandidateBuilder`).

    Returns
    -------
    list of list of TrackEvent
        Up to 4 window variants.  Each inner list has length 4.
    """
    if len(window_events) != 4:
        raise ValueError(
            f"Expected exactly 4 window events, got {len(window_events)}."
        )

    # Build B candidates for events that have detections
    b_candidates: List[Optional[TrackEvent]] = [None] * 4
    for i, evt in enumerate(window_events):
        if evt.has_detection:
            # Build candidates for this event; we need the full candidates
            # matrix etc.  Here we rely on the caller having wired up the
            # candidate_builder appropriately.
            try:
                candidates = candidate_builder.build_candidates(
                    evt, np.empty((0, 3)), [], -1, None, None, None
                )
                b_candidates[i] = candidates.get("b")
            except Exception:
                b_candidates[i] = None

    variants: List[List[TrackEvent]] = [list(window_events)]

    # Variant 2: position 2 → B
    if b_candidates[2] is not None:
        var = list(window_events)
        var[2] = b_candidates[2]
        variants.append(var)

    # Variant 3: position 3 → B
    if b_candidates[3] is not None:
        var = list(window_events)
        var[3] = b_candidates[3]
        variants.append(var)

    # Variant 4: both positions 2 and 3 → B
    if b_candidates[2] is not None and b_candidates[3] is not None:
        var = list(window_events)
        var[2] = b_candidates[2]
        var[3] = b_candidates[3]
        variants.append(var)

    return variants
