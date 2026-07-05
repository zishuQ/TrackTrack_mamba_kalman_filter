from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.rollout.losses import appearance_loss


def compute_appearance_benefit(
    event: TrackEvent,
    future_oracle_data: Dict[str, Any],
    identity_prototype: np.ndarray,
    alpha: float = 0.95,
    future_frames: int = 5,
) -> Tuple[float, List[float], List[float], List[np.ndarray], List[np.ndarray]]:
    """Compute appearance benefit for a single event.

    Two branches are simulated from the event's ``pre_update_state`` feature:

    * **Write branch** — ``appearance_gate=1``: the current event's detection
      feature is written into the track feature via an EMA update **before**
      the future rollout.
    * **Skip branch**  — ``appearance_gate=0``: the track feature stays
      unchanged.

    Both branches then continue with EMA updates using oracle detection
    features for *future_frames*.

    Per-frame loss
        ``L = 1 - cos_sim(feature, identity_prototype)``

    Benefit
        ``B_a = L_skip - L_write``

        If *negative* the write branch is better (we gain from writing).

    Parameters
    ----------
    event : TrackEvent
        The event to evaluate.
    future_oracle_data : dict
        Must contain keys:

        * ``future_oracle_features`` — list of ``(1, D)`` ndarray or ``None``,
          length >= *future_frames*.
        * ``future_oracle_scores`` — list of float, length >= *future_frames*.
    identity_prototype : ndarray, shape ``(D,)``
        The L2-normalised identity prototype vector for this track.
    alpha : float
        EMA decay factor (default 0.95).
    future_frames : int
        Number of future frames to simulate (default 5).

    Returns
    -------
    B_a : float
        Appearance benefit (= L_skip - L_write).
    write_losses : list of float
        Per-frame appearance losses for the write branch.
    skip_losses : list of float
        Per-frame appearance losses for the skip branch.
    write_features : list of ndarray (D,)
        Features after each future frame for the write branch.
    skip_features : list of ndarray (D,)
        Features after each future frame for the skip branch.
    """
    pre_state = event.pre_update_state
    if pre_state is None:
        raise ValueError(
            "Event has no pre_update_state — cannot compute appearance benefit."
        )

    old_feat = pre_state.feature.ravel().copy()
    oracle_features: List[Optional[np.ndarray]] = future_oracle_data.get(
        "future_oracle_features", []
    )
    oracle_scores: List[float] = future_oracle_data.get("future_oracle_scores", [])

    num_frames = min(future_frames, len(oracle_features), len(oracle_scores))
    oracle_features = oracle_features[:num_frames]
    oracle_scores = oracle_scores[:num_frames]

    # --- Write branch initial state ---
    if event.has_detection and event.detection is not None:
        det_feat = event.detection.feature.ravel()
        det_score = event.detection.score
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
        oracle_feat = oracle_features[i]
        oracle_score = oracle_scores[i]

        # --- Write branch ---
        if oracle_feat is not None:
            w_feat = ema_update(write_feat, oracle_feat.ravel(), oracle_score, alpha)
        else:
            w_feat = write_feat.copy()
        w_loss = appearance_loss(w_feat, identity_prototype)
        write_losses.append(w_loss)
        write_features.append(w_feat)
        write_feat = w_feat

        # --- Skip branch ---
        if oracle_feat is not None:
            s_feat = ema_update(skip_feat, oracle_feat.ravel(), oracle_score, alpha)
        else:
            s_feat = skip_feat.copy()
        s_loss = appearance_loss(s_feat, identity_prototype)
        skip_losses.append(s_loss)
        skip_features.append(s_feat)
        skip_feat = s_feat

    B_a = sum(skip_losses) - sum(write_losses)
    return B_a, write_losses, skip_losses, write_features, skip_features


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
