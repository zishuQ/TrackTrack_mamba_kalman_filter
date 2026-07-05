from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.contracts.events import TrackEvent
from agentguard.rollout.losses import sigmoid

# ---------------------------------------------------------------------------
#  Dataset-wide statistics
# ---------------------------------------------------------------------------

#: Minimum absolute benefit before clipping (prevents division by near-zero).
_BENEFIT_EPS: float = 1e-3
#: Maximum absolute benefit after clipping (keeps tau in a sensible range).
_BENEFIT_CLIP_MAX: float = 0.1


def compute_dataset_stats(all_benefits: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute ``tau_motion`` and ``tau_appearance`` from the full dataset.

    For each benefit type, the statistic is the clipped median of absolute
    non-zero values:

    .. math::

        \\tau_m = \\text{clip}\\big(\\text{median}(|B_m| \\neq 0),\\,
            \\varepsilon,\\, \\text{max}\\big)
        \\\\
        \\tau_a = \\text{clip}\\big(\\text{median}(|B_a| \\neq 0),\\,
            \\varepsilon,\\, \\text{max}\\big)

    The result is saved to
    ``outputs/agentguard/labels/<dataset>/dataset_stats.json`` **once**
    (not per-sequence).  The dataset name is read from an environment
    variable ``AG_GUARD_DATASET`` or defaults to ``"default"``.

    Parameters
    ----------
    all_benefits : dict
        Must contain keys ``"motion_benefits"`` and
        ``"appearance_benefits"``, each mapping to a ``list[float]`` of
        benefits collected across the full dataset.

    Returns
    -------
    dict
        Keys ``tau_motion`` (float) and ``tau_appearance`` (float).
    """
    tau_m = _compute_tau(all_benefits.get("motion_benefits", []))
    tau_a = _compute_tau(all_benefits.get("appearance_benefits", []))

    stats = {"tau_motion": tau_m, "tau_appearance": tau_a}

    # Save once to a dataset-specific path
    dataset_name = os.environ.get("AG_GUARD_DATASET", "default")
    save_dir = Path("outputs") / "agentguard" / "labels" / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "dataset_stats.json"
    if not save_path.exists():
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
#  Soft targets
# ---------------------------------------------------------------------------


def compute_soft_target(benefit: float, tau: float) -> float:
    """Convert a benefit value into a soft binary target.

    Benefit sign: ``B = L_skip - L_write``.

    .. math::

        y = \\sigma\\left(\\frac{\\text{benefit}}{\\tau}\\right)

    where :math:`\\sigma` is the sigmoid function.  Values > 0.5 favour
    the write branch (skip loss dominates).

    Parameters
    ----------
    benefit : float
        Motion or appearance benefit (``B_m`` or ``B_a``), computed as
        ``L_skip - L_write``.
    tau : float
        Scaling temperature (e.g. from :func:`compute_dataset_stats`).

    Returns
    -------
    float
        Soft target in ``(0, 1)``.  Values > 0.5 favour the write branch.
    """
    return sigmoid(benefit / max(tau, 1e-12))


# ---------------------------------------------------------------------------
#  Policy soft targets
# ---------------------------------------------------------------------------


def compute_policy_soft_target(
    gate: np.ndarray,
    prototypes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a soft policy distribution from a gate vector.

    For each policy prototype :math:`p_k` (a 2-D point in
    ``(motion_gate, appearance_gate)`` space):

    .. math::

        q_k = \\frac{
            \\exp\\big(-\\|\\text{gate} - p_k\\|^2 / 0.1\\big)
        }{
            \\sum_j \\exp\\big(-\\|\\text{gate} - p_j\\|^2 / 0.1\\big)
        }

    Parameters
    ----------
    gate : ndarray, shape ``(2,)``
        The gate vector ``[motion_gate, appearance_gate]``, each in
        ``[0, 1]``.
    prototypes : ndarray, shape ``(5, 2)`` or None
        Policy prototypes.  Defaults to
        :data:`~agentguard.contracts.enums.POLICY_PROTOTYPE_MATRIX`.

    Returns
    -------
    ndarray, shape ``(5,)``
        Normalised softmax distribution over 5 policies.
    """
    if prototypes is None:
        prototypes = POLICY_PROTOTYPE_MATRIX

    # Squared Euclidean distances, shape (5,)
    diffs = prototypes - gate.reshape(1, 2)  # (5, 2)
    sq_dists = np.sum(diffs ** 2, axis=1)  # (5,)

    logits = -sq_dists / 0.1
    # Numerically stable softmax
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits)
    return probs


# ---------------------------------------------------------------------------
#  Build rollout labels
# ---------------------------------------------------------------------------


def build_rollout_labels(
    events: List[TrackEvent],
    motion_benefits: List[float],
    appearance_benefits: List[float],
    dataset_stats: Dict[str, float],
    identity_prototypes: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    """Build rollout labels for each event.

    For every event the following label fields are produced:

    * ``motion_target`` — ``sigmoid(B_m / tau_m)``
    * ``appearance_target`` — ``sigmoid(B_a / tau_a)``
    * ``policy_soft_target`` — ``(5,)`` distribution over policies derived
      from ``[motion_target, appearance_target]``
    * ``sample_type`` — one of ``"matched"``, ``"unmatched"``, ``"A"``,
      ``"B"``, ``"C"``
    * ``valid_motion`` — always ``True`` for now
    * ``valid_appearance`` — always ``True`` for now
    * ``sample_weight`` — always ``1.0`` for now

    Parameters
    ----------
    events : list of TrackEvent
        All events to label.
    motion_benefits : list of float
        Per-event motion benefits (aligned with *events* by index).
    appearance_benefits : list of float
        Per-event appearance benefits.
    dataset_stats : dict
        Must contain keys ``"tau_motion"`` and ``"tau_appearance"``.
    identity_prototypes : dict
        Maps ``track_id -> prototype`` (ndarray of shape ``(D,)``).

    Returns
    -------
    list of dict
        One label dict per event.  Each dict contains the keys listed above.
    """
    tau_m = dataset_stats.get("tau_motion", 0.01)
    tau_a = dataset_stats.get("tau_appearance", 0.01)

    labels: List[Dict[str, Any]] = []
    for idx, event in enumerate(events):
        B_m = motion_benefits[idx] if idx < len(motion_benefits) else 0.0
        B_a = appearance_benefits[idx] if idx < len(appearance_benefits) else 0.0

        motion_target = compute_soft_target(B_m, tau_m)
        appearance_target = compute_soft_target(B_a, tau_a)

        gate = np.array([motion_target, appearance_target], dtype=np.float64)
        policy_soft_target = compute_policy_soft_target(gate)

        # Determine sample type
        if event.has_detection:
            sample_type = "matched"
        else:
            sample_type = "unmatched"

        label: Dict[str, Any] = {
            "motion_target": motion_target,
            "appearance_target": appearance_target,
            "policy_soft_target": policy_soft_target,
            "sample_type": sample_type,
            "valid_motion": True,
            "valid_appearance": True,
            "sample_weight": 1.0,
        }

        # Attach metadata for debugging / downstream processing
        label["event_id"] = event.event_id
        label["track_id"] = event.track_id
        label["frame_id"] = event.frame_id
        label["sequence"] = event.sequence
        label["motion_benefit"] = B_m
        label["appearance_benefit"] = B_a

        labels.append(label)

    return labels


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _compute_tau(benefits: List[float]) -> float:
    """Compute clipped median of absolute non-zero benefits.

    Parameters
    ----------
    benefits : list of float

    Returns
    -------
    float
    """
    arr = np.asarray(benefits, dtype=np.float64)
    non_zero = np.abs(arr[arr != 0.0])
    if len(non_zero) == 0:
        return _BENEFIT_CLIP_MAX
    median_val = float(np.median(non_zero))
    return float(np.clip(median_val, _BENEFIT_EPS, _BENEFIT_CLIP_MAX))
