from __future__ import annotations

from typing import Optional

import numpy as np

from agentguard.teacher.response_schema import TeacherResponse


def compute_rollout_reliability(
    event: object,
    motion_benefit: float,
    appearance_benefit: float,
    tau: float = 0.5,
) -> float:
    """Compute rollout reliability score.

    The rollout reliability *r* measures how trustworthy the online
    rollout labels are for this event, based on GT consistency, oracle
    quality, benefit of revision, and remaining horizon.

    Formula::

        r = max(0.4 * c_gt + 0.3 * c_oracle + 0.2 * c_benefit + 0.1 * c_horizon,
                0.05)
        r = min(r, 1.0)

    where:

    - ``c_gt``: fraction of future frames where GT exists.
    - ``c_oracle``: fraction of future frames where oracle IoU >= tau.
    - ``c_benefit``: average of *motion_benefit* and *appearance_benefit*
      (clamped to [0, 1]).
    - ``c_horizon``: fraction of the max horizon actually available.

    Parameters
    ----------
    event : object
        The event (accessed via ``getattr`` for oracle coverage).
    motion_benefit : float
        Benefit of motion revision (can be negative).
    appearance_benefit : float
        Benefit of appearance revision (can be negative).
    tau : float
        IoU threshold for oracle quality (default 0.5).

    Returns
    -------
    float
        Rollout reliability in [0.05, 1.0].
    """
    # c_gt: future GT coverage attached to the event
    c_gt = float(getattr(event, "future_oracle_coverage", 0.0))
    c_gt = max(0.0, min(1.0, c_gt))

    # c_oracle: oracle detection quality (stored as oracle_iou if available)
    c_oracle = float(getattr(event, "oracle_iou", c_gt))
    c_oracle = max(0.0, min(1.0, c_oracle))

    # c_benefit: average benefit clamped to [0, 1]
    c_benefit = 0.5 * (motion_benefit + appearance_benefit)
    c_benefit = max(0.0, min(1.0, c_benefit))

    # c_horizon: fraction of max horizon (default 5) available
    # Higher is better (more future data)
    max_horizon = 5.0
    future_count = float(getattr(event, "num_future_frames", max_horizon))
    c_horizon = min(future_count / max_horizon, 1.0)

    r = 0.4 * c_gt + 0.3 * c_oracle + 0.2 * c_benefit + 0.1 * c_horizon
    r = max(0.05, min(1.0, r))
    return float(r)


def compute_teacher_confidence(
    teacher_response: TeacherResponse,
    harm_rate: float,
) -> float:
    """Compute adjusted teacher confidence considering harm rate.

    If the Teacher's recommended policies would cause harm in similar
    events, the confidence is reduced accordingly.

    Formula::

        c = clip(teacher_response.confidence * (1 - harm_rate), 0, 1)

    Parameters
    ----------
    teacher_response : TeacherResponse
        The parsed Teacher response.
    harm_rate : float
        Fraction of similar events where the Teacher's recommended
        policies caused harm (in [0, 1]).

    Returns
    -------
    float
        Adjusted teacher confidence in [0, 1].
    """
    c = teacher_response.confidence * (1.0 - harm_rate)
    c = max(0.0, min(1.0, c))
    return float(c)


def fuse_labels(
    rollout_gate: np.ndarray,
    agent_gate: np.ndarray,
    rollout_reliability: float,
    teacher_confidence: float,
    teacher_abstained: bool,
) -> np.ndarray:
    """Fuse the rollout gate and agent gate into the final training label.

    If the Teacher abstained, the rollout gate is used directly.
    Otherwise, a reliability-weighted average is computed::

        if teacher_abstained:
            g* = g_rollout
        else:
            g* = (r * g_rollout + c * g_agent) / (r + c)

    where *r* is *rollout_reliability* and *c* is *teacher_confidence*.

    Parameters
    ----------
    rollout_gate : ndarray, shape ``(2,)``
        Gate from online rollout ``[motion, appearance]``.
    agent_gate : ndarray, shape ``(2,)``
        Gate from fused Teacher + Verifier ``[motion, appearance]``.
    rollout_reliability : float
        Reliability of rollout labels *r* in [0, 1].
    teacher_confidence : float
        Adjusted teacher confidence *c* in [0, 1].
    teacher_abstained : bool
        Whether the Teacher abstained from making a decision.

    Returns
    -------
    ndarray, shape ``(2,)``
        Fused gate ``[motion, appearance]``, each element in [0, 1].
    """
    if teacher_abstained:
        return rollout_gate.copy()

    total_weight = rollout_reliability + teacher_confidence
    if total_weight < 1e-12:
        return rollout_gate.copy()

    fused = (rollout_reliability * rollout_gate + teacher_confidence * agent_gate) / total_weight
    fused = np.clip(fused, 0.0, 1.0)
    return fused
