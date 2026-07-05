from __future__ import annotations

from typing import Dict

import numpy as np

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX


def compute_verifier_distribution(
    scores: Dict[str, float],
    temperature: float = 0.1,
) -> np.ndarray:
    """Convert cross-event scores into a probability distribution via
    softmax.

    Parameters
    ----------
    scores : dict
        Mapping ``policy_name -> score`` as produced by
        :func:`compute_cross_event_scores`.
    temperature : float
        Softmax temperature (default 0.1).  Lower values sharpen the
        distribution.

    Returns
    -------
    ndarray, shape ``(5,)``
        ``q_verifier`` — a probability distribution over the 5 write
        policies in the standard order: FULL_WRITE, MOTION_ONLY,
        APPEARANCE_ONLY, HOLD_BOTH, SOFT_CAUTION.
    """
    policy_names = [
        "FULL_WRITE",
        "MOTION_ONLY",
        "APPEARANCE_ONLY",
        "HOLD_BOTH",
        "SOFT_CAUTION",
    ]

    # Build logits vector in policy order
    logits_list = []
    for name in policy_names:
        s = scores.get(name, -float("inf"))
        logits_list.append(s)

    logits = np.asarray(logits_list, dtype=np.float64)

    # Softmax with temperature: q = exp(s / t) / sum(exp(s / t))
    scaled = logits / temperature
    # Numerical stability: subtract max
    scaled = scaled - np.max(scaled)
    exp_vals = np.exp(scaled)
    q = exp_vals / (np.sum(exp_vals) + 1e-12)

    return q


def fuse_distributions(
    teacher_dist: np.ndarray,
    verifier_dist: np.ndarray,
) -> np.ndarray:
    """Fuse the Teacher and Verifier distributions via geometric mean.

    Formula::

        q_agent[i] = sqrt(q_teacher[i] * q_verifier[i])
        q_agent = q_agent / sum(q_agent)

    Parameters
    ----------
    teacher_dist : ndarray, shape ``(5,)``
        Teacher policy distribution.
    verifier_dist : ndarray, shape ``(5,)``
        Verifier policy distribution.

    Returns
    -------
    ndarray, shape ``(5,)``
        Fused ``q_agent`` distribution, normalised to sum to 1.
    """
    if teacher_dist.shape != (5,) or verifier_dist.shape != (5,):
        raise ValueError(
            f"Expected (5,) distributions, got teacher {teacher_dist.shape} "
            f"and verifier {verifier_dist.shape}"
        )

    q_agent = np.sqrt(teacher_dist * verifier_dist + 1e-12)
    q_agent = q_agent / (np.sum(q_agent) + 1e-12)
    return q_agent


def compute_agent_gate(
    agent_distribution: np.ndarray,
    prototypes: np.ndarray = POLICY_PROTOTYPE_MATRIX,
) -> np.ndarray:
    """Compute the agent gate from the fused policy distribution via
    prototype mixing.

    Formula::

        g_agent[j] = sum_m q_agent[m] * prototype[m, j]

    where *j* indexes motion (0) and appearance (1) gates.

    Parameters
    ----------
    agent_distribution : ndarray, shape ``(5,)``
        Fused policy distribution ``q_agent``.
    prototypes : ndarray, shape ``(5, 2)``
        Prototype matrix (defaults to
        :data:`agentguard.contracts.enums.POLICY_PROTOTYPE_MATRIX`).

    Returns
    -------
    ndarray, shape ``(2,)``
        ``[motion_gate, appearance_gate]`` each in ``[0, 1]``.
    """
    g_agent = agent_distribution @ prototypes
    g_agent = np.clip(g_agent, 0.0, 1.0)
    return g_agent
