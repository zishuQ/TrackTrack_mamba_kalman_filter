from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.verifier.local_replay import LocalReplayVerifier


def compute_cross_event_scores(
    event: TrackEvent,
    similar_events: List[Tuple[TrackEvent, np.ndarray, float]],
    verifier: LocalReplayVerifier,
) -> Dict[str, float]:
    """Compute cross-event verification scores for each policy.

    For each similar event, the verifier replays it and computes
    delta R_m (the benefit of policy *m* over FULL_WRITE).  The
    cross-event score S_m for policy *m* is::

        S_m = Mean(delta_R_m) - 0.5 * Std(delta_R_m) - 0.5 * HarmRate_m

    A policy is accepted if ``S_m > 0.02`` **and** ``HarmRate < 0.30``.

    Parameters
    ----------
    event : TrackEvent
        The query event (used for sequence exclusion context).
    similar_events : list of (TrackEvent, ndarray, float)
        Similar events from :meth:`SimilarEventGrouper.find_similar`.
    verifier : LocalReplayVerifier
        Verifier used to replay each similar event.

    Returns
    -------
    dict
        Mapping ``policy_name -> score``.

    Notes
    -----
    **HarmRate** for policy *m* is the fraction of similar events where
    delta R_m is negative (the policy harmed performance).
    """
    if not similar_events:
        return {}

    # Collect deltas for each policy across all similar events
    all_deltas: Dict[str, List[float]] = {
        "FULL_WRITE": [],
        "MOTION_ONLY": [],
        "APPEARANCE_ONLY": [],
        "HOLD_BOTH": [],
        "SOFT_CAUTION": [],
    }

    for sim_event, sim_desc, sim_score in similar_events:
        # We need future_oracle_data for each similar event.
        # In practice this would come from a cache/lookup.
        # Here we pass an empty dict as a stand-in; a real pipeline
        # would hydrate this from a pre-computed oracle store.
        oracle_data: dict = _lookup_oracle_data(sim_event)

        try:
            result = verifier.verify_event(sim_event, oracle_data)
        except Exception:
            # If replay fails, skip this similar event
            continue

        deltas = result.get("policy_deltas", {})
        for policy_name in all_deltas:
            if policy_name in deltas:
                all_deltas[policy_name].append(deltas[policy_name])

    # Compute S_m for each policy
    scores: Dict[str, float] = {}
    for policy_name, deltas in all_deltas.items():
        if policy_name == "FULL_WRITE":
            scores[policy_name] = 0.0
            continue

        if len(deltas) == 0:
            scores[policy_name] = -float("inf")
            continue

        deltas_arr = np.asarray(deltas, dtype=np.float64)
        mean_delta = float(np.mean(deltas_arr))
        std_delta = float(np.std(deltas_arr))
        harm_rate = float(np.mean(deltas_arr < 0))

        s_m = mean_delta - 0.5 * std_delta - 0.5 * harm_rate
        scores[policy_name] = s_m

    return scores


def _lookup_oracle_data(event: TrackEvent) -> dict:
    """Look up pre-computed future oracle data for an event.

    In a full pipeline this would query a cache indexed by
    ``(event.dataset, event.sequence, event.event_id)``.

    Returns
    -------
    dict
        Oracle data structure (potentially empty).
    """
    # Stub: return empty oracle.  In production this would return
    # the oracle dict built by FutureOracleBuilder.
    return {}
