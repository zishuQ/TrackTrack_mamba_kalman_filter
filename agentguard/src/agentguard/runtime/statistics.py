from __future__ import annotations

from typing import Dict

import numpy as np


class RuntimeStatistics:
    """Track runtime statistics for AgentGuard.

    Collects counters and aggregated values across the full tracking
    session.  Call the ``record_*`` methods at the appropriate points
    during inference; obtain a snapshot via ``summary()``.
    """

    def __init__(self) -> None:
        self.total_events: int = 0
        self.iwg_calls: int = 0
        self.tgr_calls: int = 0
        self.replay_events: int = 0
        self.skipped_immature: int = 0
        self.gate_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.gate_count: int = 0

    def record_event(self) -> None:
        """Increment the total event counter."""
        self.total_events += 1

    def record_iwg(self, gate: np.ndarray) -> None:
        """Record an IWG inference call and its output gate.

        Parameters
        ----------
        gate : ndarray, shape (2,)
            ``[motion_gate, appearance_gate]`` produced by the IWG.
        """
        self.iwg_calls += 1
        self.gate_sum += gate
        self.gate_count += 1

    def record_tgr(self) -> None:
        """Increment the TGR call counter."""
        self.tgr_calls += 1

    def record_replay(self) -> None:
        """Increment the replayed event counter."""
        self.replay_events += 1

    def record_skipped_immature(self) -> None:
        """Increment the skipped-immature-track counter."""
        self.skipped_immature += 1

    def summary(self) -> Dict[str, float]:
        """Return a snapshot of all statistics as a flat dictionary.

        Returns
        -------
        dict
            Keys:
            - ``total_events``
            - ``iwg_calls``
            - ``tgr_calls``
            - ``replay_events``
            - ``skipped_immature``
            - ``avg_motion_gate``
            - ``avg_appearance_gate``
        """
        avg_gate = self.gate_sum / max(self.gate_count, 1)
        return {
            "total_events": self.total_events,
            "iwg_calls": self.iwg_calls,
            "tgr_calls": self.tgr_calls,
            "replay_events": self.replay_events,
            "skipped_immature": self.skipped_immature,
            "avg_motion_gate": float(avg_gate[0]),
            "avg_appearance_gate": float(avg_gate[1]),
        }
