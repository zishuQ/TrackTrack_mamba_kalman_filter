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
        self.replay_steps: int = 0
        self.skipped_replay_windows: int = 0
        self.skipped_tgr_windows: int = 0
        self.checkpoint_rolls: int = 0
        self.skipped_immature: int = 0
        self.revision_abs_diff_sum: float = 0.0
        self.revision_abs_diff_max: float = 0.0
        self.revision_abs_diff_count: int = 0
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

    def record_tgr(self, count: int = 1) -> None:
        """Increment the TGR call counter."""
        self.tgr_calls += int(count)

    def record_replay(self, steps: int = 0) -> None:
        """Increment replay-plan counters."""
        self.replay_events += 1
        self.replay_steps += int(steps)

    def record_checkpoint_roll(self, steps: int = 1) -> None:
        """Increment checkpoint-roll counters."""
        self.checkpoint_rolls += int(steps)

    def record_skipped_replay(self) -> None:
        """Increment skipped replay-window counter."""
        self.skipped_replay_windows += 1

    def record_skipped_tgr(self, count: int = 1) -> None:
        """Increment windows skipped before TGR for sparse full mode."""
        self.skipped_tgr_windows += int(count)

    def record_revision_diff(self, max_abs_diff: float) -> None:
        """Record the max absolute gate revision for one TGR window."""
        value = float(max_abs_diff)
        self.revision_abs_diff_sum += value
        self.revision_abs_diff_max = max(self.revision_abs_diff_max, value)
        self.revision_abs_diff_count += 1

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
        avg_revision_diff = self.revision_abs_diff_sum / max(
            self.revision_abs_diff_count, 1
        )
        return {
            "total_events": self.total_events,
            "iwg_calls": self.iwg_calls,
            "tgr_calls": self.tgr_calls,
            "replay_events": self.replay_events,
            "replay_steps": self.replay_steps,
            "skipped_replay_windows": self.skipped_replay_windows,
            "skipped_tgr_windows": self.skipped_tgr_windows,
            "checkpoint_rolls": self.checkpoint_rolls,
            "skipped_immature": self.skipped_immature,
            "avg_revision_abs_diff": float(avg_revision_diff),
            "max_revision_abs_diff": float(self.revision_abs_diff_max),
            "avg_motion_gate": float(avg_gate[0]),
            "avg_appearance_gate": float(avg_gate[1]),
        }
