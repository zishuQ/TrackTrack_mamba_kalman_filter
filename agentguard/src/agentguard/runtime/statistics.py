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
        self.iwg_base_gate_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.iwg_final_gate_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.iwg_base_final_abs_diff_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.iwg_base_final_abs_diff_max: np.ndarray = np.zeros(2, dtype=np.float64)
        self.iwg_correction_abs_sum: np.ndarray = np.zeros(2, dtype=np.float64)
        self.iwg_correction_abs_max: np.ndarray = np.zeros(2, dtype=np.float64)
        self.iwg_correction_nonzero_count: np.ndarray = np.zeros(2, dtype=np.int64)
        self.iwg_correction_saturation_count: np.ndarray = np.zeros(2, dtype=np.int64)
        self.iwg_correction_count: int = 0

    def record_event(self) -> None:
        """Increment the total event counter."""
        self.total_events += 1

    def record_iwg(
        self,
        gate: np.ndarray,
        *,
        base_gate: np.ndarray | None = None,
        final_gate: np.ndarray | None = None,
        correction: np.ndarray | None = None,
        correction_bound: float | None = None,
    ) -> None:
        """Record an IWG inference call and its output gate.

        Parameters
        ----------
        gate : ndarray, shape (2,)
            ``[motion_gate, appearance_gate]`` produced by the IWG.
        base_gate, final_gate : ndarray, shape (2), optional
            The unrefined and applied RG-CMA gates. Legacy IWG calls omit
            these values and are treated as having no refinement.
        correction : ndarray, shape (2), optional
            The raw RG-CMA correction before the final [0, 1] clamp.
        correction_bound : float, optional
            Bound used for the saturation diagnostic.
        """
        applied = np.asarray(gate, dtype=np.float64).reshape(2)
        base = (
            applied
            if base_gate is None
            else np.asarray(base_gate, dtype=np.float64).reshape(2)
        )
        final = (
            applied
            if final_gate is None
            else np.asarray(final_gate, dtype=np.float64).reshape(2)
        )
        raw_correction = (
            final - base
            if correction is None
            else np.asarray(correction, dtype=np.float64).reshape(2)
        )
        applied_difference = final - base
        self.iwg_calls += 1
        self.gate_sum += applied
        self.gate_count += 1
        self.iwg_base_gate_sum += base
        self.iwg_final_gate_sum += final
        applied_abs = np.abs(applied_difference)
        correction_abs = np.abs(raw_correction)
        self.iwg_base_final_abs_diff_sum += applied_abs
        self.iwg_base_final_abs_diff_max = np.maximum(
            self.iwg_base_final_abs_diff_max, applied_abs
        )
        self.iwg_correction_abs_sum += correction_abs
        self.iwg_correction_abs_max = np.maximum(
            self.iwg_correction_abs_max, correction_abs
        )
        self.iwg_correction_nonzero_count += (correction_abs > 1e-6).astype(
            np.int64
        )
        if correction_bound is not None and float(correction_bound) > 0.0:
            self.iwg_correction_saturation_count += (
                correction_abs >= 0.95 * float(correction_bound)
            ).astype(np.int64)
        self.iwg_correction_count += 1

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
        iwg_count = max(self.iwg_correction_count, 1)
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
            "avg_iwg_base_motion_gate": float(
                self.iwg_base_gate_sum[0] / iwg_count
            ),
            "avg_iwg_base_appearance_gate": float(
                self.iwg_base_gate_sum[1] / iwg_count
            ),
            "avg_iwg_final_motion_gate": float(
                self.iwg_final_gate_sum[0] / iwg_count
            ),
            "avg_iwg_final_appearance_gate": float(
                self.iwg_final_gate_sum[1] / iwg_count
            ),
            "avg_iwg_base_final_abs_diff_motion": float(
                self.iwg_base_final_abs_diff_sum[0] / iwg_count
            ),
            "avg_iwg_base_final_abs_diff_appearance": float(
                self.iwg_base_final_abs_diff_sum[1] / iwg_count
            ),
            "max_iwg_base_final_abs_diff_motion": float(
                self.iwg_base_final_abs_diff_max[0]
            ),
            "max_iwg_base_final_abs_diff_appearance": float(
                self.iwg_base_final_abs_diff_max[1]
            ),
            "avg_iwg_correction_abs_motion": float(
                self.iwg_correction_abs_sum[0] / iwg_count
            ),
            "avg_iwg_correction_abs_appearance": float(
                self.iwg_correction_abs_sum[1] / iwg_count
            ),
            "max_iwg_correction_abs_motion": float(
                self.iwg_correction_abs_max[0]
            ),
            "max_iwg_correction_abs_appearance": float(
                self.iwg_correction_abs_max[1]
            ),
            "iwg_correction_nonzero_rate_motion": float(
                self.iwg_correction_nonzero_count[0] / iwg_count
            ),
            "iwg_correction_nonzero_rate_appearance": float(
                self.iwg_correction_nonzero_count[1] / iwg_count
            ),
            "iwg_correction_saturation_rate_motion": float(
                self.iwg_correction_saturation_count[0] / iwg_count
            ),
            "iwg_correction_saturation_rate_appearance": float(
                self.iwg_correction_saturation_count[1] / iwg_count
            ),
        }
