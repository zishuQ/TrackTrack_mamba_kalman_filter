from __future__ import annotations

from typing import Dict

import numpy as np


class RuntimeStatistics:
    """Collect counters and gate diagnostics for the current Runtime path."""

    def __init__(self) -> None:
        self.total_events = 0
        self.rg_cma_inference_count = 0
        self.matched_events = 0
        self.unmatched_events = 0

        self.gate_sum = np.zeros(2, dtype=np.float64)
        self.base_gate_sum = np.zeros(2, dtype=np.float64)
        self.final_gate_sum = np.zeros(2, dtype=np.float64)
        self.correction_abs_sum = np.zeros(2, dtype=np.float64)
        self.correction_abs_max = np.zeros(2, dtype=np.float64)
        self.correction_nonzero_count = np.zeros(2, dtype=np.int64)
        self.correction_saturation_count = np.zeros(2, dtype=np.int64)

    def record_event(self) -> None:
        self.total_events += 1

    def record_rg_cma(
        self,
        gate: np.ndarray,
        base_gate: np.ndarray,
        final_gate: np.ndarray,
        correction: np.ndarray,
        *,
        matched: bool,
        correction_bound: float | None = None,
        inference_count: int = 1,
    ) -> None:
        """Record one current RG-CMA decision.

        ``gate`` is the selected base/final output, while the other arrays are
        retained for the correction diagnostics.  ``matched`` distinguishes
        normal accepted detections from the explicit unmatched sentinel.
        """
        count = int(inference_count)
        if count < 1:
            raise ValueError("inference_count must be positive")

        applied = np.asarray(gate, dtype=np.float64).reshape(2)
        base = np.asarray(base_gate, dtype=np.float64).reshape(2)
        final = np.asarray(final_gate, dtype=np.float64).reshape(2)
        raw_correction = np.asarray(correction, dtype=np.float64).reshape(2)

        self.rg_cma_inference_count += count
        if matched:
            self.matched_events += count
        else:
            self.unmatched_events += count

        self.gate_sum += applied * count
        self.base_gate_sum += base * count
        self.final_gate_sum += final * count

        correction_abs = np.abs(raw_correction)
        self.correction_abs_sum += correction_abs * count
        self.correction_abs_max = np.maximum(
            self.correction_abs_max, correction_abs
        )
        self.correction_nonzero_count += (
            (correction_abs > 1e-6).astype(np.int64) * count
        )
        if correction_bound is not None and float(correction_bound) > 0.0:
            self.correction_saturation_count += (
                (correction_abs >= 0.95 * float(correction_bound)).astype(np.int64)
                * count
            )

    def summary(self) -> Dict[str, float]:
        count = max(self.rg_cma_inference_count, 1)
        return {
            "total_events": self.total_events,
            "rg_cma_inference_count": self.rg_cma_inference_count,
            "matched_events": self.matched_events,
            "unmatched_events": self.unmatched_events,
            "avg_motion_gate": float(self.gate_sum[0] / count),
            "avg_appearance_gate": float(self.gate_sum[1] / count),
            "avg_rg_cma_base_motion_gate": float(self.base_gate_sum[0] / count),
            "avg_rg_cma_base_appearance_gate": float(self.base_gate_sum[1] / count),
            "avg_rg_cma_final_motion_gate": float(self.final_gate_sum[0] / count),
            "avg_rg_cma_final_appearance_gate": float(self.final_gate_sum[1] / count),
            "avg_rg_cma_correction_abs_motion": float(
                self.correction_abs_sum[0] / count
            ),
            "avg_rg_cma_correction_abs_appearance": float(
                self.correction_abs_sum[1] / count
            ),
            "max_rg_cma_correction_abs_motion": float(self.correction_abs_max[0]),
            "max_rg_cma_correction_abs_appearance": float(
                self.correction_abs_max[1]
            ),
            "rg_cma_correction_nonzero_rate_motion": float(
                self.correction_nonzero_count[0] / count
            ),
            "rg_cma_correction_nonzero_rate_appearance": float(
                self.correction_nonzero_count[1] / count
            ),
            "rg_cma_correction_saturation_rate_motion": float(
                self.correction_saturation_count[0] / count
            ),
            "rg_cma_correction_saturation_rate_appearance": float(
                self.correction_saturation_count[1] / count
            ),
        }
