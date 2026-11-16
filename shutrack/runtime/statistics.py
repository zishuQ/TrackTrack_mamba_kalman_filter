"""Independent SHUTrack runtime statistics (no policy/correction/fallback)."""

from __future__ import annotations

from typing import Dict

import numpy as np


class ShutrackRuntimeStatistics:
    def __init__(self) -> None:
        self.total_events = 0
        self.model_inferences = 0
        self.matched_events = 0
        self.unmatched_events = 0
        self.no_detection_events = 0
        self.gate_updates = 0
        self.buffer_resets = 0
        self.recovered_tracks = 0
        self.gate_sum = np.zeros(2, dtype=np.float64)
        self.gate_min = np.full(2, np.inf, dtype=np.float64)
        self.gate_max = np.full(2, -np.inf, dtype=np.float64)
        self.gate_count = np.zeros(2, dtype=np.int64)

    def record_event(self) -> None:
        self.total_events += 1

    def record_gate_update(self) -> None:
        self.gate_updates += 1

    def record_buffer_reset(self) -> None:
        self.buffer_resets += 1

    def record_recovery(self) -> None:
        self.recovered_tracks += 1

    def record_gates(
        self,
        gates: np.ndarray,
        *,
        matched: bool,
        used_model: bool,
    ) -> None:
        values = np.asarray(gates, dtype=np.float64).reshape(2)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite SHUTrack gates: {values}")
        if used_model:
            self.model_inferences += 1
        if matched:
            self.matched_events += 1
        else:
            self.unmatched_events += 1
            self.no_detection_events += 1
        self.gate_sum += values
        self.gate_min = np.minimum(self.gate_min, values)
        self.gate_max = np.maximum(self.gate_max, values)
        self.gate_count += 1

    def summary(self) -> Dict[str, float]:
        count = max(int(self.gate_count.sum()), 1)
        per_channel = np.maximum(self.gate_count, 1)
        motion_min = (
            float(self.gate_min[0]) if self.gate_count[0] > 0 else 0.0
        )
        appearance_min = (
            float(self.gate_min[1]) if self.gate_count[1] > 0 else 0.0
        )
        motion_max = (
            float(self.gate_max[0]) if self.gate_count[0] > 0 else 0.0
        )
        appearance_max = (
            float(self.gate_max[1]) if self.gate_count[1] > 0 else 0.0
        )
        return {
            "total_events": float(self.total_events),
            "model_inference_count": float(self.model_inferences),
            "matched_events": float(self.matched_events),
            "unmatched_events": float(self.unmatched_events),
            "no_detection_events": float(self.no_detection_events),
            "gate_update_count": float(self.gate_updates),
            "buffer_reset_count": float(self.buffer_resets),
            "recovered_track_count": float(self.recovered_tracks),
            "avg_motion_gate": float(self.gate_sum[0] / count),
            "avg_appearance_gate": float(self.gate_sum[1] / count),
            "min_motion_gate": motion_min,
            "max_motion_gate": motion_max,
            "min_appearance_gate": appearance_min,
            "max_appearance_gate": appearance_max,
            "motion_gate_observations": float(self.gate_count[0]),
            "appearance_gate_observations": float(self.gate_count[1]),
        }


__all__ = ["ShutrackRuntimeStatistics"]
