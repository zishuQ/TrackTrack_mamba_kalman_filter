from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.enums import DetectionSource


class TeacherEventSelector:
    """Selects events for Teacher annotation.

    Criteria (any one is sufficient for an event to be selected):
    - ``motion_target`` in [0.35, 0.65]
    - ``appearance_target`` in [0.35, 0.65]
    - ``|motion_target - appearance_target| >= 0.4``
    - ``candidate_margin <= 0.03``
    - ``row_entropy >= 0.7``
    - ``col_entropy >= 0.7``
    - ``detection_source`` is **Dlow** (1) or **Ddel** (2)
    - Student-V0 prediction L1 error >= 0.5
    - ``future_oracle_coverage < 0.4``

    Priority order (highest first):
    1. Student-V0 prediction error
    2. Motion/appearance strong conflict
    3. Low future oracle coverage
    4. High candidate competition
    5. Dlow / Ddel detection
    6. Label near 0.5

    Max 10000 events in first round.
    """

    _DET_SOURCE_DLOW: int = int(DetectionSource.LOW)       # 1
    _DET_SOURCE_DDEL: int = int(DetectionSource.NMS_DELETED_HIGH)  # 2

    def select(
        self,
        events: List[TrackEvent],
        labels: List[Any],
        student_preds: Optional[List[Optional[np.ndarray]]] = None,
        max_events: int = 10000,
    ) -> List[Tuple[int, TrackEvent, str]]:
        """Select events for Teacher annotation.

        Parameters
        ----------
        events : list of TrackEvent
            All candidate events from the data cache.
        labels : list
            Ground-truth labels or scalar targets for each event (e.g.
            ``(motion_target, appearance_target)`` tuples or objects
            with ``.motion_target``, ``.appearance_target`` attributes).
        student_preds : list of ndarray or None, optional
            Student-V0 prediction for each event.  Each element is a
            ``(2,)`` array ``[motion_pred, appearance_pred]`` or ``None``
            if unavailable.
        max_events : int
            Maximum number of events to select (default 10000).

        Returns
        -------
        list of (int, TrackEvent, str)
            Selected events sorted by priority (lower number = higher
            priority).  Each tuple is ``(priority, event, reason)`` where
            *reason* is a short human-readable string describing the
            triggering criterion.
        """
        scored: List[Tuple[int, TrackEvent, str]] = []

        for i, event in enumerate(events):
            label = labels[i] if i < len(labels) else None
            student_pred = (
                student_preds[i] if student_preds is not None and i < len(student_preds)
                else None
            )

            criteria, priority = self._evaluate(event, label, student_pred)
            if criteria is not None:
                scored.append((priority, event, criteria))

        # Sort by priority (ascending) then by event_id for stability.
        scored.sort(key=lambda x: (x[0], x[1].event_id))

        return scored[:max_events]

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        event: TrackEvent,
        label: Any,
        student_pred: Optional[np.ndarray],
    ) -> Tuple[Optional[str], int]:
        """Check whether an event meets any selection criterion.

        Returns
        -------
        (reason, priority) or (None, -1) if the event is not selected.
        """
        scalar = event.scalar_features

        # --- Extract fields from scalar features ---
        # Indices (from scalar.py):
        #   0: iou_similarity, 1: iou_distance, ...
        #   10-12: detection_source one-hot
        #   13-18: row stats (min, 2nd_min, diff, mean, std, entropy)
        #   19-24: col stats

        # Detection source: indices 10, 11, 12
        det_source_high = float(scalar[10]) if scalar.size > 10 else 0.0
        det_source_low = float(scalar[11]) if scalar.size > 11 else 0.0
        det_source_ddel = float(scalar[12]) if scalar.size > 12 else 0.0

        # Resolve the actual source integer
        if det_source_high > 0.5:
            source = 0  # HIGH
        elif det_source_low > 0.5:
            source = 1  # LOW
        elif det_source_ddel > 0.5:
            source = 2  # NMS_DELETED_HIGH
        else:
            source = -1

        # Row entropy at index 18, col entropy at index 24
        row_entropy = float(scalar[18]) if scalar.size > 18 else 0.0
        col_entropy = float(scalar[24]) if scalar.size > 24 else 0.0

        # Candidate margin = row second_min - row_min
        row_min = float(scalar[13]) if scalar.size > 13 else 0.0
        row_2nd = float(scalar[14]) if scalar.size > 14 else 1.0
        candidate_margin = row_2nd - row_min

        # --- Target / label fields ---
        motion_target = self._get_motion_target(label)
        appearance_target = self._get_appearance_target(label)

        # Future oracle coverage (may be stored in event extras)
        future_oracle_coverage = self._get_future_oracle_coverage(event)

        # --- Priority 1: Student-V0 prediction error ---
        if (
            student_pred is not None
            and len(student_pred) >= 2
            and motion_target is not None
            and appearance_target is not None
        ):
            l1_error = float(
                abs(student_pred[0] - motion_target)
                + abs(student_pred[1] - appearance_target)
            )
            if l1_error >= 0.5:
                return (
                    f"student_pred_error={l1_error:.3f}",
                    1,
                )

        # --- Priority 2: Motion/appearance strong conflict ---
        if motion_target is not None and appearance_target is not None:
            conflict = abs(motion_target - appearance_target)
            if conflict >= 0.4:
                return (
                    f"motion_appearance_conflict={conflict:.3f}",
                    2,
                )

        # --- Priority 3: Low future oracle coverage ---
        if future_oracle_coverage is not None and future_oracle_coverage < 0.4:
            return (
                f"low_oracle_coverage={future_oracle_coverage:.3f}",
                3,
            )

        # --- Priority 4: High candidate competition ---
        if candidate_margin <= 0.03:
            return (
                f"low_candidate_margin={candidate_margin:.4f}",
                4,
            )

        # --- Priority 5: Dlow / Ddel detection ---
        if source in (self._DET_SOURCE_DLOW, self._DET_SOURCE_DDEL):
            return (
                f"detection_source={source}",
                5,
            )

        # --- Priority 6: Label near 0.5 ---
        if motion_target is not None and 0.35 <= motion_target <= 0.65:
            return (
                f"motion_target_near_0.5={motion_target:.3f}",
                6,
            )
        if appearance_target is not None and 0.35 <= appearance_target <= 0.65:
            return (
                f"appearance_target_near_0.5={appearance_target:.3f}",
                6,
            )

        # --- Additional: high entropy ---
        if row_entropy >= 0.7:
            return (
                f"high_row_entropy={row_entropy:.3f}",
                6,
            )
        if col_entropy >= 0.7:
            return (
                f"high_col_entropy={col_entropy:.3f}",
                6,
            )

        return (None, -1)

    @staticmethod
    def _get_motion_target(label: Any) -> Optional[float]:
        """Extract motion target from a label object."""
        if label is None:
            return None
        if isinstance(label, (list, tuple)) and len(label) >= 2:
            return float(label[0])
        if hasattr(label, "motion_target"):
            v = getattr(label, "motion_target")
            return float(v) if v is not None else None
        return None

    @staticmethod
    def _get_appearance_target(label: Any) -> Optional[float]:
        """Extract appearance target from a label object."""
        if label is None:
            return None
        if isinstance(label, (list, tuple)) and len(label) >= 2:
            return float(label[1])
        if hasattr(label, "appearance_target"):
            v = getattr(label, "appearance_target")
            return float(v) if v is not None else None
        return None

    @staticmethod
    def _get_future_oracle_coverage(event: TrackEvent) -> Optional[float]:
        """Extract future oracle coverage from event extras if present."""
        coverage = getattr(event, "future_oracle_coverage", None)
        if coverage is not None:
            return float(coverage)
        return None
