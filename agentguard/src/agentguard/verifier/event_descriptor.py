from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.features.normalization import NormalizationStats


class EventDescriptor:
    """Builds a fixed-length descriptor for cross-event similarity search.

    The descriptor captures key characteristics of an event used to find
    similar events across different sequences.

    Features (in order):
    - ``candidate_margin``
    - ``row_entropy``
    - ``col_entropy``
    - ``max_detection_overlap``
    - ``detection_score``
    - ``detection_source_onehot`` (3-dim: HIGH, LOW, NMS_DELETED_HIGH)
    - ``track_age``
    - ``lost_gap``
    - ``normalized_speed``
    - ``motion_target``
    - ``appearance_target``
    - ``future_oracle_coverage``

    Total dimension: 1 + 1 + 1 + 1 + 1 + 3 + 1 + 1 + 1 + 1 + 1 + 1 = 14

    Parameters
    ----------
    norm_stats : NormalizationStats or None, optional
        Optional normalisation statistics.  If provided, features are
        z-score normalised before returning.
    """

    _DESCRIPTOR_DIM: int = 14

    def __init__(
        self,
        norm_stats: Optional[NormalizationStats] = None,
    ) -> None:
        self.norm_stats = norm_stats

    @property
    def descriptor_dim(self) -> int:
        return self._DESCRIPTOR_DIM

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def describe(
        self,
        event: TrackEvent,
        label: Any,
    ) -> np.ndarray:
        """Build a descriptor vector for *event*.

        Parameters
        ----------
        event : TrackEvent
            The event to describe.
        label : any
            Label object containing ``motion_target`` and
            ``appearance_target`` attributes (or a tuple).

        Returns
        -------
        ndarray, shape ``(14,)``
            Descriptor vector, optionally z-score normalised.
        """
        scalar = event.scalar_features
        features = np.zeros(self._DESCRIPTOR_DIM, dtype=np.float64)

        # --- 0: candidate_margin = row_2nd_min - row_min ---
        if scalar.size >= 15:
            row_min = float(scalar[13])
            row_2nd = float(scalar[14])
            features[0] = row_2nd - row_min

        # --- 1: row_entropy ---
        if scalar.size >= 19:
            features[1] = float(scalar[18])

        # --- 2: col_entropy ---
        if scalar.size >= 25:
            features[2] = float(scalar[24])

        # --- 3: max_detection_overlap ---
        # Index 25 in scalar features (max_iou_with_other)
        if scalar.size >= 26:
            features[3] = float(scalar[25])

        # --- 4: detection_score ---
        if event.has_detection and event.detection is not None:
            features[4] = event.detection.score

        # --- 5-7: detection_source one-hot (HIGH=0, LOW=1, NMS_DELETED_HIGH=2) ---
        if event.has_detection and event.association is not None:
            source = int(event.association.detection_source)
            if source == 0:
                features[5] = 1.0
            elif source == 1:
                features[6] = 1.0
            elif source == 2:
                features[7] = 1.0

        # --- 8: track_age ---
        features[8] = self._compute_track_age(event)

        # --- 9: lost_gap ---
        features[9] = self._compute_lost_gap(event)

        # --- 10: normalized_speed ---
        features[10] = self._compute_normalized_speed(event)

        # --- 11: motion_target ---
        motion_target = self._get_motion_target(label)
        if motion_target is not None:
            features[11] = motion_target

        # --- 12: appearance_target ---
        appearance_target = self._get_appearance_target(label)
        if appearance_target is not None:
            features[12] = appearance_target

        # --- 13: future_oracle_coverage ---
        coverage = getattr(event, "future_oracle_coverage", None)
        if coverage is not None:
            features[13] = float(coverage)

        # Optional normalisation
        if self.norm_stats is not None:
            features = self.norm_stats.transform(features)

        return features

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_track_age(event: TrackEvent) -> float:
        """Track age as number of frames in history."""
        state = event.pre_update_state
        if state is not None and state.history:
            return float(len(state.history))
        return 0.0

    @staticmethod
    def _compute_lost_gap(event: TrackEvent) -> float:
        """Gap (in frames) since the track was last observed."""
        state = event.pre_update_state
        if state is not None:
            gap = event.frame_id - state.end_frame_id
            return float(max(0, gap))
        return 0.0

    @staticmethod
    def _compute_normalized_speed(event: TrackEvent) -> float:
        """Compute normalised speed from KF velocity.

        Speed = sqrt(vx^2 + vy^2) normalised by image dimensions.
        """
        state = event.pre_update_state
        if state is None or state.velocity is None:
            return 0.0

        # velocity is (4, 2): column 0 = x components, column 1 = y components
        vx = float(state.velocity[0, 0])  # dx
        vy = float(state.velocity[1, 1])  # dy
        speed = np.sqrt(vx**2 + vy**2)
        # Normalise by image diagonal
        diag = np.sqrt(event.image_width**2 + event.image_height**2)
        if diag > 0:
            return float(speed / diag)
        return 0.0

    @staticmethod
    def _get_motion_target(label: Any) -> Optional[float]:
        """Extract motion target from label."""
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
        """Extract appearance target from label."""
        if label is None:
            return None
        if isinstance(label, (list, tuple)) and len(label) >= 2:
            return float(label[1])
        if hasattr(label, "appearance_target"):
            v = getattr(label, "appearance_target")
            return float(v) if v is not None else None
        return None
