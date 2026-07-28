from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agentguard.contracts.states import (
    AssociationContext,
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)


@dataclass
class TrackEvent:
    """Mutable container holding all data associated with a single track event.

    Fields are populated incrementally during tracking and can be persisted
    by the compact event cache.
    """

    # --- Identity & frame metadata ---
    event_id: str
    dataset: str
    sequence: str
    frame_id: int
    track_id: int
    image_width: int
    image_height: int
    has_detection: bool

    # --- State snapshots ---
    frame_start_state: Optional[TrackStateSnapshot] = None
    pre_update_state: Optional[TrackStateSnapshot] = None

    # --- Detection & association ---
    detection: Optional[DetectionObservation] = None
    association: Optional[AssociationPairFeatures] = None
    association_context: Optional[AssociationContext] = None
    warp_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(2, 3, dtype=np.float64)
    )

    # --- Feature vectors ---
    scalar_features: Optional[np.ndarray] = None
    track_feature: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    detection_feature: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    # --- IWG outputs (set during online inference or rollout) ---
    iwg_policy_probs: Optional[np.ndarray] = None
    iwg_gate: Optional[np.ndarray] = None
