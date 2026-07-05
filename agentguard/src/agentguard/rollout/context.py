"""Rollout context dataclass."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from agentguard.contracts.states import TrackStateSnapshot, DetectionObservation


@dataclass
class RolloutContext:
    """Aggregated context for computing motion and appearance benefits."""

    frame_id: int
    target_gt_id: int
    pre_update_state: TrackStateSnapshot
    current_candidate: DetectionObservation
    current_gt_box: np.ndarray
    future_gt_boxes: List[Optional[np.ndarray]]
    future_oracle_detections: List[Optional[DetectionObservation]]
    future_warp_matrices: List[np.ndarray]
    identity_prototype: Optional[np.ndarray]
    appearance_alpha: float = 0.95
