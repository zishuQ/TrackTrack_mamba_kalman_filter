from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TrackStateSnapshot:
    """Immutable snapshot of a track's state at a given point in time.

    All numpy arrays provided to the constructor are deep-copied to ensure
    the snapshot cannot be mutated after creation.
    """

    track_id: int
    box: np.ndarray  # (4,) x1y1x2y2
    score: float
    mean: Optional[np.ndarray]  # (8,) KF mean or None
    covariance: Optional[np.ndarray]  # (8, 8) KF cov or None
    velocity: np.ndarray  # (4, 2)
    feature: np.ndarray  # (1, D) ReID feature
    history: Dict[int, Any]  # {frame_id: [box, score, mean, cov, feat]}
    end_frame_id: int
    state: int  # TrackState enum value

    def __post_init__(self) -> None:
        # Deep-copy all numpy arrays to enforce immutability.
        object.__setattr__(self, "box", copy.deepcopy(self.box))
        object.__setattr__(self, "velocity", copy.deepcopy(self.velocity))
        object.__setattr__(self, "feature", copy.deepcopy(self.feature))

        if self.mean is not None:
            object.__setattr__(self, "mean", copy.deepcopy(self.mean))
        if self.covariance is not None:
            object.__setattr__(self, "covariance", copy.deepcopy(self.covariance))

        # Rebuild history with deep copies of all array entries.
        history_copy: Dict[int, Any] = {}
        for frame_id, entry in self.history.items():
            copied_entry: list[Any] = []
            for item in entry:
                if isinstance(item, np.ndarray):
                    copied_entry.append(copy.deepcopy(item))
                else:
                    copied_entry.append(item)
            history_copy[frame_id] = copied_entry
        object.__setattr__(self, "history", history_copy)


@dataclass(frozen=True)
class DetectionObservation:
    """A single detection from one source, with associated geometric and appearance data."""

    detection_index: int
    box: np.ndarray  # (4,) x1y1x2y2, copied
    score: float
    feature: np.ndarray  # (1, D), copied
    source: int  # DetectionSource
    class_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "box", copy.deepcopy(self.box))
        object.__setattr__(self, "feature", copy.deepcopy(self.feature))

    @property
    def x1y1x2y2(self) -> np.ndarray:
        return self.box

    @property
    def x1y1wh(self) -> np.ndarray:
        return np.array(
            [
                self.box[0],
                self.box[1],
                self.box[2] - self.box[0],
                self.box[3] - self.box[1],
            ]
        )

    @property
    def cxcywh(self) -> np.ndarray:
        return np.array(
            [
                (self.box[0] + self.box[2]) / 2,
                (self.box[1] + self.box[3]) / 2,
                self.box[2] - self.box[0],
                self.box[3] - self.box[1],
            ]
        )


@dataclass(frozen=True)
class AssociationPairFeatures:
    """Aggregated matching features computed between a track and a detection."""

    iou_similarity: float
    iou_distance: float
    cosine_distance: float
    confidence_distance: float
    angle_distance: float
    raw_cost: float
    final_cost: float
    assignment_round: int  # -1 if unmatched
    assignment_threshold: float  # -1.0 if unmatched
    detection_source: int
