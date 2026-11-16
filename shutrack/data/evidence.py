"""Per-event evidence shared by offline windows and the online buffer.

An ``EventEvidence`` is captured strictly before the track write of its frame:
the appearance vector is the pre-update track template, the detection vector
is the event's own detection (or zeros with ``has_detection=False``), and all
scalar evidence is in the model's normalization space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shutrack.schema import CONTEXT_DIM, MOTION_DIM


@dataclass(frozen=True)
class EventEvidence:
    motion: np.ndarray            # (12,) float32, normalized space
    context: np.ndarray           # (8,) float32, fixed schema order
    track_feature: np.ndarray     # (D,) float32, pre-update template
    detection_feature: np.ndarray  # (D,) float32, zeros without detection
    has_detection: bool
    frame_id: int
    track_id: int
    segment_id: int
    observation_count: int

    def __post_init__(self) -> None:
        motion = np.asarray(self.motion, dtype=np.float32).reshape(-1)
        context = np.asarray(self.context, dtype=np.float32).reshape(-1)
        track = np.asarray(self.track_feature, dtype=np.float32).reshape(-1)
        detection = np.asarray(self.detection_feature, dtype=np.float32).reshape(-1)
        if motion.shape != (MOTION_DIM,):
            raise ValueError(f"motion must have shape ({MOTION_DIM},), got {motion.shape}")
        if context.shape != (CONTEXT_DIM,):
            raise ValueError(
                f"context must have shape ({CONTEXT_DIM},), got {context.shape}"
            )
        if track.shape != detection.shape:
            raise ValueError(
                "track_feature and detection_feature must share a dimension, got "
                f"{track.shape} and {detection.shape}"
            )
        if track.size == 0:
            raise ValueError("appearance features must not be empty")
        for name, value in (
            ("motion", motion),
            ("context", context),
            ("track_feature", track),
            ("detection_feature", detection),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "track_feature", track)
        object.__setattr__(self, "detection_feature", detection)
        object.__setattr__(self, "has_detection", bool(self.has_detection))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "segment_id", int(self.segment_id))
        object.__setattr__(self, "observation_count", int(self.observation_count))

    @property
    def reid_dim(self) -> int:
        return int(self.track_feature.shape[0])


__all__ = ["EventEvidence"]
