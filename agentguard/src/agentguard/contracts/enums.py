from enum import IntEnum

import numpy as np


class DetectionSource(IntEnum):
    HIGH = 0
    LOW = 1
    NMS_DELETED_HIGH = 2


class TrackLifecycle(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class WritePolicy(IntEnum):
    FULL_WRITE = 0
    MOTION_ONLY = 1
    APPEARANCE_ONLY = 2
    HOLD_BOTH = 3
    SOFT_CAUTION = 4


class EventType(IntEnum):
    CLEAN_OBSERVATION = 0
    BOX_JITTER = 1
    PARTIAL_BOX = 2
    MOTION_OUTLIER = 3
    APPEARANCE_CONTAMINATION = 4
    HEAVY_OCCLUSION = 5
    MOTION_APPEARANCE_CONFLICT = 6
    IDENTITY_AMBIGUITY = 7
    LOST_REAPPEARANCE = 8
    INSUFFICIENT_EVIDENCE = 9


POLICY_PROTOTYPE_MATRIX: np.ndarray = np.array(
    [
        [1.0, 1.0],  # FULL_WRITE
        [1.0, 0.0],  # MOTION_ONLY
        [0.0, 1.0],  # APPEARANCE_ONLY
        [0.0, 0.0],  # HOLD_BOTH
        [0.5, 0.5],  # SOFT_CAUTION
    ],
    dtype=np.float64,
)
