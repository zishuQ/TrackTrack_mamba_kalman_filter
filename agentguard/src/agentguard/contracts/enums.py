from enum import IntEnum

import numpy as np


class DetectionSource(IntEnum):
    HIGH = 0
    LOW = 1
    NMS_DELETED_HIGH = 2


POLICY_PROTOTYPE_MATRIX: np.ndarray = np.array(
    [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5],
    ],
    dtype=np.float64,
)
