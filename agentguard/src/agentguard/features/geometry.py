from __future__ import annotations

import numpy as np


def _boxes_xyxy(value: np.ndarray) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if boxes.size % 4 != 0:
        raise ValueError(f"xyxy boxes must contain groups of four values, got {boxes.shape}")
    boxes = boxes.reshape(-1, 4)
    if not np.all(np.isfinite(boxes)):
        raise ValueError("xyxy boxes must contain only finite values")
    return boxes


def pairwise_iou_xyxy(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Match TrackTrack ``bbox_overlaps`` inclusive ``+1`` semantics."""
    a = _boxes_xyxy(boxes_a)
    b = _boxes_xyxy(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_wh = np.maximum(right_bottom - left_top + 1.0, 0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area_a_wh = np.maximum(a[:, 2:] - a[:, :2] + 1.0, 0.0)
    area_b_wh = np.maximum(b[:, 2:] - b[:, :2] + 1.0, 0.0)
    area_a = area_a_wh[:, 0] * area_a_wh[:, 1]
    area_b = area_b_wh[:, 0] * area_b_wh[:, 1]
    union = area_a[:, None] + area_b[None, :] - intersection
    overlaps = np.divide(
        intersection,
        np.maximum(union, np.finfo(np.float64).eps),
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    if not np.all(np.isfinite(overlaps)):
        raise ValueError("pairwise IoU produced non-finite values")
    return overlaps
