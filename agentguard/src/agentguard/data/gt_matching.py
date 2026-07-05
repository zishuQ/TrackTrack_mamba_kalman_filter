from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
#  IoU helpers
# ---------------------------------------------------------------------------


def _box_area(box: np.ndarray) -> float:
    """Area of an ``(x1, y1, x2, y2)`` box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Union between two ``(4,)`` boxes (x1y1x2y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _iou_matrix(
    det_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
) -> np.ndarray:
    """Compute ``(len(det_boxes), len(gt_boxes))`` IoU matrix."""
    n_det = len(det_boxes)
    n_gt = len(gt_boxes)
    matrix = np.zeros((n_det, n_gt), dtype=np.float64)
    for i, db in enumerate(det_boxes):
        for j, gb in enumerate(gt_boxes):
            matrix[i, j] = _iou(db, gb)
    return matrix


# ---------------------------------------------------------------------------
#  Matching class
# ---------------------------------------------------------------------------


class GTMatching:
    """Matches detections to ground truth using Hungarian assignment on IoU.

    Only matches with IoU >= 0.5 are considered valid; unmatched detections
    receive GT ID = -1.
    """

    _IOU_THRESHOLD: float = 0.5

    @staticmethod
    def match_frame(
        detections: List[Tuple[np.ndarray, float]],
        gt_boxes: List[np.ndarray],
        gt_ids: List[int],
    ) -> List[int]:
        """Match detections to ground-truth boxes for a single frame.

        Parameters
        ----------
        detections : list of (ndarray, float)
            Each element is ``(x1y1x2y2 box, detection_score)``.
        gt_boxes : list of ndarray
            Ground-truth boxes in ``(x1y1x2y2)`` format.
        gt_ids : list of int
            Ground-truth track IDs corresponding to each box in *gt_boxes*.

        Returns
        -------
        list of int
            For each detection, the matched GT track ID, or ``-1`` if the
            IoU with the assigned GT is below 0.5.
        """
        n_det = len(detections)
        if n_det == 0:
            return []
        if len(gt_boxes) == 0:
            return [-1] * n_det

        det_boxes = [d[0] for d in detections]

        # Build cost matrix: cost = 1 - IoU
        iou_mat = _iou_matrix(det_boxes, gt_boxes)
        cost_mat = 1.0 - iou_mat

        # Hungarian assignment
        det_indices, gt_indices = linear_sum_assignment(cost_mat)

        # Build result array initialised to -1
        matched_ids = [-1] * n_det

        for det_idx, gt_idx in zip(det_indices, gt_indices):
            iou_val = iou_mat[det_idx, gt_idx]
            if iou_val >= GTMatching._IOU_THRESHOLD:
                matched_ids[det_idx] = gt_ids[gt_idx]

        return matched_ids

    @staticmethod
    def match_frame_return_all(
        detections: List[Tuple[np.ndarray, float]],
        gt_boxes: List[np.ndarray],
        gt_ids: List[int],
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
        """Extended version that also returns the cost / IoU matrices.

        Returns
        -------
        matched_ids : list of int
            GT ID per detection (``-1`` for unmatched).
        iou_matrix : ndarray, shape ``(n_det, n_gt)``
        cost_matrix : ndarray, shape ``(n_det, n_gt)``
        assignment : ndarray, shape ``(2, n_matched)``
            ``assignment[0]`` = detection indices, ``assignment[1]`` = gt indices.
        """
        n_det = len(detections)
        if n_det == 0:
            return [], np.empty((0, 0)), np.empty((0, 0)), np.empty((2, 0), dtype=int)
        if len(gt_boxes) == 0:
            return (
                [-1] * n_det,
                np.empty((n_det, 0)),
                np.empty((n_det, 0)),
                np.empty((2, 0), dtype=int),
            )

        det_boxes = [d[0] for d in detections]
        iou_mat = _iou_matrix(det_boxes, gt_boxes)
        cost_mat = 1.0 - iou_mat

        det_idx, gt_idx = linear_sum_assignment(cost_mat)
        assignment = np.stack([det_idx, gt_idx], axis=0)

        matched_ids = [-1] * n_det
        for d_idx, g_idx in zip(det_idx, gt_idx):
            if iou_mat[d_idx, g_idx] >= GTMatching._IOU_THRESHOLD:
                matched_ids[d_idx] = gt_ids[g_idx]

        return matched_ids, iou_mat, cost_mat, assignment
