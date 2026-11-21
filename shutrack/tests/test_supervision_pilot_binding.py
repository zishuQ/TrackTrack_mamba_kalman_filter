"""Unit tests for detection-to-GT binding in the supervision replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from sportsmot_supervision_pilot import (  # noqa: E402
    bind_gt_box,
    max_box_iou,
    recovered_detection_rows,
)


def test_no_gt_boxes_is_unbound() -> None:
    gt_id, iou = bind_gt_box(np.array([0.0, 0.0, 10.0, 10.0]), {})
    assert gt_id == -1
    assert iou == 0.0


def test_no_intersection_is_unbound_with_zero_iou() -> None:
    gt_id, iou = bind_gt_box(
        np.array([100.0, 100.0, 110.0, 110.0]), {1: np.array([0.0, 0.0, 10.0, 10.0])}
    )
    assert gt_id == -1
    assert iou == 0.0


def test_exact_half_iou_is_bound() -> None:
    # 5x10 detection inside a 10x10 GT: intersection 50, union 100 -> IoU 0.5.
    gt_id, iou = bind_gt_box(
        np.array([0.0, 0.0, 5.0, 10.0]), {3: np.array([0.0, 0.0, 10.0, 10.0])}
    )
    assert iou == pytest.approx(0.5, abs=0.0)
    assert gt_id == 3


def test_just_below_half_iou_is_unbound_but_reports_true_iou() -> None:
    gt_id, iou = bind_gt_box(
        np.array([0.0, 0.0, 4.999, 10.0]), {3: np.array([0.0, 0.0, 10.0, 10.0])}
    )
    assert gt_id == -1
    assert iou == pytest.approx(0.4999, abs=1e-9)


def test_binds_to_best_gt() -> None:
    boxes = {
        1: np.array([0.0, 0.0, 10.0, 10.0]),
        2: np.array([0.0, 0.0, 8.0, 10.0]),
    }
    gt_id, iou = bind_gt_box(np.array([0.0, 0.0, 7.0, 10.0]), boxes)
    assert gt_id == 2
    assert iou == pytest.approx(0.875, abs=1e-9)


def test_max_box_iou_matches_pairwise_maximum() -> None:
    source = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    target = np.array([[0.0, 0.0, 10.0, 10.0]])
    result = max_box_iou(source, target)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
    assert max_box_iou(np.zeros((0, 4)), target).size == 0


def test_recovered_pool_uses_online_inclusive_iou_boundary() -> None:
    # Exclusive-convention label IoU would be 32/33 = 0.969697 (<0.97), but
    # the online bbox_overlaps uses +1 pixels and gives 33/34 = 0.970588
    # (>=0.97), so the online pool does NOT recover this source detection.
    target = np.array([[0.0, 0.0, 32.0, 10.0, 0.9, 0.0]])
    source = np.array([[0.0, 0.0, 33.0, 10.0, 0.9, 0.0]])
    rows = recovered_detection_rows(target, source, np.array([7]), det_thr=0.55)
    assert rows == []


def test_recovered_pool_keeps_source_index_and_score_gate() -> None:
    target = np.array([[0.0, 0.0, 32.0, 10.0, 0.9, 0.0]])
    source = np.array(
        [
            [0.0, 0.0, 31.0, 10.0, 0.9, 0.0],
            [50.0, 50.0, 60.0, 60.0, 0.5, 0.0],
        ]
    )
    rows = recovered_detection_rows(target, source, np.array([11, 12]), det_thr=0.55)
    assert [index for index, _ in rows] == [11]
    assert float(rows[0][1][4]) == pytest.approx(0.9)


def test_recovered_pool_is_empty_for_empty_target_frame() -> None:
    source = np.array([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]])
    assert recovered_detection_rows(None, source, np.array([3]), det_thr=0.55) == []
    assert (
        recovered_detection_rows(np.zeros((0, 6)), source, np.array([3]), det_thr=0.55)
        == []
    )
