"""Numeric and association-level regression for vectorized bbox_overlaps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "../3. Tracker"))

from trackers.utils import (
    bbox_overlaps,
    find_deleted_detections,
    iou_distance,
    iterative_assignment,
    track_aware_nms,
)


def bbox_overlaps_loop(a_x1y1x2y2, b_x1y1x2y2):
    """Independent copy of the original TrackTrack Python double loop."""
    num_a = a_x1y1x2y2.shape[0]
    num_b = b_x1y1x2y2.shape[0]
    overlaps = np.zeros((num_a, num_b))

    for n_b in range(num_b):
        box_area = (b_x1y1x2y2[n_b, 2] - b_x1y1x2y2[n_b, 0] + 1) * (
            b_x1y1x2y2[n_b, 3] - b_x1y1x2y2[n_b, 1] + 1
        )
        for n_a in range(num_a):
            iw = (
                min(a_x1y1x2y2[n_a, 2], b_x1y1x2y2[n_b, 2])
                - max(a_x1y1x2y2[n_a, 0], b_x1y1x2y2[n_b, 0])
                + 1
            )
            if iw > 0:
                ih = (
                    min(a_x1y1x2y2[n_a, 3], b_x1y1x2y2[n_b, 3])
                    - max(a_x1y1x2y2[n_a, 1], b_x1y1x2y2[n_b, 1])
                    + 1
                )
                if ih > 0:
                    ua = (
                        (a_x1y1x2y2[n_a, 2] - a_x1y1x2y2[n_a, 0] + 1)
                        * (a_x1y1x2y2[n_a, 3] - a_x1y1x2y2[n_a, 1] + 1)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n_a, n_b] = iw * ih / ua
    return overlaps


class _AssocTrack:
    """Minimal track/detection stub for iou_distance and iterative_assignment."""

    def __init__(self, box, score=0.9, feat=None, history=None, velocity=None):
        self.x1y1x2y2 = np.asarray(box, dtype=np.float64)
        self.score = float(score)
        self.feat = (
            np.asarray(feat, dtype=np.float64)
            if feat is not None
            else np.ones((1, 8), dtype=np.float64)
        )
        box_copy = self.x1y1x2y2.copy()
        self.history = history if history is not None else {
            0: [box_copy, self.score],
            1: [box_copy, self.score],
        }
        self.velocity = (
            np.asarray(velocity, dtype=np.float64)
            if velocity is not None
            else np.zeros((4, 2), dtype=np.float64)
        )


def _random_valid_boxes(rng, count):
    starts = rng.uniform(-20.0, 100.0, size=(count, 2))
    sizes = rng.uniform(0.0, 80.0, size=(count, 2))
    return np.concatenate([starts, starts + sizes], axis=1)


def test_bbox_overlaps_matches_original_loop_on_random_valid_boxes():
    rng = np.random.default_rng(20260908)
    boxes_a = _random_valid_boxes(rng, 50)
    boxes_b = _random_valid_boxes(rng, 37)

    actual = bbox_overlaps(boxes_a, boxes_b)
    expected = bbox_overlaps_loop(boxes_a, boxes_b)

    assert actual.shape == expected.shape == (50, 37)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_bbox_overlaps_edge_cases_match_original_loop():
    disjoint = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    identical = np.array([[0.0, 0.0, 10.0, 10.0]])
    contained = np.array([[2.0, 2.0, 8.0, 8.0]])
    touching = np.array([[10.0, 0.0, 20.0, 10.0]])
    small = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    inverted = np.array([[8.0, 8.0, 2.0, 2.0], [5.0, 1.0, 4.0, 9.0]])
    empty = np.empty((0, 4), dtype=np.float64)
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    cases = [
        (empty, boxes),
        (boxes, empty),
        (empty, empty),
        (identical, identical),
        (identical, contained),
        (identical, touching),
        (disjoint, disjoint),
        (small, small),
        (boxes, inverted),
        (inverted, boxes),
    ]
    for left, right in cases:
        actual = bbox_overlaps(left, right)
        expected = bbox_overlaps_loop(left, right)
        assert actual.shape == expected.shape
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, expected)

    # Inclusive +1: horizontally adjacent boxes share one pixel column.
    adjacent = bbox_overlaps(identical, touching)
    assert adjacent.shape == (1, 1)
    assert adjacent[0, 0] > 0.0
    np.testing.assert_array_equal(adjacent, bbox_overlaps_loop(identical, touching))


def test_bbox_overlaps_noncontiguous_and_float64_output():
    rng = np.random.default_rng(7)
    boxes_a = np.asfortranarray(_random_valid_boxes(rng, 9))
    boxes_b = _random_valid_boxes(rng, 6)[::-1]
    assert not boxes_a.flags["C_CONTIGUOUS"]
    assert not boxes_b.flags["C_CONTIGUOUS"]

    actual = bbox_overlaps(boxes_a, boxes_b)
    expected = bbox_overlaps_loop(
        np.asarray(boxes_a, dtype=np.float64),
        np.asarray(boxes_b, dtype=np.float64),
    )
    assert actual.dtype == np.float64
    assert actual.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(actual, expected)


def test_bbox_overlaps_does_not_import_agentguard_or_torch():
    import ast
    import trackers.utils as utils_mod

    tree = ast.parse(Path(utils_mod.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".", 1)[0])
    assert "agentguard" not in imported
    assert "torch" not in imported
    assert imported == ["lap", "numpy"]


def _run_assignment(tracks, dets_high, dets_low, dets_del):
    return iterative_assignment(
        tracks,
        dets_high,
        dets_low,
        dets_del,
        match_thr=0.7,
        penalty_p=0.20,
        penalty_q=0.40,
        reduce_step=0.05,
        frame_id=3,
        no_reid=False,
    )


def test_association_and_nms_match_original_iou_loop(monkeypatch):
    rng = np.random.default_rng(11)
    track_boxes = _random_valid_boxes(rng, 8)
    det_boxes = _random_valid_boxes(rng, 12)
    feats = rng.normal(size=(20, 1, 8))
    scores = rng.uniform(0.2, 0.99, size=20)

    tracks = [
        _AssocTrack(track_boxes[i], score=float(scores[i]), feat=feats[i])
        for i in range(8)
    ]
    dets = [
        _AssocTrack(det_boxes[i], score=float(scores[8 + i]), feat=feats[8 + i])
        for i in range(12)
    ]
    dets_high = dets[:5]
    dets_low = dets[5:9]
    dets_del = dets[9:]

    import trackers.utils as utils_mod

    matches, u_tracks, u_dets = _run_assignment(
        tracks, dets_high, dets_low, dets_del
    )
    monkeypatch.setattr(utils_mod, "bbox_overlaps", bbox_overlaps_loop)
    expected_matches, expected_u_tracks, expected_u_dets = _run_assignment(
        tracks, dets_high, dets_low, dets_del
    )
    assert matches == expected_matches
    assert u_tracks == expected_u_tracks
    assert u_dets == expected_u_dets
    monkeypatch.undo()

    pair_sims = iou_distance(tracks + dets, tracks + dets)[0]
    allow = track_aware_nms(
        pair_sims,
        np.array([d.score for d in dets]),
        len(tracks),
        nms_thresh=0.55,
        score_thresh=0.6,
    )
    monkeypatch.setattr(utils_mod, "bbox_overlaps", bbox_overlaps_loop)
    expected_pair = iou_distance(tracks + dets, tracks + dets)[0]
    expected_allow = track_aware_nms(
        expected_pair,
        np.array([d.score for d in dets]),
        len(tracks),
        nms_thresh=0.55,
        score_thresh=0.6,
    )
    np.testing.assert_array_equal(pair_sims, expected_pair)
    np.testing.assert_array_equal(allow, expected_allow)
    monkeypatch.undo()

    dets80 = np.concatenate(
        [track_boxes[:4], scores[:4, None], np.zeros((4, 1)), feats[:4, 0]],
        axis=1,
    )
    dets95 = np.concatenate(
        [
            np.vstack([track_boxes[:4], det_boxes[:6]]),
            scores[:10, None],
            np.zeros((10, 1)),
            feats[:10, 0],
        ],
        axis=1,
    )
    deleted = find_deleted_detections(dets80, dets95)
    monkeypatch.setattr(utils_mod, "bbox_overlaps", bbox_overlaps_loop)
    expected_deleted = find_deleted_detections(dets80, dets95)
    np.testing.assert_array_equal(deleted, expected_deleted)


def test_iou_threshold_decisions_match_original_loop():
    rng = np.random.default_rng(99)
    boxes_a = _random_valid_boxes(rng, 40)
    boxes_b = _random_valid_boxes(rng, 40)
    actual = bbox_overlaps(boxes_a, boxes_b)
    expected = bbox_overlaps_loop(boxes_a, boxes_b)
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs == 0.0
    for threshold in (0.10, 0.50, 0.70, 0.97):
        np.testing.assert_array_equal(actual > threshold, expected > threshold)
        np.testing.assert_array_equal(actual <= 0.10, expected <= 0.10)
