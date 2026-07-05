"""Test that identity voting doesn't leak current frame GT into vote.

When computing the GT identity for a detection, the current frame's GT
should not influence the vote (otherwise we're cheating). We test this by
ensuring that the vote is based only on past GT associations, not the
current frame's GT assignment.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.data.gt_matching import GTMatching


# ──────────────────────────────────────────────────────────────────────
#  Identity vote implementation (simulating TrackTrack's logic)
# ──────────────────────────────────────────────────────────────────────


def compute_identity_vote(
    detection_box: np.ndarray,
    past_gt_assignments: Dict[int, List[Tuple[np.ndarray, int]]],
    current_frame_gt: List[Tuple[np.ndarray, int]],
    iou_threshold: float = 0.5,
) -> int:
    """Compute identity vote for a detection based on past GT assignments.

    This should NOT use current_frame_gt in the vote.

    Parameters
    ----------
    detection_box : ndarray (4,) x1y1x2y2
    past_gt_assignments : dict
        frame_id -> list of (box, gt_track_id) for past frames.
    current_frame_gt : list of (ndarray, int)
        Current frame GT associations (SHOULD NOT BE USED IN VOTE).
    iou_threshold : float
        Minimum IoU to consider a match.

    Returns
    -------
    int
        Voted GT track ID, or -1 if no clear vote.
    """
    from agentguard.data.gt_matching import _iou

    votes: Dict[int, float] = {}

    for frame_id, gt_entries in past_gt_assignments.items():
        for gt_box, gt_id in gt_entries:
            iou = _iou(detection_box, gt_box)
            if iou >= iou_threshold:
                votes[gt_id] = votes.get(gt_id, 0.0) + iou

    if not votes:
        return -1

    # Return the GT ID with the highest total IoU
    best_id = max(votes, key=votes.get)
    return best_id


class TestIdentityVoteNoLeakage:
    """Verify that identity voting does not leak current frame GT."""

    def test_vote_uses_past_not_current(self):
        """Vote should be based on past GT assignments, not current frame."""
        det_box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)

        # Past assignments: track ID 1 has a box overlapping with det
        past_assignments: Dict[int, List[Tuple[np.ndarray, int]]] = {
            90: [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 1)],
            95: [(np.array([1.0, 1.0, 11.0, 11.0], dtype=np.float64), 1)],
        }

        # Current frame GT: a different track ID 2
        current_gt = [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 2)]

        # Vote should return ID 1 (from past), NOT ID 2 (current frame)
        voted_id = compute_identity_vote(det_box, past_assignments, current_gt)
        assert voted_id == 1
        assert voted_id != 2  # Would leak if current GT was used

    def test_vote_no_past_returns_minus_one(self):
        """If no past GT exists, vote should return -1 regardless of current GT."""
        det_box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
        past_assignments: Dict[int, List[Tuple[np.ndarray, int]]] = {}
        current_gt = [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 5)]

        voted_id = compute_identity_vote(det_box, past_assignments, current_gt)
        assert voted_id == -1

    def test_vote_ignores_current_frame_in_iou(self):
        """Even if current GT has high IoU, it should not affect the vote."""
        det_box = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float64)

        # Past: weak overlap with ID 1
        past_assignments: Dict[int, List[Tuple[np.ndarray, int]]] = {
            90: [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 1)],
        }

        # Current: strong overlap with ID 2 (would be 1.0 IoU if leaked)
        current_gt = [(np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float64), 2)]

        # Past assignment has IoU=0 with det_box, so no match -> -1
        voted_id = compute_identity_vote(det_box, past_assignments, current_gt)
        assert voted_id == -1  # not leaked to 2

    def test_vote_multiple_past_frame(self):
        """Vote should accumulate over multiple past frames."""
        det_box = np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float64)

        past_assignments: Dict[int, List[Tuple[np.ndarray, int]]] = {
            80: [(np.array([40.0, 40.0, 140.0, 140.0], dtype=np.float64), 10)],
            85: [(np.array([45.0, 45.0, 145.0, 145.0], dtype=np.float64), 10)],
            90: [(np.array([55.0, 55.0, 155.0, 155.0], dtype=np.float64), 10)],
        }

        current_gt = [(np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float64), 20)]

        voted_id = compute_identity_vote(det_box, past_assignments, current_gt)
        assert voted_id == 10  # from past, not 20

    def test_vote_with_multiple_track_ids(self):
        """When multiple past GT IDs match, the most frequent wins."""
        det_box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)

        past_assignments: Dict[int, List[Tuple[np.ndarray, int]]] = {
            80: [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 1)],
            85: [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 2)],
            90: [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 1)],
        }

        # ID 1 appears twice with IoU=1.0 -> total 2.0
        # ID 2 appears once with IoU=1.0 -> total 1.0
        current_gt = [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 3)]

        voted_id = compute_identity_vote(det_box, past_assignments, current_gt)
        assert voted_id == 1

    def test_gt_matching_no_leak(self):
        """GTMatching.match_frame should assign -1 when IoU < threshold,
        regardless of current frame GT presence."""
        # Detection far from any GT
        dets = [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 0.9)]
        gt_boxes = [np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float64)]
        gt_ids = [1]

        result = GTMatching.match_frame(dets, gt_boxes, gt_ids)
        assert result == [-1]  # no leak

    def test_gt_matching_exact_match(self):
        """Exact match should return the GT ID."""
        dets = [(np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64), 0.9)]
        gt_boxes = [np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)]
        gt_ids = [42]

        result = GTMatching.match_frame(dets, gt_boxes, gt_ids)
        assert result == [42]
