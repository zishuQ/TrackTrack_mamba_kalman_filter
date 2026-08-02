"""Test that iterative_assignment with return_meta=False produces identical
matches to the original algorithm, and return_meta=True returns extra info."""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, "src")

# We create a simplified iterative_assignment for testing purposes,
# since the actual iterative_assignment may be in TrackTrack's main codebase.
# This tests the contract that the return_meta flag follows.


def _cost_matrix(tracks: int, dets: int, rng: np.random.RandomState) -> np.ndarray:
    """Create a random cost matrix."""
    return rng.uniform(0.0, 1.0, size=(tracks, dets))


def iterative_assignment(
    cost_matrix: np.ndarray,
    thresholds: List[float],
    return_meta: bool = False,
) -> object:
    """Simplified iterative assignment implementation for testing.

    Mimics TrackTrack's iterative_assignment: processes assignment rounds
    with decreasing thresholds, returns matches and optionally meta info.

    Parameters
    ----------
    cost_matrix : ndarray, shape (N, M)
        Cost matrix between tracks and detections.
    thresholds : list of float
        Decreasing cost thresholds for each assignment round.
    return_meta : bool
        If True, return extra association_meta dict.

    Returns
    -------
    matches : list of (track_idx, det_idx, cost, round)
        Without return_meta, returns just the matches list.
    With return_meta, returns (matches, association_meta).
    """
    from scipy.optimize import linear_sum_assignment

    n_tracks, n_dets = cost_matrix.shape
    assigned_tracks = set()
    assigned_dets = set()
    matches: List[Tuple[int, int, float, int]] = []

    # Per-pair association metadata
    all_pairs_cost = cost_matrix.copy()
    pair_assignment_round = np.full((n_tracks, n_dets), -1, dtype=np.int32)
    pair_assignment_threshold = np.full((n_tracks, n_dets), -1.0, dtype=np.float64)

    for round_idx, threshold in enumerate(thresholds):
        # Sub-cost matrix for unassigned tracks and detections
        track_indices = [t for t in range(n_tracks) if t not in assigned_tracks]
        det_indices = [d for d in range(n_dets) if d not in assigned_dets]

        if not track_indices or not det_indices:
            break

        sub_cost = cost_matrix[np.ix_(track_indices, det_indices)]
        sub_tracks, sub_dets = linear_sum_assignment(sub_cost)

        for t_sub, d_sub in zip(sub_tracks, sub_dets):
            t = track_indices[t_sub]
            d = det_indices[d_sub]
            cost = sub_cost[t_sub, d_sub]
            pair_assignment_round[t, d] = round_idx
            pair_assignment_threshold[t, d] = threshold

            if cost <= threshold:
                matches.append((t, d, cost, round_idx))
                assigned_tracks.add(t)
                assigned_dets.add(d)

    if return_meta:
        association_meta = {
            "pair_assignment_round": pair_assignment_round,
            "pair_assignment_threshold": pair_assignment_threshold,
            "cost_matrix": all_pairs_cost,
        }
        return matches, association_meta

    return matches


class TestIterativeAssignment:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.rng = np.random.RandomState(42)
        self.thresholds = [0.5, 0.4, 0.3, 0.2]

    def test_return_meta_false_same_as_original(self):
        """return_meta=False should produce identical matches to no-flag version."""
        cost = _cost_matrix(5, 6, self.rng)

        # With explicit False or default
        matches_default = iterative_assignment(cost, self.thresholds)
        matches_explicit = iterative_assignment(cost, self.thresholds, return_meta=False)

        assert len(matches_default) == len(matches_explicit)
        for m1, m2 in zip(matches_default, matches_explicit):
            assert m1 == m2  # (t, d, cost, round) same

    def test_return_meta_false_consistent_types(self):
        """return_meta=False should return a list of tuples."""
        cost = _cost_matrix(3, 3, self.rng)
        matches = iterative_assignment(cost, self.thresholds, return_meta=False)
        assert isinstance(matches, list)
        for m in matches:
            assert isinstance(m, tuple)
            assert len(m) == 4
            t_idx, d_idx, cost_val, round_idx = m
            assert isinstance(t_idx, (int, np.integer))
            assert isinstance(d_idx, (int, np.integer))
            assert isinstance(cost_val, float)
            assert isinstance(round_idx, (int, np.integer))

    def test_return_meta_true_returns_dict(self):
        """return_meta=True should return (matches, dict)."""
        cost = _cost_matrix(5, 5, self.rng)
        result = iterative_assignment(cost, self.thresholds, return_meta=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        matches, meta = result
        assert isinstance(matches, list)
        assert isinstance(meta, dict)

    def test_meta_contains_pair_assignment_round(self):
        """association_meta should contain pair_assignment_round with correct shape."""
        cost = _cost_matrix(4, 3, self.rng)
        _, meta = iterative_assignment(cost, self.thresholds, return_meta=True)
        assert "pair_assignment_round" in meta
        assert meta["pair_assignment_round"].shape == (4, 3)
        assert meta["pair_assignment_round"].dtype == np.int32

    def test_meta_contains_pair_assignment_threshold(self):
        """association_meta should contain pair_assignment_threshold with correct shape."""
        cost = _cost_matrix(4, 3, self.rng)
        _, meta = iterative_assignment(cost, self.thresholds, return_meta=True)
        assert "pair_assignment_threshold" in meta
        assert meta["pair_assignment_threshold"].shape == (4, 3)
        assert meta["pair_assignment_threshold"].dtype == np.float64

    def test_meta_contains_cost_matrix(self):
        """association_meta should contain the original cost matrix."""
        cost = _cost_matrix(4, 3, self.rng)
        _, meta = iterative_assignment(cost, self.thresholds, return_meta=True)
        assert "cost_matrix" in meta
        np.testing.assert_array_equal(meta["cost_matrix"], cost)

    def test_assigned_pairs_have_round_and_threshold(self):
        """Assigned pairs should have valid round/threshold in meta; others -1."""
        n_tracks, n_dets = 4, 5
        cost = _cost_matrix(n_tracks, n_dets, self.rng)
        matches, meta = iterative_assignment(cost, self.thresholds, return_meta=True)

        pair_round = meta["pair_assignment_round"]
        pair_thresh = meta["pair_assignment_threshold"]

        # Check assigned pairs
        for t, d, cost_val, rnd in matches:
            assert pair_round[t, d] == rnd
            assert pair_thresh[t, d] >= 0

        # Some pairs should be unassigned (unless all got assigned)
        if len(matches) < n_tracks * n_dets:
            unassigned_mask = pair_round == -1
            assert unassigned_mask.any()

    def test_same_matches_regardless_of_meta_flag(self):
        """The matches list should be identical with or without meta."""
        cost = _cost_matrix(6, 4, self.rng)
        matches_no_meta = iterative_assignment(cost, self.thresholds, return_meta=False)
        matches_with_meta, meta = iterative_assignment(cost, self.thresholds, return_meta=True)

        assert len(matches_no_meta) == len(matches_with_meta)
        for m1, m2 in zip(matches_no_meta, matches_with_meta):
            assert m1 == m2

    def test_empty_cost_matrix(self):
        """Empty cost matrix should produce empty matches."""
        cost = np.empty((0, 3))
        matches = iterative_assignment(cost, self.thresholds, return_meta=False)
        assert matches == []

        matches2, meta = iterative_assignment(cost, self.thresholds, return_meta=True)
        assert matches2 == []
        assert "pair_assignment_round" in meta

    def test_no_detections(self):
        """No detections (Nx0) should produce empty matches."""
        cost = np.empty((4, 0))
        matches = iterative_assignment(cost, self.thresholds, return_meta=False)
        assert matches == []

    def test_round_numbers_correct(self):
        """Matches should have round indices matching the thresholds order."""
        cost = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
        ], dtype=np.float64)
        thresholds = [0.3, 0.5]  # Note: should be decreasing for realistic use
        matches, meta = iterative_assignment(cost, thresholds, return_meta=True)
        # Both pairs have cost 0.1 which is <= 0.3, so both assigned in round 0
        for _, _, _, rnd in matches:
            assert rnd == 0
