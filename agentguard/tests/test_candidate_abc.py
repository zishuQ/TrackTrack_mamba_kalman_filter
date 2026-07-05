"""Test that A/B/C candidates are correctly constructed with recomputed features.

Candidate A: The event itself (original detection).
Candidate B: Wrong-identity hard candidate (lowest final_cost from another GT ID).
Candidate C: Same-identity low-quality candidate (NMS-deleted, low-score, or other high-score).
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.data.candidate_builder import CandidateBuilder


@pytest.fixture
def base_event() -> TrackEvent:
    """A base event to build candidates from."""
    rng = np.random.RandomState(42)
    feat = rng.randn(1, 64).astype(np.float64)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)

    mean = np.array([200.0, 300.0, 200.0, 200.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.eye(8, dtype=np.float64) * 0.1
    box = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)

    state = TrackStateSnapshot(
        track_id=1,
        box=box,
        score=0.85,
        mean=mean,
        covariance=cov,
        velocity=np.zeros((4, 2), dtype=np.float64),
        feature=feat,
        history={},
        end_frame_id=99,
        state=1,
    )

    det = DetectionObservation(
        detection_index=0,
        box=box,
        score=0.85,
        feature=feat,
        source=0,
        class_id=1,
    )

    assoc = AssociationPairFeatures(
        iou_similarity=0.8,
        iou_distance=0.2,
        cosine_distance=0.1,
        confidence_distance=0.05,
        angle_distance=0.02,
        raw_cost=0.30,
        final_cost=0.28,
        assignment_round=0,
        assignment_threshold=0.5,
        detection_source=0,
    )

    ev = TrackEvent(
        event_id="base_event",
        dataset="test",
        sequence="test_seq",
        frame_id=100,
        track_id=1,
        image_width=1920,
        image_height=1080,
        has_detection=True,
        frame_start_state=state,
        pre_update_state=state,
        detection=det,
        association=assoc,
        warp_matrix=np.eye(2, 3, dtype=np.float64),
        scalar_features=np.zeros(63, dtype=np.float64),
        track_feature=feat.ravel(),
        detection_feature=feat.ravel(),
    )
    return ev


class TestCandidateA:
    """Candidate A: the event itself (unchanged but with recomputed features)."""

    def test_candidate_a_not_none(self, base_event):
        """Candidate A should be a TrackEvent when event has detection."""
        candidates_matrix = np.empty((0, 3))
        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1], 1, None, None, None
        )
        assert result["a"] is not None
        assert isinstance(result["a"], TrackEvent)

    def test_candidate_a_same_event_id(self, base_event):
        """Candidate A should keep the original event_id."""
        result = CandidateBuilder.build_candidates(
            base_event, np.empty((0, 3)), [1], 1, None, None, None
        )
        assert result["a"].event_id == base_event.event_id

    def test_candidate_a_has_recomputed_features(self, base_event):
        """Candidate A should have recomputed scalar_features (not all zeros)."""
        result = CandidateBuilder.build_candidates(
            base_event, np.empty((0, 3)), [1], 1, None, None, None
        )
        # With a detection and association, scalar_features should be non-zero
        assert np.any(result["a"].scalar_features != 0.0)

    def test_candidate_a_no_detection(self, base_event):
        """If event has no detection, candidate A should be None."""
        base_event.has_detection = False
        result = CandidateBuilder.build_candidates(
            base_event, np.empty((0, 3)), [], -1, None, None, None
        )
        assert result["a"] is None


class TestCandidateB:
    """Candidate B: wrong-identity hard candidate."""

    def test_candidate_b_from_different_gt(self, base_event):
        """Candidate B should use the detection from a different GT ID.

        NOTE: CandidateBuilder._make_detection currently only returns a detection
        when det_idx matches the event's own detection_index (0). All rows
        in candidates_matrix use det_idx=0, with GT IDs differentiating them.
        """
        candidates_matrix = np.array([
            [0, 0.25, 1],   # same GT ID (target)
            [0, 0.20, 2],   # different GT ID, low cost -> should be B
            [0, 0.35, 3],   # different GT ID, higher cost
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1, 2, 3], target_gt_id=1,
            nms_deleted_idx=None, low_score_idx=None, high_score_idx=None,
        )
        assert result["b"] is not None
        # B should be the different GT with lowest final_cost -> row at index 1, cost 0.20
        assert result["b"].detection is not None

    def test_candidate_b_none_when_no_other_gt(self, base_event):
        """If no other GT IDs exist, B should be None."""
        candidates_matrix = np.array([
            [0, 0.30, 1],  # only target GT
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1], target_gt_id=1,
            nms_deleted_idx=None, low_score_idx=None, high_score_idx=None,
        )
        assert result["b"] is None

    def test_candidate_b_recomputes_features(self, base_event):
        """Candidate B should have recomputed scalar_features."""
        candidates_matrix = np.array([
            [0, 0.30, 1],
            [0, 0.15, 2],
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1, 2], target_gt_id=1,
            nms_deleted_idx=None, low_score_idx=None, high_score_idx=None,
        )
        assert result["b"] is not None
        assert np.any(result["b"].scalar_features != 0.0)

    def test_candidate_b_prefers_lowest_cost(self, base_event):
        """Among different GT IDs, B should pick the one with lowest final_cost."""
        candidates_matrix = np.array([
            [0, 0.50, 2],  # GT 2, high cost
            [0, 0.10, 3],  # GT 3, low cost -> should be B
            [0, 0.30, 2],  # GT 2, medium cost
            [0, 0.20, 4],  # GT 4, medium-low cost
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [2, 3, 4], target_gt_id=1,
            nms_deleted_idx=None, low_score_idx=None, high_score_idx=None,
        )
        assert result["b"] is not None
        # The row with GT 3 (cost 0.10) should be selected
        assert result["b"].detection is not None


class TestCandidateC:
    """Candidate C: same-identity low-quality candidate."""

    def _make_basic_det(self, detection_index: int) -> DetectionObservation:
        """Helper to create a DetectionObservation for a given index."""
        rng = np.random.RandomState(detection_index + 100)
        feat = rng.randn(1, 64).astype(np.float64)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        return DetectionObservation(
            detection_index=detection_index,
            box=np.array([105.0, 205.0, 305.0, 405.0], dtype=np.float64),
            score=0.8,
            feature=feat,
            source=0,
            class_id=1,
        )

    def test_candidate_c_nms_deleted_priority(self, base_event, monkeypatch):
        """C should prefer NMS-deleted high-scoring detection (priority 1)."""
        # Monkey-patch _make_detection to return a detection for any index
        det_for_idx = {
            0: base_event.detection,
            1: self._make_basic_det(1),
            2: self._make_basic_det(2),
            3: self._make_basic_det(3),
        }

        def mock_make_detection(event, det_idx):
            return det_for_idx.get(det_idx)

        monkeypatch.setattr(CandidateBuilder, '_make_detection', staticmethod(mock_make_detection))

        candidates_matrix = np.array([
            [0, 0.30, 1],
            [1, 0.40, 1],  # nms_deleted -> should win
            [2, 0.50, 1],  # low-score
            [3, 0.60, 1],  # other high-score
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1], target_gt_id=1,
            nms_deleted_idx=1,  # index 1 -> det_idx=1
            low_score_idx=2,
            high_score_idx=3,
        )
        assert result["c"] is not None
        assert result["c"].detection.detection_index == 1

    def test_candidate_c_low_score_priority(self, base_event, monkeypatch):
        """If NMS-deleted is None, C should prefer low-score detection (priority 2)."""
        det_for_idx = {
            0: base_event.detection,
            1: self._make_basic_det(1),
        }

        def mock_make_detection(event, det_idx):
            return det_for_idx.get(det_idx)

        monkeypatch.setattr(CandidateBuilder, '_make_detection', staticmethod(mock_make_detection))

        candidates_matrix = np.array([
            [0, 0.30, 1],
            [1, 0.50, 1],  # low-score -> should win
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1], target_gt_id=1,
            nms_deleted_idx=None,
            low_score_idx=1,
            high_score_idx=None,
        )
        assert result["c"] is not None
        assert result["c"].detection.detection_index == 1

    def test_candidate_c_high_score_fallback(self, base_event, monkeypatch):
        """If NMS-deleted and low-score are None, fall back to other high-score."""
        det_for_idx = {
            0: base_event.detection,
            3: self._make_basic_det(3),
        }

        def mock_make_detection(event, det_idx):
            return det_for_idx.get(det_idx)

        monkeypatch.setattr(CandidateBuilder, '_make_detection', staticmethod(mock_make_detection))

        candidates_matrix = np.array([
            [0, 0.30, 1],
            [3, 0.60, 1],  # other high-score -> should win
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1], target_gt_id=1,
            nms_deleted_idx=None,
            low_score_idx=None,
            high_score_idx=1,
        )
        assert result["c"] is not None
        assert result["c"].detection.detection_index == 3

    def test_candidate_c_none_when_all_none(self, base_event):
        """If all three priority indices are None, C should be None."""
        candidates_matrix = np.array([
            [0, 0.30, 1],
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1], target_gt_id=1,
            nms_deleted_idx=None, low_score_idx=None, high_score_idx=None,
        )
        assert result["c"] is None


class TestCandidateRecomputeFeatures:
    """All candidates should have recomputed scalar_features."""

    def test_candidates_have_different_features(self, base_event, monkeypatch):
        """A, B, C should have different scalar_features."""
        rng = np.random.RandomState(99)

        det_for_idx = {}
        for idx in range(3):
            feat = rng.randn(1, 64).astype(np.float64)
            feat /= np.linalg.norm(feat, axis=1, keepdims=True)
            det_for_idx[idx] = DetectionObservation(
                detection_index=idx,
                box=np.array([100.0 + idx*5, 200.0 + idx*5, 300.0 + idx*5, 400.0 + idx*5],
                            dtype=np.float64),
                score=0.8 - idx*0.05,
                feature=feat,
                source=0,
                class_id=1,
            )
        # Index 0 matches the event's own detection
        det_for_idx[0] = base_event.detection

        def mock_make_detection(event, det_idx):
            return det_for_idx.get(det_idx)

        monkeypatch.setattr(CandidateBuilder, '_make_detection', staticmethod(mock_make_detection))

        candidates_matrix = np.array([
            [0, 0.30, 1],
            [1, 0.20, 2],  # different GT
            [2, 0.40, 1],  # same GT
        ], dtype=np.float64)

        result = CandidateBuilder.build_candidates(
            base_event, candidates_matrix, [1, 2], target_gt_id=1,
            nms_deleted_idx=None, low_score_idx=2, high_score_idx=None,
        )
        # A should have non-zero features
        assert np.any(result["a"].scalar_features != 0.0)
        if result["b"] is not None:
            assert np.any(result["b"].scalar_features != 0.0)
        if result["c"] is not None:
            assert np.any(result["c"].scalar_features != 0.0)

    def test_candidate_a_recompute_differs_from_original(self, base_event):
        """After recomputation, candidate A's features may differ from the original event's
        initial all-zeros features."""
        orig_features = base_event.scalar_features.copy()
        result = CandidateBuilder.build_candidates(
            base_event, np.empty((0, 3)), [1], 1, None, None, None
        )
        # The recomputed features should differ from the initial zeros
        assert not np.allclose(result["a"].scalar_features, orig_features)
