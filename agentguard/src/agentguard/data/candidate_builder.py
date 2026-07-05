from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
)
from agentguard.features.scalar import compute_scalar_features


class CandidateBuilder:
    """Builds A/B/C candidates for each reliable-identity event.

    Candidates
    ----------
    **A** — TrackTrack's actual accepted detection (unchanged event).
    **B** — Wrong-identity hard candidate: the detection from the lowest
            ``final_cost`` association that belongs to a *different* GT ID.
    **C** — Same-identity low-quality candidate: priority order:
            1. NMS-deleted high-scoring detection,
            2. low-scoring detection,
            3. a different high-scoring detection with the same GT ID.

    All candidates recompute their detection, association, and scalar
    features independently so that the model sees a self-consistent input.
    """

    @staticmethod
    def build_candidates(
        event: TrackEvent,
        candidates_matrix: np.ndarray,
        gt_ids: List[int],
        target_gt_id: int,
        nms_deleted_idx: Optional[int],
        low_score_idx: Optional[int],
        high_score_idx: Optional[int],
    ) -> Dict[str, Optional[TrackEvent]]:
        """Build A/B/C candidate events.

        Parameters
        ----------
        event : TrackEvent
            The original reliable-identity event.  This is the **A** candidate.
        candidates_matrix : ndarray, shape ``(M, 3)``
            Row per detection candidate.  The three columns are:
            ``[detection_index, final_cost, gt_id]``.
        gt_ids : list of int
            All distinct GT track IDs appearing in *candidates_matrix*.
        target_gt_id : int
            The GT ID that the tracker *should* have used (the ground-truth
            identity at this frame).
        nms_deleted_idx : int or None
            Index into *candidates_matrix* for the NMS-deleted high-scoring
            detection (``-1`` if none).
        low_score_idx : int or None
            Index for a low-scoring detection with the target GT ID.
        high_score_idx : int or None
            Index for a different high-scoring detection with the target GT ID.

        Returns
        -------
        dict
            Keys ``'a'``, ``'b'``, ``'c'``, each mapping to a ``TrackEvent``
            or ``None`` if the candidate could not be built.
        """
        result: Dict[str, Optional[TrackEvent]] = {
            "a": None,
            "b": None,
            "c": None,
        }

        # --- Candidate A: the event itself (or a copy with recomputed features)
        if event.has_detection:
            result["a"] = CandidateBuilder._recompute_event(event, event.detection)

        # --- Candidate B: wrong-identity hard candidate
        result["b"] = CandidateBuilder._build_candidate_b(
            event, candidates_matrix, gt_ids, target_gt_id
        )

        # --- Candidate C: same-identity low-quality candidate
        result["c"] = CandidateBuilder._build_candidate_c(
            event,
            candidates_matrix,
            target_gt_id,
            nms_deleted_idx,
            low_score_idx,
            high_score_idx,
        )

        return result

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candidate_b(
        event: TrackEvent,
        candidates_matrix: np.ndarray,
        gt_ids: List[int],
        target_gt_id: int,
    ) -> Optional[TrackEvent]:
        """Build candidate B: lowest final_cost from another GT ID."""
        other_ids = [gid for gid in gt_ids if gid != target_gt_id]
        if not other_ids:
            return None

        best_row: Optional[np.ndarray] = None
        best_cost = float("inf")

        for row in candidates_matrix:
            # row = [det_idx, final_cost, gt_id]
            gt_id = int(row[2])
            if gt_id not in other_ids:
                continue
            cost = float(row[1])
            if cost < best_cost:
                best_cost = cost
                best_row = row

        if best_row is None:
            return None

        # Build a detection observation for this candidate
        det_idx = int(best_row[0])
        candidate_det = CandidateBuilder._make_detection(event, det_idx)
        if candidate_det is None:
            return None

        return CandidateBuilder._recompute_event(event, candidate_det)

    @staticmethod
    def _build_candidate_c(
        event: TrackEvent,
        candidates_matrix: np.ndarray,
        target_gt_id: int,
        nms_deleted_idx: Optional[int],
        low_score_idx: Optional[int],
        high_score_idx: Optional[int],
    ) -> Optional[TrackEvent]:
        """Build candidate C following the priority:

        1. NMS-deleted high-scoring detection.
        2. Low-scoring detection.
        3. A different high-scoring detection (same GT ID).
        """
        # Priority 1: NMS-deleted
        if nms_deleted_idx is not None and nms_deleted_idx >= 0:
            det_idx = int(candidates_matrix[nms_deleted_idx, 0])
            candidate_det = CandidateBuilder._make_detection(event, det_idx)
            if candidate_det is not None:
                return CandidateBuilder._recompute_event(event, candidate_det)

        # Priority 2: low-score
        if low_score_idx is not None and low_score_idx >= 0:
            det_idx = int(candidates_matrix[low_score_idx, 0])
            candidate_det = CandidateBuilder._make_detection(event, det_idx)
            if candidate_det is not None:
                return CandidateBuilder._recompute_event(event, candidate_det)

        # Priority 3: other high-score
        if high_score_idx is not None and high_score_idx >= 0:
            det_idx = int(candidates_matrix[high_score_idx, 0])
            candidate_det = CandidateBuilder._make_detection(event, det_idx)
            if candidate_det is not None:
                return CandidateBuilder._recompute_event(event, candidate_det)

        return None

    @staticmethod
    def _make_detection(
        event: TrackEvent, det_idx: int
    ) -> Optional[DetectionObservation]:
        """Create a ``DetectionObservation`` for a candidate detection index.

        The detection data (box, score, feature) must be recoverable from
        the event's state snapshots.  Currently we fall back to the event's
        own detection when a specific lookup mechanism is not wired in.
        In a full pipeline this would cross-reference a per-frame detection
        store; here we signal that a custom lookup callback is needed.

        Returns
        -------
        DetectionObservation or None
        """
        # In the full pipeline the detection should be fetched from a
        # pre-computed per-frame detection buffer indexed by `det_idx`.
        # For now we can only reuse the event's own detection.
        if event.detection is not None and event.detection.detection_index == det_idx:
            return event.detection
        return None

    @staticmethod
    def _recompute_event(
        event: TrackEvent,
        detection: Optional[DetectionObservation],
    ) -> TrackEvent:
        """Return a deep copy of *event* with *detection* plugged in,
        and all derived fields (association, scalar_features) recomputed.

        When *detection* is ``None``, has_detection is set to ``False``,
        and association / detection fields are cleared.
        """
        new_event = copy.deepcopy(event)

        if detection is not None:
            new_event.has_detection = True
            new_event.detection = detection
        else:
            new_event.has_detection = False
            new_event.detection = None
            new_event.association = None

        # Recompute scalar features from scratch so they reflect the new
        # detection / association state.
        new_event.scalar_features = compute_scalar_features(new_event)

        return new_event
