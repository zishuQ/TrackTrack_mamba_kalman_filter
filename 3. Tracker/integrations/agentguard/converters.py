import copy
from typing import Any, Dict, Optional

import numpy as np

from agentguard.contracts.states import (
    TrackStateSnapshot,
    DetectionObservation,
    AssociationPairFeatures,
    AssociationContext,
)
from agentguard.contracts.enums import DetectionSource


# ---------------------------------------------------------------------------
# Track ↔ TrackStateSnapshot
# ---------------------------------------------------------------------------


def export_track_state(track) -> TrackStateSnapshot:
    """Export a Track object's current state as a TrackStateSnapshot.

    Deep-copies all numpy arrays and the history dict.  Does **not** save the
    KalmanFilter object, ``args``, or any Tracker reference.
    """
    history_copy: Dict[int, Any] = {}
    for frame_id, entry in track.history.items():
        copied_entry: list[Any] = []
        for item in entry:
            if isinstance(item, np.ndarray):
                copied_entry.append(copy.deepcopy(item))
            else:
                copied_entry.append(item)
        history_copy[frame_id] = copied_entry

    return TrackStateSnapshot(
        track_id=track.track_id,
        box=track.box.copy(),
        score=track.score,
        mean=track.mean.copy() if track.mean is not None else None,
        covariance=track.covariance.copy() if track.covariance is not None else None,
        velocity=track.velocity.copy(),
        feature=track.feat.copy(),
        history=history_copy,
        end_frame_id=track.end_frame_id,
        state=track.state,
    )


def restore_track_state(track, snapshot: TrackStateSnapshot) -> None:
    """Restore a Track object's state from a TrackStateSnapshot.

    Copies everything back: ``track_id``, ``box``, ``score``, ``mean``,
    ``covariance``, ``velocity``, ``feat``, ``history``, ``end_frame_id``,
    and ``state``.  Does **not** restore the KalmanFilter (it should already
    exist on the Track object).
    """
    track.track_id = snapshot.track_id
    track.box = snapshot.box.copy()
    track.score = snapshot.score
    track.mean = snapshot.mean.copy() if snapshot.mean is not None else None
    track.covariance = (
        snapshot.covariance.copy() if snapshot.covariance is not None else None
    )
    track.velocity = snapshot.velocity.copy()
    track.feat = snapshot.feature.copy()

    # Rebuild history with copied arrays.
    history_copy: Dict[int, Any] = {}
    for frame_id, entry in snapshot.history.items():
        copied_entry: list[Any] = []
        for item in entry:
            if isinstance(item, np.ndarray):
                copied_entry.append(copy.deepcopy(item))
            else:
                copied_entry.append(item)
        history_copy[frame_id] = copied_entry
    track.history = history_copy

    track.end_frame_id = snapshot.end_frame_id
    track.state = snapshot.state


# ---------------------------------------------------------------------------
# Detection → DetectionObservation
# ---------------------------------------------------------------------------


def export_detection(
    detection,
    source: int,
    detection_index: int,
) -> DetectionObservation:
    """Export a Track object representing a detection as DetectionObservation.

    Parameters
    ----------
    detection : Track
        A Track instance initialised from a raw detection array whose layout is
        ``[x1, y1, x2, y2, score, class_id, feat1, feat2, ...]``.
    source : int
        A ``DetectionSource`` enum value indicating which detection tier the
        observation came from.
    detection_index : int
        Index of this detection within the current frame's detection list.
    """
    # The raw detection array from which the Track was constructed has the
    # class_id at index 5.  Try subscript access for compatibility with the
    # raw-array convention; fall back to 0 when the object is not subscriptable
    # (i.e. a pure Track instance without __getitem__).
    try:
        class_id = int(detection[5])
    except (TypeError, IndexError):
        class_id = 0

    return DetectionObservation(
        detection_index=detection_index,
        box=detection.box.copy(),
        score=detection.score,
        feature=detection.feat.copy(),
        source=source,
        class_id=class_id,
    )


# ---------------------------------------------------------------------------
# Association metadata → AssociationPairFeatures
# ---------------------------------------------------------------------------


def export_association_pair(
    meta: dict,
    track_index: int,
    detection_index: int,
) -> AssociationPairFeatures:
    """Export association metadata for a (track, detection) pair.

    The *meta* dict is expected to contain the following keys (each maps to a
    2-D numpy array whose shape is ``(num_tracks, num_detections)``):

    - ``iou_similarity``
    - ``iou_distance``
    - ``cosine_distance``
    - ``confidence_distance``
    - ``angle_distance``
    - ``raw_cost``
    - ``final_cost``
    - ``assignment_round``
    - ``assignment_threshold``
    - ``detection_source``
    """
    return AssociationPairFeatures(
        iou_similarity=float(meta["iou_similarity"][track_index, detection_index]),
        iou_distance=float(meta["iou_distance"][track_index, detection_index]),
        cosine_distance=float(meta["cosine_distance"][track_index, detection_index]),
        confidence_distance=float(
            meta["confidence_distance"][track_index, detection_index]
        ),
        angle_distance=float(meta["angle_distance"][track_index, detection_index]),
        raw_cost=float(meta["raw_cost"][track_index, detection_index]),
        final_cost=float(meta["final_cost"][track_index, detection_index]),
        assignment_round=int(meta["assignment_round"][track_index, detection_index]),
        assignment_threshold=float(
            meta["assignment_threshold"][track_index, detection_index]
        ),
        detection_source=int(meta["detection_source"][track_index, detection_index]),
    )


# ---------------------------------------------------------------------------
# DetectionObservation → TrackTrack update input helper
# ---------------------------------------------------------------------------


def detection_to_update_input(
    detection: DetectionObservation,
) -> dict:
    """Convert a ``DetectionObservation`` to a dict suitable for TrackTrack's
    ``update_with_gates`` interface (box/score/feature).

    Returns a dict with keys ``box``, ``score``, ``feat``.
    """
    return {
        "box": detection.box.copy(),
        "score": detection.score,
        "feat": np.asarray(detection.feature, dtype=np.float64).reshape(-1),
    }


# ---------------------------------------------------------------------------
# AssociationContext builder from raw association matrices
# ---------------------------------------------------------------------------


def build_association_context(
    meta: dict,
    track_index: int,
    detection_index: int,
    num_tracks: int,
    num_detections: int,
    no_reid: bool = False,
    detection_overlap_row: Optional[np.ndarray] = None,
) -> AssociationContext:
    """Build an ``AssociationContext`` from raw association matrices.

    Matrices must be *copied before iterative matching mutates them* —
    the caller must supply pre-mutation copies of ``final_cost`` and
    ``cosine_distance`` matrices.

    Parameters
    ----------
    meta : dict
        Must contain keys ``final_cost`` (copied pre-mutation), ``cosine_distance``
        (copied pre-mutation or zeros if no reid), and optionally
        ``detection_source`` or ``iou_similarity``.
    track_index : int
        Index of the accepted track in the matrix.
    detection_index : int
        Index of the accepted detection.
    num_tracks : int
        Total number of tracks in the association.
    num_detections : int
        Total number of detections in the association.
    no_reid : bool
        If True, cosine_distance_matrix is set to zeros.
    detection_overlap_row : np.ndarray, optional
        IoU of the accepted detection against every detection in the current
        detection pool.  This is a detection-to-detection row and must not be
        derived from the track-to-detection IoU matrix.

    Returns
    -------
    AssociationContext
    """
    final_cost_matrix = np.asarray(meta["final_cost"], dtype=np.float64)
    expected_shape = (num_tracks, num_detections)
    if final_cost_matrix.shape != expected_shape:
        raise ValueError(
            "final_cost shape must match association dimensions: "
            f"{final_cost_matrix.shape} != {expected_shape}"
        )
    if not (0 <= track_index < num_tracks):
        raise IndexError(f"track_index {track_index} outside [0, {num_tracks})")
    if not (0 <= detection_index < num_detections):
        raise IndexError(
            f"detection_index {detection_index} outside [0, {num_detections})"
        )
    track_cost_row = final_cost_matrix[track_index, :].copy()
    detection_cost_col = final_cost_matrix[:, detection_index].copy()

    if no_reid:
        cos_matrix = np.zeros_like(final_cost_matrix, dtype=np.float64)
        reid_available = False
    else:
        cos_matrix = np.asarray(meta.get("cosine_distance", meta.get("cosine_distance_matrix",
            np.zeros_like(final_cost_matrix))), dtype=np.float64)
        reid_available = True

    raw_cost_row = np.asarray(meta.get("raw_cost", final_cost_matrix), dtype=np.float64)[track_index, :].copy()

    if detection_overlap_row is None:
        overlap_row = np.zeros(num_detections, dtype=np.float64)
    else:
        overlap_row = np.asarray(detection_overlap_row, dtype=np.float64).reshape(-1).copy()
        if overlap_row.shape != (num_detections,):
            raise ValueError(
                "detection_overlap_row must have one value per detection: "
                f"expected {(num_detections,)}, got {overlap_row.shape}"
            )

    return AssociationContext(
        track_cost_row=track_cost_row,
        detection_cost_col=detection_cost_col,
        detection_overlap_row=overlap_row,
        accepted_detection_index=detection_index,
        num_tracks=num_tracks,
        num_detections=num_detections,
        reid_available=reid_available,
    )
