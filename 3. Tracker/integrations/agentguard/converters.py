import copy
from typing import Any, Dict

import numpy as np

from agentguard.contracts.states import (
    TrackStateSnapshot,
    DetectionObservation,
    AssociationPairFeatures,
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
