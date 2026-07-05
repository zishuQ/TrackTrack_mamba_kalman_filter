from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _to_list(value: Any) -> Any:
    """Recursively convert numpy arrays to nested Python lists."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_list(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_list(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_list(v) for v in value)
    return value


def _to_array(value: Any) -> Any:
    """Recursively convert nested lists back to numpy arrays (leaf lists only)."""
    if isinstance(value, dict):
        return {k: _to_array(v) for k, v in value.items()}
    if isinstance(value, list):
        # If all elements are numeric (or nested lists that eventually become numeric)
        # attempt conversion; otherwise recurse.
        if len(value) == 0:
            return np.array(value)
        # Try converting nested lists; if it fails, recurse on each element.
        try:
            arr = np.asarray(value)
            if arr.dtype.kind in ("f", "i", "u", "b"):
                return arr
            # Object array — likely a mixed list; recurse
            return [_to_array(v) for v in value]
        except (ValueError, TypeError):
            return [_to_array(v) for v in value]
    return value


# ---------------------------------------------------------------------------
#  Serialization
# ---------------------------------------------------------------------------

def _snapshot_to_dict(snapshot: TrackStateSnapshot) -> dict:
    return {
        "track_id": snapshot.track_id,
        "box": _to_list(snapshot.box),
        "score": snapshot.score,
        "mean": _to_list(snapshot.mean) if snapshot.mean is not None else None,
        "covariance": _to_list(snapshot.covariance) if snapshot.covariance is not None else None,
        "velocity": _to_list(snapshot.velocity),
        "feature": _to_list(snapshot.feature),
        "history": _to_list(snapshot.history),
        "end_frame_id": snapshot.end_frame_id,
        "state": snapshot.state,
    }


def _detection_to_dict(detection: DetectionObservation) -> dict:
    return {
        "detection_index": detection.detection_index,
        "box": _to_list(detection.box),
        "score": detection.score,
        "feature": _to_list(detection.feature),
        "source": detection.source,
        "class_id": detection.class_id,
    }


def _association_to_dict(assoc: AssociationPairFeatures) -> dict:
    return {
        "iou_similarity": assoc.iou_similarity,
        "iou_distance": assoc.iou_distance,
        "cosine_distance": assoc.cosine_distance,
        "confidence_distance": assoc.confidence_distance,
        "angle_distance": assoc.angle_distance,
        "raw_cost": assoc.raw_cost,
        "final_cost": assoc.final_cost,
        "assignment_round": assoc.assignment_round,
        "assignment_threshold": assoc.assignment_threshold,
        "detection_source": assoc.detection_source,
    }


def serialize_event(event: TrackEvent) -> dict:
    """Convert a ``TrackEvent`` to a JSON-serialisable dictionary.

    All numpy arrays are converted to nested Python lists.
    ``None`` fields remain ``None``.
    """
    out: Dict[str, Any] = {
        "event_id": event.event_id,
        "dataset": event.dataset,
        "sequence": event.sequence,
        "frame_id": event.frame_id,
        "track_id": event.track_id,
        "image_width": event.image_width,
        "image_height": event.image_height,
        "has_detection": event.has_detection,
        "frame_start_state": (
            _snapshot_to_dict(event.frame_start_state)
            if event.frame_start_state is not None
            else None
        ),
        "pre_update_state": (
            _snapshot_to_dict(event.pre_update_state)
            if event.pre_update_state is not None
            else None
        ),
        "detection": (
            _detection_to_dict(event.detection)
            if event.detection is not None
            else None
        ),
        "association": (
            _association_to_dict(event.association)
            if event.association is not None
            else None
        ),
        "warp_matrix": _to_list(event.warp_matrix),
        "scalar_features": _to_list(event.scalar_features),
        "track_feature": _to_list(event.track_feature),
        "detection_feature": _to_list(event.detection_feature),
        "iwg_policy_probs": (
            _to_list(event.iwg_policy_probs)
            if event.iwg_policy_probs is not None
            else None
        ),
        "iwg_gate": (
            _to_list(event.iwg_gate) if event.iwg_gate is not None else None
        ),
        "revised_gate": (
            _to_list(event.revised_gate) if event.revised_gate is not None else None
        ),
    }
    return out


# ---------------------------------------------------------------------------
#  Deserialization
# ---------------------------------------------------------------------------

def _dict_to_snapshot(data: dict) -> TrackStateSnapshot:
    return TrackStateSnapshot(
        track_id=data["track_id"],
        box=np.asarray(data["box"]),
        score=data["score"],
        mean=np.asarray(data["mean"]) if data.get("mean") is not None else None,
        covariance=(
            np.asarray(data["covariance"]) if data.get("covariance") is not None else None
        ),
        velocity=np.asarray(data["velocity"]),
        feature=np.asarray(data["feature"]),
        history=_to_array(data["history"]),
        end_frame_id=data["end_frame_id"],
        state=data["state"],
    )


def _dict_to_detection(data: dict) -> DetectionObservation:
    return DetectionObservation(
        detection_index=data["detection_index"],
        box=np.asarray(data["box"]),
        score=data["score"],
        feature=np.asarray(data["feature"]),
        source=data["source"],
        class_id=data["class_id"],
    )


def _dict_to_association(data: dict) -> AssociationPairFeatures:
    return AssociationPairFeatures(
        iou_similarity=data["iou_similarity"],
        iou_distance=data["iou_distance"],
        cosine_distance=data["cosine_distance"],
        confidence_distance=data["confidence_distance"],
        angle_distance=data["angle_distance"],
        raw_cost=data["raw_cost"],
        final_cost=data["final_cost"],
        assignment_round=data["assignment_round"],
        assignment_threshold=data["assignment_threshold"],
        detection_source=data["detection_source"],
    )


def deserialize_event(data: dict) -> TrackEvent:
    """Reconstruct a ``TrackEvent`` from a dictionary produced by ``serialize_event``."""
    event = TrackEvent(
        event_id=data["event_id"],
        dataset=data["dataset"],
        sequence=data["sequence"],
        frame_id=data["frame_id"],
        track_id=data["track_id"],
        image_width=data["image_width"],
        image_height=data["image_height"],
        has_detection=data["has_detection"],
        frame_start_state=(
            _dict_to_snapshot(data["frame_start_state"])
            if data.get("frame_start_state") is not None
            else None
        ),
        pre_update_state=(
            _dict_to_snapshot(data["pre_update_state"])
            if data.get("pre_update_state") is not None
            else None
        ),
        detection=(
            _dict_to_detection(data["detection"])
            if data.get("detection") is not None
            else None
        ),
        association=(
            _dict_to_association(data["association"])
            if data.get("association") is not None
            else None
        ),
        warp_matrix=np.asarray(data["warp_matrix"], dtype=np.float64),
        scalar_features=np.asarray(data["scalar_features"], dtype=np.float64),
        track_feature=np.asarray(data["track_feature"], dtype=np.float64),
        detection_feature=np.asarray(data["detection_feature"], dtype=np.float64),
        iwg_policy_probs=(
            np.asarray(data["iwg_policy_probs"])
            if data.get("iwg_policy_probs") is not None
            else None
        ),
        iwg_gate=(
            np.asarray(data["iwg_gate"])
            if data.get("iwg_gate") is not None
            else None
        ),
        revised_gate=(
            np.asarray(data["revised_gate"])
            if data.get("revised_gate") is not None
            else None
        ),
    )
    return event


# ---------------------------------------------------------------------------
#  Batch helpers
# ---------------------------------------------------------------------------

def serialize_events(events: List[TrackEvent]) -> List[dict]:
    """Serialize a list of ``TrackEvent`` s."""
    return [serialize_event(ev) for ev in events]


def deserialize_events(data_list: List[dict]) -> List[TrackEvent]:
    """Deserialize a list of dictionaries back to ``TrackEvent`` s."""
    return [deserialize_event(d) for d in data_list]
