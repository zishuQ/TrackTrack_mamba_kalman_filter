"""Read-only compatibility boundary with the legacy AgentGuard data layer.

This is the only SHUTrack module allowed to import AgentGuard.  It reuses
stable, read-only pieces:

* ``agentguard.features.scalar.compute_scalar_features`` (legacy 63-dim
  definition, including the packed training tensors already normalized with
  the same convention);
* ``agentguard.data.cache_reader.CompactEventCacheReader`` and compact label
  loaders when a source event cache is actually present;
* legacy contract dataclasses only where they are the stable interchange
  format.

No legacy model, trainer, runtime, cache builder or index builder is imported
or called from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from shutrack.data.evidence import EventEvidence
from shutrack.features.core import (
    build_core_features,
    normalize_scalar63,
    sanitize_float32,
)
from shutrack.features.margins import (
    INVALID_MARGIN,
    MarginResult,
    compute_margins_from_association_record,
    compute_selected_margins,
)


def _legacy_scalar_module():
    try:
        from agentguard.features.scalar import compute_scalar_features
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "AgentGuard's read-only scalar feature definition is unavailable; "
            "SHUTrack online inference requires the installed agentguard package"
        ) from exc
    return compute_scalar_features


def legacy_shutrack_scalar63(params: Mapping[str, Any]) -> np.ndarray:
    """Compute the legacy 63-dim scalar vector from a plain parameter dict."""

    compute_scalar_features = _legacy_scalar_module()
    return np.asarray(compute_scalar_features(dict(params)), dtype=np.float64)


def scalar_params_from_state(
    *,
    frame_id: int,
    track_id: int,
    image_width: int,
    image_height: int,
    has_detection: bool,
    pre_update_state: Any,
    detection: Any = None,
    association: Any = None,
    association_context: Any = None,
) -> dict[str, Any]:
    """Build the flat/nested dict accepted by the legacy scalar function.

    The inputs are the same objects the online tracker holds immediately
    before ``update_with_gates``: the pre-update snapshot, the accepted
    detection and the association pair/context of the current frame.
    """

    params: dict[str, Any] = {
        "has_detection": bool(has_detection),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "frame_id": int(frame_id),
    }
    if association is not None:
        params["association"] = {
            "iou_similarity": float(association.iou_similarity),
            "iou_distance": float(association.iou_distance),
            "cosine_distance": float(association.cosine_distance),
            "confidence_distance": float(association.confidence_distance),
            "angle_distance": float(association.angle_distance),
            "raw_cost": float(association.raw_cost),
            "final_cost": float(association.final_cost),
            "assignment_round": int(association.assignment_round),
            "assignment_threshold": float(association.assignment_threshold),
            "detection_source": int(association.detection_source),
        }
    if detection is not None and has_detection:
        params["detection"] = {
            "box": np.asarray(detection.box, dtype=np.float64),
            "score": float(detection.score),
            "source": int(detection.source),
        }
    if pre_update_state is not None:
        history = getattr(pre_update_state, "history", {}) or {}
        params["pre_update_state"] = {
            "mean": (
                None
                if getattr(pre_update_state, "mean", None) is None
                else np.asarray(pre_update_state.mean, dtype=np.float64)
            ),
            "covariance": (
                None
                if getattr(pre_update_state, "covariance", None) is None
                else np.asarray(pre_update_state.covariance, dtype=np.float64)
            ),
            "velocity": np.asarray(
                getattr(pre_update_state, "velocity", np.zeros((4, 2))),
                dtype=np.float64,
            ),
            "score": float(getattr(pre_update_state, "score", 0.0)),
            "history": history,
            "end_frame_id": int(getattr(pre_update_state, "end_frame_id", 0)),
            "state": int(getattr(pre_update_state, "state", 0)),
        }
    if association_context is not None:
        params["track_cost_row"] = np.asarray(
            association_context.track_cost_row, dtype=np.float64
        )
        params["detection_cost_col"] = np.asarray(
            association_context.detection_cost_col, dtype=np.float64
        )
        overlap = np.asarray(association_context.detection_overlap_row, dtype=np.float64)
        params["max_iou_with_other"] = (
            float(np.max(overlap)) if overlap.size > 0 else 0.0
        )
    return params


def normalizer_from_metadata(metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(metadata.get("normalization_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(metadata.get("normalization_std", []), dtype=np.float64).reshape(-1)
    if mean.shape != (63,) or std.shape != (63,):
        raise ValueError(
            "SHUTrack requires the legacy 63-dim normalization statistics; got "
            f"mean={mean.shape}, std={std.shape}"
        )
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return mean, std


def build_online_evidence(
    *,
    scalar_params: Mapping[str, Any],
    track_feature: np.ndarray,
    detection_feature: np.ndarray,
    has_detection: bool,
    margins: MarginResult,
    margin_stats: Mapping[str, Any] | None,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    frame_id: int,
    track_id: int,
    segment_id: int,
    observation_count: int,
) -> EventEvidence:
    """Normalize raw online scalars and assemble one evidence record."""

    raw_scalar = legacy_shutrack_scalar63(scalar_params)
    normalized = normalize_scalar63(raw_scalar, scalar_mean, scalar_std)
    motion, context = build_core_features(normalized, margins, margin_stats)
    track = sanitize_float32(track_feature)
    detection = (
        sanitize_float32(detection_feature)
        if has_detection
        else np.zeros_like(track)
    )
    return EventEvidence(
        motion=sanitize_float32(motion),
        context=sanitize_float32(context),
        track_feature=track,
        detection_feature=detection,
        has_detection=has_detection,
        frame_id=frame_id,
        track_id=track_id,
        segment_id=segment_id,
        observation_count=observation_count,
    )


def open_compact_event_reader(
    cache_dir: str | Path,
    detection_cache_dir: str | Path | None = None,
    *,
    max_cached_shards: int = 2,
):
    """Open the legacy compact event reader with a bounded shard cache."""

    try:
        from agentguard.data.cache_reader import CompactEventCacheReader
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "CompactEventCacheReader is unavailable; the source event cache "
            "path requires the installed agentguard package"
        ) from exc
    return CompactEventCacheReader(
        cache_dir,
        detection_cache_dir=detection_cache_dir,
        max_cached_shards=int(max_cached_shards),
    )


def load_compact_labels(label_dir: str | Path):
    """Read-only compact label arrays (never rebuilds labels)."""

    try:
        from agentguard.data.compact_iwg_labels import load_compact_label_arrays
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Compact label loading is unavailable; the source cache path "
            "requires the installed agentguard package"
        ) from exc
    return load_compact_label_arrays(label_dir)


def margins_from_cached_event(reader: Any, record: Mapping[str, Any]) -> MarginResult:
    """Margins for one cached event, using only the recorded candidate pool."""

    if not bool(record.get("matched", False)):
        return INVALID_MARGIN
    association = reader.get_association(
        int(record.get("association_shard_id", -1)),
        int(record.get("association_offset", -1)),
    )
    if association is None:
        return INVALID_MARGIN
    return compute_margins_from_association_record(
        association,
        track_row=int(record.get("association_track_row", -1)),
        accepted_detection_index=int(record.get("accepted_detection_index", -1)),
    )


def evidence_from_cached_event(
    reader: Any,
    record: Mapping[str, Any],
    *,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    margin_stats: Mapping[str, Any] | None,
    segment_id: int,
) -> EventEvidence:
    """Build evidence from one cached event without writing anything.

    This is the read-only source-cache path used when compact event shards and
    labels are present.  The packed fast path lives in
    ``shutrack.data.packed_dataset``.
    """

    raw_scalar = np.asarray(record.get("scalar_features", []), dtype=np.float64).reshape(-1)
    if raw_scalar.shape != (63,):
        raise ValueError(
            f"cached event {record.get('event_id')} has no valid 63-dim scalar"
        )
    normalized = normalize_scalar63(raw_scalar, scalar_mean, scalar_std)
    margins = margins_from_cached_event(reader, record)
    motion, context = build_core_features(normalized, margins, margin_stats)

    track_feature = np.asarray(record.get("track_feature", []), dtype=np.float32).reshape(-1)
    matched = bool(record.get("matched", False))
    accepted = int(record.get("accepted_detection_index", -1))
    if matched and accepted >= 0:
        detection_feature = np.asarray(
            reader.get_detection(accepted)["feature"], dtype=np.float32
        ).reshape(-1)
    else:
        detection_feature = np.zeros_like(track_feature)

    return EventEvidence(
        motion=sanitize_float32(motion),
        context=sanitize_float32(context),
        track_feature=sanitize_float32(track_feature),
        detection_feature=sanitize_float32(detection_feature),
        has_detection=matched,
        frame_id=int(record.get("frame_id", -1)),
        track_id=int(record.get("track_id", -1)),
        segment_id=int(segment_id),
        observation_count=int(record.get("history_count", 0)),
    )


def iter_cached_margin_observations(
    reader: Any,
    *,
    limit: int | None = None,
) -> Iterator[MarginResult]:
    """Stream margin observations from a source event cache (bounded reader)."""

    yielded = 0
    for record in reader.iter_event_records():
        if not bool(record.get("matched", False)):
            continue
        yield margins_from_cached_event(reader, record)
        yielded += 1
        if limit is not None and yielded >= int(limit):
            return


def online_margins_from_association_meta(
    association_meta: Mapping[str, Any] | None,
    *,
    track_row: int,
    detection_col: int,
) -> MarginResult:
    """Margins from the live association metadata of the current frame."""

    if association_meta is None:
        return INVALID_MARGIN
    raw_cost = association_meta.get("raw_cost")
    iou_similarity = association_meta.get("iou_similarity")
    if raw_cost is None or iou_similarity is None:
        return INVALID_MARGIN
    try:
        return compute_selected_margins(
            np.asarray(raw_cost),
            np.asarray(iou_similarity),
            int(track_row),
            int(detection_col),
        )
    except ValueError:
        return INVALID_MARGIN


__all__ = [
    "legacy_shutrack_scalar63",
    "scalar_params_from_state",
    "normalizer_from_metadata",
    "build_online_evidence",
    "open_compact_event_reader",
    "load_compact_labels",
    "margins_from_cached_event",
    "evidence_from_cached_event",
    "iter_cached_margin_observations",
    "online_margins_from_association_meta",
]
