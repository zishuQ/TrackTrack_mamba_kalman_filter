"""Read-only compatibility boundary with the legacy AgentGuard data layer.

This is the only SHUTrack module allowed to import AgentGuard.  It reuses
stable, read-only pieces:

* ``agentguard.features.scalar.compute_scalar_features`` (legacy 63-dim
  definition, including the packed training tensors already normalized with
  the same convention);
* ``agentguard.features.geometry.pairwise_iou_xyxy`` for the competition
  token's max-IoU column;
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
    build_competition,
    build_evidence,
    normalize_scalar63,
    sanitize_float32,
)
from shutrack.schema import COMPETITION_DIM


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


def detection_overlap_row(
    selected_box: np.ndarray,
    pool_boxes: np.ndarray,
    selected_column: int,
) -> np.ndarray:
    """Legacy max-IoU row: IoU of the accepted detection vs the pool.

    The accepted detection's own entry is set to zero, matching
    ``CompactEventCacheReader._association_detection_overlap_row``.
    """

    boxes = np.asarray(pool_boxes, dtype=np.float64).reshape(-1, 4)
    selected = np.asarray(selected_box, dtype=np.float64).reshape(1, 4)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    try:
        from agentguard.features.geometry import pairwise_iou_xyxy
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "AgentGuard pairwise IoU is unavailable; the competition token "
            "requires the installed agentguard package"
        ) from exc
    overlap = np.asarray(pairwise_iou_xyxy(selected, boxes)[0], dtype=np.float64)
    column = int(selected_column)
    if 0 <= column < overlap.shape[0]:
        overlap[column] = 0.0
    return overlap


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
    competition_enabled: bool,
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
    evidence = build_evidence(normalized, has_detection)
    competition = (
        build_competition(normalized)
        if competition_enabled
        else np.zeros(COMPETITION_DIM, dtype=np.float32)
    )
    track = sanitize_float32(track_feature)
    detection = (
        sanitize_float32(detection_feature)
        if has_detection
        else np.zeros_like(track)
    )
    return EventEvidence(
        evidence=sanitize_float32(evidence),
        competition=sanitize_float32(competition),
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


def evidence_from_cached_event(
    reader: Any,
    record: Mapping[str, Any],
    *,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    competition_enabled: bool,
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
    matched = bool(record.get("matched", False))
    evidence = build_evidence(normalized, matched)
    competition = (
        build_competition(normalized)
        if competition_enabled
        else np.zeros(COMPETITION_DIM, dtype=np.float32)
    )

    track_feature = np.asarray(record.get("track_feature", []), dtype=np.float32).reshape(-1)
    accepted = int(record.get("accepted_detection_index", -1))
    if matched and accepted >= 0:
        detection_feature = np.asarray(
            reader.get_detection(accepted)["feature"], dtype=np.float32
        ).reshape(-1)
    else:
        detection_feature = np.zeros_like(track_feature)

    return EventEvidence(
        evidence=sanitize_float32(evidence),
        competition=sanitize_float32(competition),
        track_feature=sanitize_float32(track_feature),
        detection_feature=sanitize_float32(detection_feature),
        has_detection=matched,
        frame_id=int(record.get("frame_id", -1)),
        track_id=int(record.get("track_id", -1)),
        segment_id=int(segment_id),
        observation_count=int(record.get("history_count", 0)),
    )


def iter_cached_evidence(
    reader: Any,
    *,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    competition_enabled: bool,
    segment_id: int = 0,
    limit: int | None = None,
) -> Iterator[EventEvidence]:
    """Stream cached event evidence (bounded reader), for diagnostics/tests."""

    yielded = 0
    for record in reader.iter_event_records():
        yield evidence_from_cached_event(
            reader,
            record,
            scalar_mean=scalar_mean,
            scalar_std=scalar_std,
            competition_enabled=competition_enabled,
            segment_id=segment_id,
        )
        yielded += 1
        if limit is not None and yielded >= int(limit):
            return


__all__ = [
    "legacy_shutrack_scalar63",
    "detection_overlap_row",
    "normalizer_from_metadata",
    "build_online_evidence",
    "open_compact_event_reader",
    "load_compact_labels",
    "evidence_from_cached_event",
    "iter_cached_evidence",
]
