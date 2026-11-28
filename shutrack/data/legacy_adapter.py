"""Online SHUTrack features built from the shared AgentGuard scalar definitions."""

from __future__ import annotations

from typing import Any, Mapping

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


__all__ = [
    "legacy_shutrack_scalar63",
    "detection_overlap_row",
    "normalizer_from_metadata",
    "build_online_evidence",
]
