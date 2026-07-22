"""Shared compact-timeline helpers for current AgentGuard datasets."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.label_schema import validate_rollout_label
from agentguard.features.normalization import NormalizationStats


IWG_CONTEXT_SIZE = 6


def event_key(sequence: str, event_shard_id: int, event_offset: int) -> str:
    return f"{sequence}|{int(event_shard_id)}|{int(event_offset)}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_set(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def compact_timeline_record(sequence: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "event_shard_id": int(record["event_shard_id"]),
        "event_offset": int(record["event_offset"]),
        "event_id": str(record["event_id"]),
        "frame_id": int(record["frame_id"]),
        "track_id": int(record["track_id"]),
        "matched": bool(record.get("matched", False)),
        "history_count": int(record.get("history_count", 0)),
    }


def segment_track_timelines(
    records: Iterable[dict[str, Any]],
    *,
    max_frame_gap: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["sequence"]), int(record["track_id"]))].append(record)

    segments: list[list[dict[str, Any]]] = []
    for key in sorted(grouped):
        timeline = sorted(
            grouped[key],
            key=lambda item: (
                int(item["frame_id"]),
                int(item["event_shard_id"]),
                int(item["event_offset"]),
            ),
        )
        current: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        for record in timeline:
            reset = previous is None
            if previous is not None:
                gap = int(record["frame_id"]) - int(previous["frame_id"])
                reset = (
                    gap <= 0
                    or gap > int(max_frame_gap)
                    or int(record.get("history_count", 0))
                    < int(previous.get("history_count", 0))
                )
            if reset and current:
                segments.append(current)
                current = []
            current.append(record)
            previous = record
        if current:
            segments.append(current)
    return segments


def _load_labels(label_dir: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in sorted(label_dir.glob("*_labels.json")):
        sequence = path.name[: -len("_labels.json")]
        records = json.loads(path.read_text())
        for index, label in enumerate(records):
            try:
                validate_rollout_label(label)
            except ValueError as exc:
                raise ValueError(f"invalid rollout label {path}:{index}: {exc}") from exc
            if str(label.get("candidate_type", "A")) != "A":
                continue
            key = event_key(sequence, label["event_shard_id"], label["event_offset"])
            if key in labels:
                raise ValueError(f"duplicate candidate-A rollout label key: {key}")
            labels[key] = label
    if not labels:
        raise RuntimeError(f"No candidate-A rollout labels found in {label_dir}")
    return labels


def _fit_train_normalization(
    event_cache_root: Path,
    dataset: str,
    split: str,
    train_sequences: list[str],
) -> NormalizationStats:
    count = 0
    total = np.zeros(63, dtype=np.float64)
    total_sq = np.zeros(63, dtype=np.float64)
    for sequence in train_sequences:
        reader = CompactEventCacheReader(event_cache_root / dataset / split / sequence)
        try:
            for record in reader.iter_event_records():
                scalar = np.asarray(record.get("scalar_features", []), dtype=np.float64)
                if scalar.shape != (63,) or not np.isfinite(scalar).all():
                    raise ValueError(
                        f"invalid scalar feature in {sequence} shard="
                        f"{record['event_shard_id']} offset={record['event_offset']}"
                    )
                count += 1
                total += scalar
                total_sq += scalar * scalar
        finally:
            reader.close()
    if count == 0:
        raise RuntimeError("No train timeline events available for normalization")
    stats = NormalizationStats()
    stats.mean = total / count
    variance = np.maximum(total_sq / count - stats.mean * stats.mean, 0.0)
    stats.std = np.sqrt(variance)
    stats.std[stats.std < 1e-8] = 1.0
    return stats


__all__ = [
    "IWG_CONTEXT_SIZE",
    "compact_timeline_record",
    "event_key",
    "segment_track_timelines",
]
