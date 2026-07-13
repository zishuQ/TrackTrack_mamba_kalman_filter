from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    validate_rollout_label,
)
from agentguard.features.normalization import NormalizationStats


JOINT_DATASET_SCHEMA_VERSION = 3
IWG_CONTEXT_SIZE = 6
IWG_WARMUP_EVENTS = IWG_CONTEXT_SIZE - 1
MOT17_FRCNN_TRAIN_SEQUENCES = [
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-13-FRCNN",
]
MOT17_FRCNN_VAL_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-11-FRCNN",
]
JOINT_DATASET_SCHEMA_DESCRIPTOR = {
    "name": "agentguard_iwg_tsrm_windows",
    "version": JOINT_DATASET_SCHEMA_VERSION,
    "timeline": "compact_cache_matched_and_unmatched",
    "label_join_key": ["sequence", "event_shard_id", "event_offset"],
    "unmatched_policy": "hold_both_sentinel",
    "iwg_context_size": IWG_CONTEXT_SIZE,
    "iwg_warmup_events": IWG_WARMUP_EVENTS,
    "iwg_input": "five_segment_local_warmup_events_plus_tsrm_window",
    "tsrm_input": "formal_window_only",
    "temporal_supervision": "last_valid_endpoint_only",
    "scalar_dim": 63,
    "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
}
JOINT_DATASET_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        JOINT_DATASET_SCHEMA_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

HOLD_BOTH_POLICY = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
UNMATCHED_CUE = np.zeros(3, dtype=np.float32)
UNMATCHED_RISK = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


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


def build_window_index(
    segments: Iterable[list[dict[str, Any]]],
    *,
    window_size: int,
    window_stride: int,
) -> list[dict[str, Any]]:
    if window_size < 1 or window_stride < 1:
        raise ValueError("window_size and window_stride must be positive")
    windows: list[dict[str, Any]] = []
    segment_id = 0
    for segment in segments:
        if not segment:
            continue
        endpoints = list(range(0, len(segment), window_stride))
        if endpoints[-1] != len(segment) - 1:
            endpoints.append(len(segment) - 1)
        for endpoint in endpoints:
            start = max(0, endpoint - window_size + 1)
            events = segment[start : endpoint + 1]
            iwg_start = max(0, start - IWG_WARMUP_EVENTS)
            iwg_events = segment[iwg_start : endpoint + 1]
            iwg_input_size = window_size + IWG_WARMUP_EVENTS
            windows.append(
                {
                    "sequence": str(events[0]["sequence"]),
                    "track_id": int(events[0]["track_id"]),
                    "segment_id": segment_id,
                    "pad_left": window_size - len(events),
                    "events": events,
                    "iwg_pad_left": iwg_input_size - len(iwg_events),
                    "iwg_events": iwg_events,
                }
            )
        segment_id += 1
    return windows


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


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


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


def resolve_joint_sequences(
    *,
    dataset: str,
    split: str,
    available_sequences: Iterable[str],
    val_sequences: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    available = sorted(set(available_sequences))
    validation = sorted(set(val_sequences))
    if dataset == "MOT17" and split == "all":
        expected_val = sorted(MOT17_FRCNN_VAL_SEQUENCES)
        if validation != expected_val:
            raise ValueError(
                "MOT17/all joint holdout requires exact validation sequences "
                f"{expected_val}, got {validation}"
            )
        train = sorted(MOT17_FRCNN_TRAIN_SEQUENCES)
        selected = sorted(train + expected_val)
    else:
        selected = available
        train = sorted(set(selected).difference(validation))
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"required sequences are absent from event cache: {missing}")
    if not train or not validation:
        raise ValueError("explicit holdout requires non-empty train and validation sequences")
    if not set(train).isdisjoint(validation):
        raise AssertionError("train and validation sequences must be disjoint")
    return selected, train, validation


def limit_windows_sequence_balanced(
    windows: list[dict[str, Any]], max_windows: int
) -> list[dict[str, Any]]:
    """Deterministically round-robin sequence groups for bounded smoke sets."""
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        grouped[str(window["sequence"])].append(window)
    selected: list[dict[str, Any]] = []
    offset = 0
    sequence_names = sorted(grouped)
    while len(selected) < max_windows:
        added = False
        for sequence in sequence_names:
            group = grouped[sequence]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == max_windows:
                    break
        if not added:
            break
        offset += 1
    return selected


def build_iwg_tsrm_dataset(
    *,
    dataset: str,
    split: str,
    event_cache_root: str | Path,
    detection_cache_root: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    val_sequences: list[str],
    window_size: int = 16,
    window_stride: int = 4,
    max_frame_gap: int = 30,
) -> dict[str, Any]:
    event_cache_root = Path(event_cache_root).resolve()
    detection_cache_root = Path(detection_cache_root).resolve()
    label_dir = Path(label_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cache_split_root = event_cache_root / dataset / split
    available_sequences = sorted(
        path.name for path in cache_split_root.iterdir() if path.is_dir()
    )
    sequences, train_sequences, val_sequences = resolve_joint_sequences(
        dataset=dataset,
        split=split,
        available_sequences=available_sequences,
        val_sequences=val_sequences,
    )

    labels = _load_labels(label_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        expected = {
            "joint_dataset_schema_sha256": JOINT_DATASET_SCHEMA_SHA256,
            "train_sequences": train_sequences,
            "val_sequences": val_sequences,
            "window_size": int(window_size),
            "iwg_warmup_events": IWG_WARMUP_EVENTS,
            "iwg_input_size": int(window_size) + IWG_WARMUP_EVENTS,
            "temporal_supervision": "last_valid_endpoint_only",
            "window_stride": int(window_stride),
            "max_frame_gap": int(max_frame_gap),
        }
        incompatible = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if incompatible:
            raise ValueError(f"refusing to reuse incompatible dataset directory: {incompatible}")

    split_windows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    timeline_counts: dict[str, dict[str, int]] = {}
    reid_dims: set[int] = set()
    manifest_paths: list[Path] = []
    for sequence in sequences:
        cache_dir = event_cache_root / dataset / split / sequence
        detection_dir = detection_cache_root / dataset / split / sequence
        if not detection_dir.is_dir():
            raise FileNotFoundError(f"detection cache not found: {detection_dir}")
        reader = CompactEventCacheReader(cache_dir, detection_dir)
        try:
            if not bool(reader.manifest.get("complete", False)):
                raise ValueError(f"event cache is incomplete: {cache_dir}")
            reid_dims.add(int(reader.manifest["reid_dim"]))
            timeline = [
                compact_timeline_record(sequence, record)
                for record in reader.iter_event_records()
            ]
        finally:
            reader.close()
        segments = segment_track_timelines(timeline, max_frame_gap=max_frame_gap)
        windows = build_window_index(
            segments,
            window_size=window_size,
            window_stride=window_stride,
        )
        destination = "val" if sequence in val_sequences else "train"
        split_windows[destination].extend(windows)
        timeline_counts[sequence] = {
            "events": len(timeline),
            "matched": sum(bool(record["matched"]) for record in timeline),
            "unmatched": sum(not bool(record["matched"]) for record in timeline),
            "segments": len(segments),
            "windows": len(windows),
            "labels": sum(
                event_key(sequence, record["event_shard_id"], record["event_offset"])
                in labels
                for record in timeline
            ),
        }
        manifest_paths.append(cache_dir / "manifest.json")
    if len(reid_dims) != 1:
        raise ValueError(f"inconsistent reid dimensions across caches: {sorted(reid_dims)}")
    if any(counts["unmatched"] == 0 for counts in timeline_counts.values()):
        missing = [seq for seq, counts in timeline_counts.items() if counts["unmatched"] == 0]
        raise ValueError(f"timeline is missing unmatched events for sequences: {missing}")

    _write_jsonl(output_dir / "train_windows.jsonl", split_windows["train"])
    _write_jsonl(output_dir / "val_windows.jsonl", split_windows["val"])
    norm_stats = _fit_train_normalization(
        event_cache_root, dataset, split, train_sequences
    )
    norm_stats.save(str(output_dir / "norm_stats.npz"))

    metadata = {
        "dataset": dataset,
        "split": split,
        "split_policy": "explicit_sequence_holdout",
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "event_cache_root": str(event_cache_root),
        "detection_cache_root": str(detection_cache_root),
        "label_dir": str(label_dir),
        "window_size": int(window_size),
        "iwg_context_size": IWG_CONTEXT_SIZE,
        "iwg_warmup_events": IWG_WARMUP_EVENTS,
        "iwg_input_size": int(window_size) + IWG_WARMUP_EVENTS,
        "temporal_supervision": "last_valid_endpoint_only",
        "window_stride": int(window_stride),
        "max_frame_gap": int(max_frame_gap),
        "reid_dim": reid_dims.pop(),
        "scalar_dim": 63,
        "event_dim": 128,
        "num_train_windows": len(split_windows["train"]),
        "num_val_windows": len(split_windows["val"]),
        "timeline_counts": timeline_counts,
        "joint_dataset_schema_version": JOINT_DATASET_SCHEMA_VERSION,
        "joint_dataset_schema_sha256": JOINT_DATASET_SCHEMA_SHA256,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "label_files_sha256": _sha256_file_set(label_dir.glob("*_labels.json")),
        "event_cache_manifests_sha256": _sha256_file_set(manifest_paths),
        "train_index_file": "train_windows.jsonl",
        "val_index_file": "val_windows.jsonl",
        "normalization_file": "norm_stats.npz",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "metadata.sha256").write_text(_sha256_file(metadata_path) + "\n")
    return metadata


class CompactIWGTSRMWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        max_windows: int = 0,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.dataset_dir = Path(dataset_dir)
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text())
        self._validate_metadata()
        self.split = split
        index_path = self.dataset_dir / self.metadata[f"{split}_index_file"]
        with index_path.open() as handle:
            self.windows = [json.loads(line) for line in handle if line.strip()]
        if max_windows > 0:
            self.windows = limit_windows_sequence_balanced(
                self.windows, int(max_windows)
            )
        selected_sequences = set(self.metadata[f"{split}_sequences"])
        self.labels = {
            key: label
            for key, label in _load_labels(Path(self.metadata["label_dir"])).items()
            if key.split("|", 1)[0] in selected_sequences
        }
        self.norm_stats = NormalizationStats.load(
            str(self.dataset_dir / self.metadata["normalization_file"])
        )
        self._readers: dict[str, CompactEventCacheReader] = {}

    def _validate_metadata(self) -> None:
        expected = {
            "joint_dataset_schema_version": JOINT_DATASET_SCHEMA_VERSION,
            "joint_dataset_schema_sha256": JOINT_DATASET_SCHEMA_SHA256,
            "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
            "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "iwg_context_size": IWG_CONTEXT_SIZE,
            "iwg_warmup_events": IWG_WARMUP_EVENTS,
            "iwg_input_size": int(self.metadata.get("window_size", 0))
            + IWG_WARMUP_EVENTS,
            "temporal_supervision": "last_valid_endpoint_only",
        }
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"joint dataset metadata schema mismatch: {mismatches}")
        train = set(self.metadata["train_sequences"])
        val = set(self.metadata["val_sequences"])
        if not train.isdisjoint(val):
            raise ValueError(f"joint dataset sequence leakage: {sorted(train & val)}")
        if self.metadata.get("dataset") == "MOT17" and self.metadata.get("split") == "all":
            expected_train = set(MOT17_FRCNN_TRAIN_SEQUENCES)
            expected_val = set(MOT17_FRCNN_VAL_SEQUENCES)
            if train != expected_train or val != expected_val:
                raise ValueError(
                    "MOT17/all joint dataset must use the fixed FRCNN split: "
                    f"train={sorted(expected_train)}, val={sorted(expected_val)}"
                )

    def _reader(self, sequence: str) -> CompactEventCacheReader:
        if sequence not in self._readers:
            self._readers[sequence] = CompactEventCacheReader(
                Path(self.metadata["event_cache_root"])
                / self.metadata["dataset"]
                / self.metadata["split"]
                / sequence,
                Path(self.metadata["detection_cache_root"])
                / self.metadata["dataset"]
                / self.metadata["split"]
                / sequence,
            )
        return self._readers[sequence]

    def close(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        if hasattr(self, "_readers"):
            self._readers.clear()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        length = int(self.metadata["window_size"])
        iwg_length = int(self.metadata["iwg_input_size"])
        reid_dim = int(self.metadata["reid_dim"])
        pad_left = int(window["pad_left"])
        iwg_pad_left = int(window["iwg_pad_left"])
        iwg_track_feats = np.zeros((iwg_length, reid_dim), dtype=np.float32)
        iwg_det_feats = np.zeros((iwg_length, reid_dim), dtype=np.float32)
        iwg_scalar_feats = np.zeros((iwg_length, 63), dtype=np.float64)
        iwg_padding_mask = np.ones(iwg_length, dtype=np.bool_)
        iwg_has_detection = np.zeros(iwg_length, dtype=np.bool_)
        padding_mask = np.ones(length, dtype=np.bool_)
        has_detection = np.zeros(length, dtype=np.bool_)
        label_mask = np.zeros(length, dtype=np.bool_)
        valid_motion = np.zeros(length, dtype=np.bool_)
        valid_appearance = np.zeros(length, dtype=np.bool_)
        base_target = np.zeros((length, 2), dtype=np.float32)
        final_target = np.zeros((length, 2), dtype=np.float32)
        policy_target = np.zeros((length, 5), dtype=np.float32)
        cue_target = np.zeros((length, 3), dtype=np.float32)
        risk_target = np.zeros((length, 4), dtype=np.float32)
        sample_weight = np.zeros(length, dtype=np.float32)
        frame_ids = np.full(length, -1, dtype=np.int64)
        track_ids = np.full(length, -1, dtype=np.int64)
        reset_mask = np.zeros(length, dtype=np.bool_)
        temporal_endpoint_mask = np.zeros(length, dtype=np.bool_)

        reader = self._reader(window["sequence"])
        for position, reference in enumerate(
            window["iwg_events"], start=iwg_pad_left
        ):
            record = reader.get_event_record(
                reference["event_shard_id"], reference["event_offset"]
            )
            track = np.asarray(record.get("track_feature", []), dtype=np.float32)
            scalar = np.asarray(record.get("scalar_features", []), dtype=np.float64)
            if track.shape != (reid_dim,) or scalar.shape != (63,):
                raise ValueError(
                    f"invalid cached feature shape for {window['sequence']} "
                    f"{reference['event_shard_id']}:{reference['event_offset']}"
                )
            if not np.isfinite(track).all() or not np.isfinite(scalar).all():
                raise ValueError("joint dataset encountered non-finite cached features")
            iwg_track_feats[position] = track
            iwg_scalar_feats[position] = scalar
            matched = bool(record.get("matched", False))
            iwg_has_detection[position] = matched
            if matched:
                detection_index = int(record.get("accepted_detection_index", -1))
                if detection_index < 0:
                    raise ValueError("matched event has no accepted detection index")
                iwg_det_feats[position] = np.asarray(
                    reader.get_detection(detection_index)["feature"], dtype=np.float32
                ).reshape(-1)
            iwg_padding_mask[position] = False

        formal_offset = iwg_length - length
        track_feats = iwg_track_feats[formal_offset:].copy()
        det_feats = iwg_det_feats[formal_offset:].copy()
        scalar_feats = iwg_scalar_feats[formal_offset:].copy()
        has_detection[:] = iwg_has_detection[formal_offset:]
        padding_mask[:] = iwg_padding_mask[formal_offset:]

        for position, reference in enumerate(window["events"], start=pad_left):
            matched = bool(has_detection[position])
            if not matched:
                policy_target[position] = HOLD_BOTH_POLICY
                cue_target[position] = UNMATCHED_CUE
                risk_target[position] = UNMATCHED_RISK
            frame_ids[position] = int(reference["frame_id"])
            track_ids[position] = int(reference["track_id"])
            key = event_key(
                window["sequence"],
                reference["event_shard_id"],
                reference["event_offset"],
            )
            label = self.labels.get(key)
            if label is not None and matched:
                label_mask[position] = True
                valid_motion[position] = bool(label["valid_motion"])
                valid_appearance[position] = bool(label["valid_appearance"])
                base_target[position] = [
                    label["motion_soft_target"],
                    label["appearance_soft_target"],
                ]
                final_target[position] = [
                    label["motion_safe_target"],
                    label["appearance_safe_target"],
                ]
                policy_target[position] = label["policy_soft_target"]
                cue_target[position] = label["cue_target"]
                risk_target[position] = label["risk_targets"]
                sample_weight[position] = float(label["sample_weight"])

        iwg_scalar_feats = self.norm_stats.transform(iwg_scalar_feats).astype(np.float32)
        iwg_scalar_feats[iwg_padding_mask] = 0.0
        scalar_feats = iwg_scalar_feats[formal_offset:].copy()
        scalar_feats[padding_mask] = 0.0
        if pad_left < length:
            reset_mask[pad_left] = True
            temporal_endpoint_mask[length - 1] = True
        return {
            "track_feats": torch.from_numpy(track_feats),
            "det_feats": torch.from_numpy(det_feats),
            "scalar_feats": torch.from_numpy(scalar_feats),
            "iwg_track_feats": torch.from_numpy(iwg_track_feats),
            "iwg_det_feats": torch.from_numpy(iwg_det_feats),
            "iwg_scalar_feats": torch.from_numpy(iwg_scalar_feats),
            "iwg_padding_mask": torch.from_numpy(iwg_padding_mask),
            "iwg_has_detection_mask": torch.from_numpy(iwg_has_detection),
            "padding_mask": torch.from_numpy(padding_mask),
            "mask": torch.from_numpy(padding_mask),
            "has_detection_mask": torch.from_numpy(has_detection),
            "label_mask": torch.from_numpy(label_mask),
            "valid_motion": torch.from_numpy(valid_motion),
            "valid_appearance": torch.from_numpy(valid_appearance),
            "base_gate_target": torch.from_numpy(base_target),
            "final_gate_target": torch.from_numpy(final_target),
            "policy_soft_target": torch.from_numpy(policy_target),
            "cue_target": torch.from_numpy(cue_target),
            "risk_target": torch.from_numpy(risk_target),
            "sample_weight": torch.from_numpy(sample_weight),
            "reset_mask": torch.from_numpy(reset_mask),
            "temporal_endpoint_mask": torch.from_numpy(temporal_endpoint_mask),
            "frame_id": torch.from_numpy(frame_ids),
            "track_id": torch.from_numpy(track_ids),
            "sequence": window["sequence"],
            "segment_id": int(window["segment_id"]),
        }
