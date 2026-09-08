"""Replay-complete train_data v2 with track-only ReID and shared detections.

All replay arrays are tensors inside the uncompressed torch archive. Training
opens it with mmap and retains only training tensors; replay matrices are not
deserialized into one Python dictionary per historical event.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.detection_cache import SequenceDetectionCache

V2_VERSION = 2
RULES = {"motion_normalization": "gt", "prototype_mode": "leave_one_out"}
DETECTION_FILES = ("manifest.json", "boxes.npy", "features.npy", "scores.npy",
                   "class_ids.npy", "sources.npy", "frame_offsets.npy",
                   "target_detection_indices.npy", "target_frame_offsets.npy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_detection(directory: Path) -> dict:
    return {name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
            for name in DETECTION_FILES}


def resolve_detection_dir(sequence_dir: Path, metadata: dict, root=None) -> Path:
    root = root or os.environ.get("AGENTGUARD_DETECTION_CACHE_ROOT")
    if root:
        return Path(root).resolve() / metadata["dataset"] / metadata["source_split"] / metadata["sequence"]
    return (sequence_dir / metadata["references"]["detection_cache"]).resolve()


def check_detection_reference(directory: Path, metadata: dict, *, full: bool = False) -> None:
    expected = metadata["references"]["detection_fingerprint"]
    for name, record in expected.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"detection cache missing/size mismatch: {path}")
        if (full or name == "manifest.json") and sha256(path) != record["sha256"]:
            raise ValueError(f"detection cache fingerprint mismatch: {path}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("dataset", "sequence"):
        if manifest.get(name) != metadata[name]:
            raise ValueError(f"wrong {name} in detection cache: {directory}")
    if not manifest.get("complete"):
        raise ValueError(f"incomplete detection cache: {directory}")


def pack_table(records: list[dict]) -> dict:
    """Column-pack a shard, including variable-sized association matrices."""
    if not records:
        return {"rows": 0, "columns": {}}
    columns = {}
    keys = sorted(set().union(*(r.keys() for r in records)))
    for key in keys:
        values = [r.get(key) for r in records]
        if all(isinstance(v, np.ndarray) for v in values):
            shapes = [list(v.shape) for v in values]
            if all(s == shapes[0] for s in shapes):
                columns[key] = {"kind": "stack", "data": torch.from_numpy(np.stack(values).copy())}
            else:
                offsets = np.cumsum([0] + [v.size for v in values], dtype=np.int64)
                data = np.concatenate([v.reshape(-1) for v in values])
                columns[key] = {"kind": "ragged", "data": torch.from_numpy(data.copy()),
                                "offsets": torch.from_numpy(offsets), "shapes": shapes}
        elif all(isinstance(v, (bool, np.bool_)) for v in values):
            columns[key] = {"kind": "scalar", "data": torch.tensor(values, dtype=torch.bool)}
        elif all(isinstance(v, (int, np.integer)) for v in values):
            columns[key] = {"kind": "scalar", "data": torch.tensor(values, dtype=torch.int64)}
        elif all(isinstance(v, (float, np.floating)) for v in values):
            columns[key] = {"kind": "scalar", "data": torch.tensor(values, dtype=torch.float64)}
        else:
            # IDs, optional scalars and identity keys are small built-in objects.
            columns[key] = {"kind": "objects", "data": values}
    return {"rows": len(records), "columns": columns}


def table_row(table: dict, row: int) -> dict:
    if not 0 <= row < table["rows"]:
        raise IndexError(row)
    result = {}
    for key, column in table["columns"].items():
        kind, data = column["kind"], column["data"]
        if kind == "stack":
            result[key] = data[row].numpy()
        elif kind == "ragged":
            start, end = int(column["offsets"][row]), int(column["offsets"][row + 1])
            result[key] = data[start:end].numpy().reshape(column["shapes"][row])
        elif kind == "scalar":
            result[key] = data[row].item()
        else:
            result[key] = data[row]
    return result


class PackedReplayReader(CompactEventCacheReader):
    """The existing rollout interface, backed entirely by v2 data.pt + refs."""

    def __init__(self, sequence_dir, detection_cache_dir=None, *, max_cached_shards=4):
        self.cache_dir = Path(sequence_dir).resolve()
        self._packed = torch.load(self.cache_dir / "data.pt", map_location="cpu", mmap=True, weights_only=True)
        metadata = self._packed["metadata"]
        if metadata.get("schema_version") != V2_VERSION:
            raise ValueError("PackedReplayReader requires train_data v2")
        self.manifest = self._packed["replay"]["manifest"]
        self._tables = self._packed["replay"]["tables"]
        self.event_shards = list(range(len(self._tables["events"])))
        self.state_shards = list(range(len(self._tables["states"])))
        self.frame_shards = list(range(len(self._tables["frames"])))
        self.association_shards = list(range(len(self._tables["associations"])))
        self._loaded = {}
        self._frame_by_index = None
        self.max_cached_shards = max_cached_shards
        self.allow_legacy_schema = False
        self._track_features = np.load(self.cache_dir / "reid_features.npy", mmap_mode="r", allow_pickle=False)
        expected = (int(self.manifest["num_events"]), int(self.manifest["reid_dim"]))
        if self._track_features.shape != expected:
            raise ValueError(f"track ReID shape mismatch: {self._track_features.shape} != {expected}")
        directory = Path(detection_cache_dir) if detection_cache_dir else resolve_detection_dir(self.cache_dir, metadata)
        check_detection_reference(directory, metadata)
        self.detection_cache = SequenceDetectionCache(directory)

    def _load_shard(self, kind: str, shard_id: int) -> list[dict]:
        shard_id = int(shard_id)
        key = (kind, shard_id)
        if key in self._loaded:
            value = self._loaded.pop(key)
            self._loaded[key] = value
            return value
        if not 0 <= shard_id < len(self._tables[kind]):
            raise IndexError(key)
        table = self._tables[kind][shard_id]
        rows = [table_row(table, i) for i in range(table["rows"])]
        if kind == "events":
            for row in rows:
                index = int(row["feature_index"])
                row["track_feature"] = self._track_features[index]
                row["scalar_features"] = self._packed["arrays"]["timeline_scalar_feats"][index].numpy()
        self._loaded[key] = rows
        while len(self._loaded) > self.max_cached_shards:
            self._loaded.pop(next(iter(self._loaded)))
        return rows

    def iter_event_records(self, limit=None):
        count = 0
        for shard_id in self.event_shards:
            for record in self._load_shard("events", shard_id):
                yield record
                count += 1
                if limit is not None and count >= limit:
                    return

    def close(self):
        super().close()
        self._frame_by_index = None
        self._tables = {}
        self._packed = None
        if getattr(self, "_track_features", None) is not None:
            self._track_features._mmap.close()
            self._track_features = None


def validate_replay(reader) -> dict:
    m = reader.manifest
    if not m.get("complete") or m.get("truncated", False):
        raise ValueError("replay requires a complete, untruncated sequence")
    if not reader.frame_shards or not reader.association_shards or not reader.state_shards:
        raise ValueError("full frame/association/state records are required")
    det = reader.detection_cache
    if int(m["num_frames"]) != det.num_frames:
        raise ValueError("frame count differs from detection cache")
    for i in range(det.num_frames):
        frame = reader.get_frame_record(i)
        if frame is None or int(frame["frame_id"]) != i + 1:
            raise ValueError(f"missing frame {i + 1}")
        warp = np.asarray(frame.get("effective_warp"))
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            raise ValueError(f"invalid warp in frame {i + 1}")
    count = 0
    for record in reader.iter_event_records():
        frame_index = int(record["frame_index"])
        if not 0 <= frame_index < det.num_frames or int(record["frame_id"]) != frame_index + 1:
            raise ValueError("invalid event frame reference")
        for field, shape in (("track_feature", (int(m["reid_dim"]),)), ("scalar_features", (63,))):
            value = np.asarray(record[field])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid event {count}: {field}")
        index = int(record["accepted_detection_index"])
        if bool(record["matched"]):
            start, end = int(det.frame_offsets[frame_index]), int(det.frame_offsets[frame_index + 1])
            if not start <= index < end:
                raise ValueError(f"detection is not in event frame: {count}")
        elif index != -1:
            raise ValueError("unmatched event must have detection index -1")
        for name in ("frame_start_state_offset", "pre_update_state_offset"):
            shard, offset = int(record["state_shard_id"]), int(record[name])
            if shard < 0 or offset < 0:
                raise ValueError("negative state reference")
            state = reader.get_state(shard, offset)
            for field, shape in (("mean", (8,)), ("covariance", (8, 8)), ("box", (4,))):
                value = np.asarray(state[field])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid state {field} in event {count}")
        association = reader.get_association(record["association_shard_id"], record["association_offset"])
        # Frames with no detections can legitimately have no association record.
        if association is None and bool(record["matched"]):
            raise ValueError("invalid association frame reference")
        if association is not None and int(association["frame_id"]) != frame_index + 1:
            raise ValueError("invalid association frame reference")
        count += 1
    if count != int(m["num_events"]):
        raise ValueError("event count mismatch")
    return {"events": count, "frames": det.num_frames, "complete": True}


def training_arrays(reader, labels: list[dict], context_size: int, max_frame_gap: int):
    """Build causal windows; raw scalar features are normalized at read time."""
    keyed = {(int(l["event_shard_id"]), int(l["event_offset"])): l for l in labels}
    if len(keyed) != len(labels) or not labels:
        raise ValueError("empty or duplicate labels")
    segment_counts, previous = {}, {}
    for r in reader.iter_event_records():
        tid, fid, hist = int(r["track_id"]), int(r["frame_id"]), int(r["history_count"])
        p = previous.get(tid)
        reset = p is None or not 0 < fid-p[0] <= max_frame_gap or hist < p[1]
        segment_counts[tid] = segment_counts.get(tid, 0) + int(reset)
        previous[tid] = (fid, hist)
    bases, total_segments = {}, 0
    for tid in sorted(segment_counts):
        bases[tid] = total_segments
        total_segments += segment_counts[tid]
    n = int(reader.manifest["num_events"])
    scalar = np.empty((n, 63), np.float32)
    matched = np.zeros(n, bool)
    detection_indices = np.full(n, -1, np.int64)
    windows, track_ids, segments, ordered = [], [], [], []
    history, positions, previous = {}, {}, {}
    unmatched_history = 0
    for i, r in enumerate(reader.iter_event_records()):
        tid, fid, hist = int(r["track_id"]), int(r["frame_id"]), int(r["history_count"])
        p = previous.get(tid)
        if p is None or not 0 < fid-p[0] <= max_frame_gap or hist < p[1]:
            history[tid] = deque(maxlen=context_size)
            positions[tid] = positions.get(tid, -1) + 1
        previous[tid] = (fid, hist)
        history[tid].append(i)
        scalar[i] = r["scalar_features"]
        matched[i] = bool(r["matched"])
        detection_indices[i] = int(r["accepted_detection_index"])
        key = (int(r["event_shard_id"]), int(r["event_offset"]))
        if key in keyed:
            if not matched[i]:
                raise ValueError("label points to unmatched event")
            indices = list(history[tid])
            windows.append([-1] * (context_size-len(indices)) + indices)
            track_ids.append(tid)
            segments.append(bases[tid] + positions[tid])
            ordered.append(keyed[key])
            unmatched_history += int(any(not matched[x] for x in indices[:-1]))
    if len(ordered) != len(labels):
        raise ValueError("not all labels join to the current timeline")
    arrays = {"event_indices": torch.tensor(windows, dtype=torch.int64),
              "track_ids": torch.tensor(track_ids, dtype=torch.int64),
              "segment_ids": torch.tensor(segments, dtype=torch.int64),
              "timeline_scalar_feats": torch.from_numpy(scalar),
              "timeline_has_detection": torch.from_numpy(matched),
              "timeline_detection_indices": torch.from_numpy(detection_indices)}
    for name, fields in {
        "safe_gate_target": ("motion_safe_target", "appearance_safe_target"),
        "oracle_gate_target": ("motion_soft_target", "appearance_soft_target"),
        "gate_confidence": ("motion_label_confidence", "appearance_label_confidence"),
        "valid_channels": ("valid_motion", "valid_appearance"),
    }.items():
        arrays[name] = torch.tensor([[l[f] for f in fields] for l in ordered],
                                   dtype=torch.bool if name == "valid_channels" else torch.float32)
    for name, field in {"policy_safe_soft_target": "policy_safe_soft_target", "cue_target": "cue_target",
                        "risk_target": "risk_targets", "sample_weight": "sample_weight"}.items():
        arrays[name] = torch.tensor([l[field] for l in ordered], dtype=torch.float32)
    counts = {"events": n, "matched": int(matched.sum()), "unmatched": int((~matched).sum()),
              "segments": total_segments, "labeled_endpoints": len(labels),
              "samples_with_unmatched_history": unmatched_history}
    sums = {"count": n, "sum": scalar.astype(np.float64).sum(axis=0).tolist(),
            "sum_sq": np.square(scalar.astype(np.float64)).sum(axis=0).tolist()}
    return arrays, counts, sums


def write_packed_sequence(reader, stage: Path, metadata: dict, labels: list[dict], summary: dict) -> dict:
    """Write a new stage directory, never touching an existing dataset."""
    stage.mkdir(parents=True, exist_ok=False)
    arrays, counts, sums = training_arrays(reader, labels, int(metadata["context_size"]), int(metadata["max_frame_gap"]))
    reid = np.lib.format.open_memmap(stage / "reid_features.npy", mode="w+", dtype=np.float32,
                                   shape=(counts["events"], int(reader.manifest["reid_dim"])))
    tables = {}
    try:
        event_index = 0
        for kind, shards in (("events", reader.event_shards), ("states", reader.state_shards),
                             ("frames", reader.frame_shards), ("associations", reader.association_shards)):
            tables[kind] = []
            for shard_id in range(len(shards)):
                records = []
                for original in reader._load_shard(kind, shard_id):
                    record = dict(original)
                    if kind == "events":
                        record.setdefault("event_shard_id", shard_id)
                        record.setdefault("event_offset", len(records))
                        reid[event_index] = np.asarray(record.pop("track_feature"), dtype=np.float32)
                        record.pop("scalar_features")
                        record["feature_index"] = event_index
                        event_index += 1
                    records.append(record)
                tables[kind].append(pack_table(records))
        if event_index != counts["events"]:
            raise ValueError("packed event count mismatch")
        reid.flush()
    finally:
        reid._mmap.close()
    metadata = {**metadata, "format": "train_data", "index_format": "train_data_v2",
                "schema_version": V2_VERSION, "scalar_storage": "raw",
                "reid_layout": "track_only", "reid_feature_file": "reid_features.npy",
                "reid_dim": int(reader.manifest["reid_dim"]), "scalar_dim": 63, "event_dim": 128,
                "timeline_counts": counts, "num_train_samples": len(labels)}
    torch.save({"metadata": metadata, "arrays": arrays,
                "replay": {"manifest": reader.manifest, "tables": tables},
                "labels_raw": pack_table(labels), "label_summary": summary}, stage / "data.pt")
    manifest = {**metadata, "complete": True, "scalar_statistics": sums,
                "files": {name: {"sha256": sha256(stage / name), "bytes": (stage / name).stat().st_size}
                          for name in ("data.pt", "reid_features.npy")}}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def compare_replay(source, packed) -> None:
    """Verify every stored field, including ragged matrices, before cleanup."""
    for kind in ("events", "states", "frames", "associations"):
        shards = source.event_shards if kind == "events" else getattr(source, kind[:-1] + "_shards")
        for i in range(len(shards)):
            original, restored = source._load_shard(kind, i), packed._load_shard(kind, i)
            if len(original) != len(restored):
                raise ValueError(f"replay length mismatch: {kind}/{i}")
            for a, b in zip(original, restored):
                for key, value in a.items():
                    if key not in b or not np.array_equal(np.asarray(value), np.asarray(b[key]), equal_nan=False):
                        raise ValueError(f"replay field mismatch: {kind}/{i}/{key}")
