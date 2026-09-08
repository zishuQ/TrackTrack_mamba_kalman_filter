from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)

SCHEMA_VERSION = COMPACT_CACHE_SCHEMA_VERSION


def _np(value: Any, dtype=None) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=dtype or np.float32)
    return np.asarray(value, dtype=dtype)


def _compact_state(snapshot: dict | None) -> dict:
    if snapshot is None:
        return {}
    history = snapshot.get("history") or {}
    frames = sorted(int(k) for k in history.keys())[-6:]
    boxes = []
    scores = []
    for fid in frames:
        item = history.get(fid) or history.get(str(fid))
        if item is None or len(item) < 2:
            raise ValueError(f"compact history frame {fid} is missing box or observation score")
        box = np.asarray(item[0], dtype=np.float32).reshape(-1)
        score = float(item[1])
        if box.shape != (4,) or not np.all(np.isfinite(box)) or not np.isfinite(score):
            raise ValueError(f"invalid compact history entry at frame {fid}")
        boxes.append(box)
        scores.append(score)
    return {
        "mean": _np(snapshot.get("mean"), np.float32),
        "covariance": _np(snapshot.get("covariance"), np.float32),
        "box": _np(snapshot.get("box"), np.float32),
        "velocity": _np(snapshot.get("velocity"), np.float32),
        "score": float(snapshot.get("score", 0.0)),
        "state": int(snapshot.get("state", -1)),
        "end_frame_id": int(snapshot.get("end_frame_id", -1)),
        "recent_history_frames": np.asarray(frames, dtype=np.int32),
        "recent_history_boxes": (
            np.stack(boxes, axis=0).astype(np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        ),
        "recent_history_scores": np.asarray(scores, dtype=np.float32),
        "history_count": int(
            snapshot["observation_count"]
            if snapshot.get("observation_count") is not None
            else len(history)
        ),
    }


def _compact_association(record: dict, record_id: int, frame_id: int) -> dict:
    return {
        "association_record_id": int(record_id),
        "frame_id": int(frame_id),
        "track_ids": _np(record.get("track_ids"), np.int64),
        "detection_indices": _np(record.get("detection_indices"), np.int64),
        "raw_cost": _np(record.get("raw_cost"), np.float32),
        "final_cost": _np(record.get("final_cost"), np.float32),
        "iou_similarity": _np(record.get("iou_similarity"), np.float32),
        "iou_distance": _np(record.get("iou_distance"), np.float32),
        "cosine_distance": _np(record.get("cosine_distance"), np.float32),
        "confidence_distance": _np(record.get("confidence_distance"), np.float32),
        "angle_distance": _np(record.get("angle_distance"), np.float32),
        "assignment_round": _np(record.get("assignment_round"), np.int16),
        "assignment_threshold": _np(record.get("assignment_threshold"), np.float32),
        "detection_source": _np(record.get("detection_source"), np.int8),
        "reid_available": bool(record.get("reid_available", True)),
    }


class CompactEventCacheSink:
    """Compact EventSink that references detection-cache indices.

    It intentionally drops detection boxes/scores/features from event records.
    """

    compact = True

    def __init__(
        self,
        cache_root: str | Path,
        dataset: str,
        split: str,
        sequence: str,
        reid_dim: int = 0,
        event_flush_size: int = 256,
        frame_flush_size: int = 32,
        association_flush_size: int = 32,
        config_hash: str = "",
        source_commit: str = "",
        feature_schema_sha256: str = FEATURE_SCHEMA_SHA256,
        detection_cache_manifest_sha256: str = "",
        tracker_config_sha256: str = "",
        total_sequence_frames: int = 0,
        image_width: int = 0,
        image_height: int = 0,
        tracker_config: dict | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.dataset = dataset
        self.split = split
        self.sequence = sequence
        self.reid_dim = int(reid_dim)
        self.event_flush_size = int(event_flush_size)
        self.frame_flush_size = int(frame_flush_size)
        self.association_flush_size = int(association_flush_size)
        self.config_hash = config_hash
        self.source_commit = source_commit
        self.feature_schema_sha256 = feature_schema_sha256 or FEATURE_SCHEMA_SHA256
        self.detection_cache_manifest_sha256 = detection_cache_manifest_sha256
        self.tracker_config_sha256 = tracker_config_sha256 or config_hash
        self.total_sequence_frames = int(total_sequence_frames)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.tracker_config = dict(tracker_config or {})

        self._frames: list[dict] = []
        self._events: list[dict] = []
        self._states: list[dict] = []
        self._associations: list[dict] = []
        self._frame_shard = 0
        self._event_shard = 0
        self._state_shard = 0
        self._association_shard = 0
        self._num_frames = 0
        self._num_events = 0
        self._num_matched = 0
        self._num_unmatched = 0
        self._num_associations = 0
        self._num_detections = 0
        self._truncated = False
        self._temp_dir: Path | None = None
        self._final_dir: Path | None = None

    def _seq_dir(self) -> Path:
        return self.cache_root / self.dataset / self.split / self.sequence

    def on_sequence_start(
        self,
        sequence: str,
        reid_dim: int,
        image_width: int,
        image_height: int,
    ) -> None:
        self.reid_dim = int(reid_dim or self.reid_dim)
        self.image_width = int(image_width or self.image_width)
        self.image_height = int(image_height or self.image_height)
        self._final_dir = self._seq_dir()
        self._temp_dir = self._final_dir.with_name(self._final_dir.name + ".incomplete")
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
        if self._final_dir.exists():
            manifest_path = self._final_dir / "manifest.json"
            if manifest_path.is_file():
                with manifest_path.open("r") as f:
                    manifest = json.load(f)
                if (
                    manifest.get("complete")
                    and manifest.get("schema_version") == SCHEMA_VERSION
                    and manifest.get("feature_schema_sha256") == self.feature_schema_sha256
                    and manifest.get("tracker_config_sha256") == self.tracker_config_sha256
                    and manifest.get("detection_cache_manifest_sha256")
                    == self.detection_cache_manifest_sha256
                ):
                    self._temp_dir = None
                    return
            shutil.rmtree(self._final_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(complete=False)

    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        if self._temp_dir is None:
            return

        dets = frame_record.get("detections") or []
        det_indices = [
            int(d.get("detection_index", d.get("index", -1)))
            for d in dets
            if int(d.get("detection_index", d.get("index", -1))) >= 0
        ]
        det_start = min(det_indices) if det_indices else -1
        det_end = max(det_indices) + 1 if det_indices else -1
        self._num_detections = max(self._num_detections, det_end)

        association_record_id = -1
        association_shard_id = -1
        association_offset = -1
        association = frame_record.get("association")
        if association is not None:
            association_record_id = self._num_associations
            association_shard_id = self._association_shard
            association_offset = len(self._associations)
            self._associations.append(
                _compact_association(
                    association,
                    association_record_id,
                    int(frame_record["frame_id"]),
                )
            )
            self._num_associations += 1

        self._frames.append(
            {
                "frame_id": int(frame_record["frame_id"]),
                "frame_index": int(frame_record.get("frame_index", int(frame_record["frame_id"]) - 1)),
                "image_width": int(frame_record.get("image_width", 0)),
                "image_height": int(frame_record.get("image_height", 0)),
                "effective_warp": _np(frame_record.get("warp_matrix"), np.float32),
                "warp_matrix": _np(frame_record.get("warp_matrix"), np.float32),
                "detection_start_index": int(det_start),
                "detection_end_index": int(det_end),
                "association_record_id": int(association_record_id),
                "association_shard_id": int(association_shard_id),
                "association_offset": int(association_offset),
            }
        )
        self._num_frames += 1

        for event in events:
            state_shard_id = self._state_shard
            fs_offset = len(self._states)
            self._states.append(_compact_state(event.get("frame_start_state")))
            pu_offset = len(self._states)
            self._states.append(_compact_state(event.get("pre_update_state")))

            matched = bool(event.get("has_detection", False))
            accepted = int(event.get("accepted_detection_index", -1) if matched else -1)
            history_count = int(self._states[pu_offset].get("history_count", 0))
            association_track_row = -1
            if association is not None:
                track_ids = np.asarray(association.get("track_ids", []), dtype=np.int64)
                rows = np.flatnonzero(track_ids == int(event["track_id"]))
                if rows.size > 0:
                    association_track_row = int(rows[0])
            self._events.append(
                {
                    "event_shard_id": int(self._event_shard),
                    "event_offset": int(len(self._events)),
                    "event_id": str(event["event_id"]),
                    "frame_id": int(event["frame_id"]),
                    "frame_index": int(event.get("frame_index", int(event["frame_id"]) - 1)),
                    "track_id": int(event["track_id"]),
                    "matched": matched,
                    "accepted_detection_index": accepted,
                    "association_record_id": int(association_record_id),
                    "association_shard_id": int(association_shard_id),
                    "association_offset": int(association_offset),
                    "association_track_row": int(association_track_row),
                    "state_shard_id": int(state_shard_id),
                    "frame_start_state_offset": int(fs_offset),
                    "pre_update_state_offset": int(pu_offset),
                    "frame_start_state_id": int(fs_offset),
                    "pre_update_state_id": int(pu_offset),
                    "scalar_features": _np(event.get("scalar_features"), np.float32),
                    "track_feature": _np(event.get("track_feature"), np.float32).reshape(-1),
                    "history_count": history_count,
                }
            )
            self._num_events += 1
            if matched:
                self._num_matched += 1
            else:
                self._num_unmatched += 1

        if len(self._frames) >= self.frame_flush_size:
            self._flush_frames()
        if len(self._events) >= self.event_flush_size:
            self._flush_events()
            self._flush_states()
        if len(self._associations) >= self.association_flush_size:
            self._flush_associations()

    def on_sequence_end(self) -> dict:
        if self._temp_dir is None:
            return {"num_frames": self._num_frames, "num_events": self._num_events}
        self._flush_frames()
        self._flush_events()
        self._flush_states()
        self._flush_associations()
        self._write_manifest(complete=True)
        self._validate_shards()
        os.rename(self._temp_dir, self._final_dir)
        self._temp_dir = None
        return {
            "num_frames": self._num_frames,
            "num_events": self._num_events,
            "num_matched_events": self._num_matched,
            "num_unmatched_events": self._num_unmatched,
            "num_association_records": self._num_associations,
            "num_detections": self._num_detections,
        }

    def _save(self, prefix: str, shard_id: int, payload: list[dict]) -> None:
        if self._temp_dir is None or not payload:
            return
        torch.save(payload, self._temp_dir / f"{prefix}_{shard_id:05d}.pt")

    def _flush_frames(self) -> None:
        self._save("frames", self._frame_shard, self._frames)
        if self._frames:
            self._frame_shard += 1
            self._frames = []

    def _flush_events(self) -> None:
        self._save("events", self._event_shard, self._events)
        if self._events:
            self._event_shard += 1
            self._events = []

    def _flush_states(self) -> None:
        self._save("states", self._state_shard, self._states)
        if self._states:
            self._state_shard += 1
            self._states = []

    def _flush_associations(self) -> None:
        self._save("associations", self._association_shard, self._associations)
        if self._associations:
            self._association_shard += 1
            self._associations = []

    def _write_manifest(self, complete: bool) -> None:
        if self._temp_dir is None:
            return
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "split": self.split,
            "sequence": self.sequence,
            "num_frames": self._num_frames,
            "num_detections": self._num_detections,
            "num_events": self._num_events,
            "num_matched_events": self._num_matched,
            "num_unmatched_events": self._num_unmatched,
            "num_association_records": self._num_associations,
            "reid_dim": self.reid_dim,
            "complete": bool(complete),
            "truncated": self._truncated,
            "config_sha256": self.config_hash,
            "source_commit": self.source_commit,
            "feature_schema_sha256": self.feature_schema_sha256,
            "detection_cache_manifest_sha256": self.detection_cache_manifest_sha256,
            "tracker_config_sha256": self.tracker_config_sha256,
            "tracker_config": self.tracker_config,
            "processed_frames": self._num_frames,
            "total_sequence_frames": self.total_sequence_frames or self._num_frames,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "kf_type": self.tracker_config.get("kf_type"),
            "disable_gmc": self.tracker_config.get("disable_gmc"),
            "det_thr": self.tracker_config.get("det_thr"),
            "match_thr": self.tracker_config.get("match_thr"),
        }
        with (self._temp_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    def _validate_shards(self) -> None:
        if self._temp_dir is None:
            return
        for path in self._temp_dir.glob("*.pt"):
            torch.load(path, weights_only=False)
