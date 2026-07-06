from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1


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
    for fid in frames:
        item = history.get(fid) or history.get(str(fid))
        if item is not None and len(item) > 0:
            boxes.append(np.asarray(item[0], dtype=np.float32))
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
        "history_count": int(len(history)),
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
        config_hash: str = "",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.dataset = dataset
        self.split = split
        self.sequence = sequence
        self.reid_dim = int(reid_dim)
        self.event_flush_size = int(event_flush_size)
        self.frame_flush_size = int(frame_flush_size)
        self.config_hash = config_hash

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
        self._final_dir = self._seq_dir()
        self._temp_dir = self._final_dir.with_name(self._final_dir.name + ".incomplete")
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
        if self._final_dir.exists():
            manifest_path = self._final_dir / "manifest.json"
            if manifest_path.is_file():
                with manifest_path.open("r") as f:
                    manifest = json.load(f)
                if manifest.get("complete") and manifest.get("schema_version") == SCHEMA_VERSION:
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
        association = frame_record.get("association")
        if association is not None:
            association_record_id = self._num_associations
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
                "image_width": int(frame_record.get("image_width", 0)),
                "image_height": int(frame_record.get("image_height", 0)),
                "effective_warp": _np(frame_record.get("warp_matrix"), np.float32),
                "detection_start_index": int(det_start),
                "detection_end_index": int(det_end),
                "association_record_id": int(association_record_id),
            }
        )
        self._num_frames += 1

        for event in events:
            fs_id = len(self._states)
            self._states.append(_compact_state(event.get("frame_start_state")))
            pu_id = len(self._states)
            self._states.append(_compact_state(event.get("pre_update_state")))

            matched = bool(event.get("has_detection", False))
            accepted = int(event.get("accepted_detection_index", -1) if matched else -1)
            history_count = int(self._states[pu_id].get("history_count", 0))
            association_track_row = -1
            if association is not None:
                track_ids = np.asarray(association.get("track_ids", []), dtype=np.int64)
                rows = np.flatnonzero(track_ids == int(event["track_id"]))
                if rows.size > 0:
                    association_track_row = int(rows[0])
            self._events.append(
                {
                    "event_id": str(event["event_id"]),
                    "frame_id": int(event["frame_id"]),
                    "track_id": int(event["track_id"]),
                    "matched": matched,
                    "accepted_detection_index": accepted,
                    "association_record_id": int(association_record_id),
                    "association_track_row": int(association_track_row),
                    "frame_start_state_id": int(fs_id),
                    "pre_update_state_id": int(pu_id),
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
        if self._associations:
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
        }
        with (self._temp_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    def _validate_shards(self) -> None:
        if self._temp_dir is None:
            return
        for path in self._temp_dir.glob("*.pt"):
            torch.load(path, weights_only=False)
