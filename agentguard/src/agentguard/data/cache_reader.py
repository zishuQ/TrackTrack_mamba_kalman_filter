from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agentguard.data.cache_schema import CacheManifest


class EventCacheReader:
    """Reads cached events from disk.

    Expects the directory layout produced by :class:`CacheEventSink`::

        <cache_dir>/
        ├── manifest.json
        ├── frames_00000.pt
        ├── events_00000.pt
        ├── events_00001.pt
        ├── candidates_00000.pt   (optional stage-2)
        ├── candidates_00001.pt   (optional stage-2)
        └── identity_prototypes.pt (optional stage-2)

    Parameters
    ----------
    cache_dir : str
        Path to the sequence-level cache directory (the one containing
        ``manifest.json``).
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    #  Path helpers
    # ------------------------------------------------------------------

    def _resolve(self, filename: str) -> str:
        path = os.path.join(self.cache_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Expected cache file not found: {path}"
            )
        return path

    def _glob_shards(self, prefix: str) -> List[str]:
        """Return sorted paths of all shards matching ``<prefix>_*.pt``."""
        matching: List[str] = []
        for fname in os.listdir(self.cache_dir):
            if fname.startswith(f"{prefix}_") and fname.endswith(".pt"):
                matching.append(os.path.join(self.cache_dir, fname))
        matching.sort()
        if not matching:
            raise FileNotFoundError(
                f"No shard files found matching '{prefix}_*.pt' in {self.cache_dir}"
            )
        return matching

    def _glob_shards_optional(self, prefix: str) -> List[str]:
        """Return sorted paths of shards matching ``<prefix>_*.pt`` or empty list."""
        matching: List[str] = []
        for fname in os.listdir(self.cache_dir):
            if fname.startswith(f"{prefix}_") and fname.endswith(".pt"):
                matching.append(os.path.join(self.cache_dir, fname))
        matching.sort()
        return matching

    def _resolve_optional(self, filename: str) -> Optional[str]:
        """Return path if *filename* exists, otherwise ``None``."""
        path = os.path.join(self.cache_dir, filename)
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------
    #  Public read methods
    # ------------------------------------------------------------------

    def read_manifest(self) -> CacheManifest:
        """Load and return the ``CacheManifest`` from ``manifest.json``."""
        path = self._resolve("manifest.json")
        with open(path, "r") as f:
            data = json.load(f)
        return CacheManifest.from_dict(data)

    def read_events(self) -> List[Any]:
        """Load all events from sharded ``.pt`` files.

        Events are deserialised from JSON-compatible dicts back to
        ``TrackEvent`` objects (see
        :func:`agentguard.contracts.serialization.deserialize_events`).

        Returns
        -------
        list
            Concatenated list of all events across all shards, preserving
            the original shard ordering.
        """
        from agentguard.contracts.serialization import deserialize_events

        shard_paths = self._glob_shards("events")
        all_events: List[Any] = []
        for sp in shard_paths:
            chunk = torch.load(sp, weights_only=False)
            if isinstance(chunk, list):
                all_events.extend(chunk)
            else:
                all_events.append(chunk)
        # Deserialise raw dicts back to TrackEvent objects
        return deserialize_events(all_events)

    def read_candidates(self) -> List[Any]:
        """Load all candidates from sharded ``.pt`` files.

        Returns an empty list when no candidates shards exist (optional
        stage-2 file).

        Returns
        -------
        list
            Concatenated list of all candidates across all shards.
        """
        shard_paths = self._glob_shards_optional("candidates")
        all_candidates: List[Any] = []
        for sp in shard_paths:
            chunk = torch.load(sp, weights_only=False)
            if isinstance(chunk, list):
                all_candidates.extend(chunk)
            else:
                all_candidates.append(chunk)
        return all_candidates

    def read_identity_prototypes(self) -> Dict[int, Any]:
        """Load identity prototypes.

        Returns an empty dict when the optional stage-2 file does not exist.

        Returns
        -------
        dict
            Mapping ``track_id -> prototype_vector``.
        """
        path = self._resolve_optional("identity_prototypes.pt")
        if path is None:
            return {}
        data = torch.load(path, weights_only=False)
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected identity_prototypes.pt to contain a dict, "
                f"got {type(data).__name__}"
            )
        return data

    def read_frames(self) -> Any:
        """Load per-frame metadata.

        Accepts both the list payload (from ``CacheEventSink``) and the
        legacy dict payload (from ``EventCacheWriter``), returning the data
        as-stored.

        Returns
        -------
        list or dict
            Per-frame metadata in the stored format.
        """
        path = self._resolve("frames_00000.pt")
        data = torch.load(path, weights_only=False)
        if not isinstance(data, (list, dict)):
            raise TypeError(
                f"Expected frames_00000.pt to contain a list or dict, "
                f"got {type(data).__name__}"
            )
        return data


class CompactEventCacheReader:
    """Lazy reader for schema-v2 compact AgentGuard event caches."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        detection_cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "manifest.json").open("r") as f:
            self.manifest: dict = json.load(f)
        if int(self.manifest.get("schema_version", 0)) < 2:
            raise ValueError(
                f"CompactEventCacheReader requires schema_version>=2, got "
                f"{self.manifest.get('schema_version')} in {self.cache_dir}"
            )

        self.event_shards = self._glob("events")
        self.state_shards = self._glob("states")
        self.association_shards = self._glob("associations", required=False)
        self.frame_shards = self._glob("frames", required=False)
        self._loaded: dict[tuple[str, int], list[dict]] = {}
        self.max_cached_shards = int(os.environ.get("AGENTGUARD_MAX_CACHED_SHARDS", "32"))
        self._frame_by_index: dict[int, dict] | None = None

        self.detection_cache = None
        if detection_cache_dir is not None:
            from agentguard.data.detection_cache import SequenceDetectionCache

            self.detection_cache = SequenceDetectionCache(detection_cache_dir)

    def close(self) -> None:
        self._loaded.clear()
        if self.detection_cache is not None:
            self.detection_cache.close()
            self.detection_cache = None

    def _glob(self, prefix: str, required: bool = True) -> list[Path]:
        paths = sorted(self.cache_dir.glob(f"{prefix}_*.pt"))
        if required and not paths:
            raise FileNotFoundError(f"No {prefix}_*.pt shards in {self.cache_dir}")
        return paths

    def _load_shard(self, kind: str, shard_id: int) -> list[dict]:
        shard_id = int(shard_id)
        key = (kind, shard_id)
        if key in self._loaded:
            return self._loaded[key]
        paths = {
            "events": self.event_shards,
            "states": self.state_shards,
            "associations": self.association_shards,
            "frames": self.frame_shards,
        }[kind]
        if shard_id < 0 or shard_id >= len(paths):
            raise IndexError(f"{kind} shard {shard_id} outside [0, {len(paths)})")
        data = torch.load(paths[shard_id], weights_only=False)
        if not isinstance(data, list):
            raise TypeError(f"{paths[shard_id]} must contain a list, got {type(data).__name__}")
        if self.max_cached_shards == 0:
            return data
        self._loaded[key] = data
        while self.max_cached_shards > 0 and len(self._loaded) > self.max_cached_shards:
            self._loaded.pop(next(iter(self._loaded)))
        return data

    def iter_event_records(self, limit: int | None = None):
        yielded = 0
        for shard_id, path in enumerate(self.event_shards):
            records = torch.load(path, weights_only=False)
            if not isinstance(records, list):
                raise TypeError(f"{path} must contain a list")
            for offset, record in enumerate(records):
                record.setdefault("event_shard_id", shard_id)
                record.setdefault("event_offset", offset)
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def get_event_record(self, event_shard_id: int, event_offset: int) -> dict:
        return self._load_shard("events", event_shard_id)[int(event_offset)]

    def get_state(self, state_shard_id: int, offset: int) -> dict:
        return self._load_shard("states", state_shard_id)[int(offset)]

    def get_association(self, association_shard_id: int, offset: int) -> dict | None:
        if int(association_shard_id) < 0 or int(offset) < 0:
            return None
        return self._load_shard("associations", association_shard_id)[int(offset)]

    def get_frame_record(self, frame_index: int) -> dict | None:
        frame_index = int(frame_index)
        if self._frame_by_index is None:
            self._frame_by_index = {}
            for shard_id in range(len(self.frame_shards)):
                for frame in self._load_shard("frames", shard_id):
                    idx = int(frame.get("frame_index", int(frame.get("frame_id", 0)) - 1))
                    self._frame_by_index[idx] = frame
        return self._frame_by_index.get(frame_index)

    def get_detection(self, detection_index: int) -> dict:
        if self.detection_cache is None:
            raise RuntimeError("detection_cache_dir is required to read detections")
        return self.detection_cache.get_detection(detection_index)

    def _max_detection_iou(self, frame_index: int, detection_index: int) -> float:
        if self.detection_cache is None:
            return 0.0
        frame = self.detection_cache.get_frame(int(frame_index), view="source")
        det_indices = np.asarray(frame.get("detection_indices", []), dtype=np.int64)
        boxes = np.asarray(frame.get("boxes", []), dtype=np.float64)
        matches = np.flatnonzero(det_indices == int(detection_index))
        if matches.size == 0 or boxes.shape[0] <= 1:
            return 0.0
        box_idx = int(matches[0])
        box = boxes[box_idx]
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
        area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = area_a + area_b - inter
        iou = np.divide(inter, np.maximum(union, 1e-12))
        iou[box_idx] = 0.0
        return float(np.max(iou)) if iou.size else 0.0

    def _snapshot_from_record(self, record: dict, feature: np.ndarray, track_id: int):
        from agentguard.contracts.states import TrackStateSnapshot

        history = {}
        frames = np.asarray(record.get("recent_history_frames", []), dtype=np.int32)
        boxes = np.asarray(record.get("recent_history_boxes", []), dtype=np.float32)
        for frame_id, box in zip(frames.tolist(), boxes):
            history[int(frame_id)] = [np.asarray(box, dtype=np.float64)]
        return TrackStateSnapshot(
            track_id=int(track_id),
            box=np.asarray(record.get("box", np.zeros(4)), dtype=np.float64),
            score=float(record.get("score", 0.0)),
            mean=np.asarray(record["mean"], dtype=np.float64) if np.asarray(record.get("mean", [])).size else None,
            covariance=(
                np.asarray(record["covariance"], dtype=np.float64)
                if np.asarray(record.get("covariance", [])).size
                else None
            ),
            velocity=np.asarray(record.get("velocity", np.zeros((4, 2))), dtype=np.float64),
            feature=np.asarray(feature, dtype=np.float64).reshape(1, -1),
            history=history,
            end_frame_id=int(record.get("end_frame_id", -1)),
            state=int(record.get("state", -1)),
        )

    def materialize_training_event(self, record: dict, candidate_detection_index: int | None = None):
        from agentguard.contracts.events import TrackEvent
        from agentguard.contracts.states import (
            AssociationContext,
            AssociationPairFeatures,
            DetectionObservation,
        )

        track_feature = np.asarray(record.get("track_feature", []), dtype=np.float64).reshape(-1)
        frame_start = self.get_state(
            record["state_shard_id"],
            record["frame_start_state_offset"],
        )
        pre_update = self.get_state(
            record["state_shard_id"],
            record["pre_update_state_offset"],
        )
        association = self.get_association(
            record.get("association_shard_id", -1),
            record.get("association_offset", -1),
        )
        frame_record = self.get_frame_record(record.get("frame_index", int(record["frame_id"]) - 1))

        detection = None
        detection_feature = np.zeros_like(track_feature)
        pair = None
        ctx = None
        accepted = int(record.get("accepted_detection_index", -1))
        selected = (
            int(candidate_detection_index)
            if candidate_detection_index is not None and int(candidate_detection_index) >= 0
            else accepted
        )
        row_idx = int(record.get("association_track_row", -1))
        col_idx = -1
        has_detection = bool(record.get("matched", False)) or selected >= 0
        if has_detection and selected >= 0:
            det = self.get_detection(selected)
            detection_feature = np.asarray(det["feature"], dtype=np.float64).reshape(-1)
            detection = DetectionObservation(
                detection_index=selected,
                box=np.asarray(det["box"], dtype=np.float64),
                score=float(det["score"]),
                feature=detection_feature.reshape(1, -1),
                source=int(det["source"]),
                class_id=int(det["class_id"]),
            )

        if association is not None and row_idx >= 0:
            det_indices = np.asarray(association.get("detection_indices", []), dtype=np.int64)
            matches = np.flatnonzero(det_indices == selected)
            col_idx = int(matches[0]) if matches.size else -1
            final_cost = np.asarray(association["final_cost"], dtype=np.float64)
            raw_cost = np.asarray(association.get("raw_cost", final_cost), dtype=np.float64)
            iou_similarity = np.asarray(association.get("iou_similarity", np.zeros_like(final_cost)), dtype=np.float64)
            iou_distance = np.asarray(association.get("iou_distance", np.zeros_like(final_cost)), dtype=np.float64)
            cosine_distance = np.asarray(association.get("cosine_distance", np.zeros_like(final_cost)), dtype=np.float64)
            confidence_distance = np.asarray(association.get("confidence_distance", np.zeros_like(final_cost)), dtype=np.float64)
            angle_distance = np.asarray(association.get("angle_distance", np.zeros_like(final_cost)), dtype=np.float64)
            assignment_round = np.asarray(association.get("assignment_round", np.full(final_cost.shape, -1)), dtype=np.int16)
            assignment_threshold = np.asarray(association.get("assignment_threshold", np.full(final_cost.shape, -1.0)), dtype=np.float64)
            if col_idx >= 0 and row_idx < final_cost.shape[0]:
                pair = AssociationPairFeatures(
                    iou_similarity=float(iou_similarity[row_idx, col_idx]),
                    iou_distance=float(iou_distance[row_idx, col_idx]),
                    cosine_distance=float(cosine_distance[row_idx, col_idx]),
                    confidence_distance=float(confidence_distance[row_idx, col_idx]),
                    angle_distance=float(angle_distance[row_idx, col_idx]),
                    raw_cost=float(raw_cost[row_idx, col_idx]),
                    final_cost=float(final_cost[row_idx, col_idx]),
                    assignment_round=int(assignment_round[row_idx, col_idx]),
                    assignment_threshold=float(assignment_threshold[row_idx, col_idx]),
                    detection_source=int(detection.source if detection is not None else -1),
                )
                ctx = AssociationContext(
                    track_cost_row=final_cost[row_idx, :].copy(),
                    detection_cost_col=final_cost[:, col_idx].copy(),
                    detection_overlap_row=np.asarray(
                        [
                            self._max_detection_iou(
                                int(record.get("frame_index", int(record["frame_id"]) - 1)),
                                selected,
                            )
                        ],
                        dtype=np.float64,
                    ),
                    accepted_detection_index=selected,
                    num_tracks=int(final_cost.shape[0]),
                    num_detections=int(final_cost.shape[1]),
                    reid_available=bool(association.get("reid_available", True)),
                )

        event = TrackEvent(
            event_id=str(record["event_id"]),
            dataset=str(self.manifest.get("dataset", "")),
            sequence=str(self.manifest.get("sequence", "")),
            frame_id=int(record["frame_id"]),
            track_id=int(record["track_id"]),
            image_width=int(self.manifest.get("image_width", 0)),
            image_height=int(self.manifest.get("image_height", 0)),
            has_detection=has_detection,
            frame_start_state=self._snapshot_from_record(frame_start, track_feature, record["track_id"]),
            pre_update_state=self._snapshot_from_record(pre_update, track_feature, record["track_id"]),
            detection=detection,
            association=pair,
            association_context=ctx,
            warp_matrix=(
                np.asarray(
                    frame_record.get("warp_matrix", frame_record.get("effective_warp")),
                    dtype=np.float64,
                )
                if frame_record is not None
                else np.eye(2, 3, dtype=np.float64)
            ),
            scalar_features=np.asarray(record.get("scalar_features", []), dtype=np.float64),
            track_feature=track_feature,
            detection_feature=detection_feature,
        )
        if candidate_detection_index is not None and int(candidate_detection_index) >= 0:
            from agentguard.features.scalar import compute_scalar_features

            event.scalar_features = compute_scalar_features(event)
        return event
