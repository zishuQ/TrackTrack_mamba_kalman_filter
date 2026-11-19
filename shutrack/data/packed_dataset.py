"""Read-only adapter over the existing packed AgentGuard ``train_data``.

The packed directory is the only large training source used by SHUTrack:

    metadata.json
    <sequence>/data.pt          # small per-sample/label/timeline arrays
    <sequence>/reid_features.npy  # [T, 2, D] pre-update track + detection ReID

Nothing here builds caches, indexes, labels, windows on disk, or copies
feature arrays.  Arrays are either loaded lazily (``data.pt`` is small) or
memory-mapped (``reid_features.npy``); the source directory is never written.

The competition branch reads normalized legacy scalar columns 15/21/25.  When
``competition_enabled=False`` the dataset still returns the same tensor layout
but fills the branch with zeros, so the lean and full variants share one
window interface.  The effective branch configuration is recorded in the
dataset descriptor and must match the model/checkpoint.
"""

from __future__ import annotations

import bisect
import json
import mmap
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from shutrack.data.evidence import EventEvidence
from shutrack.data.legacy_adapter import normalizer_from_metadata
from shutrack.features.core import build_competition, build_evidence
from shutrack.features.windows import WindowArrays, build_window
from shutrack.schema import (
    COMPETITION_DIM,
    COMPETITION_SCALAR_INDICES,
    CONTEXT_SIZE,
    EVIDENCE_DIM,
    EVIDENCE_SCALAR_INDICES,
    FEATURE_SCHEMA_VERSION,
    HISTORY_SIZE,
    LEGACY_SCALAR_DIM,
)

PACKED_FORMAT = "train_data"
_REQUIRED_PACKED_ARRAYS = (
    "event_indices",
    "track_ids",
    "segment_ids",
    "safe_gate_target",
    "valid_channels",
    "sample_weight",
    "timeline_scalar_feats",
    "timeline_has_detection",
)


class PackedShutrackDataset(Dataset):
    """Lazy, read-only training view over one packed dataset directory."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        max_samples: int = 0,
        competition_enabled: bool = True,
        sequence_cache_size: int = 2,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"packed SHUTrack dataset metadata not found: {metadata_path}"
            )
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        if self.metadata.get("format") != PACKED_FORMAT:
            raise ValueError(
                "SHUTrack only reads packed AgentGuard train_data; "
                f"got format={self.metadata.get('format')!r} in {metadata_path}"
            )
        if int(self.metadata.get("context_size", -1)) != CONTEXT_SIZE:
            raise ValueError(
                "SHUTrack requires context_size=6 packed windows, got "
                f"{self.metadata.get('context_size')!r} in {metadata_path}"
            )
        if int(self.metadata.get("scalar_dim", -1)) != LEGACY_SCALAR_DIM:
            raise ValueError(
                "SHUTrack requires the legacy 63-dim scalar cache, got "
                f"scalar_dim={self.metadata.get('scalar_dim')!r}"
            )
        self.reid_dim = int(self.metadata.get("reid_dim", 0))
        if self.reid_dim <= 0:
            raise ValueError("packed metadata must declare a positive reid_dim")
        self.scalar_mean, self.scalar_std = normalizer_from_metadata(self.metadata)
        self.competition_enabled = bool(competition_enabled)

        sequences = list(self.metadata.get("train_sequences", []))
        if not sequences:
            raise ValueError(f"packed metadata lists no train_sequences: {metadata_path}")
        counts = self.metadata.get("timeline_counts", {})
        missing_counts = [seq for seq in sequences if seq not in counts]
        if missing_counts:
            raise ValueError(
                f"packed metadata is missing timeline_counts for {missing_counts[:3]}"
            )
        self.sequences = sequences
        self._sequence_counts = [
            int(counts[seq].get("labeled_endpoints", 0)) for seq in sequences
        ]
        if any(count <= 0 for count in self._sequence_counts):
            raise ValueError("packed dataset contains an empty sequence")
        self._boundaries: list[int] = []
        running = 0
        for count in self._sequence_counts:
            running += count
            self._boundaries.append(running)
        self.total_samples = running
        self._length = (
            min(self.total_samples, int(max_samples)) if int(max_samples) > 0 else self.total_samples
        )

        self._sequence_cache_size = max(1, int(sequence_cache_size))
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.sequence_load_count = 0
        # Worker processes each hold their own dataset copy; the trainer passes
        # a shared counter so loads in workers are counted too.
        self._shared_load_counter = None
        self._load_lock = threading.Lock()

    def set_shared_load_counter(self, counter: Any) -> None:
        self._shared_load_counter = counter

    def total_sequence_loads(self) -> int:
        shared = 0
        if self._shared_load_counter is not None:
            shared = int(self._shared_load_counter.value)
        return int(self.sequence_load_count) + shared

    @property
    def sample_boundaries(self) -> tuple[int, ...]:
        """Cumulative sample boundaries, one entry per sequence."""

        return tuple(self._boundaries)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def dataset_name(self) -> str:
        return str(self.metadata.get("dataset", "unknown"))

    @property
    def split(self) -> str:
        return str(self.metadata.get("split", "unknown"))

    def descriptor(self) -> dict[str, Any]:
        return {
            "format": PACKED_FORMAT,
            "dataset": self.dataset_name,
            "split": self.split,
            "dataset_dir": str(self.dataset_dir),
            "train_sequences": list(self.sequences),
            "context_size": CONTEXT_SIZE,
            "reid_dim": self.reid_dim,
            "scalar_dim": LEGACY_SCALAR_DIM,
            "evidence_dim": EVIDENCE_DIM,
            "competition_dim": COMPETITION_DIM,
            "competition_enabled": self.competition_enabled,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_source": "packed_train_data_safe_gate_target",
            "normalization_mean": self.scalar_mean.tolist(),
            "normalization_std": self.scalar_std.tolist(),
            "max_frame_gap": int(self.metadata.get("max_frame_gap", 30)),
        }

    # ------------------------------------------------------------------
    # Sequence cache
    # ------------------------------------------------------------------
    def _load_sequence(self, sequence_position: int) -> dict[str, Any]:
        with self._load_lock:
            cached = self._cache.pop(sequence_position, None)
            if cached is not None:
                self._cache[sequence_position] = cached
                return cached

            sequence = self.sequences[sequence_position]
            sequence_dir = self.dataset_dir / str(sequence)
            data_path = sequence_dir / "data.pt"
            if not data_path.is_file():
                raise FileNotFoundError(f"packed sequence data not found: {data_path}")
            if self._shared_load_counter is not None:
                with self._shared_load_counter.get_lock():
                    self._shared_load_counter.value += 1
            else:
                self.sequence_load_count += 1
            try:
                packed = torch.load(
                    data_path, map_location="cpu", weights_only=False, mmap=True
                )
            except (RuntimeError, TypeError, ValueError):
                packed = torch.load(data_path, map_location="cpu", weights_only=False)
            if not isinstance(packed, dict) or not isinstance(packed.get("arrays"), dict):
                raise ValueError(f"invalid packed sequence payload: {data_path}")
            arrays_raw = packed["arrays"]
            missing = [name for name in _REQUIRED_PACKED_ARRAYS if name not in arrays_raw]
            if missing:
                raise ValueError(f"packed sequence {sequence} missing arrays: {missing}")
            arrays: dict[str, np.ndarray] = {}
            for name in _REQUIRED_PACKED_ARRAYS:
                value = arrays_raw[name]
                arrays[name] = (
                    value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                )

            expected_samples = int(self._sequence_counts[sequence_position])
            if int(arrays["event_indices"].shape[0]) != expected_samples:
                raise ValueError(
                    f"packed sample count mismatch for {sequence}: "
                    f"{arrays['event_indices'].shape[0]} != {expected_samples}"
                )
            if int(arrays["event_indices"].shape[1]) != CONTEXT_SIZE:
                raise ValueError(
                    f"packed context mismatch for {sequence}: "
                    f"{arrays['event_indices'].shape[1]} != {CONTEXT_SIZE}"
                )
            timeline_count = int(arrays["timeline_scalar_feats"].shape[0])
            if arrays["timeline_scalar_feats"].shape[1] != LEGACY_SCALAR_DIM:
                raise ValueError(f"packed scalar dimension mismatch for {sequence}")

            reid_path = sequence_dir / str(
                self.metadata.get("reid_feature_file", "reid_features.npy")
            )
            if not reid_path.is_file():
                raise FileNotFoundError(f"packed ReID cache not found: {reid_path}")
            reid_features = np.load(reid_path, mmap_mode="r", allow_pickle=False)
            if reid_features.shape != (timeline_count, 2, self.reid_dim):
                raise ValueError(
                    f"packed ReID shape mismatch in {reid_path}: "
                    f"{reid_features.shape} != {(timeline_count, 2, self.reid_dim)}"
                )

            cached = {"arrays": arrays, "reid_features": reid_features, "packed": packed}
            self._cache[sequence_position] = cached
            while len(self._cache) > self._sequence_cache_size:
                _, evicted = self._cache.popitem(last=False)
                mapped = getattr(evicted.get("reid_features"), "_mmap", None)
                if mapped is not None:
                    mapped.close()
            return cached

    def close(self) -> None:
        for evicted in self._cache.values():
            mapped = getattr(evicted.get("reid_features"), "_mmap", None)
            if mapped is not None:
                mapped.close()
        self._cache.clear()

    def release_cached_pages(self) -> dict[str, int]:
        advised = 0
        for evicted in self._cache.values():
            mapped = getattr(evicted.get("reid_features"), "_mmap", None)
            if mapped is None:
                continue
            try:
                mapped.madvise(mmap.MADV_DONTNEED)
                advised += 1
            except (AttributeError, OSError, ValueError):
                pass
        return {"mmap_regions": advised, "files": 0}

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sample access
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._length)

    def _locate(self, index: int) -> tuple[int, int]:
        sequence_position = bisect.bisect_right(self._boundaries, index)
        previous = self._boundaries[sequence_position - 1] if sequence_position else 0
        return sequence_position, index - previous

    def _evidence_for_timeline(
        self,
        arrays: Mapping[str, np.ndarray],
        reid_features: np.ndarray,
        timeline_index: int,
        *,
        track_id: int,
        segment_id: int,
    ) -> EventEvidence:
        scalar = np.asarray(
            arrays["timeline_scalar_feats"][timeline_index], dtype=np.float64
        )
        if scalar.shape != (LEGACY_SCALAR_DIM,) or not np.isfinite(scalar).all():
            raise ValueError(
                f"non-finite or wrong-shape packed scalar at timeline index "
                f"{timeline_index}"
            )
        has_detection = bool(arrays["timeline_has_detection"][timeline_index])
        evidence = build_evidence(scalar, has_detection)
        competition = (
            build_competition(scalar)
            if self.competition_enabled
            else np.zeros(COMPETITION_DIM, dtype=np.float32)
        )
        track_feature = np.array(
            reid_features[timeline_index, 0, :], dtype=np.float32, copy=True
        )
        detection_feature = np.array(
            reid_features[timeline_index, 1, :], dtype=np.float32, copy=True
        )
        if not has_detection:
            detection_feature = np.zeros_like(track_feature)
        return EventEvidence(
            evidence=evidence,
            competition=competition,
            track_feature=track_feature,
            detection_feature=detection_feature,
            has_detection=has_detection,
            frame_id=int(timeline_index),
            track_id=int(track_id),
            segment_id=int(segment_id),
            observation_count=0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sequence_position, local_index = self._locate(index)
        cached = self._load_sequence(sequence_position)
        arrays = cached["arrays"]
        reid_features = cached["reid_features"]
        sequence = self.sequences[sequence_position]

        event_indices = np.asarray(arrays["event_indices"][local_index], dtype=np.int64)
        if event_indices.shape != (CONTEXT_SIZE,):
            raise ValueError(f"packed event window has wrong shape for {sequence}")
        current_index = int(event_indices[-1])
        if current_index < 0:
            raise ValueError(
                f"packed sample {sequence}:{local_index} has no current event"
            )
        track_id = int(arrays["track_ids"][local_index])
        segment_id = int(arrays["segment_ids"][local_index])

        history: list[EventEvidence] = []
        for position in range(CONTEXT_SIZE - 1):
            timeline_index = int(event_indices[position])
            if timeline_index < 0:
                continue
            history.append(
                self._evidence_for_timeline(
                    arrays,
                    reid_features,
                    timeline_index,
                    track_id=track_id,
                    segment_id=segment_id,
                )
            )
        current = self._evidence_for_timeline(
            arrays,
            reid_features,
            current_index,
            track_id=track_id,
            segment_id=segment_id,
        )
        if not current.has_detection:
            raise ValueError(
                f"packed supervised sample {sequence}:{local_index} points to an "
                "unmatched current event"
            )
        window: WindowArrays = build_window(history, current)

        safe_gate = np.asarray(
            arrays["safe_gate_target"][local_index], dtype=np.float32
        )
        valid = np.asarray(arrays["valid_channels"][local_index], dtype=np.bool_).reshape(2)
        weight = np.asarray(arrays["sample_weight"][local_index], dtype=np.float32).reshape(())
        current_scalar = np.asarray(
            arrays["timeline_scalar_feats"][current_index], dtype=np.float64
        )
        current_match_distance = torch.tensor(
            float(current_scalar[2]), dtype=torch.float32
        )

        return {
            "history_evidence": torch.from_numpy(window.history_evidence),
            "history_track": torch.from_numpy(window.history_track),
            "history_detection": torch.from_numpy(window.history_detection),
            "history_mask": torch.from_numpy(window.history_mask),
            "history_has_detection": torch.from_numpy(window.history_has_detection),
            "history_reset": torch.from_numpy(window.history_reset),
            "current_evidence": torch.from_numpy(window.current_evidence),
            "current_competition": torch.from_numpy(window.current_competition),
            "current_track": torch.from_numpy(window.current_track),
            "current_detection": torch.from_numpy(window.current_detection),
            "current_has_detection": torch.tensor(window.current_has_detection),
            "safe_gate_target": torch.from_numpy(safe_gate),
            "valid_channels": torch.from_numpy(valid),
            "sample_weight": torch.from_numpy(weight),
            "current_match_distance": current_match_distance,
            "sequence": sequence,
            "track_id": track_id,
            "segment_id": segment_id,
            "local_index": int(local_index),
        }

    def batch(self, indices: Sequence[int]) -> dict[str, Any]:
        """Vectorised gather for a whole batch of packed sample indices.

        Groups the batch by sequence, gathers the per-sample windows with
        numpy fancy indexing on the mmap-backed arrays and scatters the rows
        back into the original sample order (duplicates preserved).  The
        result is tensor-for-tensor identical to stacking ``__getitem__``.
        """

        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        count = int(index_array.size)
        if count == 0:
            raise ValueError("batch must contain at least one sample index")
        if int(index_array.min()) < 0 or int(index_array.max()) >= len(self):
            raise IndexError("batch contains out-of-range sample indices")
        boundaries = np.asarray(self._boundaries, dtype=np.int64)
        sequence_positions = np.searchsorted(boundaries, index_array, side="right")
        starts = np.zeros(count, dtype=np.int64)
        non_first = sequence_positions > 0
        starts[non_first] = boundaries[sequence_positions[non_first] - 1]
        local_indices = index_array - starts

        history = HISTORY_SIZE
        reid_dim = self.reid_dim
        out = {
            "history_evidence": np.zeros((count, history, EVIDENCE_DIM), np.float32),
            "history_track": np.zeros((count, history, reid_dim), np.float32),
            "history_detection": np.zeros((count, history, reid_dim), np.float32),
            "history_mask": np.ones((count, history), np.bool_),
            "history_has_detection": np.zeros((count, history), np.bool_),
            "history_reset": np.zeros((count, history), np.bool_),
            "current_evidence": np.zeros((count, EVIDENCE_DIM), np.float32),
            "current_competition": np.zeros((count, COMPETITION_DIM), np.float32),
            "current_track": np.zeros((count, reid_dim), np.float32),
            "current_detection": np.zeros((count, reid_dim), np.float32),
            "current_has_detection": np.zeros((count,), np.bool_),
            "safe_gate_target": np.zeros((count, 2), np.float32),
            "valid_channels": np.zeros((count, 2), np.bool_),
            "sample_weight": np.zeros((count,), np.float32),
            "current_match_distance": np.zeros((count,), np.float32),
        }
        track_ids = np.zeros(count, np.int64)
        segment_ids = np.zeros(count, np.int64)
        local_out = np.zeros(count, np.int64)
        sequences: list[str] = [""] * count

        for position in np.unique(sequence_positions):
            rows = np.nonzero(sequence_positions == position)[0]
            cached = self._load_sequence(int(position))
            arrays = cached["arrays"]
            reid_features = cached["reid_features"]
            sequence = self.sequences[int(position)]
            previous = int(boundaries[position - 1]) if position > 0 else 0
            local = local_indices[rows]
            event_indices = np.asarray(
                arrays["event_indices"][local], dtype=np.int64
            )

            def gather(timeline_position: int):
                timeline_index = event_indices[:, timeline_position]
                valid = timeline_index >= 0
                safe_index = np.where(valid, timeline_index, 0)
                scalar = np.asarray(
                    arrays["timeline_scalar_feats"][safe_index], dtype=np.float64
                )
                has_detection = (
                    np.asarray(
                        arrays["timeline_has_detection"][safe_index], dtype=np.bool_
                    )
                    & valid
                )
                track = np.asarray(
                    reid_features[safe_index, 0, :], dtype=np.float32
                )
                detection = np.asarray(
                    reid_features[safe_index, 1, :], dtype=np.float32
                )
                track[~valid] = 0.0
                detection[~has_detection] = 0.0
                evidence = np.concatenate(
                    [
                        scalar[:, EVIDENCE_SCALAR_INDICES],
                        has_detection[:, None].astype(np.float64),
                    ],
                    axis=1,
                ).astype(np.float32)
                competition = (
                    np.asarray(
                        scalar[:, COMPETITION_SCALAR_INDICES], dtype=np.float32
                    )
                    if self.competition_enabled
                    else np.zeros((safe_index.size, COMPETITION_DIM), np.float32)
                )
                match_distance = scalar[:, 2].astype(np.float32)
                evidence[~valid] = 0.0
                competition[~valid] = 0.0
                match_distance[~valid] = 0.0
                return (
                    evidence,
                    competition,
                    track,
                    detection,
                    has_detection,
                    valid,
                    match_distance,
                )

            for history_position in range(history):
                evidence, _, track, detection, has_detection, valid, _ = gather(
                    history_position
                )
                out["history_evidence"][rows, history_position] = evidence
                out["history_track"][rows, history_position] = track
                out["history_detection"][rows, history_position] = detection
                out["history_has_detection"][rows, history_position] = has_detection
                out["history_mask"][rows, history_position] = ~valid

            (
                current_evidence,
                current_competition,
                current_track,
                current_detection,
                current_has_detection,
                current_valid,
                current_match_distance,
            ) = gather(history)
            if not bool((current_valid & current_has_detection).all()):
                bad = int((~(current_valid & current_has_detection)).sum())
                raise ValueError(
                    f"{bad} packed samples in {sequence} point to a missing or "
                    "unmatched current event"
                )
            out["current_evidence"][rows] = current_evidence
            out["current_competition"][rows] = current_competition
            out["current_track"][rows] = current_track
            out["current_detection"][rows] = current_detection
            out["current_has_detection"][rows] = current_has_detection
            out["current_match_distance"][rows] = current_match_distance
            out["safe_gate_target"][rows] = np.asarray(
                arrays["safe_gate_target"][local], dtype=np.float32
            )
            out["valid_channels"][rows] = np.asarray(
                arrays["valid_channels"][local], dtype=np.bool_
            ).reshape(-1, 2)
            out["sample_weight"][rows] = np.asarray(
                arrays["sample_weight"][local], dtype=np.float32
            ).reshape(-1)
            track_ids[rows] = np.asarray(arrays["track_ids"][local], dtype=np.int64)
            segment_ids[rows] = np.asarray(
                arrays["segment_ids"][local], dtype=np.int64
            )
            local_out[rows] = local
            for row in rows.tolist():
                sequences[row] = str(sequence)

        result = {key: torch.from_numpy(value) for key, value in out.items()}
        result["track_id"] = [int(value) for value in track_ids]
        result["segment_id"] = [int(value) for value in segment_ids]
        result["sequence"] = sequences
        result["local_index"] = [int(value) for value in local_out]
        return result

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        tensor_keys = [key for key, value in batch[0].items() if torch.is_tensor(value)]
        result: dict[str, Any] = {
            key: torch.stack([item[key] for item in batch], dim=0)
            for key in tensor_keys
        }
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


__all__ = ["PackedShutrackDataset"]
