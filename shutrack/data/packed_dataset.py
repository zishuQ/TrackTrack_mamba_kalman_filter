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
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from shutrack.data.evidence import EventEvidence
from shutrack.data.legacy_adapter import normalizer_from_metadata
from shutrack.features.core import build_competition, build_evidence
from shutrack.features.windows import WindowArrays, build_window
from shutrack.schema import (
    COMPETITION_DIM,
    CONTEXT_SIZE,
    EVIDENCE_DIM,
    FEATURE_SCHEMA_VERSION,
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
        cached = self._cache.pop(sequence_position, None)
        if cached is not None:
            self._cache[sequence_position] = cached
            return cached

        sequence = self.sequences[sequence_position]
        sequence_dir = self.dataset_dir / str(sequence)
        data_path = sequence_dir / "data.pt"
        if not data_path.is_file():
            raise FileNotFoundError(f"packed sequence data not found: {data_path}")
        self.sequence_load_count += 1
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

        cached = {"arrays": arrays, "reid_features": reid_features}
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
            "sequence": sequence,
            "track_id": track_id,
            "segment_id": segment_id,
            "local_index": int(local_index),
        }

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
