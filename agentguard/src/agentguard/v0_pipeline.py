from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.label_schema import validate_rollout_label
from agentguard.features.builder import EventFeatureBuilder


SAMPLE_TYPE_TO_ID = {
    "A": 0,
    "matched": 0,
    "unmatched": 3,
}


def load_label_records(label_dir: str | os.PathLike[str], max_samples: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(Path(label_dir).glob("*_labels.json")):
        with path.open("r") as f:
            seq_records = json.load(f)
        for index, record in enumerate(seq_records):
            try:
                validate_rollout_label(record)
            except ValueError as exc:
                raise ValueError(f"invalid rollout label {path}:{index}: {exc}") from exc
            records.append(record)
            if max_samples > 0 and len(records) >= max_samples:
                return records
    return records


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_norm_stats_from_records(
    records: List[Dict[str, Any]],
    event_cache_root: str | os.PathLike[str],
    detection_cache_root: str | os.PathLike[str],
    dataset: str,
    split: str,
):
    from agentguard.features.normalization import NormalizationStats

    readers: Dict[str, CompactEventCacheReader] = {}
    scalars: List[np.ndarray] = []
    try:
        for record in records:
            seq = record["sequence"]
            if seq not in readers:
                readers[seq] = CompactEventCacheReader(
                    Path(event_cache_root) / dataset / split / seq,
                    Path(detection_cache_root) / dataset / split / seq,
                )
            event_record = readers[seq].get_event_record(
                int(record["event_shard_id"]),
                int(record["event_offset"]),
            )
            event = readers[seq].materialize_training_event(event_record)
            if event.scalar_features is not None:
                scalar = np.asarray(event.scalar_features, dtype=np.float64).reshape(-1)
                if scalar.size == 63:
                    scalars.append(scalar)
    finally:
        for reader in readers.values():
            reader.close()
    stats = NormalizationStats()
    stats.fit(scalars)
    return stats


def student_v0_record_key(record: Dict[str, Any]) -> str:
    """Return a stable key for one Student-V0 index record."""
    return "|".join(
        [
            str(record.get("event_id", "")),
            str(record.get("candidate_type", "A")),
            str(record.get("candidate_detection_index", -1)),
        ]
    )


class StudentV0IWGOutputCache:
    """Small on-disk cache of frozen IWG outputs keyed by index records."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.keys = np.asarray(data["keys"]).astype(str)
            self.policy_probs = np.asarray(data["policy_probs"], dtype=np.float32)
            self.gates = np.asarray(data["gates"], dtype=np.float32)
        if self.policy_probs.shape != (len(self.keys), 5):
            raise ValueError(f"Invalid policy_probs shape in {self.path}: {self.policy_probs.shape}")
        if self.gates.shape != (len(self.keys), 2):
            raise ValueError(f"Invalid gates shape in {self.path}: {self.gates.shape}")
        self._indices = {key: idx for idx, key in enumerate(self.keys)}

    def get(self, record: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        key = student_v0_record_key(record)
        idx = self._indices.get(key)
        if idx is None:
            raise KeyError(f"Record key is missing from IWG output cache: {key}")
        return self.policy_probs[idx].astype(np.float64), self.gates[idx].astype(np.float64)


class _CompactDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        event_cache_root: str | os.PathLike[str],
        detection_cache_root: str | os.PathLike[str],
        dataset: str,
        split: str,
        feature_builder: EventFeatureBuilder,
        iwg_output_cache: Optional[StudentV0IWGOutputCache] = None,
    ) -> None:
        for index, record in enumerate(records):
            try:
                validate_rollout_label(record)
            except ValueError as exc:
                raise ValueError(f"invalid rollout label at record {index}: {exc}") from exc
        self.records = records
        self.event_cache_root = Path(event_cache_root)
        self.detection_cache_root = Path(detection_cache_root)
        self.dataset = dataset
        self.split = split
        self.feature_builder = feature_builder
        self.iwg_output_cache = iwg_output_cache
        self._readers: Dict[str, CompactEventCacheReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def _reader(self, sequence: str) -> CompactEventCacheReader:
        if sequence not in self._readers:
            self._readers[sequence] = CompactEventCacheReader(
                self.event_cache_root / self.dataset / self.split / sequence,
                self.detection_cache_root / self.dataset / self.split / sequence,
            )
        return self._readers[sequence]

    def _event(self, record: Dict[str, Any]):
        reader = self._reader(record["sequence"])
        event_record = reader.get_event_record(
            int(record["event_shard_id"]),
            int(record["event_offset"]),
        )
        candidate_detection_index = int(record.get("candidate_detection_index", -1))
        candidate_type = str(record.get("candidate_type", "A"))
        if candidate_detection_index >= 0 and (
            candidate_type != "A"
            or candidate_detection_index != int(event_record.get("accepted_detection_index", -1))
        ):
            event = reader.materialize_training_event(
                event_record,
                candidate_detection_index=candidate_detection_index,
            )
        else:
            event = reader.materialize_training_event(event_record)
        if self.iwg_output_cache is not None:
            policy_probs, gate = self.iwg_output_cache.get(record)
            event.iwg_gate = gate
            event.iwg_policy_probs = policy_probs
        else:
            event.iwg_gate = np.array(
                [record.get("motion_soft_target", 1.0), record.get("appearance_soft_target", 1.0)],
                dtype=np.float64,
            )
            event.iwg_policy_probs = np.asarray(
                record.get("policy_soft_target", [0.2] * 5),
                dtype=np.float64,
            )
        return event

    @staticmethod
    def _targets(label: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        validate_rollout_label(label)
        sample_type = label.get("candidate_type") or label.get("sample_type", "matched")
        return {
            "motion_target": torch.tensor(float(label["motion_soft_target"]), dtype=torch.float),
            "appearance_target": torch.tensor(float(label["appearance_soft_target"]), dtype=torch.float),
            "motion_soft_target": torch.tensor(float(label["motion_soft_target"]), dtype=torch.float),
            "appearance_soft_target": torch.tensor(float(label["appearance_soft_target"]), dtype=torch.float),
            "motion_safe_target": torch.tensor(float(label["motion_safe_target"]), dtype=torch.float),
            "appearance_safe_target": torch.tensor(float(label["appearance_safe_target"]), dtype=torch.float),
            "motion_label_confidence": torch.tensor(float(label["motion_label_confidence"]), dtype=torch.float),
            "appearance_label_confidence": torch.tensor(float(label["appearance_label_confidence"]), dtype=torch.float),
            "cue_target": torch.tensor(label["cue_target"], dtype=torch.float),
            "risk_target": torch.tensor(label["risk_targets"], dtype=torch.float),
            "policy_soft_target": torch.tensor(label["policy_soft_target"], dtype=torch.float),
            "policy_safe_soft_target": torch.tensor(label["policy_safe_soft_target"], dtype=torch.float),
            "sample_type": torch.tensor(SAMPLE_TYPE_TO_ID.get(str(sample_type), 0), dtype=torch.long),
            "valid_motion": torch.tensor(bool(label["valid_motion"]), dtype=torch.bool),
            "valid_appearance": torch.tensor(bool(label["valid_appearance"]), dtype=torch.bool),
            "sample_weight": torch.tensor(float(label["sample_weight"]), dtype=torch.float),
        }


class CompactV0IWGDataset(_CompactDatasetBase):
    def __init__(self, *args, max_history: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_history = int(max_history)
        self._track_index: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
        for idx, record in enumerate(self.records):
            self._track_index[(record["sequence"], int(record["track_id"]))].append(
                (int(record["frame_id"]), idx)
            )
        for entries in self._track_index.values():
            entries.sort(key=lambda item: item[0])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        entries = self._track_index[(record["sequence"], int(record["track_id"]))]
        pos = next((i for i, (_, rec_idx) in enumerate(entries) if rec_idx == idx), -1)
        prev_indices: List[int] = []
        if pos >= 0:
            prev_indices = [rec_idx for _, rec_idx in entries[max(0, pos - self.max_history):pos]]
        seq_records = [self.records[i] for i in prev_indices] + [record]
        events = [self._event(r) for r in seq_records]
        seq_len = self.max_history + 1
        if len(events) < seq_len:
            events = [None] * (seq_len - len(events)) + events
        inputs = self.feature_builder.build_iwg_input(events)
        return {
            "track_feats": inputs["track_feats"].squeeze(0),
            "det_feats": inputs["det_feats"].squeeze(0),
            "scalar_feats": inputs["scalar_feats"].squeeze(0),
            "mask": inputs["mask"].squeeze(0),
            "targets": self._targets(record),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from agentguard.datasets.iwg_dataset import IWGDataset

        return IWGDataset.collate_fn(batch)


class CompactV0TGRDataset(_CompactDatasetBase):
    def __init__(self, *args, window_size: int = 4, window_stride: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_size = int(window_size)
        self.window_stride = max(int(window_stride), 1)
        self._windows = self._build_windows()

    def _build_windows(self) -> List[List[int]]:
        groups: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
        for idx, record in enumerate(self.records):
            groups[(record["sequence"], int(record["track_id"]))].append((int(record["frame_id"]), idx))
        windows: List[List[int]] = []
        for entries in groups.values():
            entries.sort(key=lambda item: item[0])
            for i in range(0, max(0, len(entries) - self.window_size + 1), self.window_stride):
                chunk = entries[i:i + self.window_size]
                if len(chunk) != self.window_size:
                    continue
                if all(chunk[j + 1][0] - chunk[j][0] <= 1 for j in range(self.window_size - 1)):
                    windows.append([idx for _, idx in chunk])
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self._windows[idx]
        records = [self.records[i] for i in indices]
        events = [self._event(r) for r in records]
        inputs = self.feature_builder.build_tgr_batch_input([events])
        target_gate = np.asarray([r.get("target_gate", [1.0, 1.0]) for r in records], dtype=np.float64)
        sample_weight = np.asarray([r.get("sample_weight", 1.0) for r in records], dtype=np.float64)
        valid_motion = np.asarray([r.get("valid_motion", True) for r in records], dtype=bool)
        valid_appearance = np.asarray([r.get("valid_appearance", True) for r in records], dtype=bool)
        return {
            "track_feats": inputs["track_feats"].squeeze(0),
            "det_feats": inputs["det_feats"].squeeze(0),
            "scalar_feats": inputs["scalar_feats"].squeeze(0),
            "iwg_policy_probs": inputs["iwg_policy_probs"].squeeze(0),
            "iwg_gates": inputs["iwg_gates"].squeeze(0),
            "has_detection_mask": inputs["has_detection_mask"].squeeze(0),
            "target_gate": torch.from_numpy(target_gate).float(),
            "sample_weight": torch.from_numpy(sample_weight).float(),
            "valid_motion": torch.from_numpy(valid_motion),
            "valid_appearance": torch.from_numpy(valid_appearance),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from agentguard.datasets.tgr_dataset import TGRDataset

        stacked = TGRDataset.collate_fn(batch)
        stacked["valid_motion"] = torch.stack([b["valid_motion"] for b in batch])
        stacked["valid_appearance"] = torch.stack([b["valid_appearance"] for b in batch])
        return stacked
