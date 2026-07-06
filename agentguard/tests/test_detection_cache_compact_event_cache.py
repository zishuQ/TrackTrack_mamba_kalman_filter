from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from agentguard.data.compact_event_cache import CompactEventCacheSink
from agentguard.data.detection_cache import SequenceDetectionCache


def _load_split_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "agentguard" / "00_split_detection_cache.py"
    spec = importlib.util.spec_from_file_location("split_detection_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _det(x: float, score: float = 0.9, dim: int = 4) -> np.ndarray:
    feat = np.arange(dim, dtype=np.float32) + x
    return np.concatenate(
        [
            np.array([x, x + 1, x + 10, x + 20, score, 1.0], dtype=np.float32),
            feat,
        ]
    )


def test_split_pickle_by_sequence_preserves_empty_frame_offsets(tmp_path):
    split = _load_split_module()
    frames = {
        2: np.stack([_det(2.0)], axis=0),
        4: np.stack([_det(4.0), _det(5.0)], axis=0),
    }
    manifest = split._write_sequence(
        tmp_path / "MOT17-XX-FRCNN",
        "MOT17-XX-FRCNN",
        frames,
        compact_index=None,
        dataset="MOT17",
        split="all",
        source_pickle=tmp_path / "source.pkl",
    )

    assert manifest["complete"] is True
    offsets = np.load(tmp_path / "MOT17-XX-FRCNN" / "frame_offsets.npy")
    assert offsets.tolist() == [0, 0, 1, 1, 3]


def test_detection_cache_frame_slice_and_mmap(tmp_path):
    split = _load_split_module()
    frames = {
        1: np.stack([_det(1.0), _det(2.0)], axis=0),
        2: np.stack([_det(3.0)], axis=0),
    }
    compact_index = {
        "index": {
            "seq": {
                1: np.array([1], dtype=np.int64),
                2: np.array([0], dtype=np.int64),
            }
        }
    }
    split._write_sequence(
        tmp_path / "seq",
        "seq",
        frames,
        compact_index=compact_index,
        dataset="MOT17",
        split="all",
        source_pickle=tmp_path / "source.pkl",
    )

    cache = SequenceDetectionCache(tmp_path / "seq")
    try:
        assert isinstance(cache.features, np.memmap)
        source = cache.get_frame(0, view="source")
        target = cache.get_frame(0, view="target")
        assert source["detection_indices"].tolist() == [0, 1]
        assert target["detection_indices"].tolist() == [1]
        assert target["boxes"].shape == (1, 4)
        arr = cache.get_frame_array(1, view="target")
        assert arr.shape == (1, 10)
    finally:
        cache.close()


def _state(history_len: int = 8) -> dict:
    return {
        "track_id": 7,
        "box": np.array([1, 2, 3, 4], dtype=np.float32),
        "score": 0.8,
        "mean": np.ones(8, dtype=np.float32),
        "covariance": np.eye(8, dtype=np.float32),
        "velocity": np.zeros((4, 2), dtype=np.float32),
        "feature": np.ones((1, 4), dtype=np.float32),
        "history": {
            i: [
                np.array([i, i + 1, i + 2, i + 3], dtype=np.float32),
                0.7,
                np.ones(8, dtype=np.float32),
                np.eye(8, dtype=np.float32),
                np.ones((1, 4), dtype=np.float32),
            ]
            for i in range(history_len)
        },
        "end_frame_id": 6,
        "state": 1,
    }


def _event(event_id: str = "MOT17/seq/000001/000007") -> dict:
    return {
        "event_id": event_id,
        "frame_id": 1,
        "track_id": 7,
        "has_detection": True,
        "accepted_detection_index": 42,
        "frame_start_state": _state(),
        "pre_update_state": _state(),
        "detection": {
            "detection_index": 42,
            "box": np.array([1, 2, 3, 4], dtype=np.float32),
            "score": 0.9,
            "feature": np.ones(4, dtype=np.float32),
        },
        "scalar_features": np.ones(63, dtype=np.float32),
        "track_feature": np.ones(4, dtype=np.float32),
    }


def _frame_record() -> dict:
    return {
        "frame_id": 1,
        "image_width": 1920,
        "image_height": 1080,
        "warp_matrix": np.eye(2, 3, dtype=np.float32),
        "detections": [
            {"index": 0, "detection_index": 42, "score": 0.9, "source": 0},
            {"index": 1, "detection_index": 43, "score": 0.8, "source": 1},
        ],
        "association": {
            "track_ids": np.array([7, 8], dtype=np.int64),
            "detection_indices": np.array([42, 43], dtype=np.int64),
            "raw_cost": np.ones((2, 2), dtype=np.float32),
            "final_cost": np.ones((2, 2), dtype=np.float32) * 0.5,
            "iou_similarity": np.ones((2, 2), dtype=np.float32) * 0.7,
            "iou_distance": np.ones((2, 2), dtype=np.float32) * 0.3,
            "cosine_distance": np.zeros((2, 2), dtype=np.float32),
            "confidence_distance": np.zeros((2, 2), dtype=np.float32),
            "angle_distance": np.zeros((2, 2), dtype=np.float32),
            "assignment_round": np.zeros((2, 2), dtype=np.int16),
            "assignment_threshold": np.ones((2, 2), dtype=np.float32) * 0.5,
            "reid_available": True,
        },
    }


def test_compact_event_does_not_copy_detection_feature_or_full_history(tmp_path):
    sink = CompactEventCacheSink(
        cache_root=tmp_path,
        dataset="MOT17",
        split="all",
        sequence="seq",
        reid_dim=4,
        event_flush_size=1,
        frame_flush_size=1,
    )
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)
    sink.on_frame(_frame_record(), [_event()])
    sink.on_sequence_end()

    seq_dir = tmp_path / "MOT17" / "all" / "seq"
    events = torch.load(seq_dir / "events_00000.pt", weights_only=False)
    states = torch.load(seq_dir / "states_00000.pt", weights_only=False)
    frames = torch.load(seq_dir / "frames_00000.pt", weights_only=False)

    assert "detection" not in events[0]
    assert "detection_feature" not in events[0]
    assert events[0]["accepted_detection_index"] == 42
    assert events[0]["association_track_row"] == 0
    assert "feature" not in states[0]
    assert "history" not in states[0]
    assert states[0]["history_count"] == 8
    assert states[0]["recent_history_boxes"].shape[0] == 6
    assert "detections" not in frames[0]


def test_association_saved_once_per_frame_and_manifest_counts(tmp_path):
    sink = CompactEventCacheSink(
        cache_root=tmp_path,
        dataset="MOT17",
        split="all",
        sequence="seq",
        reid_dim=4,
        event_flush_size=2,
        frame_flush_size=2,
    )
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)
    sink.on_frame(_frame_record(), [_event("e1"), _event("e2")])
    assert sink._events == []
    sink.on_sequence_end()

    seq_dir = tmp_path / "MOT17" / "all" / "seq"
    associations = torch.load(seq_dir / "associations_00000.pt", weights_only=False)
    assert len(associations) == 1
    assert associations[0]["final_cost"].shape == (2, 2)
    assert associations[0]["detection_indices"].tolist() == [42, 43]

    with (seq_dir / "manifest.json").open("r") as f:
        manifest = json.load(f)
    assert manifest["complete"] is True
    assert manifest["num_events"] == 2
    assert manifest["num_association_records"] == 1
