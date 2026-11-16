"""Read-only compact-cache adapter: recorded candidate-pool margins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from shutrack.data.legacy_adapter import (
    evidence_from_cached_event,
    margins_from_cached_event,
    open_compact_event_reader,
)


def _write_cache(root: Path) -> tuple[Path, Path]:
    reid_dim = 8
    event_dir = root / "event_cache" / "Tiny" / "all" / "seq1"
    detection_dir = root / "detection_cache" / "Tiny" / "all" / "seq1"
    event_dir.mkdir(parents=True)
    detection_dir.mkdir(parents=True)

    (event_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "dataset": "Tiny",
                "split": "all",
                "sequence": "seq1",
                "num_events": 1,
                "num_frames": 1,
                "num_detections": 2,
                "reid_dim": reid_dim,
                "complete": True,
                "image_width": 640,
                "image_height": 480,
            }
        )
    )
    scalar = np.zeros(63, dtype=np.float32)
    scalar[list(range(12))] = np.arange(12, dtype=np.float32)
    scalar[27:31] = [0.1, 0.2, 0.3, 0.4]
    scalar[39:43] = [1.0, 2.0, 3.0, 4.0]
    scalar[43:47] = [-1.0, -2.0, -3.0, -4.0]
    scalar[9] = 0.8
    scalar[26] = 1.0
    scalar[58] = 0.2
    scalar[59] = 0.1
    record = {
        "event_shard_id": 0,
        "event_offset": 0,
        "event_id": "tiny/seq1/1/1",
        "frame_id": 1,
        "frame_index": 0,
        "track_id": 1,
        "matched": True,
        "accepted_detection_index": 1,
        "association_record_id": 0,
        "association_shard_id": 0,
        "association_offset": 0,
        "association_track_row": 0,
        "state_shard_id": 0,
        "pre_update_state_offset": 0,
        "frame_start_state_offset": 0,
        "scalar_features": scalar,
        "track_feature": np.ones(reid_dim, dtype=np.float32),
        "history_count": 3,
    }
    torch.save([record], event_dir / "events_00000.pt")
    torch.save([{}], event_dir / "states_00000.pt")
    association = {
        "track_ids": np.array([1, 2], dtype=np.int64),
        "detection_indices": np.array([0, 1], dtype=np.int64),
        "raw_cost": np.array([[0.9, 0.2], [0.1, 0.8]], dtype=np.float32),
        "final_cost": np.array([[0.9, 0.2], [0.1, 0.8]], dtype=np.float32),
        "iou_similarity": np.ones((2, 2), dtype=np.float32),
    }
    torch.save([association], event_dir / "associations_00000.pt")

    (detection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "Tiny",
                "split": "all",
                "sequence": "seq1",
                "num_frames": 1,
                "num_detections": 2,
                "reid_dim": reid_dim,
            }
        )
    )
    np.save(detection_dir / "frame_offsets.npy", np.array([0, 2], dtype=np.int64))
    np.save(detection_dir / "boxes.npy", np.full((2, 4), 10.0, dtype=np.float32))
    np.save(detection_dir / "scores.npy", np.array([0.7, 0.9], dtype=np.float32))
    np.save(
        detection_dir / "features.npy",
        np.stack(
            [
                np.ones(reid_dim, dtype=np.float32) * 0.25,
                np.ones(reid_dim, dtype=np.float32) * 0.5,
            ]
        ),
    )
    np.save(detection_dir / "sources.npy", np.zeros(2, dtype=np.int8))
    np.save(detection_dir / "class_ids.npy", np.ones(2, dtype=np.int16))
    return event_dir, detection_dir


def test_cached_margins_and_evidence_are_read_only(tmp_path: Path):
    event_dir, detection_dir = _write_cache(tmp_path)
    reader = open_compact_event_reader(event_dir, detection_dir, max_cached_shards=2)
    try:
        record = reader.get_event_record(0, 0)
        margins = margins_from_cached_event(reader, record)
        assert margins.row_valid and margins.col_valid
        assert margins.row_margin == pytest.approx(0.7, abs=1e-6)
        assert margins.col_margin == pytest.approx(0.6, abs=1e-6)

        evidence = evidence_from_cached_event(
            reader,
            record,
            scalar_mean=np.zeros(63),
            scalar_std=np.ones(63),
            margin_stats={
                "mean": [0.0, 0.0],
                "std": [1.0, 1.0],
            },
            segment_id=0,
        )
        assert evidence.has_detection
        assert evidence.context[0] == pytest.approx(0.8)
        assert evidence.context[1] == pytest.approx(0.7)
        assert evidence.context[2] == pytest.approx(0.6)
        assert evidence.context[3] == pytest.approx(0.2)
        assert evidence.context[4] == pytest.approx(0.1)
        assert evidence.context[5] == pytest.approx(1.0)
        assert evidence.context[6] == pytest.approx(1.0)
        assert evidence.context[7] == pytest.approx(1.0)
        np.testing.assert_allclose(evidence.motion[:4], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(evidence.motion[4:8], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(evidence.motion[8:], [-1.0, -2.0, -3.0, -4.0])
        np.testing.assert_allclose(evidence.detection_feature, 0.5)
    finally:
        reader.close()


def test_missing_association_yields_invalid_margins(tmp_path: Path):
    event_dir, detection_dir = _write_cache(tmp_path)
    reader = open_compact_event_reader(event_dir, detection_dir)
    try:
        record = dict(reader.get_event_record(0, 0))
        record["association_shard_id"] = -1
        record["association_offset"] = -1
        margins = margins_from_cached_event(reader, record)
        assert not margins.row_valid and not margins.col_valid
    finally:
        reader.close()
