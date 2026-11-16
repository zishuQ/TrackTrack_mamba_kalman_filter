"""Read-only compact-cache adapter: evidence and competition columns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from shutrack.data.legacy_adapter import (
    evidence_from_cached_event,
    open_compact_event_reader,
)
from shutrack.schema import COMPETITION_SCALAR_INDICES, EVIDENCE_SCALAR_INDICES


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
    scalar[27:31] = [0.1, 0.2, 0.3, 0.4]
    scalar[39:43] = [1.0, 2.0, 3.0, 4.0]
    scalar[43:45] = [-1.0, -2.0]
    scalar[9] = 0.8
    scalar[59] = 0.1
    scalar[15] = 0.6
    scalar[21] = 0.5
    scalar[25] = 0.7
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


def test_cached_evidence_uses_normalized_evidence_and_competition_columns(tmp_path: Path):
    event_dir, detection_dir = _write_cache(tmp_path)
    reader = open_compact_event_reader(event_dir, detection_dir, max_cached_shards=2)
    try:
        record = reader.get_event_record(0, 0)
        scalar = np.asarray(record["scalar_features"], dtype=np.float64)
        evidence = evidence_from_cached_event(
            reader,
            record,
            scalar_mean=np.zeros(63),
            scalar_std=np.ones(63),
            competition_enabled=True,
            segment_id=0,
        )
        assert evidence.has_detection
        expected_evidence = [float(scalar[index]) for index in EVIDENCE_SCALAR_INDICES] + [1.0]
        np.testing.assert_allclose(evidence.evidence, expected_evidence, atol=1e-6)
        expected_competition = [float(scalar[index]) for index in COMPETITION_SCALAR_INDICES]
        np.testing.assert_allclose(
            evidence.competition, expected_competition, atol=1e-6
        )
        np.testing.assert_allclose(evidence.detection_feature, 0.5)
    finally:
        reader.close()


def test_cached_evidence_zeros_competition_when_branch_disabled(tmp_path: Path):
    event_dir, detection_dir = _write_cache(tmp_path)
    reader = open_compact_event_reader(event_dir, detection_dir)
    try:
        record = reader.get_event_record(0, 0)
        evidence = evidence_from_cached_event(
            reader,
            record,
            scalar_mean=np.zeros(63),
            scalar_std=np.ones(63),
            competition_enabled=False,
            segment_id=0,
        )
        assert evidence.competition.tolist() == [0.0, 0.0, 0.0]
    finally:
        reader.close()
