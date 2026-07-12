from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.data.compact_event_cache import CompactEventCacheSink
from agentguard.data.cache_reader import CompactEventCacheReader
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


def test_target_view_requires_target_indices(tmp_path):
    split = _load_split_module()
    split._write_sequence(
        tmp_path / "seq",
        "seq",
        {1: np.stack([_det(1.0)], axis=0)},
        compact_index=None,
        dataset="MOT17",
        split="all",
        source_pickle=tmp_path / "source.pkl",
    )
    cache = SequenceDetectionCache(tmp_path / "seq")
    try:
        import pytest

        with pytest.raises(FileNotFoundError):
            cache.get_frame(0, view="target")
    finally:
        cache.close()


def test_split_with_target_index_missing_sequence_fails(tmp_path):
    split = _load_split_module()
    import pytest

    with pytest.raises(RuntimeError, match="Target compact index does not contain sequence"):
        split._write_sequence(
            tmp_path / "seq",
            "seq",
            {1: np.stack([_det(1.0)], axis=0)},
            compact_index={"index": {}},
            dataset="MOT17",
            split="all",
            source_pickle=tmp_path / "source.pkl",
        )


def test_track_compact_snapshot_copies_recent_6_only():
    from types import SimpleNamespace

    from integrations.agentguard.adapter import AgentGuardTrackerAdapter
    from trackers.track import Track

    track = Track.__new__(Track)
    track.track_id = 1
    track.box = np.array([0, 1, 2, 3], dtype=np.float64)
    track.score = 0.9
    track.mean = np.ones(8, dtype=np.float64)
    track.covariance = np.eye(8, dtype=np.float64)
    track.velocity = np.zeros((4, 2), dtype=np.float64)
    track.feat = np.ones((1, 4), dtype=np.float64)
    track.end_frame_id = 10
    track.state = 1
    track.history = {
        i: [
            np.array([i, i, i + 1, i + 1], dtype=np.float64),
            0.1 * (i + 1),
            None,
            None,
            np.ones((1, 4)),
        ]
        for i in range(10)
    }
    snap = track.snapshot_state(compact_history=True)
    assert sorted(snap.history.keys()) == [4, 5, 6, 7, 8, 9]
    assert all(len(v) == 2 for v in snap.history.values())
    np.testing.assert_allclose(
        [snap.history[i][1] for i in range(4, 10)],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    compact = AgentGuardTrackerAdapter(
        SimpleNamespace(dataset="MOT17"),
        "seq",
    )._compact_snapshot_dict(snap)
    np.testing.assert_allclose(
        [compact["history"][i][1] for i in range(4, 10)],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )


def test_sequence_enumeration_from_detection_manifest(tmp_path, monkeypatch):
    from agentguard import cli

    root = tmp_path / "det"
    manifest_dir = root / "MOT17" / "all"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sequences": {
                    "MOT17-02-FRCNN": {},
                    "MOT17-02-SDP": {},
                    "MOT17-04-FRCNN": {},
                }
            }
        )
    )
    assert cli._resolve_detection_cache_sequences(
        "MOT17",
        "all",
        str(root),
        detector="FRCNN",
    ) == ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]

    train_dir = root / "MOT17" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sequences": {
                    "MOT17-02-FRCNN": {},
                    "MOT17-04-FRCNN": {},
                    "MOT17-99-FRCNN": {},
                }
            }
        )
    )
    monkeypatch.setattr(cli, "_resolve_sequences", lambda dataset, mode: ["MOT17-02"])
    assert cli._resolve_detection_cache_sequences(
        "MOT17",
        "train_custom",
        str(root),
        detector="FRCNN",
    ) == ["MOT17-02-FRCNN"]


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
                0.11 * (i + 1),
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
            "detection_source": np.array([[0, 1], [0, 1]], dtype=np.int8),
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
    np.testing.assert_allclose(
        states[0]["recent_history_scores"],
        [0.33, 0.44, 0.55, 0.66, 0.77, 0.88],
    )
    assert "detections" not in frames[0]
    assert events[0]["state_shard_id"] == 0
    assert events[0]["frame_start_state_offset"] == 0
    assert events[0]["pre_update_state_offset"] == 1
    assert events[0]["association_shard_id"] == 0
    assert events[0]["association_offset"] == 0
    assert frames[0]["frame_id"] == 1
    assert frames[0]["frame_index"] == 0
    assert "warp_matrix" in frames[0]


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
    assert manifest["schema_version"] == COMPACT_CACHE_SCHEMA_VERSION
    assert manifest["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256


def test_association_flushes_every_32_frames(tmp_path):
    sink = CompactEventCacheSink(
        cache_root=tmp_path,
        dataset="MOT17",
        split="all",
        sequence="seq",
        reid_dim=4,
        association_flush_size=32,
    )
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)
    for frame_id in range(64):
        fr = _frame_record()
        fr["frame_id"] = frame_id + 1
        sink.on_frame(fr, [])
    sink.on_sequence_end()
    seq_dir = tmp_path / "MOT17" / "all" / "seq"
    assert len(sorted(seq_dir.glob("associations_*.pt"))) == 2


def test_compact_reader_random_access_restores_event_state_detection_and_association(
    tmp_path,
    monkeypatch,
):
    split = _load_split_module()
    det_dir = tmp_path / "det" / "MOT17" / "all" / "seq"
    frames = {
        1: np.stack([_det(1.0), _det(2.0)], axis=0),
    }
    split._write_sequence(
        det_dir,
        "seq",
        frames,
        compact_index={"index": {"seq": {1: np.array([0, 1], dtype=np.int64)}}},
        dataset="MOT17",
        split="all",
        source_pickle=tmp_path / "source.pkl",
    )

    sink = CompactEventCacheSink(
        cache_root=tmp_path / "events",
        dataset="MOT17",
        split="all",
        sequence="seq",
        reid_dim=4,
        event_flush_size=1,
        frame_flush_size=1,
        association_flush_size=32,
        image_width=1920,
        image_height=1080,
    )
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)
    fr = _frame_record()
    fr["detections"][0]["detection_index"] = 0
    fr["detections"][1]["detection_index"] = 1
    fr["association"]["detection_indices"] = np.array([1, 0], dtype=np.int64)
    ev = _event()
    ev["accepted_detection_index"] = 1
    sink.on_frame(fr, [ev])
    sink.on_sequence_end()

    reader = CompactEventCacheReader(
        tmp_path / "events" / "MOT17" / "all" / "seq",
        det_dir,
    )
    try:
        import agentguard.data.cache_reader as cache_reader_module

        real_pairwise_iou = cache_reader_module.pairwise_iou_xyxy
        iou_input_shapes = []

        def recording_pairwise_iou(boxes_a, boxes_b):
            iou_input_shapes.append((boxes_a.shape, boxes_b.shape))
            return real_pairwise_iou(boxes_a, boxes_b)

        monkeypatch.setattr(
            cache_reader_module,
            "pairwise_iou_xyxy",
            recording_pairwise_iou,
        )
        record = next(reader.iter_event_records())
        event = reader.materialize_training_event(record)
        assert event.event_id == ev["event_id"]
        assert event.frame_start_state is not None
        assert event.pre_update_state is not None
        assert event.detection is not None
        assert event.detection.detection_index == 1
        assert event.detection.feature.reshape(-1).shape[0] == 4
        assert event.association_context is not None
        assert event.association_context.track_cost_row.shape == (2,)
        assert event.association_context.detection_cost_col.shape == (2,)
        assert event.association_context.detection_overlap_row.shape == (2,)
        assert event.association_context.accepted_detection_index == 0
        assert event.association_context.accepted_detection_column == 0
        from agentguard.features.geometry import pairwise_iou_xyxy

        association_boxes = np.stack(
            [reader.get_detection(1)["box"], reader.get_detection(0)["box"]],
            axis=0,
        )
        expected_overlap = pairwise_iou_xyxy(association_boxes, association_boxes)
        np.fill_diagonal(expected_overlap, 0.0)
        np.testing.assert_allclose(
            event.association_context.detection_overlap_row,
            expected_overlap[0],
            rtol=0.0,
            atol=1e-12,
        )
        assert iou_input_shapes == [((1, 4), (2, 4))]
        assert event.pre_update_state.history[7][1] == pytest.approx(0.88)
        from agentguard.features.scalar import compute_scalar_features

        scalar = compute_scalar_features(event)
        assert scalar[56] == pytest.approx(0.88)
        assert scalar[57] == pytest.approx((0.66 + 0.77 + 0.88) / 3.0)
        assert event.warp_matrix.shape == (2, 3)
    finally:
        reader.close()


def test_schema_v2_cache_is_not_reused_by_v3_sink(tmp_path):
    seq_dir = tmp_path / "MOT17" / "all" / "seq"
    seq_dir.mkdir(parents=True)
    (seq_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "complete": True,
                "feature_schema_sha256": "legacy",
            }
        )
    )
    (seq_dir / "legacy.marker").write_text("old")

    sink = CompactEventCacheSink(
        cache_root=tmp_path,
        dataset="MOT17",
        split="all",
        sequence="seq",
    )
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)

    assert sink._temp_dir is not None
    assert not seq_dir.exists()
    assert (tmp_path / "MOT17" / "all" / "seq.incomplete" / "manifest.json").is_file()


def test_reader_rejects_legacy_schema_by_default(tmp_path):
    seq_dir = tmp_path / "legacy"
    seq_dir.mkdir()
    (seq_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "feature_schema_sha256": "legacy"})
    )
    torch.save([], seq_dir / "events_00000.pt")
    torch.save([], seq_dir / "states_00000.pt")

    with pytest.raises(ValueError, match="current schema_version"):
        CompactEventCacheReader(seq_dir)

    reader = CompactEventCacheReader(seq_dir, allow_legacy_schema=True)
    reader.close()


def test_candidate_a_keeps_cached_scalar_and_alternative_recomputes(tmp_path):
    split = _load_split_module()
    det_dir = tmp_path / "det" / "MOT17" / "all" / "seq"
    split._write_sequence(
        det_dir,
        "seq",
        {1: np.stack([_det(1.0), _det(2.0)], axis=0)},
        compact_index={"index": {"seq": {1: np.array([0, 1], dtype=np.int64)}}},
        dataset="MOT17",
        split="all",
        source_pickle=tmp_path / "source.pkl",
    )
    sink = CompactEventCacheSink(tmp_path / "events", "MOT17", "all", "seq", reid_dim=4)
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)
    frame = _frame_record()
    frame["association"]["detection_indices"] = np.array([0, 1], dtype=np.int64)
    event_record = _event()
    event_record["accepted_detection_index"] = 1
    sink.on_frame(frame, [event_record])
    sink.on_sequence_end()

    reader = CompactEventCacheReader(
        tmp_path / "events" / "MOT17" / "all" / "seq",
        det_dir,
    )
    try:
        record = next(reader.iter_event_records())
        cached = np.asarray(record["scalar_features"], dtype=np.float64)
        candidate_a = reader.materialize_training_event(record)
        candidate_a_explicit = reader.materialize_training_event(
            record,
            candidate_detection_index=1,
        )
        alternative = reader.materialize_training_event(
            record,
            candidate_detection_index=0,
        )
        np.testing.assert_array_equal(candidate_a.scalar_features, cached)
        np.testing.assert_array_equal(candidate_a_explicit.scalar_features, cached)
        assert alternative.scalar_features is not None
        assert alternative.scalar_features.shape == (63,)
        assert not np.array_equal(alternative.scalar_features, cached)
    finally:
        reader.close()


def test_unmatched_event_does_not_build_association_context(tmp_path):
    split = _load_split_module()
    det_dir = tmp_path / "det" / "MOT17" / "all" / "seq"
    split._write_sequence(
        det_dir,
        "seq",
        {1: np.stack([_det(1.0), _det(2.0)], axis=0)},
        compact_index={"index": {"seq": {1: np.array([0, 1], dtype=np.int64)}}},
        dataset="MOT17",
        split="all",
        source_pickle=tmp_path / "source.pkl",
    )
    sink = CompactEventCacheSink(tmp_path / "events", "MOT17", "all", "seq", reid_dim=4)
    sink.on_sequence_start("seq", reid_dim=4, image_width=1920, image_height=1080)
    event_record = _event()
    event_record["has_detection"] = False
    event_record["accepted_detection_index"] = -1
    sink.on_frame(_frame_record(), [event_record])
    sink.on_sequence_end()

    reader = CompactEventCacheReader(
        tmp_path / "events" / "MOT17" / "all" / "seq",
        det_dir,
    )
    try:
        event = reader.materialize_training_event(next(reader.iter_event_records()))
        assert event.has_detection is False
        assert event.detection is None
        assert event.association is None
        assert event.association_context is None
    finally:
        reader.close()
