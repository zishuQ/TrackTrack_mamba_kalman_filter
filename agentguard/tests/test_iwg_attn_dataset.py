from __future__ import annotations

import json

import numpy as np
import torch

import agentguard.datasets.iwg_attn_dataset as iwg_attn_dataset
from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.datasets.iwg_attn_dataset import (
    COMPACT_INDEX_FORMAT,
    DANCETRACK_IWG_ATTN_DATASET_SCHEMA_SHA256,
    DANCETRACK_TRAIN_SEQUENCES,
    IWG_ATTN_DATASET_SCHEMA_SHA256,
    MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256,
    SPORTSMOT_IWG_ATTN_DATASET_SCHEMA_SHA256,
    SPORTSMOT_TRAINVAL_IWG_ATTN_DATASET_SCHEMA_SHA256,
    SPORTSMOT_TRAINVAL_SEQUENCES,
    SPORTSMOT_TRAIN_SEQUENCES,
    SPORTSMOT_VAL_SEQUENCES,
    StreamingIWGAttnDataset,
    build_streaming_sample_index,
    event_key,
    resolve_iwg_attn_dataset_spec,
    segment_track_timelines,
)


def _record(frame: int, *, track: int = 1, matched: bool = True, history: int | None = None):
    return {
        "sequence": "seq",
        "event_shard_id": 0,
        "event_offset": frame,
        "event_id": f"event-{frame}",
        "frame_id": frame,
        "track_id": track,
        "matched": matched,
        "history_count": frame if history is None else history,
    }


def test_streaming_index_includes_unmatched_history_without_crossing_segments():
    timeline = [
        _record(1),
        _record(2, matched=False),
        _record(3),
        _record(100, history=1),
        _record(101, matched=False, history=2),
        _record(102, history=3),
    ]
    segments = segment_track_timelines(timeline, max_frame_gap=30)
    labels = {
        event_key("seq", 0, 3): {},
        event_key("seq", 0, 102): {},
    }
    samples = build_streaming_sample_index(segments, labels)
    assert len(samples) == 2
    assert [event["frame_id"] for event in samples[0]["events"]] == [1, 2, 3]
    assert [event["matched"] for event in samples[0]["events"]] == [True, False, True]
    assert [event["frame_id"] for event in samples[1]["events"]] == [100, 101, 102]
    assert samples[0]["segment_id"] != samples[1]["segment_id"]
    for sample in samples:
        assert len({event["sequence"] for event in sample["events"]}) == 1
        assert len({event["track_id"] for event in sample["events"]}) == 1


def test_mot17_schema_hash_is_stable_and_mot20_is_separate():
    assert (
        IWG_ATTN_DATASET_SCHEMA_SHA256
        == "cf1d1fc4731a53902157ee3a44584e384e5b61453dc51b2796a6437782440158"
    )
    assert MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256 != IWG_ATTN_DATASET_SCHEMA_SHA256
    assert DANCETRACK_IWG_ATTN_DATASET_SCHEMA_SHA256 not in {
        IWG_ATTN_DATASET_SCHEMA_SHA256,
        MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256,
    }
    assert len(DANCETRACK_TRAIN_SEQUENCES) == 40
    assert len(set(DANCETRACK_TRAIN_SEQUENCES)) == 40
    assert SPORTSMOT_IWG_ATTN_DATASET_SCHEMA_SHA256 not in {
        IWG_ATTN_DATASET_SCHEMA_SHA256,
        MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256,
        DANCETRACK_IWG_ATTN_DATASET_SCHEMA_SHA256,
    }
    assert len(SPORTSMOT_TRAIN_SEQUENCES) == 45
    assert len(set(SPORTSMOT_TRAIN_SEQUENCES)) == 45


def test_sportsmot_trainval_schema_has_exact_disjoint_source_mapping():
    assert len(SPORTSMOT_VAL_SEQUENCES) == 45
    assert set(SPORTSMOT_TRAIN_SEQUENCES).isdisjoint(SPORTSMOT_VAL_SEQUENCES)
    assert len(SPORTSMOT_TRAINVAL_SEQUENCES) == 90
    sequences, schema, source_splits = resolve_iwg_attn_dataset_spec(
        "SportsMOT", "trainval"
    )
    assert sequences == SPORTSMOT_TRAINVAL_SEQUENCES
    assert schema == SPORTSMOT_TRAINVAL_IWG_ATTN_DATASET_SCHEMA_SHA256
    assert {source_splits[item] for item in SPORTSMOT_TRAIN_SEQUENCES} == {"train"}
    assert {source_splits[item] for item in SPORTSMOT_VAL_SEQUENCES} == {"val"}


def test_compact_reader_bounds_lru_shard_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "complete": True,
            }
        )
    )
    for shard in range(3):
        torch.save([{"value": shard}], cache_dir / f"events_{shard:05d}.pt")
    torch.save([{}], cache_dir / "states_00000.pt")
    reader = CompactEventCacheReader(cache_dir, max_cached_shards=2)
    try:
        assert reader.get_event_record(0, 0)["value"] == 0
        assert reader.get_event_record(1, 0)["value"] == 1
        assert reader.get_event_record(0, 0)["value"] == 0
        assert reader.get_event_record(2, 0)["value"] == 2
        assert list(reader._loaded) == [("events", 0), ("events", 2)]
    finally:
        reader.close()


def test_jsonl_dataset_reader_keeps_event_shards_resident(monkeypatch):
    calls = []

    class ReaderStub:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.max_cached_shards = kwargs["max_cached_shards"]

        def close(self):
            pass

    monkeypatch.setattr(iwg_attn_dataset, "CompactEventCacheReader", ReaderStub)
    dataset = object.__new__(StreamingIWGAttnDataset)
    dataset.metadata = {
        "event_cache_root": "/events",
        "detection_cache_root": "/detections",
        "dataset": "MOT17",
        "split": "train",
    }
    dataset._readers = {}

    reader = dataset._reader("MOT17-02-FRCNN")

    assert reader.max_cached_shards is None
    assert calls[0][1]["max_cached_shards"] is None


def test_compact_index_getitem_does_not_open_event_shard_reader():
    class DetectionReaderStub:
        def get_detection(self, index):
            assert index == 7
            return {"feature": np.asarray([0.25, 0.75], dtype=np.float32)}

    class NormStatsStub:
        @staticmethod
        def transform(value):
            return value

    dataset = object.__new__(StreamingIWGAttnDataset)
    dataset.index_format = COMPACT_INDEX_FORMAT
    dataset._length = 1
    dataset._compact_boundaries = [1]
    dataset.metadata = {"train_sequences": ["seq"], "reid_dim": 2}
    dataset.norm_stats = NormStatsStub()
    dataset._compact_indexes = {
        "seq": {
            "event_indices": np.asarray([[-1, -1, -1, -1, -1, 0]]),
            "event_shard_ids": np.asarray([[-1, -1, -1, -1, -1, 0]]),
            "event_offsets": np.asarray([[-1, -1, -1, -1, -1, 0]]),
            "safe_gate_target": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "policy_safe_soft_target": np.asarray([[0.5, 0.5]], dtype=np.float32),
            "cue_target": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "risk_target": np.asarray([[0.0, 0.0]], dtype=np.float32),
            "valid_channels": np.asarray([[True, True]]),
            "sample_weight": np.asarray([1.0], dtype=np.float32),
            "track_ids": np.asarray([3]),
            "segment_ids": np.asarray([4]),
            "timeline_track_feats": np.asarray([[0.1, 0.2]], dtype=np.float32),
            "timeline_scalar_feats": np.zeros((1, 63), dtype=np.float64),
            "timeline_has_detection": np.asarray([True]),
            "timeline_detection_indices": np.asarray([7]),
        }
    }
    dataset._reader = lambda sequence: (_ for _ in ()).throw(
        AssertionError("compact index must not open an event-shard reader")
    )
    dataset._detection_reader = lambda sequence: DetectionReaderStub()

    sample = dataset[0]

    assert sample["label_key"] == "seq|0|0"
    assert torch.equal(sample["det_feats"][-1], torch.tensor([0.25, 0.75]))
