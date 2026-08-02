from __future__ import annotations

import json

import numpy as np
import torch

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.datasets.iwg_rg_cma_dataset import (
    COMPACT_INDEX_FORMAT,
    DANCETRACK_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    DANCETRACK_TRAIN_SEQUENCES,
    IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    LEGACY_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT,
    MOT20_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    SPORTSMOT_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    SPORTSMOT_TRAINVAL_SEQUENCES,
    SPORTSMOT_TRAIN_SEQUENCES,
    SPORTSMOT_VAL_SEQUENCES,
    StreamingIWGRGCMADataset,
    resolve_iwg_rg_cma_dataset_spec,
)


def test_mot17_schema_hash_is_stable_and_mot20_is_separate():
    assert (
        IWG_RG_CMA_DATASET_SCHEMA_SHA256
        == "7b6d571293275f03c04686b5a88b2525a8fc2022111d12ca91bbb91aa94e9367"
    )
    assert (
        "cf1d1fc4731a53902157ee3a44584e384e5b61453dc51b2796a6437782440158"
        in LEGACY_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT[("MOT17", "all", 6)]
    )
    assert MOT20_IWG_RG_CMA_DATASET_SCHEMA_SHA256 != IWG_RG_CMA_DATASET_SCHEMA_SHA256
    assert DANCETRACK_IWG_RG_CMA_DATASET_SCHEMA_SHA256 not in {
        IWG_RG_CMA_DATASET_SCHEMA_SHA256,
        MOT20_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    }
    assert len(DANCETRACK_TRAIN_SEQUENCES) == 40
    assert len(set(DANCETRACK_TRAIN_SEQUENCES)) == 40
    assert SPORTSMOT_IWG_RG_CMA_DATASET_SCHEMA_SHA256 not in {
        IWG_RG_CMA_DATASET_SCHEMA_SHA256,
        MOT20_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
        DANCETRACK_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    }
    assert len(SPORTSMOT_TRAIN_SEQUENCES) == 45
    assert len(set(SPORTSMOT_TRAIN_SEQUENCES)) == 45


def test_pre_rename_dataset_hashes_remain_explicitly_loadable():
    assert {
        "cf1d1fc4731a53902157ee3a44584e384e5b61453dc51b2796a6437782440158",
        "3321c5bba6d06512614247d79e36faa483ffa0feb939c6a8d4b07427c5fd6709",
        "e0ac984bee62c819fe84b7b25c652e4af9d020b60b52fe1f49b9c7dee3214175",
        "8135595698deb753202a6a884419838e19d6fc624a4e6829810e6f41e42fe574",
    }.issubset(
        {
            schema
            for schemas in LEGACY_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT.values()
            for schema in schemas
        }
    )


def test_sportsmot_trainval_schema_has_exact_disjoint_source_mapping():
    assert len(SPORTSMOT_VAL_SEQUENCES) == 45
    assert set(SPORTSMOT_TRAIN_SEQUENCES).isdisjoint(SPORTSMOT_VAL_SEQUENCES)
    assert len(SPORTSMOT_TRAINVAL_SEQUENCES) == 90
    sequences, schema, source_splits = resolve_iwg_rg_cma_dataset_spec(
        "SportsMOT", "trainval"
    )
    assert sequences == SPORTSMOT_TRAINVAL_SEQUENCES
    assert schema == SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256
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


def test_compact_index_getitem_does_not_open_event_shard_reader():
    class DetectionReaderStub:
        def get_detection(self, index):
            assert index == 7
            return {"feature": np.asarray([0.25, 0.75], dtype=np.float32)}

    class NormStatsStub:
        @staticmethod
        def transform(value):
            return value

    dataset = object.__new__(StreamingIWGRGCMADataset)
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
            "oracle_gate_target": np.asarray([[0.25, 0.75]], dtype=np.float32),
            "gate_confidence": np.asarray([[0.8, 0.9]], dtype=np.float32),
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
