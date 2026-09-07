from __future__ import annotations

import hashlib
import json
import mmap
import tempfile
from bisect import bisect_right
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.compact_iwg_labels import load_compact_label_arrays
from agentguard.data.detection_cache import SequenceDetectionCache
from agentguard.data.label_schema import ROLLOUT_LABEL_SCHEMA_SHA256
from agentguard.datasets.timeline_utils import IWG_CONTEXT_SIZE, _fit_train_normalization
from agentguard.features.normalization import NormalizationStats


IWG_RG_CMA_DATASET_SCHEMA_VERSION = 1
MOT17_FRCNN_ALL_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]
MOT20_ALL_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
DANCETRACK_TRAIN_SEQUENCES = [
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0006",
    "dancetrack0008",
    "dancetrack0012",
    "dancetrack0015",
    "dancetrack0016",
    "dancetrack0020",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0027",
    "dancetrack0029",
    "dancetrack0032",
    "dancetrack0033",
    "dancetrack0037",
    "dancetrack0039",
    "dancetrack0044",
    "dancetrack0045",
    "dancetrack0049",
    "dancetrack0051",
    "dancetrack0052",
    "dancetrack0053",
    "dancetrack0055",
    "dancetrack0057",
    "dancetrack0061",
    "dancetrack0062",
    "dancetrack0066",
    "dancetrack0068",
    "dancetrack0069",
    "dancetrack0072",
    "dancetrack0074",
    "dancetrack0075",
    "dancetrack0080",
    "dancetrack0082",
    "dancetrack0083",
    "dancetrack0086",
    "dancetrack0087",
    "dancetrack0096",
    "dancetrack0098",
    "dancetrack0099",
]
SPORTSMOT_TRAIN_SEQUENCES = [
    "v_-6Os86HzwCs_c001",
    "v_-6Os86HzwCs_c003",
    "v_-6Os86HzwCs_c007",
    "v_-6Os86HzwCs_c009",
    "v_1LwtoLPw2TU_c006",
    "v_1LwtoLPw2TU_c012",
    "v_1LwtoLPw2TU_c014",
    "v_1LwtoLPw2TU_c016",
    "v_1yHWGw8DH4A_c029",
    "v_1yHWGw8DH4A_c047",
    "v_1yHWGw8DH4A_c077",
    "v_1yHWGw8DH4A_c601",
    "v_1yHWGw8DH4A_c609",
    "v_1yHWGw8DH4A_c610",
    "v_2j7kLB-vEEk_c001",
    "v_2j7kLB-vEEk_c002",
    "v_2j7kLB-vEEk_c005",
    "v_2j7kLB-vEEk_c007",
    "v_2j7kLB-vEEk_c009",
    "v_2j7kLB-vEEk_c010",
    "v_4LXTUim5anY_c002",
    "v_4LXTUim5anY_c003",
    "v_4LXTUim5anY_c010",
    "v_4LXTUim5anY_c012",
    "v_4LXTUim5anY_c013",
    "v_ApPxnw_Jffg_c001",
    "v_ApPxnw_Jffg_c002",
    "v_ApPxnw_Jffg_c009",
    "v_ApPxnw_Jffg_c015",
    "v_ApPxnw_Jffg_c016",
    "v_CW0mQbgYIF4_c004",
    "v_CW0mQbgYIF4_c005",
    "v_CW0mQbgYIF4_c006",
    "v_Dk3EpDDa3o0_c002",
    "v_Dk3EpDDa3o0_c007",
    "v_HdiyOtliFiw_c003",
    "v_HdiyOtliFiw_c004",
    "v_HdiyOtliFiw_c008",
    "v_HdiyOtliFiw_c010",
    "v_HdiyOtliFiw_c602",
    "v_dChHNGIfm4Y_c003",
    "v_gQNyhv8y0QY_c003",
    "v_gQNyhv8y0QY_c012",
    "v_gQNyhv8y0QY_c013",
    "v_iIxMOsCGH58_c013",
]
SPORTSMOT_VAL_SEQUENCES = [
    "v_00HRwkvvjtQ_c001",
    "v_00HRwkvvjtQ_c003",
    "v_00HRwkvvjtQ_c005",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c008",
    "v_00HRwkvvjtQ_c011",
    "v_0kUtTtmLaJA_c004",
    "v_0kUtTtmLaJA_c005",
    "v_0kUtTtmLaJA_c006",
    "v_0kUtTtmLaJA_c007",
    "v_0kUtTtmLaJA_c008",
    "v_0kUtTtmLaJA_c010",
    "v_2QhNRucNC7E_c017",
    "v_4-EmEtrturE_c009",
    "v_4r8QL_wglzQ_c001",
    "v_5ekaksddqrc_c001",
    "v_5ekaksddqrc_c002",
    "v_5ekaksddqrc_c003",
    "v_5ekaksddqrc_c004",
    "v_5ekaksddqrc_c005",
    "v_9MHDmAMxO5I_c002",
    "v_9MHDmAMxO5I_c003",
    "v_9MHDmAMxO5I_c004",
    "v_9MHDmAMxO5I_c006",
    "v_9MHDmAMxO5I_c009",
    "v_BgwzTUxJaeU_c008",
    "v_BgwzTUxJaeU_c012",
    "v_BgwzTUxJaeU_c014",
    "v_G-vNjfx1GGc_c004",
    "v_G-vNjfx1GGc_c008",
    "v_G-vNjfx1GGc_c600",
    "v_G-vNjfx1GGc_c601",
    "v_ITo3sCnpw_k_c007",
    "v_ITo3sCnpw_k_c010",
    "v_ITo3sCnpw_k_c011",
    "v_ITo3sCnpw_k_c012",
    "v_cC2mHWqMcjk_c007",
    "v_cC2mHWqMcjk_c008",
    "v_cC2mHWqMcjk_c009",
    "v_dw7LOz17Omg_c053",
    "v_dw7LOz17Omg_c067",
    "v_i2_L4qquVg0_c006",
    "v_i2_L4qquVg0_c007",
    "v_i2_L4qquVg0_c009",
    "v_i2_L4qquVg0_c010",
]
SPORTSMOT_TRAINVAL_SEQUENCES = SPORTSMOT_TRAIN_SEQUENCES + SPORTSMOT_VAL_SEQUENCES
IWG_RG_CMA_TRAIN_ALL_SEQUENCES = {
    "MOT17": MOT17_FRCNN_ALL_SEQUENCES,
    "MOT20": MOT20_ALL_SEQUENCES,
    "DanceTrack": DANCETRACK_TRAIN_SEQUENCES,
    "SportsMOT": SPORTSMOT_TRAIN_SEQUENCES,
}
IWG_RG_CMA_TRAIN_SPLIT_BY_DATASET = {
    "MOT17": "all",
    "MOT20": "all",
    "DanceTrack": "train",
    "SportsMOT": "train",
}
IWG_RG_CMA_DATASET_DESCRIPTOR = {
    "name": "agentguard_iwg_rg_cma_streaming_windows",
    "version": IWG_RG_CMA_DATASET_SCHEMA_VERSION,
    "timeline": "complete_compact_cache_matched_and_unmatched",
    "sample": "each_candidate_a_labeled_matched_endpoint",
    "history": "five_prior_segment_local_events_plus_endpoint",
    "context_size": IWG_CONTEXT_SIZE,
    "split_policy": "mot17_all_frcnn_train_all",
    "unmatched_history": True,
    "unmatched_supervision": False,
    "index_format": "compact_memmap_v1",
    "label_join": "native_current_timeline_event_key",
    "scalar_dim": 63,
    "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
}
IWG_RG_CMA_DATASET_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        IWG_RG_CMA_DATASET_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
MOT20_IWG_RG_CMA_DATASET_DESCRIPTOR = {
    **IWG_RG_CMA_DATASET_DESCRIPTOR,
    "split_policy": "mot20_all_train_all",
    "index_format": "compact_memmap_v1",
    "label_join": "native_current_timeline_event_key",
}
MOT20_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        MOT20_IWG_RG_CMA_DATASET_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
DANCETRACK_IWG_RG_CMA_DATASET_DESCRIPTOR = {
    **IWG_RG_CMA_DATASET_DESCRIPTOR,
    "split_policy": "dancetrack_official_train",
    "index_format": "compact_memmap_v1",
    "label_join": "native_current_timeline_event_key",
}
DANCETRACK_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        DANCETRACK_IWG_RG_CMA_DATASET_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
SPORTSMOT_IWG_RG_CMA_DATASET_DESCRIPTOR = {
    **IWG_RG_CMA_DATASET_DESCRIPTOR,
    "split_policy": "sportsmot_official_train",
    "index_format": "compact_memmap_v1",
    "label_join": "native_current_timeline_event_key",
}
SPORTSMOT_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        SPORTSMOT_IWG_RG_CMA_DATASET_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_DESCRIPTOR = {
    **IWG_RG_CMA_DATASET_DESCRIPTOR,
    "split_policy": "sportsmot_official_train_plus_val",
    "index_format": "compact_memmap_v1",
    "label_join": "native_current_timeline_event_key",
    "source_layout": "per_sequence_official_split_mapping",
}
SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET = {
    "MOT17": IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    "MOT20": MOT20_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    "DanceTrack": DANCETRACK_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    "SportsMOT": SPORTSMOT_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
}
SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = frozenset(
    {
        *IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET.values(),
        SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    }
)
SUPPORTED_IWG_CONTEXT_SIZES = frozenset({1, 2, 4, 6, 8, 10})


def _validate_context_size(context_size: int) -> int:
    context_size = int(context_size)
    if context_size not in SUPPORTED_IWG_CONTEXT_SIZES:
        raise ValueError(
            "unsupported IWG RG-CMA context_size: "
            f"{context_size}; expected one of {sorted(SUPPORTED_IWG_CONTEXT_SIZES)}"
        )
    return context_size


def _contextual_schema_sha256(descriptor: dict[str, Any], context_size: int) -> str:
    contextual = dict(descriptor)
    contextual["context_size"] = int(context_size)
    contextual["history"] = (
        f"{int(context_size) - 1}_prior_segment_local_events_plus_endpoint"
    )
    return hashlib.sha256(
        json.dumps(
            contextual,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


IWG_RG_CMA_CONTEXT8_DATASET_SCHEMA_SHA256_BY_DATASET = {
    dataset: _contextual_schema_sha256(
        {
            "MOT17": IWG_RG_CMA_DATASET_DESCRIPTOR,
            "MOT20": MOT20_IWG_RG_CMA_DATASET_DESCRIPTOR,
            "DanceTrack": DANCETRACK_IWG_RG_CMA_DATASET_DESCRIPTOR,
            "SportsMOT": SPORTSMOT_IWG_RG_CMA_DATASET_DESCRIPTOR,
        }[dataset],
        8,
    )
    for dataset in IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET
}
SPORTSMOT_TRAINVAL_CONTEXT8_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = (
    _contextual_schema_sha256(SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_DESCRIPTOR, 8)
)
_IWG_RG_CMA_DATASET_DESCRIPTORS = {
    "MOT17": IWG_RG_CMA_DATASET_DESCRIPTOR,
    "MOT20": MOT20_IWG_RG_CMA_DATASET_DESCRIPTOR,
    "DanceTrack": DANCETRACK_IWG_RG_CMA_DATASET_DESCRIPTOR,
    "SportsMOT": SPORTSMOT_IWG_RG_CMA_DATASET_DESCRIPTOR,
}
IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT = {
    (dataset, context_size): (
        IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET[dataset]
        if context_size == IWG_CONTEXT_SIZE
        else _contextual_schema_sha256(descriptor, context_size)
    )
    for dataset, descriptor in _IWG_RG_CMA_DATASET_DESCRIPTORS.items()
    for context_size in SUPPORTED_IWG_CONTEXT_SIZES
}
SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_CONTEXT = {
    context_size: (
        SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256
        if context_size == IWG_CONTEXT_SIZE
        else _contextual_schema_sha256(
            SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_DESCRIPTOR,
            context_size,
        )
    )
    for context_size in SUPPORTED_IWG_CONTEXT_SIZES
}
SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = frozenset(
    {
        *SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
        *IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT.values(),
        *SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_CONTEXT.values(),
    }
)

# These hashes identify datasets written before the public IWG/RG-CMA rename.
# They remain loadable, while every newly built dataset uses the descriptors
# above and therefore writes only the current schema hash.
LEGACY_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT = {
    ("MOT17", "all", 6): frozenset(
        {"cf1d1fc4731a53902157ee3a44584e384e5b61453dc51b2796a6437782440158"}
    ),
    ("MOT17", "all", 8): frozenset(
        {"a92a7c4146f5ebbf563777314fc8c3cb76043cb094b13f75ffaac9e5241f4357"}
    ),
    ("MOT20", "all", 6): frozenset(
        {"3321c5bba6d06512614247d79e36faa483ffa0feb939c6a8d4b07427c5fd6709"}
    ),
    ("MOT20", "all", 8): frozenset(
        {"55f2e221dc43c3a0e261ba5925aeb0c9db206182e6a3046cb6bcb3b401db6136"}
    ),
    ("DanceTrack", "train", 6): frozenset(
        {"e0ac984bee62c819fe84b7b25c652e4af9d020b60b52fe1f49b9c7dee3214175"}
    ),
    ("DanceTrack", "train", 8): frozenset(
        {"f2a7cc928b6e1e7060ac53d373926f085a264ffda30749df1b9eb006a9e669c5"}
    ),
    ("SportsMOT", "train", 6): frozenset(
        {"17cee34dee8024a1472c97f2843e1769a70bb7321722509fb320ee3ff3654d56"}
    ),
    ("SportsMOT", "trainval", 6): frozenset(
        {"8135595698deb753202a6a884419838e19d6fc624a4e6829810e6f41e42fe574"}
    ),
    ("SportsMOT", "train", 8): frozenset(
        {"d083bd7f35caa9f3e20971bd882289337103def8008adf0dc596552c17e60a1d"}
    ),
    ("SportsMOT", "trainval", 8): frozenset(
        {"e5e6c13eb135492dac5c4bb2990c6cfadf6c0100e5ab9d97e0a32783a263f93b"}
    ),
}
SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = frozenset(
    {
        *SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
        *(
            schema_sha256
            for schemas in LEGACY_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT.values()
            for schema_sha256 in schemas
        ),
    }
)


def resolve_iwg_rg_cma_dataset_spec(
    dataset: str,
    split: str,
    context_size: int = IWG_CONTEXT_SIZE,
) -> tuple[list[str], str, dict[str, str]]:
    context_size = _validate_context_size(context_size)
    if dataset == "SportsMOT" and split == "trainval":
        source_splits = {
            **{sequence: "train" for sequence in SPORTSMOT_TRAIN_SEQUENCES},
            **{sequence: "val" for sequence in SPORTSMOT_VAL_SEQUENCES},
        }
        return (
            list(SPORTSMOT_TRAINVAL_SEQUENCES),
            SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_CONTEXT[
                context_size
            ],
            source_splits,
        )
    expected_split = IWG_RG_CMA_TRAIN_SPLIT_BY_DATASET.get(dataset)
    if expected_split is None or split != expected_split:
        supported = [
            *(f"{name}/{value}" for name, value in IWG_RG_CMA_TRAIN_SPLIT_BY_DATASET.items()),
            "SportsMOT/trainval",
        ]
        raise ValueError(f"IWG RG-CMA supports {', '.join(supported)}")
    sequences = list(IWG_RG_CMA_TRAIN_ALL_SEQUENCES[dataset])
    dataset_schema_sha256 = IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT[
        (dataset, context_size)
    ]
    return (
        sequences,
        dataset_schema_sha256,
        {sequence: split for sequence in sequences},
    )


def accepted_iwg_rg_cma_dataset_schema_sha256(
    dataset: str,
    split: str,
    context_size: int,
) -> frozenset[str]:
    """Return current and pre-rename schema hashes for one dataset contract."""
    _sequences, current, _source_splits = resolve_iwg_rg_cma_dataset_spec(
        dataset, split, context_size
    )
    return frozenset(
        {
            current,
            *LEGACY_DATASET_SCHEMA_SHA256_BY_DATASET_CONTEXT.get(
                (dataset, split, int(context_size)), frozenset()
            ),
        }
    )
COMPACT_INDEX_FORMAT = "compact_memmap_v1"
PACKED_TRAIN_DATA_FORMAT = "train_data"
COMPACT_SAMPLE_ARRAY_FILES_V1 = {
    "event_indices": "event_indices.npy",
    "event_shard_ids": "event_shard_ids.npy",
    "event_offsets": "event_offsets.npy",
    "track_ids": "track_ids.npy",
    "segment_ids": "segment_ids.npy",
    "safe_gate_target": "safe_gate_target.npy",
    "policy_safe_soft_target": "policy_safe_soft_target.npy",
    "cue_target": "cue_target.npy",
    "risk_target": "risk_target.npy",
    "valid_channels": "valid_channels.npy",
    "sample_weight": "sample_weight.npy",
}
COMPACT_RESIDUAL_TARGET_ARRAY_FILES = {
    "oracle_gate_target": "oracle_gate_target.npy",
    "gate_confidence": "gate_confidence.npy",
}
COMPACT_SAMPLE_ARRAY_FILES = {
    **COMPACT_SAMPLE_ARRAY_FILES_V1,
    **COMPACT_RESIDUAL_TARGET_ARRAY_FILES,
}
COMPACT_TIMELINE_ARRAY_FILES = {
    "timeline_track_feats": "timeline_track_feats.npy",
    "timeline_scalar_feats": "timeline_scalar_feats.npy",
    "timeline_has_detection": "timeline_has_detection.npy",
    "timeline_detection_indices": "timeline_detection_indices.npy",
}
COMPACT_INDEX_ARRAY_FILES = {
    **COMPACT_SAMPLE_ARRAY_FILES_V1,
    **COMPACT_TIMELINE_ARRAY_FILES,
}


def _fit_sequence_source_normalization(
    event_cache_root: Path,
    dataset: str,
    sequences: list[str],
    source_splits: dict[str, str],
) -> NormalizationStats:
    count = 0
    total = np.zeros(63, dtype=np.float64)
    total_sq = np.zeros(63, dtype=np.float64)
    for sequence in sequences:
        reader = CompactEventCacheReader(
            event_cache_root / dataset / source_splits[sequence] / sequence
        )
        try:
            for record in reader.iter_event_records():
                scalar = np.asarray(record.get("scalar_features", []), dtype=np.float64)
                if scalar.shape != (63,) or not np.isfinite(scalar).all():
                    raise ValueError(
                        f"invalid scalar feature in {sequence} shard="
                        f"{record['event_shard_id']} offset={record['event_offset']}"
                    )
                count += 1
                total += scalar
                total_sq += scalar * scalar
        finally:
            reader.close()
    if count == 0:
        raise RuntimeError("No train timeline events available for normalization")
    stats = NormalizationStats()
    stats.mean = total / count
    variance = np.maximum(total_sq / count - stats.mean * stats.mean, 0.0)
    stats.std = np.sqrt(variance)
    stats.std[stats.std < 1e-8] = 1.0
    return stats


def _write_compact_index_arrays(
    index_dir: Path,
    arrays: dict[str, np.ndarray],
    filenames: dict[str, str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    filenames = filenames or COMPACT_SAMPLE_ARRAY_FILES_V1
    for name, value in arrays.items():
        if name not in filenames:
            continue
        filename = filenames[name]
        path = index_dir / filename
        np.save(path, value, allow_pickle=False)
        paths.append(path)
    return paths


def _build_compact_sequence_index(
    *,
    sequence: str,
    reader: CompactEventCacheReader,
    compact_label_dir: Path,
    index_dir: Path,
    max_frame_gap: int,
    segment_offset: int,
    context_size: int,
) -> tuple[dict[str, int], int, list[Path]]:
    context_size = _validate_context_size(context_size)
    label_manifest, labels = load_compact_label_arrays(compact_label_dir)
    label_keys = labels["event_keys"]
    count = int(label_manifest["retained_labels"])
    timeline_size = int(reader.manifest["num_events"])
    reid_dim = int(reader.manifest["reid_dim"])

    # Preserve the legacy deterministic segment numbering while keeping the
    # actual index construction streaming. The old builder ordered tracks by
    # track_id before numbering their local segments; event-cache order is not
    # guaranteed to have that same ordering.
    segment_counts: dict[int, int] = {}
    previous_by_track: dict[int, tuple[int, int]] = {}
    for record in reader.iter_event_records():
        track_id = int(record["track_id"])
        frame_id = int(record["frame_id"])
        history_count = int(record.get("history_count", 0))
        previous = previous_by_track.get(track_id)
        reset = previous is None
        if previous is not None:
            previous_frame, previous_history_count = previous
            gap = frame_id - previous_frame
            reset = (
                gap <= 0
                or gap > int(max_frame_gap)
                or history_count < previous_history_count
            )
        if reset:
            segment_counts[track_id] = segment_counts.get(track_id, 0) + 1
        previous_by_track[track_id] = (frame_id, history_count)
    segment_bases: dict[int, int] = {}
    next_segment = int(segment_offset)
    for track_id in sorted(segment_counts):
        segment_bases[track_id] = next_segment
        next_segment += segment_counts[track_id]

    index_dir.mkdir(parents=True, exist_ok=False)
    timeline_track = np.lib.format.open_memmap(
        index_dir / COMPACT_TIMELINE_ARRAY_FILES["timeline_track_feats"],
        mode="w+",
        dtype=np.float32,
        shape=(timeline_size, reid_dim),
    )
    timeline_scalar = np.lib.format.open_memmap(
        index_dir / COMPACT_TIMELINE_ARRAY_FILES["timeline_scalar_feats"],
        mode="w+",
        dtype=np.float32,
        shape=(timeline_size, 63),
    )
    timeline_has_detection = np.lib.format.open_memmap(
        index_dir / COMPACT_TIMELINE_ARRAY_FILES["timeline_has_detection"],
        mode="w+",
        dtype=np.bool_,
        shape=(timeline_size,),
    )
    timeline_detection_indices = np.lib.format.open_memmap(
        index_dir / COMPACT_TIMELINE_ARRAY_FILES["timeline_detection_indices"],
        mode="w+",
        dtype=np.int64,
        shape=(timeline_size,),
    )
    event_indices = np.full((count, context_size), -1, dtype=np.int32)
    event_shards = np.full((count, context_size), -1, dtype=np.int32)
    event_offsets = np.full((count, context_size), -1, dtype=np.int32)
    track_ids = np.zeros(count, dtype=np.int64)
    segment_ids = np.zeros(count, dtype=np.int64)

    histories: dict[int, dict[str, Any]] = {}
    segment_positions: dict[int, int] = {}
    label_position = 0
    events = 0
    matched = 0
    unmatched = 0
    unmatched_history = 0
    for timeline_index, record in enumerate(reader.iter_event_records()):
        events += 1
        is_matched = bool(record.get("matched", False))
        track_feature = np.asarray(record.get("track_feature", []), dtype=np.float32)
        scalar_feature = np.asarray(record.get("scalar_features", []), dtype=np.float32)
        if track_feature.shape != (reid_dim,) or scalar_feature.shape != (63,):
            raise ValueError(f"invalid compact timeline feature in {sequence}")
        if not np.isfinite(track_feature).all() or not np.isfinite(scalar_feature).all():
            raise ValueError(f"non-finite compact timeline feature in {sequence}")
        timeline_track[timeline_index] = track_feature
        timeline_scalar[timeline_index] = scalar_feature
        timeline_has_detection[timeline_index] = is_matched
        timeline_detection_indices[timeline_index] = int(
            record.get("accepted_detection_index", -1)
        )
        matched += int(is_matched)
        unmatched += int(not is_matched)
        track_id = int(record["track_id"])
        frame_id = int(record["frame_id"])
        history_count = int(record.get("history_count", 0))
        state = histories.get(track_id)
        reset = state is None
        if state is not None:
            gap = frame_id - int(state["frame_id"])
            reset = (
                gap <= 0
                or gap > int(max_frame_gap)
                or history_count < int(state["history_count"])
            )
        if reset:
            segment_position = segment_positions.get(track_id, 0)
            state = {
                "frame_id": frame_id,
                "history_count": history_count,
                "segment_id": segment_bases[track_id] + segment_position,
                "events": deque(maxlen=context_size),
            }
            histories[track_id] = state
            segment_positions[track_id] = segment_position + 1
        state["frame_id"] = frame_id
        state["history_count"] = history_count
        shard_id = int(record["event_shard_id"])
        offset = int(record["event_offset"])
        state["events"].append((timeline_index, shard_id, offset, is_matched))

        packed_key = (shard_id << 32) | offset
        if label_position >= count or packed_key != int(label_keys[label_position]):
            continue
        if not is_matched:
            raise ValueError(f"supervised compact label points to unmatched event: {sequence}")
        history = list(state["events"])
        pad_left = context_size - len(history)
        for position, (
            history_index,
            history_shard,
            history_offset,
            _history_matched,
        ) in enumerate(history, start=pad_left):
            event_indices[label_position, position] = history_index
            event_shards[label_position, position] = history_shard
            event_offsets[label_position, position] = history_offset
        unmatched_history += int(any(not item[3] for item in history[:-1]))
        track_ids[label_position] = track_id
        segment_ids[label_position] = int(state["segment_id"])
        label_position += 1
    if label_position != count:
        next_key = int(label_keys[label_position]) if label_position < count else None
        raise ValueError(
            f"only joined {label_position}/{count} compact labels for {sequence}; "
            f"next_key={next_key}"
        )
    if events != timeline_size:
        raise ValueError(
            f"timeline size mismatch for {sequence}: {events} != {timeline_size}"
        )
    for timeline in (
        timeline_track,
        timeline_scalar,
        timeline_has_detection,
        timeline_detection_indices,
    ):
        timeline.flush()

    arrays = {
        "event_indices": event_indices,
        "event_shard_ids": event_shards,
        "event_offsets": event_offsets,
        "track_ids": track_ids,
        "segment_ids": segment_ids,
        "safe_gate_target": np.asarray(labels["safe_gate_target"]),
        "policy_safe_soft_target": np.asarray(labels["policy_safe_soft_target"]),
        "cue_target": np.asarray(labels["cue_target"]),
        "risk_target": np.asarray(labels["risk_target"]),
        "valid_channels": np.asarray(labels["valid_channels"]),
        "sample_weight": np.asarray(labels["sample_weight"]),
    }
    sample_array_files = dict(COMPACT_SAMPLE_ARRAY_FILES_V1)
    if "oracle_gate_target" in labels and "gate_confidence" in labels:
        arrays["oracle_gate_target"] = np.asarray(labels["oracle_gate_target"])
        arrays["gate_confidence"] = np.asarray(labels["gate_confidence"])
        sample_array_files.update(COMPACT_RESIDUAL_TARGET_ARRAY_FILES)
    paths = _write_compact_index_arrays(
        index_dir,
        arrays,
        filenames=sample_array_files,
    )
    paths.extend(
        index_dir / filename for filename in COMPACT_TIMELINE_ARRAY_FILES.values()
    )
    return (
        {
            "events": events,
            "matched": matched,
            "unmatched": unmatched,
            "segments": next_segment - int(segment_offset),
            "labeled_endpoints": count,
            "samples_with_unmatched_history": unmatched_history,
        },
        next_segment,
        paths,
    )


def build_iwg_rg_cma_dataset(
    *,
    dataset: str,
    split: str,
    event_cache_root: str | Path,
    detection_cache_root: str | Path | None = None,
    label_dir: str | Path,
    output_dir: str | Path,
    max_frame_gap: int = 30,
    context_size: int = IWG_CONTEXT_SIZE,
    sequence_subset: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build AgentGuard records plus an independent event-aligned ReID cache."""
    context_size = _validate_context_size(context_size)
    sequences, _schema_id, source_splits = resolve_iwg_rg_cma_dataset_spec(
        dataset, split, context_size
    )
    if sequence_subset is not None:
        requested = [str(item) for item in sequence_subset]
        unknown = sorted(set(requested).difference(sequences))
        if unknown:
            raise ValueError(f"unsupported {dataset} sequence(s): {unknown}")
        sequences = [sequence for sequence in sequences if sequence in requested]
        source_splits = {sequence: source_splits[sequence] for sequence in sequences}
    if not sequences:
        raise ValueError(f"no sequences selected for {dataset}")
    event_cache_root = Path(event_cache_root).resolve()
    if detection_cache_root is None or not str(detection_cache_root):
        raise ValueError(
            "detection_cache_root is required to preserve event-aligned ReID features"
        )
    detection_cache_root = Path(detection_cache_root).resolve()
    label_dir = Path(label_dir).resolve()
    final_output_dir = Path(output_dir).resolve()
    if final_output_dir.exists() and any(final_output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite train data: {final_output_dir}")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = (
        _fit_train_normalization(event_cache_root, dataset, split, sequences)
        if len(set(source_splits.values())) == 1
        else _fit_sequence_source_normalization(
            event_cache_root, dataset, sequences, source_splits
        )
    )
    timeline_counts: dict[str, dict[str, int]] = {}
    reid_dims: set[int] = set()
    segment_offset = 0

    def as_tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(value, copy=True))

    for sequence in sequences:
        source_split = source_splits[sequence]
        cache_dir = event_cache_root / dataset / source_split / sequence
        detection_dir = detection_cache_root / dataset / source_split / sequence
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"event cache not found: {cache_dir}")
        if not detection_dir.is_dir():
            raise FileNotFoundError(f"detection cache not found: {detection_dir}")

        with tempfile.TemporaryDirectory(prefix="agentguard-train-data-") as tmp:
            index_dir = Path(tmp) / sequence
            index_dir.parent.mkdir(parents=True, exist_ok=True)
            reader = CompactEventCacheReader(cache_dir, detection_dir)
            try:
                if not bool(reader.manifest.get("complete", False)):
                    raise ValueError(f"event cache is incomplete: {cache_dir}")
                reid_dim = int(reader.manifest["reid_dim"])
                reid_dims.add(reid_dim)
                counts, segment_offset, _paths = _build_compact_sequence_index(
                    sequence=sequence,
                    reader=reader,
                    compact_label_dir=label_dir / sequence,
                    index_dir=index_dir,
                    max_frame_gap=max_frame_gap,
                    segment_offset=segment_offset,
                    context_size=context_size,
                )
                arrays = {
                    name: np.load(path, mmap_mode="r", allow_pickle=False)
                    for name, path in (
                        ("event_indices", index_dir / "event_indices.npy"),
                        ("track_ids", index_dir / "track_ids.npy"),
                        ("segment_ids", index_dir / "segment_ids.npy"),
                        ("safe_gate_target", index_dir / "safe_gate_target.npy"),
                        ("policy_safe_soft_target", index_dir / "policy_safe_soft_target.npy"),
                        ("cue_target", index_dir / "cue_target.npy"),
                        ("risk_target", index_dir / "risk_target.npy"),
                        ("valid_channels", index_dir / "valid_channels.npy"),
                        ("sample_weight", index_dir / "sample_weight.npy"),
                        ("timeline_track_feats", index_dir / "timeline_track_feats.npy"),
                        ("timeline_scalar_feats", index_dir / "timeline_scalar_feats.npy"),
                        ("timeline_has_detection", index_dir / "timeline_has_detection.npy"),
                        ("timeline_detection_indices", index_dir / "timeline_detection_indices.npy"),
                    )
                }
                for optional in ("oracle_gate_target", "gate_confidence"):
                    path = index_dir / f"{optional}.npy"
                    if path.is_file():
                        arrays[optional] = np.load(path, mmap_mode="r", allow_pickle=False)
                normalized_scalar = norm_stats.transform(
                    np.asarray(arrays["timeline_scalar_feats"], dtype=np.float64)
                ).astype(np.float32)
                timeline_count = int(arrays["timeline_track_feats"].shape[0])
                reid_features = np.zeros(
                    (timeline_count, 2, reid_dim), dtype=np.float32
                )
                reid_features[:, 0, :] = np.asarray(
                    arrays["timeline_track_feats"], dtype=np.float32
                )
                matched = np.asarray(arrays["timeline_has_detection"], dtype=bool)
                detection_indices = np.asarray(
                    arrays["timeline_detection_indices"], dtype=np.int64
                )
                valid_detection = matched & (detection_indices >= 0)
                if valid_detection.any():
                    reid_features[valid_detection, 1, :] = np.asarray(
                        reader.detection_cache.features[
                            detection_indices[valid_detection]
                        ],
                        dtype=np.float32,
                    )
                sequence_metadata = {
                    "format": PACKED_TRAIN_DATA_FORMAT,
                    "dataset": dataset,
                    "split": split,
                    "source_split": source_split,
                    "sequence": sequence,
                    "train_sequences": sequences,
                    "context_size": context_size,
                    "max_frame_gap": int(max_frame_gap),
                    "reid_dim": reid_dim,
                    "scalar_dim": 63,
                    "event_dim": 128,
                    "num_train_samples": int(counts["labeled_endpoints"]),
                    "timeline_counts": counts,
                    "normalization_mean": norm_stats.mean.tolist(),
                    "normalization_std": norm_stats.std.tolist(),
                    "reid_feature_file": "reid_features.npy",
                    "reid_feature_fields": [
                        "track_feature_before",
                        "detection_feature",
                    ],
                    "track_feature_after_available": False,
                    "available_fields": sorted(
                        {
                            "event_indices", "track_ids", "segment_ids",
                            "safe_gate_target", "policy_safe_soft_target",
                            "cue_target", "risk_target", "valid_channels",
                            "sample_weight", "timeline_scalar_feats",
                            "timeline_has_detection",
                        }
                        | {name for name in ("oracle_gate_target", "gate_confidence") if name in arrays}
                    ),
                }
                packed = {
                    "metadata": sequence_metadata,
                    "arrays": {
                        "event_indices": as_tensor(arrays["event_indices"]),
                        "track_ids": as_tensor(arrays["track_ids"]),
                        "segment_ids": as_tensor(arrays["segment_ids"]),
                        "safe_gate_target": as_tensor(arrays["safe_gate_target"]),
                        "policy_safe_soft_target": as_tensor(arrays["policy_safe_soft_target"]),
                        "cue_target": as_tensor(arrays["cue_target"]),
                        "risk_target": as_tensor(arrays["risk_target"]),
                        "valid_channels": as_tensor(arrays["valid_channels"]),
                        "sample_weight": as_tensor(arrays["sample_weight"]),
                        "timeline_scalar_feats": as_tensor(normalized_scalar),
                        "timeline_has_detection": as_tensor(
                            arrays["timeline_has_detection"]
                        ),
                    },
                }
                for optional in ("oracle_gate_target", "gate_confidence"):
                    if optional in arrays:
                        packed["arrays"][optional] = as_tensor(arrays[optional])
            finally:
                reader.close()

        sequence_dir = final_output_dir / sequence
        sequence_dir.mkdir(parents=True, exist_ok=False)
        torch.save(packed, sequence_dir / "data.pt")
        np.save(sequence_dir / "reid_features.npy", reid_features, allow_pickle=False)
        timeline_counts[sequence] = counts

    num_train_samples = sum(
        item["labeled_endpoints"] for item in timeline_counts.values()
    )
    if len(reid_dims) != 1:
        raise ValueError(f"inconsistent ReID dimensions: {sorted(reid_dims)}")
    if num_train_samples == 0:
        raise RuntimeError("no labeled streaming IWG RG-CMA samples were built")
    metadata = {
        "format": PACKED_TRAIN_DATA_FORMAT,
        "index_format": PACKED_TRAIN_DATA_FORMAT,
        "dataset": dataset,
        "split": split,
        "train_sequences": sequences,
        "context_size": context_size,
        "max_frame_gap": int(max_frame_gap),
        "reid_dim": sorted(reid_dims)[0] if reid_dims else 0,
        "scalar_dim": 63,
        "event_dim": 128,
        "num_train_samples": num_train_samples,
        "timeline_counts": timeline_counts,
        "normalization_mean": norm_stats.mean.tolist(),
        "normalization_std": norm_stats.std.tolist(),
        "reid_feature_file": "reid_features.npy",
        "reid_feature_fields": [
            "track_feature_before",
            "detection_feature",
        ],
        "track_feature_after_available": False,
    }
    (final_output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def migrate_compact_iwg_rg_cma_sequence(
    *,
    source_dataset_dir: str | Path,
    output_dir: str | Path,
    detection_cache_dir: str | Path,
    sequence: str,
    normalization_file: str | Path | None = None,
) -> dict[str, Any]:
    """Convert one legacy compact sequence into the packed train_data layout."""
    source_root = Path(source_dataset_dir).resolve()
    source_metadata = json.loads((source_root / "metadata.json").read_text())
    sequence = str(sequence)
    if sequence not in source_metadata.get("train_sequences", []):
        raise ValueError(f"sequence is absent from legacy dataset metadata: {sequence}")
    source_dir = source_root / str(source_metadata.get("train_index_dir", "compact_index")) / sequence
    target_root = Path(output_dir).resolve()
    target_dir = target_root / sequence
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite packed sequence: {target_dir}")
    target_root.mkdir(parents=True, exist_ok=True)

    array_files = {
        **COMPACT_SAMPLE_ARRAY_FILES_V1,
        **COMPACT_TIMELINE_ARRAY_FILES,
    }
    arrays = {
        name: np.load(source_dir / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in array_files.items()
    }
    count = int(arrays["track_ids"].shape[0])
    timeline_count = int(arrays["timeline_scalar_feats"].shape[0])
    for name in COMPACT_SAMPLE_ARRAY_FILES_V1:
        if int(arrays[name].shape[0]) != count:
            raise ValueError(f"legacy sample length mismatch for {name}: {source_dir}")
    for name in COMPACT_TIMELINE_ARRAY_FILES:
        if int(arrays[name].shape[0]) != timeline_count:
            raise ValueError(f"legacy timeline length mismatch for {name}: {source_dir}")
    context_size = int(source_metadata.get("context_size", IWG_CONTEXT_SIZE))
    if arrays["event_indices"].shape[1] != context_size:
        raise ValueError(f"legacy context mismatch: {source_dir}")
    reid_dim = int(source_metadata["reid_dim"])
    norm_path = Path(normalization_file).resolve() if normalization_file else source_root / "norm_stats.npz"
    norm_stats = NormalizationStats.load(norm_path)
    normalized_scalar = norm_stats.transform(
        np.asarray(arrays["timeline_scalar_feats"], dtype=np.float64)
    ).astype(np.float32)

    target_dir.mkdir(parents=True, exist_ok=False)
    reid_path = target_dir / "reid_features.npy"
    reid_features = np.lib.format.open_memmap(
        reid_path, mode="w+", dtype=np.float32, shape=(timeline_count, 2, reid_dim)
    )
    detection_cache = SequenceDetectionCache(Path(detection_cache_dir).resolve())
    try:
        detection_indices = np.asarray(arrays["timeline_detection_indices"], dtype=np.int64)
        matched = np.asarray(arrays["timeline_has_detection"], dtype=bool)
        for start in range(0, timeline_count, 65536):
            end = min(start + 65536, timeline_count)
            reid_features[start:end, 0, :] = np.asarray(
                arrays["timeline_track_feats"][start:end], dtype=np.float32
            )
            valid = matched[start:end] & (detection_indices[start:end] >= 0)
            if valid.any():
                indices = detection_indices[start:end][valid]
                if int(indices.max()) >= detection_cache.num_detections:
                    raise ValueError(f"legacy detection index outside cache: {source_dir}")
                detection_block = reid_features[start:end, 1, :]
                detection_block[valid] = np.asarray(
                    detection_cache.features[indices], dtype=np.float32
                )
        reid_features.flush()
    finally:
        detection_cache.close()
        del reid_features

    packed_arrays: dict[str, torch.Tensor] = {
        name: torch.from_numpy(np.array(arrays[name], copy=True))
        for name in COMPACT_SAMPLE_ARRAY_FILES_V1
    }
    packed_arrays["timeline_scalar_feats"] = torch.from_numpy(normalized_scalar)
    packed_arrays["timeline_has_detection"] = torch.from_numpy(
        np.array(arrays["timeline_has_detection"], copy=True)
    )
    for optional in COMPACT_RESIDUAL_TARGET_ARRAY_FILES:
        path = source_dir / COMPACT_RESIDUAL_TARGET_ARRAY_FILES[optional]
        if path.is_file():
            packed_arrays[optional] = torch.from_numpy(
                np.array(np.load(path, mmap_mode="r", allow_pickle=False), copy=True)
            )
    counts = dict(source_metadata.get("timeline_counts", {}).get(sequence, {}))
    counts.setdefault("events", timeline_count)
    counts.setdefault("labeled_endpoints", count)
    sequence_metadata = {
        "format": PACKED_TRAIN_DATA_FORMAT,
        "index_format": PACKED_TRAIN_DATA_FORMAT,
        "dataset": str(source_metadata.get("dataset", "MOT20")),
        "split": str(source_metadata.get("split", "all")),
        "sequence": sequence,
        "train_sequences": [sequence],
        "context_size": context_size,
        "max_frame_gap": int(source_metadata.get("max_frame_gap", 30)),
        "reid_dim": reid_dim,
        "scalar_dim": 63,
        "event_dim": int(source_metadata.get("event_dim", 128)),
        "num_train_samples": count,
        "timeline_counts": counts,
        "normalization_mean": norm_stats.mean.tolist(),
        "normalization_std": norm_stats.std.tolist(),
        "reid_feature_file": "reid_features.npy",
        "reid_feature_fields": ["track_feature_before", "detection_feature"],
        "track_feature_after_available": False,
    }
    torch.save({"metadata": sequence_metadata, "arrays": packed_arrays}, target_dir / "data.pt")

    existing = target_root / "metadata.json"
    if existing.is_file():
        output_metadata = json.loads(existing.read_text())
        migrated = list(output_metadata.get("train_sequences", []))
    else:
        # A per-sequence migration must advertise only data that already exists.
        migrated = []
    if sequence not in migrated:
        migrated.append(sequence)
    source_counts = source_metadata.get("timeline_counts", {})
    timeline_counts = {
        item: source_counts[item]
        for item in migrated
        if item in source_counts
    }
    output_metadata = {
        "format": PACKED_TRAIN_DATA_FORMAT,
        "index_format": PACKED_TRAIN_DATA_FORMAT,
        "dataset": str(source_metadata.get("dataset", "MOT20")),
        "split": str(source_metadata.get("split", "all")),
        "train_sequences": migrated,
        "context_size": context_size,
        "max_frame_gap": int(source_metadata.get("max_frame_gap", 30)),
        "reid_dim": reid_dim,
        "scalar_dim": 63,
        "event_dim": int(source_metadata.get("event_dim", 128)),
        "num_train_samples": sum(
            int(item.get("labeled_endpoints", 0)) for item in timeline_counts.values()
        ),
        "timeline_counts": timeline_counts,
        "normalization_mean": norm_stats.mean.tolist(),
        "normalization_std": norm_stats.std.tolist(),
        "reid_feature_file": "reid_features.npy",
        "reid_feature_fields": ["track_feature_before", "detection_feature"],
        "track_feature_after_available": False,
    }
    (target_root / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n"
    )
    return sequence_metadata


class StreamingIWGRGCMADataset(torch.utils.data.Dataset):
    """Load the self-contained packed AgentGuard training dataset."""

    def __init__(self, dataset_dir: str | Path, *, max_samples: int = 0) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text())
        if self.metadata.get("format") != PACKED_TRAIN_DATA_FORMAT:
            raise ValueError(
                "only packed AgentGuard train_data is supported; "
                "legacy compact datasets are no longer training inputs"
            )
        self._init_packed(max_samples=max_samples)

    def _init_packed(self, *, max_samples: int) -> None:
        """Load AgentGuard arrays and the separate mmap ReID feature cache."""
        self.context_size = int(self.metadata["context_size"])
        self.index_format = PACKED_TRAIN_DATA_FORMAT
        self._packed_sequences: list[dict[str, Any]] = []
        self._packed_boundaries: list[int] = []
        total = 0
        for sequence in self.metadata["train_sequences"]:
            path = self.dataset_dir / str(sequence) / "data.pt"
            packed = torch.load(path, weights_only=False)
            if not isinstance(packed, dict) or not isinstance(packed.get("arrays"), dict):
                raise ValueError(f"invalid packed AgentGuard data: {path}")
            arrays = packed["arrays"]
            required = {
                "event_indices", "track_ids", "segment_ids",
                "safe_gate_target", "policy_safe_soft_target",
                "cue_target", "risk_target", "valid_channels",
                "sample_weight", "timeline_scalar_feats",
                "timeline_has_detection",
            }
            missing = sorted(required.difference(arrays))
            if missing:
                raise ValueError(f"packed sequence {sequence} missing fields: {missing}")
            count = int(arrays["track_ids"].shape[0])
            timeline_count = int(arrays["timeline_scalar_feats"].shape[0])
            if arrays["event_indices"].shape[1] != self.context_size:
                raise ValueError(f"packed context mismatch in {path}")
            for name in required - {"timeline_scalar_feats", "timeline_has_detection"}:
                if int(arrays[name].shape[0]) != count:
                    raise ValueError(f"packed sample length mismatch for {name} in {path}")
            for name in ("timeline_scalar_feats", "timeline_has_detection"):
                if int(arrays[name].shape[0]) != timeline_count:
                    raise ValueError(f"packed timeline length mismatch for {name} in {path}")
            for name in ("oracle_gate_target", "gate_confidence"):
                if name in arrays and int(arrays[name].shape[0]) != count:
                    raise ValueError(f"packed sample length mismatch for {name} in {path}")
            event_indices = np.asarray(arrays["event_indices"])
            if event_indices.size and (
                int(event_indices.min()) < -1
                or int(event_indices.max()) >= timeline_count
            ):
                raise ValueError(f"packed event index outside timeline in {path}")
            reid_path = path.parent / str(
                self.metadata.get("reid_feature_file", "reid_features.npy")
            )
            if not reid_path.is_file():
                raise FileNotFoundError(f"packed ReID feature cache not found: {reid_path}")
            reid_features = np.load(reid_path, mmap_mode="r", allow_pickle=False)
            expected_reid_dim = int(self.metadata.get("reid_dim", 0))
            if reid_features.shape != (timeline_count, 2, expected_reid_dim):
                raise ValueError(
                    f"packed ReID feature shape mismatch in {reid_path}: "
                    f"{reid_features.shape} != {(timeline_count, 2, expected_reid_dim)}"
                )
            self._packed_sequences.append(
                {"name": str(sequence), "arrays": arrays, "reid_features": reid_features}
            )
            total += count
            self._packed_boundaries.append(total)
        self._length = min(total, int(max_samples)) if max_samples > 0 else total
        self.norm_stats = NormalizationStats()
        self.norm_stats.mean = np.asarray(
            self.metadata.get("normalization_mean", np.zeros(63)), dtype=np.float64
        )
        self.norm_stats.std = np.asarray(
            self.metadata.get("normalization_std", np.ones(63)), dtype=np.float64
        )

    def _packed_item(self, index: int) -> dict[str, Any]:
        sequence_position = bisect_right(self._packed_boundaries, index)
        previous = self._packed_boundaries[sequence_position - 1] if sequence_position else 0
        local_index = index - previous
        item = self._packed_sequences[sequence_position]
        sequence = item["name"]
        arrays = item["arrays"]
        reid_features = item["reid_features"]
        event_indices = np.asarray(arrays["event_indices"][local_index], dtype=np.int64)
        valid_positions = event_indices >= 0
        pad_left = int((~valid_positions).sum())
        context = self.context_size
        scalar = np.zeros((context, 63), dtype=np.float32)
        has_detection = np.zeros(context, dtype=np.bool_)
        if valid_positions.any():
            indices = event_indices[valid_positions]
            scalar[pad_left:] = np.asarray(arrays["timeline_scalar_feats"][indices], dtype=np.float32)
            has_detection[pad_left:] = np.asarray(
                arrays["timeline_has_detection"][indices], dtype=np.bool_
            )
        padding = np.ones(context, dtype=np.bool_)
        padding[pad_left:] = False
        reset = np.zeros(context, dtype=np.bool_)
        reset[pad_left] = True
        reid_dim = int(self.metadata.get("reid_dim", 0))
        track_features = np.zeros((context, reid_dim), dtype=np.float32)
        detection_features = np.zeros((context, reid_dim), dtype=np.float32)
        if valid_positions.any():
            indices = event_indices[valid_positions]
            track_features[pad_left:] = np.asarray(
                reid_features[indices, 0, :], dtype=np.float32
            )
            detection_features[pad_left:] = np.asarray(
                reid_features[indices, 1, :], dtype=np.float32
            )
        result = {
            "track_feats": torch.from_numpy(track_features),
            "det_feats": torch.from_numpy(detection_features),
            "scalar_feats": torch.from_numpy(scalar),
            "padding_mask": torch.from_numpy(padding),
            "mask": torch.from_numpy(padding),
            "has_detection_mask": torch.from_numpy(has_detection),
            "reset_mask": torch.from_numpy(reset),
            "safe_gate_target": arrays["safe_gate_target"][local_index].float(),
            "policy_safe_soft_target": arrays["policy_safe_soft_target"][local_index].float(),
            "cue_target": arrays["cue_target"][local_index].float(),
            "risk_target": arrays["risk_target"][local_index].float(),
            "valid_motion": arrays["valid_channels"][local_index][0].bool(),
            "valid_appearance": arrays["valid_channels"][local_index][1].bool(),
            "sample_weight": arrays["sample_weight"][local_index].float(),
            "sequence": sequence,
            "track_id": int(arrays["track_ids"][local_index]),
            "segment_id": int(arrays["segment_ids"][local_index]),
            "label_key": f"{sequence}:{local_index}",
        }
        for name in ("oracle_gate_target", "gate_confidence"):
            if name in arrays:
                result[name] = arrays[name][local_index].float()
        return result

    def close(self) -> None:
        if not hasattr(self, "metadata"):
            return
        for item in getattr(self, "_packed_sequences", []):
            mapped = getattr(item.get("reid_features"), "_mmap", None)
            if mapped is not None:
                mapped.close()
        self._packed_sequences = []
        self._packed_boundaries = []

    def release_cached_pages(self) -> dict[str, int]:
        """Release clean mmap pages between low-memory training phases."""
        advised_mmaps = 0
        for item in getattr(self, "_packed_sequences", []):
            mapped = getattr(item.get("reid_features"), "_mmap", None)
            if mapped is None:
                continue
            try:
                mapped.madvise(mmap.MADV_DONTNEED)
                advised_mmaps += 1
            except (AttributeError, OSError, ValueError):
                pass
        return {"mmap_regions": advised_mmaps, "files": 0}

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._packed_item(index)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        tensor_keys = [key for key, value in batch[0].items() if torch.is_tensor(value)]
        result: dict[str, Any] = {
            key: torch.stack([item[key] for item in batch]) for key in tensor_keys
        }
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result
