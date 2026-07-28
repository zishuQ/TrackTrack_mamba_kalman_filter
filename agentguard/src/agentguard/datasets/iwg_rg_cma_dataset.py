from __future__ import annotations

import hashlib
import json
import mmap
import os
from bisect import bisect_right
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.compact_iwg_labels import load_compact_label_arrays
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    validate_rollout_label,
)
from agentguard.datasets.timeline_utils import (
    IWG_CONTEXT_SIZE,
    _fit_train_normalization,
    _load_labels,
    _sha256_file,
    _sha256_file_set,
    compact_timeline_record,
    event_key,
    segment_track_timelines,
)
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
SUPPORTED_IWG_CONTEXT_SIZES = frozenset({IWG_CONTEXT_SIZE, 8})


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
SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256 = frozenset(
    {
        *SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
        *IWG_RG_CMA_CONTEXT8_DATASET_SCHEMA_SHA256_BY_DATASET.values(),
        SPORTSMOT_TRAINVAL_CONTEXT8_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
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
            (
                SPORTSMOT_TRAINVAL_IWG_RG_CMA_DATASET_SCHEMA_SHA256
                if context_size == IWG_CONTEXT_SIZE
                else SPORTSMOT_TRAINVAL_CONTEXT8_IWG_RG_CMA_DATASET_SCHEMA_SHA256
            ),
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
    dataset_schema_sha256 = IWG_RG_CMA_DATASET_SCHEMA_SHA256_BY_DATASET[dataset]
    if context_size != IWG_CONTEXT_SIZE:
        dataset_schema_sha256 = IWG_RG_CMA_CONTEXT8_DATASET_SCHEMA_SHA256_BY_DATASET[
            dataset
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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _sha256_relative_file_set(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


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


def _dataset_source_contract(
    *,
    dataset_schema_sha256: str,
    index_sha256: str,
    normalization_sha256: str,
    label_files_sha256: str,
    event_cache_manifests_sha256: str,
    sequences: list[str],
    max_frame_gap: int,
    sequence_source_splits: dict[str, str] | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "dataset_schema_sha256": dataset_schema_sha256,
        "index_sha256": index_sha256,
        "normalization_sha256": normalization_sha256,
        "label_files_sha256": label_files_sha256,
        "event_cache_manifests_sha256": event_cache_manifests_sha256,
        "sequences": sequences,
        "max_frame_gap": int(max_frame_gap),
    }
    if sequence_source_splits is not None:
        contract["sequence_source_splits"] = sequence_source_splits
    return contract


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
    next_segment = int(segment_offset)
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
            state = {
                "frame_id": frame_id,
                "history_count": history_count,
                "segment_id": next_segment,
                "events": deque(maxlen=context_size),
            }
            histories[track_id] = state
            next_segment += 1
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


def build_streaming_sample_index(
    segments: Iterable[list[dict[str, Any]]],
    labels: dict[str, dict[str, Any]],
    context_size: int = IWG_CONTEXT_SIZE,
) -> list[dict[str, Any]]:
    context_size = _validate_context_size(context_size)
    samples: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        for endpoint, record in enumerate(segment):
            key = event_key(
                str(record["sequence"]),
                int(record["event_shard_id"]),
                int(record["event_offset"]),
            )
            if not bool(record.get("matched", False)) or key not in labels:
                continue
            events = segment[max(0, endpoint - context_size + 1) : endpoint + 1]
            samples.append(
                {
                    "sequence": str(record["sequence"]),
                    "track_id": int(record["track_id"]),
                    "segment_id": int(segment_id),
                    "pad_left": context_size - len(events),
                    "events": events,
                    "label_key": key,
                }
            )
    return samples


def build_iwg_rg_cma_dataset(
    *,
    dataset: str,
    split: str,
    event_cache_root: str | Path,
    detection_cache_root: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    max_frame_gap: int = 30,
    context_size: int = IWG_CONTEXT_SIZE,
) -> dict[str, Any]:
    context_size = _validate_context_size(context_size)
    sequences, dataset_schema_sha256, source_splits = resolve_iwg_rg_cma_dataset_spec(
        dataset, split, context_size
    )
    event_cache_root = Path(event_cache_root).resolve()
    detection_cache_root = Path(detection_cache_root).resolve()
    label_dir = Path(label_dir).resolve()
    final_output_dir = Path(output_dir).resolve()
    if final_output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite IWG RG-CMA dataset: {final_output_dir}"
        )
    output_dir = final_output_dir.with_name(
        f"{final_output_dir.name}.incomplete.{os.getpid()}"
    )
    if output_dir.exists():
        raise FileExistsError(f"stale IWG RG-CMA build directory exists: {output_dir}")
    missing = sorted(
        sequence
        for sequence in sequences
        if not (event_cache_root / dataset / source_splits[sequence] / sequence).is_dir()
    )
    if missing:
        raise FileNotFoundError(f"{dataset} event caches are missing: {missing}")

    timeline_counts: dict[str, dict[str, int]] = {}
    manifest_paths: list[Path] = []
    reid_dims: set[int] = set()
    residual_target_schema = "safe-v1"
    compact_label_schema_sha256 = None
    output_dir.mkdir(parents=True)
    norm_path = output_dir / "norm_stats.npz"
    if dataset == "MOT17":
        labels = _load_labels(label_dir)
        samples: list[dict[str, Any]] = []
        segment_offset = 0
        for sequence in sequences:
            source_split = source_splits[sequence]
            cache_dir = event_cache_root / dataset / source_split / sequence
            detection_dir = detection_cache_root / dataset / source_split / sequence
            if not detection_dir.is_dir():
                raise FileNotFoundError(f"detection cache not found: {detection_dir}")
            reader = CompactEventCacheReader(cache_dir, detection_dir)
            try:
                if not bool(reader.manifest.get("complete", False)):
                    raise ValueError(f"event cache is incomplete: {cache_dir}")
                reid_dims.add(int(reader.manifest["reid_dim"]))
                timeline = [
                    compact_timeline_record(sequence, record)
                    for record in reader.iter_event_records()
                ]
            finally:
                reader.close()
            segments = segment_track_timelines(timeline, max_frame_gap=max_frame_gap)
            sequence_samples = build_streaming_sample_index(
                segments, labels, context_size
            )
            for sample in sequence_samples:
                sample["segment_id"] += segment_offset
            segment_offset += len(segments)
            samples.extend(sequence_samples)
            timeline_counts[sequence] = {
                "events": len(timeline),
                "matched": sum(bool(record["matched"]) for record in timeline),
                "unmatched": sum(not bool(record["matched"]) for record in timeline),
                "segments": len(segments),
                "labeled_endpoints": len(sequence_samples),
                "samples_with_unmatched_history": sum(
                    any(
                        not bool(event["matched"])
                        for event in sample["events"][:-1]
                    )
                    for sample in sequence_samples
                ),
            }
            manifest_paths.append(cache_dir / "manifest.json")
        index_path = output_dir / "train_samples.jsonl"
        _write_jsonl(index_path, samples)
        index_format = "jsonl_v1"
        index_sha256 = _sha256_file(index_path)
        label_files_sha256 = _sha256_file_set(label_dir.glob("*_labels.json"))
        index_metadata = {"train_index_file": index_path.name}
        num_train_samples = len(samples)
    else:
        label_summary_path = label_dir / "summary.json"
        if not label_summary_path.is_file():
            raise FileNotFoundError(f"compact label summary not found: {label_summary_path}")
        label_summary = json.loads(label_summary_path.read_text())
        if label_summary.get("motion_label_mode", "nsa_rollout") != "nsa_rollout":
            raise ValueError("only NSA rollout labels are supported")
        index_root = output_dir / "compact_index"
        index_root.mkdir()
        first_manifest = json.loads(
            (label_dir / sequences[0] / "manifest.json").read_text()
        )
        compact_label_schema_sha256 = str(
            first_manifest.get("compact_label_schema_sha256", "")
        )
        first_array_files = first_manifest.get("array_files", {})
        if (
            "oracle_gate_target" in first_array_files
            and "gate_confidence" in first_array_files
        ):
            residual_target_schema = "oracle-confidence-v1"
            sample_array_files = {
                **COMPACT_SAMPLE_ARRAY_FILES_V1,
                **COMPACT_RESIDUAL_TARGET_ARRAY_FILES,
            }
        else:
            sample_array_files = dict(COMPACT_SAMPLE_ARRAY_FILES_V1)
        compact_index_array_files = {
            **sample_array_files,
            **COMPACT_TIMELINE_ARRAY_FILES,
        }
        compact_index_paths: list[Path] = []
        segment_offset = 0
        for sequence in sequences:
            source_split = source_splits[sequence]
            cache_dir = event_cache_root / dataset / source_split / sequence
            detection_dir = detection_cache_root / dataset / source_split / sequence
            if not detection_dir.is_dir():
                raise FileNotFoundError(f"detection cache not found: {detection_dir}")
            reader = CompactEventCacheReader(cache_dir, detection_dir)
            try:
                if not bool(reader.manifest.get("complete", False)):
                    raise ValueError(f"event cache is incomplete: {cache_dir}")
                reid_dims.add(int(reader.manifest["reid_dim"]))
                counts, segment_offset, paths = _build_compact_sequence_index(
                    sequence=sequence,
                    reader=reader,
                    compact_label_dir=label_dir / sequence,
                    index_dir=index_root / sequence,
                    max_frame_gap=max_frame_gap,
                    segment_offset=segment_offset,
                    context_size=context_size,
                )
            finally:
                reader.close()
            timeline_counts[sequence] = counts
            compact_index_paths.extend(paths)
            manifest_paths.append(cache_dir / "manifest.json")
        index_format = COMPACT_INDEX_FORMAT
        index_sha256 = _sha256_relative_file_set(compact_index_paths, output_dir)
        label_paths = [
            path
            for sequence in sequences
            for path in (label_dir / sequence).glob("*")
            if path.is_file()
        ]
        label_paths.append(label_dir / "summary.json")
        label_files_sha256 = _sha256_relative_file_set(label_paths, label_dir)
        index_metadata = {
            "train_index_dir": index_root.name,
            "compact_index_array_files": compact_index_array_files,
        }
        num_train_samples = sum(
            counts["labeled_endpoints"] for counts in timeline_counts.values()
        )
    if len(reid_dims) != 1:
        raise ValueError(f"inconsistent ReID dimensions: {sorted(reid_dims)}")
    if any(counts["unmatched"] == 0 for counts in timeline_counts.values()):
        raise ValueError("every source timeline must contain unmatched events")
    if num_train_samples == 0:
        raise RuntimeError("no labeled streaming IWG RG-CMA samples were built")

    norm_stats = (
        _fit_train_normalization(event_cache_root, dataset, split, sequences)
        if len(set(source_splits.values())) == 1
        else _fit_sequence_source_normalization(
            event_cache_root,
            dataset,
            sequences,
            source_splits,
        )
    )
    norm_stats.save(str(norm_path))
    multi_source_splits = source_splits if len(set(source_splits.values())) > 1 else None
    source_contract = _dataset_source_contract(
        dataset_schema_sha256=dataset_schema_sha256,
        index_sha256=index_sha256,
        normalization_sha256=_sha256_file(norm_path),
        label_files_sha256=label_files_sha256,
        event_cache_manifests_sha256=_sha256_file_set(manifest_paths),
        sequences=sequences,
        max_frame_gap=max_frame_gap,
        sequence_source_splits=multi_source_splits,
    )
    metadata = {
        "dataset": dataset,
        "split": split,
        "split_policy": "train_all",
        "candidate_types": ["A"],
        "train_sequences": sequences,
        **(
            {"sequence_source_splits": source_splits}
            if multi_source_splits is not None
            else {}
        ),
        "event_cache_root": str(event_cache_root),
        "detection_cache_root": str(detection_cache_root),
        "label_dir": str(label_dir),
        "index_format": index_format,
        **index_metadata,
        "normalization_file": norm_path.name,
        "num_train_samples": num_train_samples,
        "context_size": context_size,
        "max_frame_gap": int(max_frame_gap),
        "reid_dim": reid_dims.pop(),
        "scalar_dim": 63,
        "event_dim": 128,
        "timeline_counts": timeline_counts,
        "dataset_schema_version": IWG_RG_CMA_DATASET_SCHEMA_VERSION,
        "dataset_schema_sha256": dataset_schema_sha256,
        "residual_target_schema": residual_target_schema,
        **(
            {"compact_label_schema_sha256": compact_label_schema_sha256}
            if compact_label_schema_sha256 is not None
            else {}
        ),
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        **source_contract,
        "dataset_sha256": _canonical_sha256(source_contract),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "metadata.sha256").write_text(_sha256_file(metadata_path) + "\n")
    output_dir.rename(final_output_dir)
    return metadata


class StreamingIWGRGCMADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str | Path, *, max_samples: int = 0) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text())
        self._validate_metadata()
        self.context_size = int(self.metadata["context_size"])
        self.index_format = str(self.metadata.get("index_format", "jsonl_v1"))
        self.samples: list[dict[str, Any]] = []
        self.labels: dict[str, dict[str, Any]] = {}
        self._compact_indexes: dict[str, dict[str, np.ndarray]] = {}
        self._compact_boundaries: list[int] = []
        self._compact_index_array_files = dict(
            self.metadata.get("compact_index_array_files", COMPACT_INDEX_ARRAY_FILES)
        )
        if self.index_format == "jsonl_v1":
            with (self.dataset_dir / self.metadata["train_index_file"]).open() as handle:
                self.samples = [json.loads(line) for line in handle if line.strip()]
            if max_samples > 0:
                self.samples = self.samples[: int(max_samples)]
            self.labels = _load_labels(Path(self.metadata["label_dir"]))
            self._length = len(self.samples)
        elif self.index_format == COMPACT_INDEX_FORMAT:
            total = 0
            index_root = self.dataset_dir / self.metadata["train_index_dir"]
            timeline_names = set(COMPACT_TIMELINE_ARRAY_FILES)
            index_array_files = dict(self._compact_index_array_files)
            sample_array_files = {
                name: filename
                for name, filename in index_array_files.items()
                if name not in timeline_names
            }
            index_array_files = {
                **sample_array_files,
                **{
                    name: filename
                    for name, filename in index_array_files.items()
                    if name in timeline_names
                },
            }
            for sequence in self.metadata["train_sequences"]:
                sequence_dir = index_root / sequence
                arrays = {
                    name: np.load(
                        sequence_dir / filename,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    for name, filename in index_array_files.items()
                }
                count = int(arrays["track_ids"].shape[0])
                if any(
                    arrays[name].shape[0] != count
                    for name in sample_array_files
                ):
                    raise ValueError(f"compact index length mismatch in {sequence_dir}")
                timeline_count = int(arrays["timeline_track_feats"].shape[0])
                if any(
                    arrays[name].shape[0] != timeline_count
                    for name in COMPACT_TIMELINE_ARRAY_FILES
                ):
                    raise ValueError(f"compact timeline length mismatch in {sequence_dir}")
                event_indices = arrays["event_indices"]
                if event_indices.size and (
                    int(event_indices.max()) >= timeline_count
                    or int(event_indices.min()) < -1
                ):
                    raise ValueError(f"compact event index outside timeline in {sequence_dir}")
                self._compact_indexes[sequence] = arrays
                total += count
                self._compact_boundaries.append(total)
            self._length = min(total, int(max_samples)) if max_samples > 0 else total
        else:
            raise ValueError(f"unsupported IWG RG-CMA index format: {self.index_format}")
        self.norm_stats = NormalizationStats.load(
            str(self.dataset_dir / self.metadata["normalization_file"])
        )
        self._readers: dict[str, CompactEventCacheReader] = {}
        self._detection_readers: dict[str, Any] = {}

    def _validate_metadata(self) -> None:
        dataset = str(self.metadata.get("dataset", ""))
        if dataset not in IWG_RG_CMA_TRAIN_ALL_SEQUENCES:
            raise ValueError(f"unsupported IWG RG-CMA dataset: {dataset}")
        if str(self.metadata.get("motion_target_mode", "nsa")) != "nsa":
            raise ValueError("only NSA datasets are supported")
        split = str(self.metadata.get("split", ""))
        context_size = _validate_context_size(self.metadata.get("context_size", -1))
        sequences, _current_schema_sha256, source_splits = (
            resolve_iwg_rg_cma_dataset_spec(dataset, split, context_size)
        )
        metadata_schema_sha256 = str(self.metadata.get("dataset_schema_sha256", ""))
        accepted_schema_sha256 = accepted_iwg_rg_cma_dataset_schema_sha256(
            dataset, split, context_size
        )
        multi_source_splits = (
            source_splits if len(set(source_splits.values())) > 1 else None
        )
        expected = {
            "dataset": dataset,
            "split": split,
            "split_policy": "train_all",
            "candidate_types": ["A"],
            "train_sequences": sequences,
            "context_size": context_size,
            "dataset_schema_version": IWG_RG_CMA_DATASET_SCHEMA_VERSION,
            "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
            "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        }
        if multi_source_splits is not None:
            expected["sequence_source_splits"] = source_splits
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if metadata_schema_sha256 not in accepted_schema_sha256:
            mismatches["dataset_schema_sha256"] = (
                metadata_schema_sha256,
                sorted(accepted_schema_sha256),
            )
        if mismatches:
            raise ValueError(f"IWG RG-CMA dataset metadata mismatch: {mismatches}")
        norm_path = self.dataset_dir / self.metadata["normalization_file"]
        index_format = str(self.metadata.get("index_format", "jsonl_v1"))
        if index_format == "jsonl_v1":
            index_sha256 = _sha256_file(
                self.dataset_dir / self.metadata["train_index_file"]
            )
            label_files_sha256 = _sha256_file_set(
                Path(self.metadata["label_dir"]).glob("*_labels.json")
            )
        elif index_format == COMPACT_INDEX_FORMAT:
            index_root = self.dataset_dir / self.metadata["train_index_dir"]
            index_paths = [
                path
                for sequence in sequences
                for path in (index_root / sequence).glob("*.npy")
            ]
            index_sha256 = _sha256_relative_file_set(index_paths, self.dataset_dir)
            label_root = Path(self.metadata["label_dir"])
            label_paths = [
                path
                for sequence in sequences
                for path in (label_root / sequence).glob("*")
                if path.is_file()
            ]
            label_paths.append(label_root / "summary.json")
            label_files_sha256 = _sha256_relative_file_set(label_paths, label_root)
        else:
            raise ValueError(f"unsupported IWG RG-CMA index format: {index_format}")
        manifest_paths = [
            Path(self.metadata["event_cache_root"])
            / self.metadata["dataset"]
            / source_splits[sequence]
            / sequence
            / "manifest.json"
            for sequence in sequences
        ]
        event_cache_manifests_sha256 = _sha256_file_set(manifest_paths)
        if label_files_sha256 != self.metadata["label_files_sha256"]:
            raise ValueError("IWG RG-CMA label files changed after dataset construction")
        if (
            event_cache_manifests_sha256
            != self.metadata["event_cache_manifests_sha256"]
        ):
            raise ValueError("IWG RG-CMA event cache manifests changed after construction")
        source_contract = _dataset_source_contract(
            dataset_schema_sha256=metadata_schema_sha256,
            index_sha256=index_sha256,
            normalization_sha256=_sha256_file(norm_path),
            label_files_sha256=label_files_sha256,
            event_cache_manifests_sha256=event_cache_manifests_sha256,
            sequences=sequences,
            max_frame_gap=int(self.metadata["max_frame_gap"]),
            sequence_source_splits=multi_source_splits,
        )
        if self.metadata.get("dataset_sha256") != _canonical_sha256(source_contract):
            raise ValueError("IWG RG-CMA dataset content hash mismatch")

    def _reader(self, sequence: str) -> CompactEventCacheReader:
        if sequence not in self._readers:
            source_split = self.metadata.get("sequence_source_splits", {}).get(
                sequence, self.metadata["split"]
            )
            self._readers[sequence] = CompactEventCacheReader(
                Path(self.metadata["event_cache_root"])
                / self.metadata["dataset"]
                / source_split
                / sequence,
                Path(self.metadata["detection_cache_root"])
                / self.metadata["dataset"]
                / source_split
                / sequence,
                # JSONL training shuffles across hundreds of event shards. A
                # bounded LRU repeatedly deserializes the same shards every
                # epoch, so keep them resident for this legacy index format.
                max_cached_shards=None,
            )
        return self._readers[sequence]

    def _detection_reader(self, sequence: str):
        if sequence not in self._detection_readers:
            from agentguard.data.detection_cache import SequenceDetectionCache

            source_split = self.metadata.get("sequence_source_splits", {}).get(
                sequence, self.metadata["split"]
            )
            self._detection_readers[sequence] = SequenceDetectionCache(
                Path(self.metadata["detection_cache_root"])
                / self.metadata["dataset"]
                / source_split
                / sequence
            )
        return self._detection_readers[sequence]

    def close(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        if hasattr(self, "_readers"):
            self._readers.clear()
        for reader in getattr(self, "_detection_readers", {}).values():
            reader.close()
        if hasattr(self, "_detection_readers"):
            self._detection_readers.clear()
        for arrays in getattr(self, "_compact_indexes", {}).values():
            for array in arrays.values():
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        if hasattr(self, "_compact_indexes"):
            self._compact_indexes.clear()

    def release_cached_pages(self) -> dict[str, int]:
        """Release clean mmap pages between low-memory training phases."""
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        if hasattr(self, "_readers"):
            self._readers.clear()
        for reader in getattr(self, "_detection_readers", {}).values():
            reader.close()
        if hasattr(self, "_detection_readers"):
            self._detection_readers.clear()

        advised_mmaps = 0
        seen_mmaps: set[int] = set()
        for arrays in getattr(self, "_compact_indexes", {}).values():
            for array in arrays.values():
                mapped = getattr(array, "_mmap", None)
                if mapped is None or id(mapped) in seen_mmaps:
                    continue
                seen_mmaps.add(id(mapped))
                try:
                    mapped.madvise(mmap.MADV_DONTNEED)
                    advised_mmaps += 1
                except (AttributeError, OSError, ValueError):
                    pass

        advised_files = 0
        if (
            self.index_format == COMPACT_INDEX_FORMAT
            and hasattr(os, "posix_fadvise")
            and hasattr(os, "POSIX_FADV_DONTNEED")
        ):
            index_root = self.dataset_dir / self.metadata["train_index_dir"]
            for sequence in self.metadata["train_sequences"]:
                for filename in self._compact_index_array_files.values():
                    path = index_root / sequence / filename
                    descriptor = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(
                            descriptor, 0, 0, os.POSIX_FADV_DONTNEED
                        )
                        advised_files += 1
                    except OSError:
                        pass
                    finally:
                        os.close(descriptor)
        return {"mmap_regions": advised_mmaps, "files": advised_files}

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        oracle_gate = None
        gate_confidence = None
        if self.index_format == "jsonl_v1":
            sample = self.samples[index]
            sequence = str(sample["sequence"])
            pad_left = int(sample["pad_left"])
            references = [
                (int(item["event_shard_id"]), int(item["event_offset"]))
                for item in sample["events"]
            ]
            label = self.labels[sample["label_key"]]
            safe_gate = np.asarray(
                [label["motion_safe_target"], label["appearance_safe_target"]],
                dtype=np.float32,
            )
            oracle_gate = np.asarray(
                [label["motion_soft_target"], label["appearance_soft_target"]],
                dtype=np.float32,
            )
            gate_confidence = np.asarray(
                [
                    label["motion_label_confidence"],
                    label["appearance_label_confidence"],
                ],
                dtype=np.float32,
            )
            policy_target = np.asarray(
                label["policy_safe_soft_target"], dtype=np.float32
            )
            cue_target = np.asarray(label["cue_target"], dtype=np.float32)
            risk_target = np.asarray(label["risk_targets"], dtype=np.float32)
            valid_motion = bool(label["valid_motion"])
            valid_appearance = bool(label["valid_appearance"])
            sample_weight = float(label["sample_weight"])
            track_id = int(sample["track_id"])
            segment_id = int(sample["segment_id"])
            label_key = str(sample["label_key"])
            timeline_indices = None
        else:
            sequence_position = bisect_right(self._compact_boundaries, index)
            previous = (
                self._compact_boundaries[sequence_position - 1]
                if sequence_position > 0
                else 0
            )
            local_index = index - previous
            sequence = str(self.metadata["train_sequences"][sequence_position])
            compact = self._compact_indexes[sequence]
            timeline_row = compact["event_indices"][local_index]
            shard_ids = compact["event_shard_ids"][local_index]
            offsets = compact["event_offsets"][local_index]
            valid_positions = timeline_row >= 0
            pad_left = int((~valid_positions).sum())
            timeline_indices = timeline_row[valid_positions]
            references = [
                (int(shard_id), int(offset))
                for shard_id, offset in zip(
                    shard_ids[valid_positions], offsets[valid_positions]
                )
            ]
            safe_gate = np.array(
                compact["safe_gate_target"][local_index], dtype=np.float32, copy=True
            )
            policy_target = np.array(
                compact["policy_safe_soft_target"][local_index],
                dtype=np.float32,
                copy=True,
            )
            if "oracle_gate_target" in compact:
                oracle_gate = np.array(
                    compact["oracle_gate_target"][local_index],
                    dtype=np.float32,
                    copy=True,
                )
                gate_confidence = np.array(
                    compact["gate_confidence"][local_index],
                    dtype=np.float32,
                    copy=True,
                )
            cue_target = np.array(
                compact["cue_target"][local_index], dtype=np.float32, copy=True
            )
            risk_target = np.array(
                compact["risk_target"][local_index], dtype=np.float32, copy=True
            )
            if oracle_gate is None and gate_confidence is None:
                # Legacy v3 compact labels already contain both signals in
                # encoded form: risk[0:2] = 1 - oracle gate and cue[0:2]
                # = motion/appearance label confidence. Keep the old files
                # immutable and materialize the fields only for this sample.
                if risk_target.shape != (4,) or cue_target.shape != (3,):
                    raise ValueError(
                        "legacy compact labels cannot derive oracle-confidence "
                        f"targets: risk={risk_target.shape}, cue={cue_target.shape}"
                    )
                oracle_gate = np.clip(1.0 - risk_target[:2], 0.0, 1.0)
                gate_confidence = np.clip(cue_target[:2], 0.0, 1.0)
            valid = compact["valid_channels"][local_index]
            valid_motion = bool(valid[0])
            valid_appearance = bool(valid[1])
            sample_weight = float(compact["sample_weight"][local_index])
            track_id = int(compact["track_ids"][local_index])
            segment_id = int(compact["segment_ids"][local_index])
            endpoint_shard, endpoint_offset = references[-1]
            label_key = event_key(sequence, endpoint_shard, endpoint_offset)
        reid_dim = int(self.metadata["reid_dim"])
        context_size = int(
            getattr(self, "context_size", self.metadata.get("context_size", 6))
        )
        track = np.zeros((context_size, reid_dim), dtype=np.float32)
        detection = np.zeros((context_size, reid_dim), dtype=np.float32)
        scalar = np.zeros((context_size, 63), dtype=np.float64)
        padding = np.ones(context_size, dtype=np.bool_)
        has_detection = np.zeros(context_size, dtype=np.bool_)
        reset = np.zeros(context_size, dtype=np.bool_)
        reader = (
            self._reader(sequence)
            if timeline_indices is None
            else self._detection_reader(sequence)
        )
        for history_position, (shard_id, offset) in enumerate(references):
            position = pad_left + history_position
            if timeline_indices is None:
                record = reader.get_event_record(shard_id, offset)
                track_value = np.asarray(
                    record.get("track_feature", []), dtype=np.float32
                )
                scalar_value = np.asarray(
                    record.get("scalar_features", []), dtype=np.float64
                )
                matched = bool(record.get("matched", False))
                detection_index = int(record.get("accepted_detection_index", -1))
            else:
                timeline_index = int(timeline_indices[history_position])
                track_value = np.asarray(
                    compact["timeline_track_feats"][timeline_index], dtype=np.float32
                )
                scalar_value = np.asarray(
                    compact["timeline_scalar_feats"][timeline_index], dtype=np.float64
                )
                matched = bool(compact["timeline_has_detection"][timeline_index])
                detection_index = int(
                    compact["timeline_detection_indices"][timeline_index]
                )
            if track_value.shape != (reid_dim,) or scalar_value.shape != (63,):
                raise ValueError("cached IWG RG-CMA feature has an invalid shape")
            if not np.isfinite(track_value).all() or not np.isfinite(scalar_value).all():
                raise ValueError("cached IWG RG-CMA feature is non-finite")
            track[position] = track_value
            scalar[position] = scalar_value
            has_detection[position] = matched
            if matched:
                if detection_index < 0:
                    raise ValueError("matched event lacks an accepted detection")
                detection[position] = np.asarray(
                    reader.get_detection(detection_index)["feature"], dtype=np.float32
                ).reshape(-1)
            padding[position] = False
        reset[pad_left] = True
        scalar = self.norm_stats.transform(scalar).astype(np.float32)
        scalar[padding] = 0.0
        result = {
            "track_feats": torch.from_numpy(track),
            "det_feats": torch.from_numpy(detection),
            "scalar_feats": torch.from_numpy(scalar),
            "padding_mask": torch.from_numpy(padding),
            "mask": torch.from_numpy(padding),
            "has_detection_mask": torch.from_numpy(has_detection),
            "reset_mask": torch.from_numpy(reset),
            "safe_gate_target": torch.from_numpy(safe_gate),
            "policy_safe_soft_target": torch.from_numpy(policy_target),
            "cue_target": torch.from_numpy(cue_target),
            "risk_target": torch.from_numpy(risk_target),
            "valid_motion": torch.tensor(valid_motion),
            "valid_appearance": torch.tensor(valid_appearance),
            "sample_weight": torch.tensor(sample_weight),
            "sequence": sequence,
            "track_id": track_id,
            "segment_id": segment_id,
            "label_key": label_key,
        }
        if oracle_gate is not None and gate_confidence is not None:
            result["oracle_gate_target"] = torch.from_numpy(oracle_gate)
            result["gate_confidence"] = torch.from_numpy(gate_confidence)
        return result

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
