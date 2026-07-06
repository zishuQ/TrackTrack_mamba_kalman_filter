import os
import pickle

import numpy as np

_SOURCE_PICKLE_CACHE = {}
_COMPACT_INDEX_CACHE = {}
_DETECTION_PAIR_CACHE = {}


def clear_detection_pair_cache():
    _SOURCE_PICKLE_CACHE.clear()
    _COMPACT_INDEX_CACHE.clear()
    _DETECTION_PAIR_CACHE.clear()


def _cache_key(target_pickle_path, source_pickle_path):
    return (os.path.abspath(target_pickle_path), os.path.abspath(source_pickle_path))


def _load_source_pickle(source_pickle_path):
    source_key = os.path.abspath(source_pickle_path)
    if source_key not in _SOURCE_PICKLE_CACHE:
        with open(source_pickle_path, "rb") as f:
            _SOURCE_PICKLE_CACHE[source_key] = pickle.load(f)
    return _SOURCE_PICKLE_CACHE[source_key]


def _load_compact_index(target_pickle_path, source_pickle_path):
    idx_path = compact_index_path(target_pickle_path, source_pickle_path)
    idx_key = os.path.abspath(idx_path)
    if idx_key not in _COMPACT_INDEX_CACHE:
        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"Missing compact index {idx_path}. "
                "Run scripts/build_nms_idx_from_95.py to create it from the 0.95 pickle."
            )
        with open(idx_path, "rb") as f:
            _COMPACT_INDEX_CACHE[idx_key] = pickle.load(f)
    return _COMPACT_INDEX_CACHE[idx_key]


def compact_index_path(target_pickle_path, source_pickle_path):
    target_base = os.path.basename(target_pickle_path).replace(".pickle", "")
    source_base = os.path.basename(source_pickle_path).replace(".pickle", "")
    return os.path.join(os.path.dirname(source_pickle_path), f"{target_base}.from_{source_base}.idx.pickle")


def shard_path(target_pickle_path, source_pickle_path, sequence_name):
    target_base = os.path.basename(target_pickle_path).replace(".pickle", "")
    source_base = os.path.basename(source_pickle_path).replace(".pickle", "")
    return os.path.join(
        os.path.dirname(source_pickle_path),
        f"{target_base}.from_{source_base}.{sequence_name}.shard.pickle",
    )


def _all_shards_exist(target_pickle_path, source_pickle_path, sequence_names):
    for seq_name in sequence_names:
        if not os.path.isfile(shard_path(target_pickle_path, source_pickle_path, seq_name)):
            return False
    return True


def _load_shard(target_pickle_path, source_pickle_path, sequence_name):
    sp = shard_path(target_pickle_path, source_pickle_path, sequence_name)
    with open(sp, "rb") as f:
        return pickle.load(f)


def _load_from_shards(target_pickle_path, source_pickle_path, sequence_names):
    detections_80 = {}
    detections_95 = {}
    for seq_name in sequence_names:
        shard = _load_shard(target_pickle_path, source_pickle_path, seq_name)
        detections_80[seq_name] = shard["detections_80"]
        detections_95[seq_name] = shard["detections_95"]
    return detections_80, detections_95


def build_shards(target_pickle_path, source_pickle_path, sequence_names=None, overwrite=False):
    detections_95 = _load_source_pickle(source_pickle_path)
    compact_index = _load_compact_index(target_pickle_path, source_pickle_path)

    available_seqs = set(compact_index["index"].keys())
    if sequence_names is None:
        sequence_names = sorted(available_seqs)

    built = []
    skipped = []
    for seq_name in sequence_names:
        if seq_name not in available_seqs:
            skipped.append((seq_name, "not in source"))
            continue
        sp = shard_path(target_pickle_path, source_pickle_path, seq_name)
        if os.path.isfile(sp) and not overwrite:
            skipped.append((seq_name, "already exists"))
            continue

        det80_single = apply_compact_index(detections_95, compact_index, sequence_names=[seq_name])
        shard = {
            "detections_80": det80_single.get(seq_name, {}),
            "detections_95": detections_95.get(seq_name, {}),
        }
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        with open(sp, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        built.append(seq_name)

    return built, skipped


def apply_compact_index(detections_95, compact_index, sequence_names=None):
    requested_sequences = None
    if sequence_names is not None:
        requested_sequences = set(sequence_names)

    detections_80 = {}
    for vid_name, frames in compact_index["index"].items():
        if requested_sequences is not None and vid_name not in requested_sequences:
            continue
        detections_80[vid_name] = {}
        for frame_id, frame_index in frames.items():
            if frame_index is None:
                detections_80[vid_name][frame_id] = None
            else:
                detections_80[vid_name][frame_id] = detections_95[vid_name][frame_id][frame_index]
    return detections_80


def load_detection_pair(target_pickle_path, source_pickle_path, sequence_names=None):
    sequence_key = None if sequence_names is None else tuple(sorted(sequence_names))
    key = _cache_key(target_pickle_path, source_pickle_path) + (sequence_key,)
    if key not in _DETECTION_PAIR_CACHE:
        if sequence_names is not None and _all_shards_exist(
            target_pickle_path, source_pickle_path, sequence_names
        ):
            _DETECTION_PAIR_CACHE[key] = _load_from_shards(
                target_pickle_path, source_pickle_path, sequence_names
            )
            return _DETECTION_PAIR_CACHE[key]

        detections_95 = _load_source_pickle(source_pickle_path)
        compact_index = _load_compact_index(target_pickle_path, source_pickle_path)
        _DETECTION_PAIR_CACHE[key] = (
            apply_compact_index(
                detections_95,
                compact_index,
                sequence_names=sequence_names,
            ),
            detections_95,
        )
    return _DETECTION_PAIR_CACHE[key]
