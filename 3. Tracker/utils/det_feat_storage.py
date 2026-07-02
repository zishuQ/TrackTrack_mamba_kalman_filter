import os
import pickle

import numpy as np


def compact_index_path(target_pickle_path, source_pickle_path):
    target_base = os.path.basename(target_pickle_path).replace(".pickle", "")
    source_base = os.path.basename(source_pickle_path).replace(".pickle", "")
    return os.path.join(os.path.dirname(source_pickle_path), f"{target_base}.from_{source_base}.idx.pickle")


def apply_compact_index(detections_95, compact_index):
    detections_80 = {}
    for vid_name, frames in compact_index["index"].items():
        detections_80[vid_name] = {}
        for frame_id, frame_index in frames.items():
            if frame_index is None:
                detections_80[vid_name][frame_id] = None
            else:
                detections_80[vid_name][frame_id] = detections_95[vid_name][frame_id][frame_index]
    return detections_80


def load_detection_pair(target_pickle_path, source_pickle_path):
    with open(source_pickle_path, "rb") as f:
        detections_95 = pickle.load(f)

    idx_path = compact_index_path(target_pickle_path, source_pickle_path)
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Missing compact index {idx_path}. "
            "Run scripts/build_nms_idx_from_95.py to create it from the 0.95 pickle."
        )

    with open(idx_path, "rb") as f:
        compact_index = pickle.load(f)
    return apply_compact_index(detections_95, compact_index), detections_95
