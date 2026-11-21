"""Read-only, CPU comparison of old SportsMOT 0.80 detections with reconstructed cache."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3. Tracker"))
from utils.det_feat_storage import list_sequence_caches, open_sequence_detection_pair

TARGET = ROOT / "outputs" / "2. det_feat" / "sportsmot_val_0.80.pickle"
OUT = ROOT / "outputs" / "3. track" / "reproduce_paper_checkpoint_legacy080_20261004_retry" / "detection_input_audit.json"

def main():
    with TARGET.open("rb") as handle:
        legacy = pickle.load(handle)
    split, dirs = list_sequence_caches("SportsMOT", "val")
    report = {
        "legacy": str(TARGET), "cache_split": split, "sequences": len(dirs),
        "frames": 0, "legacy_detections": 0, "cache_detections": 0,
        "shape_mismatch_frames": 0, "raw_equal_frames": 0,
        "float32_equal_frames": 0, "float32_equal_box_score_class_frames": 0,
        "same_box_set_after_float32_frames": 0,
        "same_box_set_different_features_frames": 0,
        "max_feature_difference_for_matching_boxes": 0.0,
        "legacy_dtypes": [], "cache_dtypes": [], "examples": [], "per_sequence": [],
    }
    for directory in dirs:
        target, _, cache = open_sequence_detection_pair(directory)
        entry = {"sequence": directory.name, "frames": 0, "float32_equal_frames": 0, "shape_mismatch_frames": 0, "legacy_detections": 0, "cache_detections": 0}
        try:
            for frame in target.keys():
                a = legacy[directory.name].get(frame)
                b = target[frame]
                na, nb = (0 if a is None else len(a)), (0 if b is None else len(b))
                report["frames"] += 1
                entry["frames"] += 1
                report["legacy_detections"] += na
                report["cache_detections"] += nb
                entry["legacy_detections"] += na
                entry["cache_detections"] += nb
                if a is not None and str(a.dtype) not in report["legacy_dtypes"]:
                    report["legacy_dtypes"].append(str(a.dtype))
                if b is not None and str(b.dtype) not in report["cache_dtypes"]:
                    report["cache_dtypes"].append(str(b.dtype))
                if na != nb or (na and a.shape != b.shape):
                    report["shape_mismatch_frames"] += 1
                    entry["shape_mismatch_frames"] += 1
                    if len(report["examples"]) < 12:
                        report["examples"].append({"sequence": directory.name, "frame": frame, "kind": "shape", "legacy_count": na, "cache_count": nb})
                    continue
                if not na:
                    for key in ["raw_equal_frames", "float32_equal_frames", "float32_equal_box_score_class_frames", "same_box_set_after_float32_frames"]:
                        report[key] += 1
                    entry["float32_equal_frames"] += 1
                    continue
                if np.array_equal(a, b):
                    report["raw_equal_frames"] += 1
                af = a.astype(np.float32, copy=False)
                if np.array_equal(af, b):
                    report["float32_equal_frames"] += 1
                    entry["float32_equal_frames"] += 1
                box_same_order = np.array_equal(af[:, :6], b[:, :6])
                if box_same_order:
                    report["float32_equal_box_score_class_frames"] += 1
                    ai = bi = np.arange(na)
                    box_same_set = True
                else:
                    ai = np.lexsort(af[:, :6].T[::-1])
                    bi = np.lexsort(b[:, :6].T[::-1])
                    box_same_set = np.array_equal(af[ai, :6], b[bi, :6])
                if box_same_set:
                    report["same_box_set_after_float32_frames"] += 1
                    features_equal = np.array_equal(af[ai, 6:], b[bi, 6:])
                    if not features_equal:
                        report["same_box_set_different_features_frames"] += 1
                        diff = float(np.max(np.abs(af[ai, 6:] - b[bi, 6:])))
                        report["max_feature_difference_for_matching_boxes"] = max(report["max_feature_difference_for_matching_boxes"], diff)
                        if len(report["examples"]) < 12:
                            report["examples"].append({"sequence": directory.name, "frame": frame, "kind": "features", "max_abs_difference": diff})
                elif len(report["examples"]) < 12:
                    report["examples"].append({"sequence": directory.name, "frame": frame, "kind": "box_score_class_content", "max_abs_difference_same_order": float(np.max(np.abs(af[:, :6] - b[:, :6])))})
        finally:
            cache.close()
        report["per_sequence"].append(entry)
    h = hashlib.sha256()
    with TARGET.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    report["legacy_sha256"] = h.hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "per_sequence"}
    summary["report"] = str(OUT)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
