#!/usr/bin/env python3
"""Build a joint 4-D detection/GT residual pool for SportsMOT train45.

For every sample row we resolve the tracked target's GT box at frame t and,
when a real detection is present and its IoU with that GT is >= 0.5, keep the
joint residual

    [ (det_cx-gt_cx)/gt_w, (det_cy-gt_cy)/gt_h,
      log(det_w/gt_w), log(det_h/gt_h) ]

and the raw pixel residual.  The relative form preserves the position/size
correlation while staying comparable across object scales.  Per-sequence
`gt_box_t0.npy` / `gt_valid_t0.npy` are written for training-time simulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = Path.home() / "datasets/SportsMOT/dataset"


def load_gt(sequence: str):
    frames: dict[int, dict[int, np.ndarray]] = {}
    path = GT_ROOT / "train" / sequence / "gt/gt.txt"
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        gt_id = int(float(fields[1]))
        if len(fields) >= 8 and (float(fields[6]) <= 0 or float(fields[7]) != 1):
            continue
        x, y, w, h = map(float, fields[2:6])
        frames.setdefault(frame, {})[gt_id] = np.array([x, y, x + w, y + h])
    return frames


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    relative = []
    raw = []
    per_sequence = {}
    total_rows = 0
    total_pairs = 0
    for seq_dir in sorted(data_root.iterdir()):
        samples_path = seq_dir / "samples.npz"
        if not samples_path.is_file():
            continue
        sequence = seq_dir.name
        gt = load_gt(sequence)
        with np.load(samples_path, allow_pickle=False) as samples:
            frame = np.asarray(samples["frame"], np.int64)
            gt_id = np.asarray(samples["gt_id"], np.int64)
            event_index = np.asarray(samples["current_event_index"], np.int64)
            gt_t1_valid = np.asarray(samples["motion_gt_valid"], bool)
        event_measurement = np.load(seq_dir / "event_det_measurement.npy", mmap_mode="r")
        event_has = np.load(seq_dir / "event_has_detection.npy", mmap_mode="r")
        measurements = np.asarray(event_measurement[event_index], np.float32)
        has = np.asarray(event_has[event_index], bool)
        n = len(frame)
        gt_box_t0 = np.zeros((n, 4), np.float32)
        gt_valid_t0 = np.zeros(n, bool)
        det_box_t0 = np.zeros((n, 4), np.float32)
        det_has_t0 = has.copy()
        pairs = 0
        candidates = 0
        for row in range(n):
            boxes = gt.get(int(frame[row]), {})
            box = boxes.get(int(gt_id[row]))
            if box is None:
                continue
            x1, y1, x2, y2 = box
            gt_cx = (x1 + x2) / 2.0; gt_cy = (y1 + y2) / 2.0
            gt_w = max(x2 - x1, 1.0); gt_h = max(y2 - y1, 1.0)
            gt_box_t0[row] = (gt_cx, gt_cy, gt_w, gt_h)
            gt_valid_t0[row] = True
            if not has[row]:
                continue
            candidates += 1
            det_cx, det_cy, det_w, det_h = measurements[row]
            det_box_t0[row] = measurements[row]
            det_xyxy = np.array([det_cx - det_w / 2, det_cy - det_h / 2,
                                 det_cx + det_w / 2, det_cy + det_h / 2])
            if iou_xyxy(det_xyxy, box) < 0.5:
                continue
            pairs += 1
            relative.append(((det_cx - gt_cx) / gt_w, (det_cy - gt_cy) / gt_h,
                             np.log(max(det_w, 1e-3) / gt_w),
                             np.log(max(det_h, 1e-3) / gt_h)))
            raw.append((det_cx - gt_cx, det_cy - gt_cy, det_w - gt_w, det_h - gt_h))
        np.save(seq_dir / "gt_box_t0.npy", gt_box_t0)
        np.save(seq_dir / "gt_valid_t0.npy", gt_valid_t0)
        per_sequence[sequence] = {
            "rows": n, "gt_t0": int(gt_valid_t0.sum()),
            "detections": int(has.sum()), "aligned_pairs": pairs,
            "pair_coverage_of_gt_det": pairs / max(1, candidates),
        }
        total_rows += n
        total_pairs += pairs
    relative = np.asarray(relative, np.float64)
    raw = np.asarray(raw, np.float64)
    np.savez(out / "error_pool.npz", relative=relative.astype(np.float32),
             raw=raw.astype(np.float32))
    corr = np.corrcoef(relative, rowvar=False).tolist() if len(relative) > 1 else None
    report = {
        "data_root": str(data_root), "sequences": len(per_sequence),
        "rows": total_rows, "pairs": total_pairs,
        "pair_coverage": total_pairs / max(1, total_rows),
        "relative": {
            "mean": relative.mean(0).tolist(), "std": relative.std(0).tolist(),
            "p01": np.percentile(relative, 1, axis=0).tolist(),
            "p50": np.percentile(relative, 50, axis=0).tolist(),
            "p99": np.percentile(relative, 99, axis=0).tolist(),
            "correlation": corr,
        },
        "raw_pixels": {
            "mean": raw.mean(0).tolist(), "std": raw.std(0).tolist(),
            "p01": np.percentile(raw, 1, axis=0).tolist(),
            "p50": np.percentile(raw, 50, axis=0).tolist(),
            "p99": np.percentile(raw, 99, axis=0).tolist(),
        },
        "per_sequence": per_sequence,
    }
    (out / "pool_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("sequences", "rows", "pairs", "pair_coverage", "relative",
                       "raw_pixels")}, indent=2))


if __name__ == "__main__":
    main()
