#!/usr/bin/env python3
"""Compare optimized GT+LOO labels against the frozen pre-change builder.

Writes a report into an independent directory. Never overwrites train_data_v2,
detection cache, ReID, or experiment outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "agentguard/src"))

from agentguard.data.gt_reader import GTReader
from agentguard.data.rollout_label_builder import (
    best_oracle_detection,
    build_compact_rollout_labels_for_sequence,
    gt_box_for_id,
)
from agentguard.data.train_data_v2 import PackedReplayReader, RULES, table_row

FROZEN_DIR = REPO / "agentguard/tests/frozen_unoptimized"
FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-10
DISCRETE_FIELDS = (
    "event_id", "event_shard_id", "event_offset", "sequence", "frame_id",
    "frame_index", "track_id", "candidate_type", "candidate_detection_index",
    "target_gt_id", "target_identity_key", "detection_gt_id", "valid_motion",
    "valid_appearance", "valid_horizon_count", "sample_type",
    "motion_oracle_hard", "appearance_oracle_hard", "target_gate",
    "motion_label_mode",
)
FLOAT_FIELDS = (
    "detection_gt_iou", "motion_benefit", "appearance_benefit", "gt_coverage",
    "oracle_detection_coverage", "motion_soft_target", "appearance_soft_target",
    "motion_safe_target", "appearance_safe_target", "motion_label_confidence",
    "appearance_label_confidence", "sample_weight",
)
VECTOR_FIELDS = (
    "cue_target", "risk_targets", "policy_soft_target", "policy_safe_soft_target",
)


def load_frozen_builder():
    spec = importlib.util.spec_from_file_location(
        "frozen_unoptimized_rollout_label_builder",
        FROZEN_DIR / "rollout_label_builder.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compare_labels(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError(f"label count {len(actual)} != {len(expected)}")
    max_err = 0.0
    max_field = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        for field in DISCRETE_FIELDS:
            if got[field] != want[field]:
                raise AssertionError(f"{field} mismatch at {index}: {got[field]!r} vs {want[field]!r}")
        for field in FLOAT_FIELDS:
            err = abs(float(got[field]) - float(want[field]))
            if err > max_err:
                max_err, max_field = err, field
            if not np.allclose(got[field], want[field], rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
                raise AssertionError(f"{field} mismatch at {index}: {got[field]} vs {want[field]} err={err}")
        for field in VECTOR_FIELDS:
            a = np.asarray(got[field], dtype=np.float64)
            b = np.asarray(want[field], dtype=np.float64)
            err = float(np.max(np.abs(a - b))) if a.size else 0.0
            if err > max_err:
                max_err, max_field = err, field
            if not np.allclose(a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
                raise AssertionError(f"{field} mismatch at {index}: err={err}")
    return max_err, max_field


def rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def collect_oracle_indices(reader, gt_reader, labels, future_frames, selector):
    rows = []
    for label in labels:
        frame_id = int(label["frame_id"])
        gt_id = int(label["target_gt_id"])
        indices = []
        total_frames = int(
            reader.manifest.get("total_sequence_frames")
            or reader.manifest.get("num_frames")
            or 0
        )
        for step in range(1, future_frames + 1):
            fid = frame_id + step
            if total_frames and fid > total_frames:
                break
            gt_box = gt_box_for_id(gt_reader, fid, gt_id)
            det = selector(reader, fid - 1, gt_box)
            indices.append(None if det is None else int(det.detection_index))
        rows.append(indices)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="MOT17-09-FRCNN")
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=REPO / "outputs/agentguard/train_data_v2",
    )
    parser.add_argument(
        "--detection-cache-root",
        type=Path,
        default=REPO / "outputs/agentguard/detection_cache",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/home/shang/datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp") / "agentguard_gt_loo_compare",
    )
    parser.add_argument("--future-frames", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    sequence = args.sequence
    seq_dir = args.v2_root.resolve() / "MOT17" / sequence
    det_dir = args.detection_cache_root.resolve() / "MOT17/all" / sequence
    gt_root = args.data_dir.resolve() / "MOT17/train"
    if not (seq_dir / "data.pt").is_file():
        raise FileNotFoundError(seq_dir / "data.pt")
    if not (det_dir / "manifest.json").is_file():
        raise FileNotFoundError(det_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.resolve() == (REPO / "outputs/agentguard/train_data_v2").resolve():
        raise ValueError("refusing to write comparison output into train_data_v2")

    frozen = load_frozen_builder()
    os.environ["AGENTGUARD_LABEL_PROFILE"] = "1"

    report: dict = {
        "sequence": sequence,
        "v2_dir": str(seq_dir),
        "detection_dir": str(det_dir),
        "output_dir": str(args.output_dir.resolve()),
        "float_atol": FLOAT_ATOL,
        "float_rtol": FLOAT_RTOL,
        "cache": {
            "oracle_key": "(replay_dir, detection_dir, frame_index, gt_id, min_iou)",
            "oracle_lru": 4096,
            "frame_box_limit": "max(16, future_frames+3), evict frame_index < current frame_id",
            "warp_limit": "max(16, future_frames+3), same eviction",
            "loo": "prepared L2-normalised candidates per identity; cache final prototype only for identical excluded detection",
            "release": "per-sequence object, dropped when the builder returns; mmap closed with the reader",
        },
    }

    rss0 = rss_kb()
    t0 = time.perf_counter()
    old_reader = PackedReplayReader(seq_dir, det_dir)
    old_labels, old_summary = frozen.build_compact_rollout_labels_for_sequence(
        seq_dir,
        det_dir,
        gt_root,
        reader=old_reader,
        persist_stats=False,
        max_events=args.max_events,
        future_frames=args.future_frames,
        **RULES,
    )
    old_s = time.perf_counter() - t0
    old_rss = rss_kb()
    report["frozen"] = {
        "seconds": old_s,
        "peak_rss_kb": old_rss,
        "rss_delta_kb": old_rss - rss0,
        "num_labels": len(old_labels),
        "cache": "cold process start; OS page cache empty-ish; no in-process reuse",
        "summary": {k: v for k, v in old_summary.items() if k != "dataset_stats"},
    }

    t1 = time.perf_counter()
    new_reader = PackedReplayReader(seq_dir, det_dir)
    new_labels, new_summary = build_compact_rollout_labels_for_sequence(
        seq_dir,
        det_dir,
        gt_root,
        reader=new_reader,
        persist_stats=False,
        max_events=args.max_events,
        future_frames=args.future_frames,
        **RULES,
    )
    new_s = time.perf_counter() - t1
    new_rss = rss_kb()
    report["optimized"] = {
        "seconds": new_s,
        "peak_rss_kb": new_rss,
        "rss_delta_kb_from_start": new_rss - rss0,
        "num_labels": len(new_labels),
        "cache": "hot OS page cache after frozen run; in-process caches start empty",
        "profile": new_summary.get("profile"),
        "summary": {k: v for k, v in new_summary.items() if k not in {"dataset_stats", "profile"}},
    }

    t2 = time.perf_counter()
    hot_reader = PackedReplayReader(seq_dir, det_dir)
    hot_labels, hot_summary = build_compact_rollout_labels_for_sequence(
        seq_dir,
        det_dir,
        gt_root,
        reader=hot_reader,
        persist_stats=False,
        max_events=args.max_events,
        future_frames=args.future_frames,
        **RULES,
    )
    hot_s = time.perf_counter() - t2
    report["optimized_hot"] = {
        "seconds": hot_s,
        "peak_rss_kb": rss_kb(),
        "cache": "second optimized run in the same process; OS page cache hot; in-process caches still rebuilt per call",
        "profile": hot_summary.get("profile"),
    }

    max_err, max_field = compare_labels(new_labels, old_labels)
    compare_labels(hot_labels, new_labels)
    report["label_compare"] = {
        "count_match": len(new_labels) == len(old_labels),
        "discrete_fields": list(DISCRETE_FIELDS),
        "max_abs_float_error": max_err,
        "max_error_field": max_field,
        "atol": FLOAT_ATOL,
        "rtol": FLOAT_RTOL,
    }

    packed = torch.load(seq_dir / "data.pt", map_location="cpu", mmap=True, weights_only=True)
    stored_err = 0.0
    stored_field = None
    if args.max_events == 0:
        stored = [table_row(packed["labels_raw"], i) for i in range(packed["labels_raw"]["rows"])]
        stored_err, stored_field = compare_labels(new_labels, stored)
    report["stored_labels_raw"] = {
        "compared": args.max_events == 0,
        "max_abs_float_error": stored_err,
        "max_error_field": stored_field,
    }

    gt_reader = GTReader(str(gt_root), sequence)
    oracle_reader = PackedReplayReader(seq_dir, det_dir)
    try:
        new_oracle = collect_oracle_indices(
            oracle_reader, gt_reader, new_labels, args.future_frames, best_oracle_detection
        )
        frozen_oracle = collect_oracle_indices(
            oracle_reader,
            gt_reader,
            new_labels,
            args.future_frames,
            frozen.best_oracle_detection,
        )
    finally:
        oracle_reader.close()
    oracle_mismatch = sum(1 for a, b in zip(new_oracle, frozen_oracle) if a != b)
    report["oracle_indices"] = {
        "compared_labels": len(new_oracle),
        "mismatched_events": oracle_mismatch,
    }

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    torch.save({"labels": new_labels, "summary": new_summary}, args.output_dir / "optimized_labels.pt")
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
