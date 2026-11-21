#!/usr/bin/env python3
"""Validate a formal B-recipe paired dataset for one dataset/split.

Checks the expected sequence set, required shard files, event-key uniqueness,
sample/detection-frame consistency, mask coverage and the schema-3 full-write
protocol.  Exit code is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from paired_training_input import official_sequences  # noqa: E402
from paired_collection_io import atomic_json, inspect_shard, stream_nonfinite_rows
from agentguard.data.detection_cache import SequenceDetectionCache  # noqa: E402

CACHE = MAIN / "outputs/agentguard/detection_cache"
REQUIRED_FILES = (
    "samples.npz",
    "event_evidence.npy",
    "event_track.npy",
    "event_detection_index.npy",
    "event_has_detection.npy",
    "summary.json",
)
SAMPLE_ROWS = 200


def expected_sequences(dataset: str, split: str) -> set[str]:
    if dataset == "SportsMOT":
        sequences = official_sequences("train", dataset)
        if split == "trainval":
            sequences |= official_sequences("val", dataset)
        return sequences
    if split != "train":
        raise SystemExit(f"{dataset} only supports split=train")
    return official_sequences("train", dataset)


def cache_dir(dataset: str, split: str, sequence: str) -> Path:
    cache_split = "all" if dataset != "SportsMOT" else split
    return CACHE / dataset / cache_split / sequence


def sequence_split(dataset: str, split: str, sequence: str) -> str:
    if dataset != "SportsMOT":
        return "all"
    if split == "train":
        return "train"
    return "val" if sequence in official_sequences("val", dataset) else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("SportsMOT", "MOT17", "MOT20"), required=True)
    parser.add_argument("--split", choices=("train", "trainval"), required=True)
    parser.add_argument("--expect-full-write", action="store_true")
    parser.add_argument("--require-rank", action="store_true", default=True)
    parser.add_argument("--require-column", action="store_true")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Explicit sequence list; the data root must match it exactly.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    expected = expected_sequences(args.dataset, args.split)
    if args.sequences:
        requested = set(args.sequences)
        unknown = sorted(requested - expected)
        if unknown:
            raise SystemExit(
                f"Requested sequences are not in the {args.split} split: {unknown}"
            )
        expected = requested
    found = {
        path.name for path in data_root.iterdir() if (path / "samples.npz").is_file()
    } if data_root.is_dir() else set()
    reasons: list[str] = []
    if not found:
        reasons.append(f"no sequences under {data_root}")
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing:
        reasons.append(f"missing {len(missing)} sequences: {missing[:8]}")
    if extra:
        reasons.append(f"unexpected {len(extra)} sequences: {extra[:8]}")
    totals = {
        "dataset": args.dataset,
        "split": args.split,
        "expected_sequences": len(expected),
        "found_sequences": len(found),
        "samples": 0,
        "valid_motion": 0,
        "motion_gt_valid": 0,
        "valid_appearance": 0,
        "rank_valid": 0,
        "column_valid": 0,
        "column_checked": 0,
        "column_mismatch": 0,
        "finite_bad": 0,
        "key_checked": 0,
        "key_mismatch": 0,
        "index_checked": 0,
        "index_mismatch": 0,
        "per_sequence": {},
        "reasons": reasons,
    }
    for sequence in sorted(expected & found):
        directory = data_root / sequence
        missing_files = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
        if missing_files:
            reasons.append(f"{sequence}: missing files {missing_files}")
            continue
        summary = json.loads((directory / "summary.json").read_text())
        protocol = summary.get("protocol") or {}
        if not summary.get("complete"):
            reasons.append(f"{sequence}: summary not complete")
        if protocol.get("schema") != 3:
            reasons.append(f"{sequence}: schema != 3 ({protocol.get('schema')})")
        if args.expect_full_write and not protocol.get("full_write"):
            reasons.append(f"{sequence}: not a full-write replay")
        if protocol.get("dataset", "SportsMOT") != args.dataset:
            reasons.append(f"{sequence}: protocol dataset mismatch")
        if args.require_rank and not protocol.get("rank_arrays"):
            reasons.append(f"{sequence}: rank arrays missing")
        try:
            inspect_shard(directory)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            reasons.append(f"{sequence}: invalid collected shard: {exc}")
            continue
        # Keep small metadata once; large rank features are checked as streams.
        rank_features = {"rank_template0", "rank_detection", "rank_positive", "rank_competitor"}
        with np.load(directory / "samples.npz", allow_pickle=False) as archive:
            sample_members = set(archive.files)
            samples = {name: archive[name] for name in archive.files if name not in rank_features}
        count = int(samples["frame"].size)
        if count == 0:
            reasons.append(f"{sequence}: zero samples")
            continue
        totals["samples"] += count
        required_arrays = (
            "frame",
            "track_id",
            "detection_index",
            "gt_id",
            "window_indices",
            "current_event_index",
            "current_competition",
            "motion_gt_valid",
            "valid_motion",
            "valid_appearance",
            "rank_valid",
            "pred_box_g0",
            "pred_box_g1",
            "gt_box_t1",
        )
        for key in required_arrays:
            if key not in sample_members:
                reasons.append(f"{sequence}: missing array {key}")
            elif samples[key].shape[0] != count:
                reasons.append(f"{sequence}: {key} length mismatch")
        keys = json.loads(str(samples["keys_json"]))
        totals["key_checked"] += len(keys)
        unique = {
            (item.get("sequence"), item.get("frame"), item.get("track_id"), item.get("detection_index"))
            for item in keys
        }
        if len(keys) != count or len(unique) != len(keys):
            totals["key_mismatch"] += 1
            reasons.append(f"{sequence}: duplicate or mis-sized event keys")
        rank_rows = samples["rank_rows"]
        if rank_rows.size != int(samples["rank_valid"].sum()):
            reasons.append(f"{sequence}: rank_rows != rank_valid count")
        if rank_rows.size and (
            rank_rows.min() < 0 or rank_rows.max() >= count
        ):
            reasons.append(f"{sequence}: rank_rows out of range")
        if args.require_column:
            if not protocol.get("column"):
                reasons.append(f"{sequence}: protocol column missing")
            column_keys = ("column_det", "column_offset", "column_rival", "column_score")
            missing_column = [key for key in column_keys if key not in sample_members]
            if missing_column:
                reasons.append(f"{sequence}: missing arrays {missing_column}")
            else:
                column_det = samples["column_det"]
                if column_det.shape[0] != count:
                    reasons.append(f"{sequence}: column_det length mismatch")
                if column_det.size and column_det.min() < -1:
                    reasons.append(f"{sequence}: column_det below -1")
                totals["column_valid"] += int((column_det >= 0).sum())
            column_counts = summary.get("column_counts") or {}
            if column_counts.get("actual_mismatch") != 0:
                reasons.append(
                    f"{sequence}: column actual_mismatch="
                    f"{column_counts.get('actual_mismatch')}"
                )

        def nonfinite_rows(array, mask) -> int:
            values = np.asarray(array)
            if values.ndim > 1:
                return int((~np.isfinite(values[mask])).any(axis=1).sum())
            return int((~np.isfinite(values[mask])).sum())

        finite_checks = (
            ("motion_benefit", samples["motion_benefit"], samples["valid_motion"].astype(bool)),
            ("appearance_benefit", samples["appearance_benefit"], samples["valid_appearance"].astype(bool)),
            ("appearance_delta", samples["appearance_delta"], samples["appearance_diag_valid"].astype(bool)),
            ("pred_box_g0", samples["pred_box_g0"], samples["motion_gt_valid"].astype(bool)),
            ("pred_box_g1", samples["pred_box_g1"], samples["motion_gt_valid"].astype(bool)),
            ("gt_box_t1", samples["gt_box_t1"], samples["motion_gt_valid"].astype(bool)),
        )
        if "rank_template0" in sample_members:
            for key in sorted(rank_features & sample_members):
                bad = stream_nonfinite_rows(directory / "samples.npz", key)
                if bad:
                    totals["finite_bad"] += bad
                    reasons.append(f"{sequence}: {bad} non-finite {key} values")
        if args.require_column and "column_det" in sample_members:
            column_mask = samples["column_det"] >= 0
            for key in ("column_offset", "column_rival", "column_score"):
                if key in sample_members:
                    finite_checks += ((key, samples[key], column_mask),)
        for key, values, mask in finite_checks:
            bad = nonfinite_rows(values, mask)
            if bad:
                totals["finite_bad"] += bad
                reasons.append(f"{sequence}: {bad} non-finite {key} values")
        totals["valid_motion"] += int(samples["valid_motion"].sum())
        totals["motion_gt_valid"] += int(samples["motion_gt_valid"].sum())
        totals["valid_appearance"] += int(samples["valid_appearance"].sum())
        totals["rank_valid"] += int(samples["rank_valid"].sum())

        index_before = totals["index_mismatch"]
        column_before = totals["column_mismatch"]
        split = sequence_split(args.dataset, args.split, sequence)
        cache = SequenceDetectionCache(cache_dir(args.dataset, split, sequence))
        try:
            rows = np.arange(count)
            if rows.size > SAMPLE_ROWS:
                rows = rows[:: max(1, rows.size // SAMPLE_ROWS)][:SAMPLE_ROWS]
            if args.require_column and "column_det" in sample_members:
                column_rows = np.nonzero(samples["column_det"] >= 0)[0]
                if column_rows.size:
                    step = max(1, column_rows.size // SAMPLE_ROWS)
                    rows = np.unique(
                        np.concatenate([rows, column_rows[::step][:SAMPLE_ROWS]])
                    )
            frames = samples["frame"]
            for row in rows:
                index = int(samples["detection_index"][row])
                totals["index_checked"] += 1
                if index < 0 or index >= cache.num_detections:
                    totals["index_mismatch"] += 1
                    continue
                frame_index = int(
                    np.searchsorted(cache.frame_offsets, index, side="right") - 1
                )
                if frame_index + 1 != int(frames[row]):
                    totals["index_mismatch"] += 1
                if args.require_column and "column_det" in sample_members:
                    column_det = int(samples["column_det"][row])
                    if column_det >= 0:
                        totals["column_checked"] += 1
                        if column_det >= cache.num_detections:
                            totals["column_mismatch"] += 1
                            continue
                        column_frame = int(
                            np.searchsorted(cache.frame_offsets, column_det, side="right") - 1
                        )
                        if column_frame != int(frames[row]):
                            totals["column_mismatch"] += 1
        finally:
            cache.close()
        if totals["index_mismatch"] > index_before:
            reasons.append(f"{sequence}: detection index/frame mismatches")
        if totals["column_mismatch"] > column_before:
            reasons.append(f"{sequence}: column detection/frame mismatches")
        totals["per_sequence"][sequence] = {
            "samples": count,
            "valid_motion": int(samples["valid_motion"].sum()),
            "motion_gt_valid": int(samples["motion_gt_valid"].sum()),
            "valid_appearance": int(samples["valid_appearance"].sum()),
            "rank_valid": int(samples["rank_valid"].sum()),
            "full_write": bool(protocol.get("full_write")),
        }
    totals["data_ok"] = not reasons and len(found & expected) == len(expected)
    if args.out:
        atomic_json(args.out, totals)
    print(json.dumps({key: totals[key] for key in (
        "dataset", "split", "expected_sequences", "found_sequences", "samples",
        "valid_motion", "motion_gt_valid", "valid_appearance", "rank_valid",
        "column_valid", "column_checked", "column_mismatch", "finite_bad",
        "key_checked", "key_mismatch", "index_checked", "index_mismatch", "data_ok",
    )}, indent=1))
    if reasons:
        print(json.dumps({"reasons": reasons[:12]}, indent=1))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
