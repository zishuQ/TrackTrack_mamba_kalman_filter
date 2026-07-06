#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_DIR = REPO_ROOT / "3. Tracker"
if str(TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKER_DIR))

from utils.det_feat_storage import compact_index_path


SCHEMA_VERSION = 1


def _load_pickle_once(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return ""


def _normalise_sequences(raw: str | None, available: Iterable[str]) -> list[str]:
    if raw:
        requested = [s.strip() for s in raw.split(",") if s.strip()]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SystemExit(f"Sequences not found in source pickle: {missing}")
        return requested
    return sorted(available)


def _frame_count(frames: dict[int, Any]) -> int:
    if not frames:
        return 0
    return max(int(k) for k in frames.keys())


def _stack_sequence(frames: dict[int, Any]) -> tuple[np.ndarray, ...]:
    num_frames = _frame_count(frames)
    offsets = np.zeros(num_frames + 1, dtype=np.int64)
    boxes_parts: list[np.ndarray] = []
    scores_parts: list[np.ndarray] = []
    features_parts: list[np.ndarray] = []
    sources_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    cursor = 0
    reid_dim = None

    for frame_id in range(1, num_frames + 1):
        offsets[frame_id - 1] = cursor
        dets = frames.get(frame_id)
        if dets is None or len(dets) == 0:
            continue
        arr = np.asarray(dets)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError(f"Frame {frame_id} has invalid detection shape {arr.shape}")
        if reid_dim is None:
            reid_dim = int(arr.shape[1] - 6)
        elif reid_dim != int(arr.shape[1] - 6):
            raise ValueError(
                f"Frame {frame_id} reid_dim={arr.shape[1] - 6}, expected {reid_dim}"
            )

        n = int(arr.shape[0])
        boxes_parts.append(arr[:, :4].astype(np.float32, copy=False))
        scores_parts.append(arr[:, 4].astype(np.float32, copy=False))
        class_parts.append(arr[:, 5].astype(np.int16, copy=False))
        features_parts.append(arr[:, 6:].astype(np.float32, copy=False))
        sources_parts.append(np.zeros(n, dtype=np.uint8))
        cursor += n

    offsets[num_frames] = cursor

    if cursor == 0:
        reid_dim = 0
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        features = np.zeros((0, 0), dtype=np.float32)
        sources = np.zeros((0,), dtype=np.uint8)
        class_ids = np.zeros((0,), dtype=np.int16)
    else:
        boxes = np.concatenate(boxes_parts, axis=0).astype(np.float32, copy=False)
        scores = np.concatenate(scores_parts, axis=0).astype(np.float32, copy=False)
        features = np.concatenate(features_parts, axis=0).astype(np.float32, copy=False)
        sources = np.concatenate(sources_parts, axis=0).astype(np.uint8, copy=False)
        class_ids = np.concatenate(class_parts, axis=0).astype(np.int16, copy=False)

    return offsets, boxes, scores, features, sources, class_ids


def _build_target_indices(
    source_frames: dict[int, Any],
    compact_index: dict | None,
    sequence: str,
    frame_offsets: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if compact_index is None:
        return None, None
    index_by_frame = compact_index.get("index", {}).get(sequence)
    if index_by_frame is None:
        return None, None

    num_frames = len(frame_offsets) - 1
    target_offsets = np.zeros(num_frames + 1, dtype=np.int64)
    parts: list[np.ndarray] = []
    cursor = 0
    for frame_id in range(1, num_frames + 1):
        target_offsets[frame_id - 1] = cursor
        local = index_by_frame.get(frame_id)
        dets = source_frames.get(frame_id)
        if local is None or dets is None or len(dets) == 0:
            continue
        local_arr = np.asarray(local, dtype=np.int64)
        global_start = int(frame_offsets[frame_id - 1])
        parts.append(global_start + local_arr)
        cursor += int(local_arr.size)
    target_offsets[num_frames] = cursor
    if parts:
        return target_offsets, np.concatenate(parts, axis=0).astype(np.int64, copy=False)
    return target_offsets, np.zeros((0,), dtype=np.int64)


def _write_sequence(
    sequence_dir: Path,
    sequence: str,
    frames: dict[int, Any],
    compact_index: dict | None,
    dataset: str,
    split: str,
    source_pickle: Path,
) -> dict:
    temp_dir = sequence_dir.with_name(sequence_dir.name + ".incomplete")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if sequence_dir.exists():
        manifest_path = sequence_dir / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r") as f:
                existing = json.load(f)
            if existing.get("complete") and existing.get("schema_version") == SCHEMA_VERSION:
                return existing
        shutil.rmtree(sequence_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    offsets, boxes, scores, features, sources, class_ids = _stack_sequence(frames)
    target_offsets, target_indices = _build_target_indices(
        frames,
        compact_index,
        sequence,
        offsets,
    )
    if compact_index is not None and (target_offsets is None or target_indices is None):
        raise RuntimeError(
            f"Target compact index does not contain sequence {sequence}; refusing to "
            "write a complete cache that would silently use source detections."
        )

    np.save(temp_dir / "frame_offsets.npy", offsets)
    np.save(temp_dir / "boxes.npy", boxes)
    np.save(temp_dir / "scores.npy", scores)
    np.save(temp_dir / "features.npy", features)
    np.save(temp_dir / "sources.npy", sources)
    np.save(temp_dir / "class_ids.npy", class_ids)
    if target_offsets is not None and target_indices is not None:
        np.save(temp_dir / "target_frame_offsets.npy", target_offsets)
        np.save(temp_dir / "target_detection_indices.npy", target_indices)

    reid_dim = int(features.shape[1]) if features.ndim == 2 else 0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "source_pickle": str(source_pickle),
        "num_frames": int(len(offsets) - 1),
        "num_detections": int(boxes.shape[0]),
        "num_target_detections": int(target_indices.shape[0]) if target_indices is not None else 0,
        "reid_dim": reid_dim,
        "complete": True,
    }
    with (temp_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Verify mmap readability before publish.
    for name in ("frame_offsets", "boxes", "scores", "features", "sources", "class_ids"):
        arr = np.load(temp_dir / f"{name}.npy", mmap_mode="r")
        _ = arr.shape
    os.rename(temp_dir, sequence_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split monolithic detection+FastReID pickle by sequence.")
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-pickle", required=True)
    parser.add_argument("--target-pickle", default=None, help="Logical target pickle used to derive compact index")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "agentguard" / "detection_cache"))
    parser.add_argument("--sequences", default=None, help="Comma-separated sequence list")
    args = parser.parse_args()

    source_pickle = Path(args.source_pickle).resolve()
    target_pickle = Path(args.target_pickle).resolve() if args.target_pickle else None
    if not source_pickle.is_file():
        raise SystemExit(f"source pickle not found: {source_pickle}")

    detections = _load_pickle_once(source_pickle)
    compact_index = None
    if target_pickle is not None:
        idx_path = Path(compact_index_path(str(target_pickle), str(source_pickle)))
        if not idx_path.is_file():
            raise SystemExit(
                f"target compact index not found for {target_pickle}: {idx_path}"
            )
        compact_index = _load_pickle_once(idx_path)

    sequences = _normalise_sequences(args.sequences, detections.keys())
    out_root = Path(args.output_root) / args.dataset / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    seq_manifests = {}
    for seq in sequences:
        seq_manifests[seq] = _write_sequence(
            out_root / seq,
            seq,
            detections[seq],
            compact_index,
            args.dataset,
            args.split,
            source_pickle,
        )
        print(f"wrote {seq}: {seq_manifests[seq]['num_detections']} detections")

    reid_dims = {m["reid_dim"] for m in seq_manifests.values() if m["reid_dim"] > 0}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "reid_dim": int(next(iter(reid_dims))) if len(reid_dims) == 1 else 0,
        "sequences": seq_manifests,
        "source_pickle": str(source_pickle),
        "source_commit": _source_commit(),
        "complete": True,
    }
    with (out_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
