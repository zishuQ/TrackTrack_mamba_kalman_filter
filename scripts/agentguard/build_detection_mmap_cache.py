#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_ROOT = REPO_ROOT / "3. Tracker"
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from utils.det_feat_storage import compact_index_path


SCHEMA_VERSION = 1


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _resolve_sequences(
    requested: str | None, available: Iterable[str]
) -> list[str]:
    available = sorted(available)
    if not requested:
        return available
    sequences = [value.strip() for value in requested.split(",") if value.strip()]
    missing = sorted(set(sequences).difference(available))
    if missing:
        raise ValueError(f"sequences missing from source pickle: {missing}")
    return sequences


def _stack_sequence(frames: dict[int, Any]) -> tuple[np.ndarray, ...]:
    num_frames = max((int(frame) for frame in frames), default=0)
    offsets = np.zeros(num_frames + 1, dtype=np.int64)
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    features: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    cursor = 0
    reid_dim: int | None = None
    for frame_id in range(1, num_frames + 1):
        offsets[frame_id - 1] = cursor
        detections = frames.get(frame_id)
        if detections is None or len(detections) == 0:
            continue
        array = np.asarray(detections)
        if array.ndim != 2 or array.shape[1] < 7:
            raise ValueError(f"frame {frame_id} has invalid shape {array.shape}")
        current_reid_dim = int(array.shape[1] - 6)
        if reid_dim is None:
            reid_dim = current_reid_dim
        elif reid_dim != current_reid_dim:
            raise ValueError(
                f"frame {frame_id} reid_dim={current_reid_dim}, expected {reid_dim}"
            )
        count = int(array.shape[0])
        boxes.append(array[:, :4].astype(np.float32, copy=False))
        scores.append(array[:, 4].astype(np.float32, copy=False))
        classes.append(array[:, 5].astype(np.int16, copy=False))
        features.append(array[:, 6:].astype(np.float32, copy=False))
        sources.append(np.zeros(count, dtype=np.uint8))
        cursor += count
    offsets[num_frames] = cursor
    if cursor == 0:
        return (
            offsets,
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int16),
        )
    return (
        offsets,
        np.concatenate(boxes).astype(np.float32, copy=False),
        np.concatenate(scores).astype(np.float32, copy=False),
        np.concatenate(features).astype(np.float32, copy=False),
        np.concatenate(sources).astype(np.uint8, copy=False),
        np.concatenate(classes).astype(np.int16, copy=False),
    )


def _target_indices(
    frames: dict[int, Any],
    compact_index: dict[str, Any],
    sequence: str,
    frame_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    index_by_frame = compact_index.get("index", {}).get(sequence)
    if index_by_frame is None:
        raise ValueError(f"compact target index lacks {sequence}")
    num_frames = int(len(frame_offsets) - 1)
    offsets = np.zeros(num_frames + 1, dtype=np.int64)
    parts: list[np.ndarray] = []
    cursor = 0
    for frame_id in range(1, num_frames + 1):
        offsets[frame_id - 1] = cursor
        local_indices = index_by_frame.get(frame_id)
        detections = frames.get(frame_id)
        if local_indices is None or detections is None or len(detections) == 0:
            continue
        local = np.asarray(local_indices, dtype=np.int64)
        if local.size and (int(local.min()) < 0 or int(local.max()) >= len(detections)):
            raise ValueError(f"target index outside source frame {sequence}:{frame_id}")
        parts.append(int(frame_offsets[frame_id - 1]) + local)
        cursor += int(local.size)
    offsets[num_frames] = cursor
    indices = (
        np.concatenate(parts).astype(np.int64, copy=False)
        if parts
        else np.zeros((0,), dtype=np.int64)
    )
    return offsets, indices


def _write_sequence(
    output_dir: Path,
    *,
    dataset: str,
    split: str,
    sequence: str,
    frames: dict[int, Any],
    compact_index: dict[str, Any],
    source_pickle: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("complete") and int(existing.get("schema_version", 0)) == 1:
            return existing
        raise FileExistsError(f"refusing incomplete detection cache: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing detection cache: {output_dir}")
    temporary = output_dir.with_name(f"{output_dir.name}.incomplete.{os.getpid()}")
    temporary.mkdir(parents=True)
    offsets, boxes, scores, features, sources, classes = _stack_sequence(frames)
    target_offsets, target_indices = _target_indices(
        frames, compact_index, sequence, offsets
    )
    arrays = {
        "frame_offsets": offsets,
        "boxes": boxes,
        "scores": scores,
        "features": features,
        "sources": sources,
        "class_ids": classes,
        "target_frame_offsets": target_offsets,
        "target_detection_indices": target_indices,
    }
    for name, value in arrays.items():
        np.save(temporary / f"{name}.npy", value, allow_pickle=False)
        mapped = np.load(temporary / f"{name}.npy", mmap_mode="r")
        if mapped.shape != value.shape:
            raise ValueError(f"failed mmap verification for {sequence}/{name}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "source_pickle": str(source_pickle),
        "num_frames": int(len(offsets) - 1),
        "num_detections": int(len(boxes)),
        "num_target_detections": int(len(target_indices)),
        "reid_dim": int(features.shape[1]),
        "complete": True,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-sequence mmap detection caches from compact-index pickles."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source-pickle", required=True)
    parser.add_argument("--target-pickle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequences", default="")
    args = parser.parse_args()

    source_pickle = Path(args.source_pickle).resolve()
    target_pickle = Path(args.target_pickle).resolve()
    index_path = Path(compact_index_path(str(target_pickle), str(source_pickle)))
    if not source_pickle.is_file():
        raise FileNotFoundError(source_pickle)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    detections = _load_pickle(source_pickle)
    compact_index = _load_pickle(index_path)
    sequences = _resolve_sequences(args.sequences, detections)
    split_root = Path(args.output_root).resolve() / args.dataset / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    for sequence in sequences:
        manifest = _write_sequence(
            split_root / sequence,
            dataset=args.dataset,
            split=args.split,
            sequence=sequence,
            frames=detections[sequence],
            compact_index=compact_index,
            source_pickle=source_pickle,
        )
        manifests[sequence] = manifest
        print(
            f"{sequence}: frames={manifest['num_frames']} "
            f"source={manifest['num_detections']} "
            f"target={manifest['num_target_detections']}",
            flush=True,
        )
    reid_dims = {int(item["reid_dim"]) for item in manifests.values()}
    if reid_dims != {2048}:
        raise ValueError(f"unexpected ReID dimensions: {sorted(reid_dims)}")
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "reid_dim": 2048,
        "sequences": manifests,
        "source_pickle": str(source_pickle),
        "source_commit": _source_commit(),
        "complete": True,
    }
    (split_root / "manifest.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(f"manifest: {split_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
