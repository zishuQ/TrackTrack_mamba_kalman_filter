#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from agentguard.data.compact_iwg_labels import (
    build_current_rollout_compact_labels_for_sequence,
)
from agentguard.datasets.iwg_attn_dataset import SPORTSMOT_TRAIN_SEQUENCES
from agentguard.datasets.iwg_attn_dataset import SPORTSMOT_VAL_SEQUENCES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_sequence_labels(
    payload: tuple[str, str, str, str, str, str, str, int],
) -> tuple[str, dict[str, Any]]:
    """Build one sequence in an isolated worker process."""
    (
        sequence,
        source_split,
        event_cache_root,
        detection_cache_root,
        dataset_root,
        gt_root_override,
        output_root,
        future_frames,
    ) = payload
    gt_root = (
        Path(gt_root_override).resolve()
        if gt_root_override
        else Path(dataset_root).resolve() / source_split
    )
    manifest = build_current_rollout_compact_labels_for_sequence(
        sequence=sequence,
        event_cache_dir=(
            Path(event_cache_root) / "SportsMOT" / source_split / sequence
        ),
        detection_cache_dir=(
            Path(detection_cache_root) / "SportsMOT" / source_split / sequence
        ),
        gt_root=gt_root,
        output_root=Path(output_root),
        future_frames=int(future_frames),
    )
    return sequence, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SportsMOT candidate-A safe-direct compact labels."
    )
    parser.add_argument("--event-cache-root", required=True)
    parser.add_argument("--detection-cache-root", required=True)
    parser.add_argument("--split", choices=["train", "val", "trainval"], default="train")
    parser.add_argument(
        "--dataset-root", default="/home/shang/datasets/SportsMOT/dataset"
    )
    parser.add_argument("--gt-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument(
        "--future-frames",
        type=int,
        default=5,
        help="Number of future frames used when constructing rollout labels.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of independent sequence workers; use 1 for sequential mode.",
    )
    args = parser.parse_args()
    if args.future_frames < 1:
        raise ValueError("--future-frames must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.split == "trainval" and args.gt_root:
        raise ValueError(
            "--gt-root cannot represent both official splits; use --dataset-root "
            "with --split trainval"
        )

    event_cache_root = Path(args.event_cache_root).resolve()
    detection_cache_root = Path(args.detection_cache_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    allowed = {
        "train": list(SPORTSMOT_TRAIN_SEQUENCES),
        "val": list(SPORTSMOT_VAL_SEQUENCES),
        "trainval": list(SPORTSMOT_TRAIN_SEQUENCES + SPORTSMOT_VAL_SEQUENCES),
    }[args.split]
    source_splits = {
        **{sequence: "train" for sequence in SPORTSMOT_TRAIN_SEQUENCES},
        **{sequence: "val" for sequence in SPORTSMOT_VAL_SEQUENCES},
    }
    sequences = list(dict.fromkeys(args.sequence or allowed))
    invalid = sorted(set(sequences).difference(allowed))
    if invalid:
        raise ValueError(f"unsupported SportsMOT {args.split} sequences: {invalid}")

    payloads = [
        (
            sequence,
            source_splits[sequence],
            str(event_cache_root),
            str(detection_cache_root),
            str(Path(args.dataset_root).resolve()),
            str(Path(args.gt_root).resolve()) if args.gt_root else "",
            str(output_dir),
            int(args.future_frames),
        )
        for sequence in sequences
    ]

    def report(sequence: str, manifest: dict[str, Any]) -> None:
        print(
            f"{sequence}: retained={manifest['retained_labels']}/"
            f"{manifest['source_labels']}",
            flush=True,
        )

    started = time.monotonic()
    print(
        f"Building {len(payloads)} SportsMOT sequences with "
        f"{args.workers} worker(s); completed manifests are reused.",
        flush=True,
    )
    if args.workers == 1 or len(payloads) <= 1:
        for payload in payloads:
            sequence, manifest = _build_sequence_labels(payload)
            report(sequence, manifest)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_build_sequence_labels, payload): payload[0]
                for payload in payloads
            }
            for future in as_completed(futures):
                sequence, manifest = future.result()
                report(sequence, manifest)
    print(
        f"Label workers finished in {time.monotonic() - started:.1f}s.",
        flush=True,
    )

    available: dict[str, dict] = {}
    for sequence in sequences:
        path = output_dir / sequence / "manifest.json"
        if path.is_file():
            available[sequence] = json.loads(path.read_text())
    aggregate = {
        "dataset": "SportsMOT",
        "split": args.split,
        "future_frames": int(args.future_frames),
        "complete": set(available) == set(sequences),
        "sequences": sorted(available),
        "sequence_source_splits": {
            sequence: source_splits[sequence] for sequence in sorted(available)
        },
        "source_labels": sum(item["source_labels"] for item in available.values()),
        "retained_labels": sum(
            item["retained_labels"] for item in available.values()
        ),
        "per_sequence": available,
        "generation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "generation_source_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in (
                root / "agentguard/src/agentguard/data/cache_reader.py",
                root / "agentguard/src/agentguard/data/compact_iwg_labels.py",
                root / "agentguard/src/agentguard/data/gt_reader.py",
                root / "agentguard/src/agentguard/rollout_labels.py",
                root / "agentguard/src/agentguard/v0_pipeline.py",
            )
        },
    }
    aggregate["retention_rate"] = aggregate["retained_labels"] / max(
        aggregate["source_labels"], 1
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
