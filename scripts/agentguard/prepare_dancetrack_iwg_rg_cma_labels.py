#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from agentguard.data.compact_iwg_labels import (
    build_current_rollout_compact_labels_for_sequence,
)
from agentguard.datasets.iwg_rg_cma_dataset import DANCETRACK_TRAIN_SEQUENCES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DanceTrack/train candidate-A safe-direct labels for IWG+RG-CMA."
    )
    parser.add_argument("--event-cache-root", required=True)
    parser.add_argument("--detection-cache-root", required=True)
    parser.add_argument("--gt-root", default="/home/shang/datasets/DanceTrack/train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", action="append", default=[])
    args = parser.parse_args()

    event_cache_root = Path(args.event_cache_root).resolve()
    detection_cache_root = Path(args.detection_cache_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    sequences = args.sequence or DANCETRACK_TRAIN_SEQUENCES
    invalid = sorted(set(sequences).difference(DANCETRACK_TRAIN_SEQUENCES))
    if invalid:
        raise ValueError(f"unsupported DanceTrack train sequences: {invalid}")

    for sequence in sequences:
        manifest = build_current_rollout_compact_labels_for_sequence(
            sequence=sequence,
            event_cache_dir=event_cache_root / "DanceTrack" / "train" / sequence,
            detection_cache_dir=(
                detection_cache_root / "DanceTrack" / "train" / sequence
            ),
            gt_root=args.gt_root,
            output_root=output_dir,
        )
        print(
            f"{sequence}: retained={manifest['retained_labels']}/"
            f"{manifest['source_labels']}",
            flush=True,
        )

    available: dict[str, dict] = {}
    for sequence in DANCETRACK_TRAIN_SEQUENCES:
        path = output_dir / sequence / "manifest.json"
        if path.is_file():
            available[sequence] = json.loads(path.read_text())
    aggregate = {
        "dataset": "DanceTrack",
        "split": "train",
        "complete": set(available) == set(DANCETRACK_TRAIN_SEQUENCES),
        "sequences": sorted(available),
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
                root / "agentguard/src/agentguard/rollout_labels.py",
                root / "agentguard/src/agentguard/data/rollout_label_builder.py",
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
