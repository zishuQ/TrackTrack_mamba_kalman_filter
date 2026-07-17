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


MOT20_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build current-timeline MOT20 candidate-A v3 safe-direct rollout "
            "labels for IWG+RG-CMA."
        )
    )
    parser.add_argument("--event-cache-root", required=True)
    parser.add_argument("--detection-cache-root", required=True)
    parser.add_argument("--gt-root", default="/home/shang/datasets/MOT20/train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument(
        "--motion-label-mode",
        choices=["nsa_rollout", "mamba_native_current"],
        default="nsa_rollout",
    )
    args = parser.parse_args()

    event_cache_root = Path(args.event_cache_root).resolve()
    detection_cache_root = Path(args.detection_cache_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    sequences = args.sequence or MOT20_SEQUENCES
    invalid = sorted(set(sequences).difference(MOT20_SEQUENCES))
    if invalid:
        raise ValueError(f"unsupported MOT20 sequences: {invalid}")

    manifests = {}
    for sequence in sequences:
        manifest = build_current_rollout_compact_labels_for_sequence(
            sequence=sequence,
            event_cache_dir=event_cache_root / "MOT20" / "all" / sequence,
            detection_cache_dir=(
                detection_cache_root / "MOT20" / "all" / sequence
            ),
            gt_root=args.gt_root,
            output_root=output_dir,
            motion_label_mode=args.motion_label_mode,
        )
        manifests[sequence] = manifest
        print(
            f"{sequence}: retained={manifest['retained_labels']}/"
            f"{manifest['source_labels']} (100.00%)",
            flush=True,
        )

    available_manifests = {}
    for sequence in MOT20_SEQUENCES:
        manifest_path = output_dir / sequence / "manifest.json"
        if manifest_path.is_file():
            available_manifests[sequence] = json.loads(manifest_path.read_text())
    aggregate = {
        "dataset": "MOT20",
        "event_source": (
            "mamba_native"
            if args.motion_label_mode == "mamba_native_current"
            else "nsa"
        ),
        "motion_target_mode": (
            "mamba_native"
            if args.motion_label_mode == "mamba_native_current"
            else "nsa"
        ),
        "motion_label_mode": args.motion_label_mode,
        "complete": set(available_manifests) == set(MOT20_SEQUENCES),
        "sequences": sorted(available_manifests),
        "source_labels": sum(
            item["source_labels"] for item in available_manifests.values()
        ),
        "retained_labels": sum(
            item["retained_labels"] for item in available_manifests.values()
        ),
        "per_sequence": available_manifests,
        "generation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "generation_source_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in (
                root / "agentguard/src/agentguard/data/cache_reader.py",
                root / "agentguard/src/agentguard/data/compact_iwg_labels.py",
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
