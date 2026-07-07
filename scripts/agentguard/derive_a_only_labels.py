#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _filter_sequence_file(src: Path, dst: Path) -> tuple[int, int]:
    records = json.loads(src.read_text())
    filtered = [
        record
        for record in records
        if str(record.get("candidate_type", "A")).upper() == "A"
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(filtered, indent=2, sort_keys=True))
    return len(records), len(filtered)


def _summary_from_counts(
    *,
    dataset: str,
    mode: str,
    source_dir: Path,
    output_dir: Path,
    per_sequence: dict[str, dict[str, int]],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "mode": mode,
        "label_mode": f"{mode}_a_only",
        "candidate_types": ["A"],
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "num_labels": sum(item["a_count"] for item in per_sequence.values()),
        "num_source_labels": sum(item["source_count"] for item in per_sequence.values()),
        "num_sequences": len(per_sequence),
        "sequences": per_sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive lightweight AgentGuard A-only rollout labels from full A/B/C labels."
    )
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--labels-root", default="outputs/agentguard/labels")
    parser.add_argument("--source-dir", default="", help="Default: <labels-root>/<dataset>/<mode>")
    parser.add_argument("--output-dir", default="", help="Default: <labels-root>/<dataset>/<mode>_a_only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    source_dir = Path(args.source_dir) if args.source_dir else labels_root / args.dataset / args.mode
    output_dir = Path(args.output_dir) if args.output_dir else labels_root / args.dataset / f"{args.mode}_a_only"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source label directory not found: {source_dir}")
    if output_dir.exists() and any(output_dir.glob("*_labels.json")) and not args.overwrite:
        raise FileExistsError(f"Output label directory already exists: {output_dir}. Pass --overwrite.")

    per_sequence: dict[str, dict[str, int]] = {}
    for src in sorted(source_dir.glob("*_labels.json")):
        dst = output_dir / src.name
        source_count, a_count = _filter_sequence_file(src, dst)
        per_sequence[src.name.removesuffix("_labels.json")] = {
            "source_count": int(source_count),
            "a_count": int(a_count),
        }

    if not per_sequence:
        raise RuntimeError(f"No *_labels.json files found in {source_dir}")

    summary = _summary_from_counts(
        dataset=args.dataset,
        mode=args.mode,
        source_dir=source_dir,
        output_dir=output_dir,
        per_sequence=per_sequence,
    )
    summary_path = labels_root / args.dataset / f"{args.mode}_a_only_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
