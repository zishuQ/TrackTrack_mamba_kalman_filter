#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


HOLDOUT_SEQUENCES = ["MOT17-02-FRCNN", "MOT17-11-FRCNN"]
DEFAULT_CASES = {
    "baseline_raw": "mot17_all_0.80_holdout_baseline_raw",
    "base_only_raw": "mot17_all_0.80_holdout_base_only_raw_agentguard_iwg",
    "joint_base_raw": "mot17_all_0.80_holdout_joint_base_raw_agentguard_joint_base",
    "joint_final_raw": "mot17_all_0.80_holdout_joint_final_raw_agentguard_joint_final",
    "baseline_post": "mot17_all_0.80_holdout_baseline_post_post",
    "base_only_post": "mot17_all_0.80_holdout_base_only_post_agentguard_iwg_post",
    "joint_base_post": "mot17_all_0.80_holdout_joint_base_post_agentguard_joint_base_post",
    "joint_final_post": "mot17_all_0.80_holdout_joint_final_post_agentguard_joint_final_post",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_cases(values: list[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CASES)
    cases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"case must be NAME=TRACKER_FOLDER, got {value!r}")
        name, tracker = value.split("=", 1)
        if not name or not tracker or name in cases:
            raise ValueError(f"invalid or duplicate case: {value!r}")
        cases[name] = tracker
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate existing MOT17 tracker files on the strict 02/11 holdout."
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tracker-root", default="outputs/3. track")
    parser.add_argument("--data-dir", default="/home/shang/datasets")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tracker_root = (root / args.tracker_root).resolve()
    tracker_dir = root / "3. Tracker"
    sys.path.insert(0, str(tracker_dir))
    from utils.etc import evaluate_sequences

    cases = _parse_cases(args.case)
    eval_args = SimpleNamespace(
        mode="all",
        data_path=str(Path(args.data_dir).resolve() / "MOT17" / "train"),
        output_dir=str(tracker_root),
        print_per_sequence_metrics=False,
    )
    results = {}
    for name, tracker in cases.items():
        source_hashes = {}
        for sequence in HOLDOUT_SEQUENCES:
            source = tracker_root / tracker / f"{sequence}.txt"
            if not source.is_file():
                raise FileNotFoundError(f"missing tracker result: {source}")
            source_hashes[sequence] = _sha256(source)
        summary = evaluate_sequences(
            eval_args, tracker, "MOT17", HOLDOUT_SEQUENCES
        )
        results[name] = {
            "tracker_folder": tracker,
            "source_sha256": source_hashes,
            **summary,
        }

    manifest = {
        "schema_version": 1,
        "dataset": "MOT17",
        "mode": "all",
        "sequences": HOLDOUT_SEQUENCES,
        "leakage_free": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tracker_root": str(tracker_root),
        "cases": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
