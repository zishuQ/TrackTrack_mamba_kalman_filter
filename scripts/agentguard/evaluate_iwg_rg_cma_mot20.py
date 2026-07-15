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


SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
REQUIRED_CASES = {"base_raw", "final_raw", "base_post", "final_post"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_cases(values: list[str]) -> dict[str, str]:
    cases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"case must be NAME=TRACKER_FOLDER, got {value!r}")
        name, tracker = value.split("=", 1)
        if not name or not tracker or name in cases:
            raise ValueError(f"invalid or duplicate case: {value!r}")
        cases[name] = tracker
    if set(cases) != REQUIRED_CASES:
        raise ValueError(f"cases must be exactly {sorted(REQUIRED_CASES)}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed MOT20/all IWG+RG-CMA base/final raw/post cases."
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--tracker-root", default="outputs/3. track")
    parser.add_argument("--data-dir", default="/home/shang/datasets")
    parser.add_argument("--baseline", default="mot20_all_0.80")
    parser.add_argument("--baseline-post", default="mot20_all_0.80_post")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tracker_root = (root / args.tracker_root).resolve()
    sys.path.insert(0, str(root / "3. Tracker"))
    from utils.etc import evaluate_sequences

    eval_args = SimpleNamespace(
        mode="all",
        data_path=str(Path(args.data_dir).resolve() / "MOT20" / "train"),
        output_dir=str(tracker_root),
        print_per_sequence_metrics=False,
    )
    requested = _parse_cases(args.case)
    if args.baseline:
        requested = {"baseline_raw": args.baseline, **requested}
    if args.baseline_post:
        requested = {"baseline_post": args.baseline_post, **requested}
    results = {}
    for name, tracker in requested.items():
        hashes = {}
        for sequence in SEQUENCES:
            source = tracker_root / tracker / f"{sequence}.txt"
            if not source.is_file():
                raise FileNotFoundError(f"missing tracker result: {source}")
            hashes[sequence] = _sha256(source)
        results[name] = {
            "tracker_folder": tracker,
            "source_sha256": hashes,
            **evaluate_sequences(eval_args, tracker, "MOT20", SEQUENCES),
        }

    hota = {name: value["combined"]["HOTA"] for name, value in results.items()}
    comparisons = {
        "final_raw_minus_base_raw": hota["final_raw"] - hota["base_raw"],
        "final_post_minus_base_post": hota["final_post"] - hota["base_post"],
        "base_post_minus_base_raw": hota["base_post"] - hota["base_raw"],
        "final_post_minus_final_raw": hota["final_post"] - hota["final_raw"],
    }
    if "baseline_raw" in hota:
        comparisons.update(
            {
                "base_raw_minus_baseline_raw": hota["base_raw"]
                - hota["baseline_raw"],
                "final_raw_minus_baseline_raw": hota["final_raw"]
                - hota["baseline_raw"],
            }
        )
    if "baseline_post" in hota:
        comparisons.update(
            {
                "base_post_minus_baseline_post": hota["base_post"]
                - hota["baseline_post"],
                "final_post_minus_baseline_post": hota["final_post"]
                - hota["baseline_post"],
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": "MOT20",
        "mode": "all",
        "sequences": SEQUENCES,
        "training_evaluation": True,
        "validation_used_for_selection": False,
        "tracker_seed": 10000,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "promotion_gate": None,
        "comparisons": comparisons,
        "cases": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
