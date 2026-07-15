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


SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


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
    required = {"base_raw", "final_raw", "base_post", "final_post"}
    if set(cases) != required:
        raise ValueError(f"evaluation cases must be exactly {sorted(required)}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed MOT17/all RG-CMA cases and apply the promotion gate."
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--tracker-root", default="outputs/3. track")
    parser.add_argument("--data-dir", default="/home/shang/datasets")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tracker_root = (root / args.tracker_root).resolve()
    sys.path.insert(0, str(root / "3. Tracker"))
    from utils.etc import evaluate_sequences

    eval_args = SimpleNamespace(
        mode="all",
        data_path=str(Path(args.data_dir).resolve() / "MOT17" / "train"),
        output_dir=str(tracker_root),
        print_per_sequence_metrics=False,
    )
    results = {}
    for name, tracker in _parse_cases(args.case).items():
        hashes = {}
        for sequence in SEQUENCES:
            source = tracker_root / tracker / f"{sequence}.txt"
            if not source.is_file():
                raise FileNotFoundError(f"missing tracker result: {source}")
            hashes[sequence] = _sha256(source)
        results[name] = {
            "tracker_folder": tracker,
            "source_sha256": hashes,
            **evaluate_sequences(eval_args, tracker, "MOT17", SEQUENCES),
        }

    hota = {name: value["combined"]["HOTA"] for name, value in results.items()}
    checks = {
        "final_raw_ge_old_iwg_raw": hota["final_raw"] >= 0.784148,
        "final_post_gt_old_iwg_post": hota["final_post"] > 0.791127,
        "final_raw_ge_same_checkpoint_base": hota["final_raw"] >= hota["base_raw"],
        "final_post_ge_same_checkpoint_base": hota["final_post"] >= hota["base_post"],
    }
    manifest = {
        "schema_version": 1,
        "dataset": "MOT17",
        "mode": "all",
        "sequences": SEQUENCES,
        "training_evaluation": True,
        "validation_used_for_selection": False,
        "tracker_seed": 10000,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "promotion_thresholds": {
            "old_iwg_raw_hota": 0.784148,
            "old_iwg_post_hota": 0.791127,
        },
        "promotion_checks": checks,
        "promoted": all(checks.values()),
        "cases": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
