#!/usr/bin/env python3
"""Evaluate the original TrackTrack output on every three-sequence MOT17 subset."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-root", type=Path, default=Path("outputs/3. track"))
    parser.add_argument("--tracker-folder", default="mot17_all_0.80")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/shang/datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation/tracktrack_sequence_subset_results.csv"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tracker_root = resolve(root, args.tracker_root)
    data_dir = resolve(root, args.data_dir)
    output = resolve(root, args.output)
    tracker_path = tracker_root / args.tracker_folder
    if not tracker_path.is_dir():
        raise FileNotFoundError(tracker_path)
    missing = [
        sequence
        for sequence in SEQUENCES
        if not (tracker_path / f"{sequence}.txt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{tracker_path}: missing {missing}")

    sys.path.insert(0, str(root / "3. Tracker"))
    from utils.etc import evaluate_sequences

    eval_args = SimpleNamespace(
        mode="all",
        data_path=str(data_dir / "MOT17" / "train"),
        output_dir=str(tracker_root),
        print_per_sequence_metrics=False,
    )
    fieldnames = [
        "sequence_set",
        "seq1",
        "seq2",
        "seq3",
        "HOTA",
        "MOTA",
        "IDF1",
        "DetA",
        "AssA",
        "tracker_folder",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    combinations = list(itertools.combinations(SEQUENCES, 3))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, combo in enumerate(combinations, start=1):
            with redirect_stdout(StringIO()):
                summary = evaluate_sequences(
                    eval_args, args.tracker_folder, "MOT17", list(combo)
                )
            metrics = summary["combined"]
            writer.writerow(
                {
                    "sequence_set": "+".join(combo),
                    "seq1": combo[0],
                    "seq2": combo[1],
                    "seq3": combo[2],
                    "HOTA": metrics["HOTA"],
                    "MOTA": metrics["MOTA"],
                    "IDF1": metrics["IDF1"],
                    "DetA": metrics["DetA"],
                    "AssA": metrics["AssA"],
                    "tracker_folder": args.tracker_folder,
                }
            )
            handle.flush()
            print(f"completed {index}/{len(combinations)}: {'+'.join(combo)}", flush=True)

    print(f"wrote {len(combinations)} rows to {output.resolve()}")


if __name__ == "__main__":
    main()
