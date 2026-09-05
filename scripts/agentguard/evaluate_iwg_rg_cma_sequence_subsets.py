#!/usr/bin/env python3
"""Evaluate every three-sequence MOT17 subset for the existing checkpoints."""

from __future__ import annotations

import argparse
import csv
import itertools
import re
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
EPOCH_RE = re.compile(r"_e(\d+)_final_all_raw_iwg_rg_cma_final$")
TRACKER_TEMPLATE = "mot17_all_0.80_mot17_baseline_e{epoch:03d}_final_all_raw_iwg_rg_cma_final"


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def checkpoint_epochs(tracker_root: Path) -> list[int]:
    epochs = []
    for path in tracker_root.glob("mot17_all_0.80_mot17_baseline_e*_final_all_raw_iwg_rg_cma_final"):
        match = EPOCH_RE.search(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracker-root", type=Path, default=Path("outputs/3. track")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/home/shang/datasets")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation/sequence_subset_results.csv"
        ),
    )
    parser.add_argument("--tracktrack-hota", type=float, default=0.770209)
    parser.add_argument("--epochs", nargs="+", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tracker_root = resolve(root, args.tracker_root)
    data_dir = resolve(root, args.data_dir)
    output = resolve(root, args.output)
    epochs = sorted(args.epochs or checkpoint_epochs(tracker_root))
    if not epochs or any(epoch < 5 or epoch > 100 or epoch % 5 for epoch in epochs):
        raise SystemExit(f"epochs must be checkpoint values from 5..100 step 5, found {epochs}")

    sys.path.insert(0, str(root / "3. Tracker"))
    from utils.etc import evaluate_sequences

    eval_args = SimpleNamespace(
        mode="all",
        data_path=str(data_dir / "MOT17" / "train"),
        output_dir=str(tracker_root),
        print_per_sequence_metrics=False,
    )
    combinations = list(itertools.combinations(SEQUENCES, 3))
    fieldnames = [
        "epoch",
        "checkpoint_tracker_folder",
        "sequence_set",
        "seq1",
        "seq2",
        "seq3",
        "HOTA",
        "MOTA",
        "IDF1",
        "DetA",
        "AssA",
        "tracktrack_all_HOTA",
        "hota_delta_vs_tracktrack_all_pp",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch_index, epoch in enumerate(epochs, start=1):
            tracker_folder = TRACKER_TEMPLATE.format(epoch=epoch)
            tracker_path = tracker_root / tracker_folder
            if not tracker_path.is_dir():
                raise FileNotFoundError(tracker_path)
            missing = [
                sequence
                for sequence in SEQUENCES
                if not (tracker_path / f"{sequence}.txt").is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{tracker_path}: missing {missing}")

            for combo in combinations:
                # evaluate_sequences prints a two-line metric block; keep the
                # batch output compact while retaining progress and CSV rows.
                with redirect_stdout(StringIO()):
                    summary = evaluate_sequences(
                        eval_args, tracker_folder, "MOT17", list(combo)
                    )
                metrics = summary["combined"]
                row = {
                    "epoch": epoch,
                    "checkpoint_tracker_folder": tracker_folder,
                    "sequence_set": "+".join(combo),
                    "seq1": combo[0],
                    "seq2": combo[1],
                    "seq3": combo[2],
                    "HOTA": metrics["HOTA"],
                    "MOTA": metrics["MOTA"],
                    "IDF1": metrics["IDF1"],
                    "DetA": metrics["DetA"],
                    "AssA": metrics["AssA"],
                    "tracktrack_all_HOTA": args.tracktrack_hota,
                    "hota_delta_vs_tracktrack_all_pp": (
                        metrics["HOTA"] - args.tracktrack_hota
                    )
                    * 100.0,
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
            print(
                f"completed epoch {epoch:03d} "
                f"({epoch_index}/{len(epochs)}; {len(combinations)} subsets)",
                flush=True,
            )

    ranked_output = output.with_name("sequence_subset_results_ranked.csv")
    ranked_rows = sorted(
        rows,
        key=lambda row: (float(row["HOTA"]), float(row["hota_delta_vs_tracktrack_all_pp"])),
        reverse=True,
    )
    with ranked_output.open("w", newline="") as handle:
        ranked_fields = ["rank", *fieldnames]
        writer = csv.DictWriter(handle, fieldnames=ranked_fields)
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            writer.writerow({"rank": rank, **row})

    best = ranked_rows[0]
    print(f"wrote {len(rows)} rows to {output.resolve()}")
    print(f"wrote ranked results to {ranked_output.resolve()}")
    print(
        "best subset: "
        f"epoch {best['epoch']}, {best['sequence_set']}, "
        f"HOTA={float(best['HOTA']):.6f}, "
        f"delta={float(best['hota_delta_vs_tracktrack_all_pp']):+.4f} pp"
    )


if __name__ == "__main__":
    main()
