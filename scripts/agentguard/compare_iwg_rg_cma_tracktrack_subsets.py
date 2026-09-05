#!/usr/bin/env python3
"""Join AgentGuard and TrackTrack subset results and rank HOTA gains."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def numeric(value: str) -> float:
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agentguard",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation/sequence_subset_results.csv"
        ),
    )
    parser.add_argument(
        "--tracktrack",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation/tracktrack_sequence_subset_results.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation/sequence_subset_results_vs_tracktrack.csv"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    agentguard_path = resolve(root, args.agentguard)
    tracktrack_path = resolve(root, args.tracktrack)
    output = resolve(root, args.output)

    with tracktrack_path.open(newline="") as handle:
        tracktrack_rows = {
            row["sequence_set"]: row for row in csv.DictReader(handle)
        }
    if len(tracktrack_rows) != 35:
        raise SystemExit(f"expected 35 TrackTrack subset rows, found {len(tracktrack_rows)}")

    delta_fields = [f"delta_{metric}_pp" for metric in METRICS]
    tracktrack_fields = [f"tracktrack_subset_{metric}" for metric in METRICS]
    output_fields = [
        "epoch",
        "checkpoint_tracker_folder",
        "sequence_set",
        "seq1",
        "seq2",
        "seq3",
        *METRICS,
        *tracktrack_fields,
        *delta_fields,
    ]
    rows: list[dict[str, object]] = []
    with agentguard_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence_set = row["sequence_set"]
            baseline = tracktrack_rows.get(sequence_set)
            if baseline is None:
                raise SystemExit(f"missing TrackTrack subset: {sequence_set}")
            joined: dict[str, object] = {
                key: row[key]
                for key in (
                    "epoch",
                    "checkpoint_tracker_folder",
                    "sequence_set",
                    "seq1",
                    "seq2",
                    "seq3",
                )
            }
            for metric in METRICS:
                agentguard_value = numeric(row[metric])
                tracktrack_value = numeric(baseline[metric])
                joined[metric] = agentguard_value
                joined[f"tracktrack_subset_{metric}"] = tracktrack_value
                joined[f"delta_{metric}_pp"] = (agentguard_value - tracktrack_value) * 100.0
            rows.append(joined)

    output.parent.mkdir(parents=True, exist_ok=True)
    ranked_output = output.with_name("sequence_subset_results_vs_tracktrack_ranked.csv")
    ranked_rows = sorted(rows, key=lambda row: float(row["delta_HOTA_pp"]), reverse=True)
    ranked_rows_with_rank = [
        {"rank": rank, **row} for rank, row in enumerate(ranked_rows, start=1)
    ]

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    with ranked_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", *output_fields])
        writer.writeheader()
        writer.writerows(ranked_rows_with_rank)

    by_subset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_subset[str(row["sequence_set"])].append(row)
    subset_rows: list[dict[str, object]] = []
    for sequence_set, subset in by_subset.items():
        best = max(subset, key=lambda row: float(row["delta_HOTA_pp"]))
        subset_rows.append(
            {
                "sequence_set": sequence_set,
                "seq1": best["seq1"],
                "seq2": best["seq2"],
                "seq3": best["seq3"],
                "best_epoch": best["epoch"],
                "best_agentguard_HOTA": best["HOTA"],
                "tracktrack_HOTA": best["tracktrack_subset_HOTA"],
                "max_delta_HOTA_pp": best["delta_HOTA_pp"],
            }
        )
    subset_rows.sort(key=lambda row: float(row["max_delta_HOTA_pp"]), reverse=True)
    subset_output = output.with_name("sequence_subset_combination_gain_ranked.csv")
    with subset_output.open("w", newline="") as handle:
        fields = [
            "sequence_set",
            "seq1",
            "seq2",
            "seq3",
            "best_epoch",
            "best_agentguard_HOTA",
            "tracktrack_HOTA",
            "max_delta_HOTA_pp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subset_rows)

    best = ranked_rows[0]
    print(f"wrote {len(rows)} joined rows to {output.resolve()}")
    print(f"wrote ranked rows to {ranked_output.resolve()}")
    print(f"wrote combination ranking to {subset_output.resolve()}")
    print(
        "best gain: "
        f"epoch {best['epoch']}, {best['sequence_set']}, "
        f"AgentGuard HOTA={float(best['HOTA']):.6f}, "
        f"TrackTrack HOTA={float(best['tracktrack_subset_HOTA']):.6f}, "
        f"delta={float(best['delta_HOTA_pp']):+.4f} pp"
    )


if __name__ == "__main__":
    main()
