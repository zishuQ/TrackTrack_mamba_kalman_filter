#!/usr/bin/env python3
"""Collect per-checkpoint MOT17 evaluation logs into a long CSV table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")
NUMBER = r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"
METRIC_HEADER = re.compile(r"^HOTA\s+MOTA\s+IDF1\s+DetA\s+AssA$")
METRIC_ROW = re.compile(
    rf"^({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})$"
)
SEQUENCE_ROW = re.compile(
    rf"^(MOT17-[0-9]+-[A-Z]+)\s+({NUMBER})\s+({NUMBER})\s+"
    rf"({NUMBER})\s+({NUMBER})\s+({NUMBER})$"
)
EPOCH_FROM_LOG = re.compile(r"eval_e(\d+)_final_all_raw\.log$")


def parse_metric_row(lines: list[str], start: int) -> tuple[float, ...]:
    for line in lines[start + 1 :]:
        match = METRIC_ROW.match(line.strip())
        if match:
            return tuple(float(value) for value in match.groups())
    raise ValueError(f"metric row missing after line {start + 1}")


def parse_log(path: Path) -> tuple[int, tuple[float, ...], list[tuple[str, tuple[float, ...]]]]:
    match = EPOCH_FROM_LOG.search(path.name)
    if not match:
        raise ValueError(f"unexpected evaluation filename: {path}")
    epoch = int(match.group(1))
    lines = path.read_text(errors="replace").splitlines()

    aggregate = None
    sequence_rows: list[tuple[str, tuple[float, ...]]] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if aggregate is None and METRIC_HEADER.match(line):
            aggregate = parse_metric_row(lines, index)
            continue
        sequence_match = SEQUENCE_ROW.match(line)
        if sequence_match:
            values = tuple(float(value) for value in sequence_match.groups()[1:])
            sequence_rows.append((sequence_match.group(1), values))

    if aggregate is None:
        raise ValueError(f"aggregate metrics missing from {path}")
    if len(sequence_rows) != 7:
        raise ValueError(f"expected 7 sequence rows in {path}, found {len(sequence_rows)}")
    return epoch, aggregate, sequence_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/agentguard/experiments/"
            "iwg_rg_cma_mot17_all_baseline_fullshuffle_seed42_bs1024_100e_ckpt5/"
            "logs/evaluation/all_epochs_final_all_raw_metrics.csv"
        ),
    )
    parser.add_argument("--tracktrack-hota", type=float, default=0.770209)
    args = parser.parse_args()

    logs = sorted(
        args.evaluation_dir.glob("eval_e*_final_all_raw.log"),
        key=lambda path: int(EPOCH_FROM_LOG.search(path.name).group(1)),
    )
    if len(logs) != 20:
        raise SystemExit(f"expected 20 checkpoint logs, found {len(logs)}")

    run_root = args.evaluation_dir.parent.parent
    checkpoint_dir = run_root / "checkpoints"
    rows: list[dict[str, object]] = []
    for log_path in logs:
        epoch, aggregate, sequence_rows = parse_log(log_path)
        checkpoint = checkpoint_dir / f"iwg_rg_cma_epoch{epoch:03d}.pt"
        common = {
            "epoch": epoch,
            "checkpoint": str(checkpoint.resolve()),
            "eval_mode": "final_all_raw",
            "source_log": str(log_path.resolve()),
        }
        rows.append(
            {
                **common,
                "scope": "aggregate",
                "sequence": "ALL",
                **dict(zip(METRICS, aggregate)),
                "tracktrack_all_HOTA": args.tracktrack_hota,
                "hota_delta_vs_tracktrack_pp": (aggregate[0] - args.tracktrack_hota) * 100.0,
            }
        )
        for sequence, values in sequence_rows:
            rows.append(
                {
                    **common,
                    "scope": "sequence",
                    "sequence": sequence,
                    **dict(zip(METRICS, values)),
                    "tracktrack_all_HOTA": "",
                    "hota_delta_vs_tracktrack_pp": "",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "checkpoint",
        "eval_mode",
        "scope",
        "sequence",
        *METRICS,
        "tracktrack_all_HOTA",
        "hota_delta_vs_tracktrack_pp",
        "source_log",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output.resolve()}")
    print(f"aggregate rows: {len(logs)}; sequence rows: {len(rows) - len(logs)}")


if __name__ == "__main__":
    main()
