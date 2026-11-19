#!/usr/bin/env python3
"""Summarize multiple SportsMOT development evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", required=True)
    parser.add_argument("--baseline", default="control")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    data = json.loads(Path(args.parsed).read_text())
    combined = {name: value["combined"] for name, value in data.items()}
    per_sequence = {name: value["per_sequence"] for name, value in data.items()}
    baseline = per_sequence[args.baseline]
    groups = [name for name in data if name != args.baseline]
    keys = ("HOTA", "AssA", "IDF1", "DetA", "MOTA")
    order = [name for name in ("e200", args.baseline) if name in combined]
    order += sorted(name for name in combined if name not in order)
    lines = ["## COMBINED", "", "| group | HOTA | AssA | IDF1 | DetA | MOTA |", "|---|---|---|---|---|---|"]
    for name in order:
        row = combined[name]
        lines.append(
            f"| {name} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | "
            f"{row['DetA']:.4f} | {row['MOTA']:.4f} |"
        )
    lines += [
        "",
        f"## Per-sequence delta vs {args.baseline}",
        "",
        "| sequence | " + " | ".join(
            f"{name} HOTA (Δ) | {name} AssA (Δ)" for name in groups
        ) + " |",
        "|---|" + "---|" * (2 * len(groups)),
    ]
    sequences = sorted(baseline)
    for sequence in sequences:
        cells = []
        for name in groups:
            if sequence not in per_sequence.get(name, {}):
                cells += ["-", "-"]
                continue
            for metric in ("HOTA", "AssA"):
                value = per_sequence[name][sequence][metric]
                base = baseline[sequence][metric]
                cells.append(f"{value:.4f} ({value - base:+.4f})")
        lines.append(f"| {sequence} | " + " | ".join(cells) + " |")
    summary = {"combined": combined, "per_sequence_delta": {}}
    for name in groups:
        deltas = {key: [] for key in keys}
        for sequence in sequences:
            for key in keys:
                deltas[key].append(per_sequence[name][sequence][key] - baseline[sequence][key])
        summary["per_sequence_delta"][name] = {
            key: {
                "mean": sum(values) / len(values),
                "improved": sum(1 for value in values if value > 0),
                "regressed": sum(1 for value in values if value < 0),
            }
            for key, values in deltas.items()
        }
    text = (
        "\n".join(lines)
        + "\n\n```json\n"
        + json.dumps(summary["per_sequence_delta"], indent=1, sort_keys=True)
        + "\n```\n"
    )
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
