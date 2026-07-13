#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_DESCRIPTOR,
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    validate_rollout_label,
)


RISK_NAMES = (
    "motion_harm",
    "appearance_harm",
    "insufficient_evidence",
    "cross_modal_conflict",
)


def _records(label_dir: Path, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(label_dir.glob("*_labels.json"))
    if not paths:
        raise FileNotFoundError(f"No *_labels.json files found in {label_dir}")
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected a label list in {path}")
        for index, item in enumerate(data):
            try:
                validate_rollout_label(item)
            except ValueError as exc:
                raise ValueError(f"Invalid label {path}:{index}: {exc}") from exc
            records.append(item)
            if max_records > 0 and len(records) >= max_records:
                return records
    return records


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {name: 0.0 for name in ("p05", "p25", "p50", "p75", "p95")}

    def q(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[max(0, min(len(ordered) - 1, index))]

    return {
        "p05": q(0.05),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
    }


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else 0.0,
        **_quantiles(values),
    }


def _channel_summary(records: list[dict[str, Any]], channel: str) -> dict[str, Any]:
    valid_field = f"valid_{channel}"
    valid = [record for record in records if bool(record[valid_field])]
    benefits = [float(record[f"{channel}_benefit"]) for record in valid]
    soft = [float(record[f"{channel}_soft_target"]) for record in valid]
    safe = [float(record[f"{channel}_safe_target"]) for record in valid]
    confidence = [float(record[f"{channel}_label_confidence"]) for record in valid]
    negative = [record for record in valid if float(record[f"{channel}_benefit"]) < 0.0]
    eps = 1e-6
    return {
        "valid_count": len(valid),
        "valid_coverage": len(valid) / max(len(records), 1),
        "benefit": {
            **_distribution(benefits),
            "positive_ratio": sum(value > eps for value in benefits) / max(len(benefits), 1),
            "negative_ratio": sum(value < -eps for value in benefits) / max(len(benefits), 1),
            "near_zero_ratio": sum(abs(value) <= eps for value in benefits) / max(len(benefits), 1),
        },
        "soft_target": _distribution(soft),
        "safe_target": {
            **_distribution(safe),
            "gte_0p95_ratio": sum(value >= 0.95 for value in safe) / max(len(safe), 1),
        },
        "confidence": _distribution(confidence),
        "negative_benefit_subset": {
            "count": len(negative),
            "safe_lt_0p5_ratio": sum(
                float(record[f"{channel}_safe_target"]) < 0.5 for record in negative
            ) / max(len(negative), 1),
            "safe_gte_0p8_ratio": sum(
                float(record[f"{channel}_safe_target"]) >= 0.8 for record in negative
            ) / max(len(negative), 1),
        },
    }


def _risk_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(RISK_NAMES):
        if index == 0:
            selected = [record for record in records if bool(record["valid_motion"])]
        elif index == 1:
            selected = [record for record in records if bool(record["valid_appearance"])]
        else:
            selected = [
                record
                for record in records
                if bool(record["valid_motion"]) and bool(record["valid_appearance"])
            ]
        result[name] = _distribution(record["risk_targets"][index] for record in selected)
    return result


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_records": len(records),
        "motion": _channel_summary(records, "motion"),
        "appearance": _channel_summary(records, "appearance"),
        "risk_targets": _risk_summary(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--labels-root", default="outputs/agentguard/labels")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.label_dir:
        label_dir = Path(args.label_dir)
    else:
        if not args.dataset:
            parser.error("--dataset is required when --label-dir is not provided")
        label_dir = Path(args.labels_root) / args.dataset / args.mode
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    records = _records(label_dir, args.max_records)
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sequence[str(record.get("sequence", ""))].append(record)

    report = {
        "dataset": args.dataset or None,
        "mode": args.mode,
        "label_dir": str(label_dir),
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "label_schema_descriptor": ROLLOUT_LABEL_SCHEMA_DESCRIPTOR,
        "overall": _summary(records),
        "per_sequence": {
            sequence: _summary(sequence_records)
            for sequence, sequence_records in sorted(by_sequence.items())
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    output = Path(args.output) if args.output else label_dir.parent / f"{args.mode}_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
