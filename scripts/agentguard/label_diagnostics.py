#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _records(label_dir: Path, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(label_dir.glob("*_labels.json")):
        data = json.loads(path.read_text())
        for item in data:
            records.append(item)
            if max_records > 0 and len(records) >= max_records:
                return records
    return records


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    values = sorted(values)
    def q(frac: float) -> float:
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * frac))))
        return float(values[idx])
    return {"p05": q(0.05), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "p95": q(0.95)}


def _benefit_stats(values: list[float]) -> dict[str, Any]:
    eps = 1e-6
    near = [v for v in values if abs(v) <= eps]
    soft = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v / 0.1)))) for v in values]
    near_soft = [y for y in soft if abs(y - 0.5) <= 0.05]
    return {
        "count": len(values),
        "mean": _safe_mean(values),
        "positive": sum(v > eps for v in values),
        "negative": sum(v < -eps for v in values),
        "tie": len(near),
        "positive_ratio": sum(v > eps for v in values) / max(len(values), 1),
        "negative_ratio": sum(v < -eps for v in values) / max(len(values), 1),
        "near_zero_ratio": len(near) / max(len(values), 1),
        "soft_near_0p5_ratio_tau_0p1": len(near_soft) / max(len(values), 1),
        **_quantiles(values),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seq = Counter()
    for record in records:
        ctype = str(record.get("candidate_type", "A")).upper()
        by_type[ctype].append(record)
        by_seq[str(record.get("sequence", ""))] += 1

    summary: dict[str, Any] = {
        "total_records": len(records),
        "sequence_counts": dict(sorted(by_seq.items())),
        "candidate_counts": {k: len(v) for k, v in sorted(by_type.items())},
        "candidate": {},
    }
    for ctype, bucket in sorted(by_type.items()):
        target_known = [r for r in bucket if r.get("target_gt_id", -1) not in (-1, None)]
        det_known = [r for r in bucket if r.get("detection_gt_id", -1) not in (-1, None)]
        valid_motion = [r for r in bucket if bool(r.get("valid_motion", False))]
        valid_app = [r for r in bucket if bool(r.get("valid_appearance", False))]
        motion = [float(r.get("motion_benefit", 0.0)) for r in valid_motion]
        app = [float(r.get("appearance_benefit", 0.0)) for r in valid_app]
        same_identity = [
            r for r in bucket
            if r.get("target_gt_id", -1) not in (-1, None)
            and r.get("detection_gt_id", -1) not in (-1, None)
            and int(r.get("target_gt_id")) == int(r.get("detection_gt_id"))
        ]
        other_identity = [
            r for r in bucket
            if r.get("target_gt_id", -1) not in (-1, None)
            and r.get("detection_gt_id", -1) not in (-1, None)
            and int(r.get("target_gt_id")) != int(r.get("detection_gt_id"))
        ]
        summary["candidate"][ctype] = {
            "count": len(bucket),
            "target_known_ratio": len(target_known) / max(len(bucket), 1),
            "detection_gt_known_ratio": len(det_known) / max(len(bucket), 1),
            "same_identity_ratio": len(same_identity) / max(len(bucket), 1),
            "other_identity_ratio": len(other_identity) / max(len(bucket), 1),
            "valid_motion_ratio": len(valid_motion) / max(len(bucket), 1),
            "valid_appearance_ratio": len(valid_app) / max(len(bucket), 1),
            "gt_coverage_mean": _safe_mean([float(r.get("gt_coverage", 0.0)) for r in bucket]),
            "oracle_detection_coverage_mean": _safe_mean(
                [float(r.get("oracle_detection_coverage", 0.0)) for r in bucket]
            ),
            "motion_benefit": _benefit_stats(motion),
            "appearance_benefit": _benefit_stats(app),
            "motion_hard_write_ratio": _safe_mean(
                [float(r.get("motion_oracle_hard", 0)) for r in bucket if r.get("valid_motion", False)]
            ),
            "appearance_hard_write_ratio": _safe_mean(
                [float(r.get("appearance_oracle_hard", 0)) for r in bucket if r.get("valid_appearance", False)]
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--labels-root", default="outputs/agentguard/labels")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    label_dir = Path(args.labels_root) / args.dataset / args.mode
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    records = _records(label_dir, args.max_records)
    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "label_dir": str(label_dir),
        **_summarize(records),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
    else:
        out = label_dir.parent / f"{args.mode}_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
