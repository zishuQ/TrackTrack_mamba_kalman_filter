"""Read-only, small-array diagnosis of the existing one-step motion objective.

No torch, tracker replay, ReID feature reads or model/checkpoint changes.
"""
from __future__ import annotations
import argparse
import json
import os
import resource
import time
from pathlib import Path
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np


def losses(box, target):
    inter_wh = np.maximum(0, np.minimum(box[:, 2:], target[:, 2:]) -
                          np.maximum(box[:, :2], target[:, :2]))
    inter = inter_wh.prod(1)
    a = np.maximum(0, box[:, 2:] - box[:, :2]).prod(1)
    b = np.maximum(0, target[:, 2:] - target[:, :2]).prod(1)
    iou = inter / np.maximum(a + b - inter, 1e-9)
    centers = (box[:, :2] + box[:, 2:] - target[:, :2] - target[:, 2:]) / 2
    enclosing = np.maximum(box[:, 2:], target[:, 2:]) - np.minimum(box[:, :2], target[:, :2])
    return 1 - iou + np.square(centers).sum(1) / np.maximum(np.square(enclosing).sum(1), 1e-9)


def compute(path):
    with np.load(path, allow_pickle=False) as archive:
        mask = archive["motion_gt_valid"].astype(bool)
        b0 = archive["pred_box_g0"][mask].astype(np.float64)
        b1 = archive["pred_box_g1"][mask].astype(np.float64)
        target = archive["gt_box_t1"][mask].astype(np.float64)
    if not (np.isfinite(b0).all() and np.isfinite(b1).all() and np.isfinite(target).all()):
        raise ValueError(f"nonfinite valid endpoint: {path}")
    n = len(b0)
    grid = np.unique(np.r_[np.linspace(0, 1, 21), .9, .99, .999, .9999])
    delta = b1 - b0
    all_losses = np.stack([losses(b0 + g * delta, target) for g in grid])
    best = all_losses.min(0)
    optimal = grid[all_losses.argmin(0)]
    l0, l1 = all_losses[0], all_losses[-1]
    h = 1e-5
    derivatives = {}
    for g in (.5, .99, .9999):
        d = (losses(b0 + (g+h)*delta, target) -
             losses(b0 + (g-h)*delta, target)) / (2*h)
        derivatives[str(g)] = {
            "negative_count_push_open": int((d < -1e-7).sum()),
            "positive_count_push_close": int((d > 1e-7).sum()),
            "sum_dloss_dgate": float(d.sum()),
            "negative_gradient_sum": float(d[d < 0].sum()),
            "positive_gradient_sum": float(d[d > 0].sum()),
            "sum_dloss_dlogit": float((d * g * (1-g)).sum()),
        }
    return {
        "samples": len(mask), "valid": n,
        "loss_sum_by_gate": {str(float(g)): float(v.sum()) for g,v in zip(grid, all_losses)},
        "g1_better_g0_count": int((l1 < l0 - 1e-6).sum()),
        "g0_better_g1_count": int((l0 < l1 - 1e-6).sum()),
        "g1_matches_grid_oracle_count": int((l1 <= best + 1e-6).sum()),
        "oracle_gate_sum": float(optimal.sum()),
        "oracle_gate_le_05_count": int((optimal <= .5).sum()),
        "oracle_loss_sum": float(best.sum()),
        "g1_to_oracle_improvement_sum": float((l1-best).sum()),
        "g1_to_oracle_improvement_gt_001_count": int((l1-best > .01).sum()),
        "endpoint_max_pixel_gap_sum": float(np.abs(delta).max(1).sum()),
        "derivatives": derivatives,
    }


def aggregate(items):
    n = sum(x["valid"] for x in items)
    samples = sum(x["samples"] for x in items)
    def rate(field):
        return sum(x[field] for x in items) / max(1,n)
    gate_names = next(iter(items))["loss_sum_by_gate"]
    curve = {g: sum(x["loss_sum_by_gate"][g] for x in items)/max(1,n) for g in gate_names}
    result = {
        "samples": samples, "valid": n, "valid_ratio": n/max(1,samples),
        "loss_mean_by_gate": curve,
        "best_constant_gate_on_grid": min(curve, key=curve.get),
        "g1_better_g0_fraction": rate("g1_better_g0_count"),
        "g0_better_g1_fraction": rate("g0_better_g1_count"),
        "g1_matches_grid_oracle_fraction": rate("g1_matches_grid_oracle_count"),
        "grid_oracle_gate_mean": rate("oracle_gate_sum"),
        "grid_oracle_gate_le_05_fraction": rate("oracle_gate_le_05_count"),
        "grid_oracle_loss_mean": rate("oracle_loss_sum"),
        "g1_to_grid_oracle_loss_reduction": rate("g1_to_oracle_improvement_sum"),
        "g1_to_grid_oracle_improvement_gt_001_fraction": rate("g1_to_oracle_improvement_gt_001_count"),
        "endpoint_max_pixel_gap_mean": rate("endpoint_max_pixel_gap_sum"),
        "derivatives": {}
    }
    for g in items[0]["derivatives"]:
        result["derivatives"][g] = {
            k: sum(x["derivatives"][g][k] for x in items)/max(1,n)
            for k in items[0]["derivatives"][g]
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.nice(10)
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["data_root"])
    names = config["sequence_names"]
    per_sequence = {}
    for name in names:
        per_sequence[name] = compute(root / name / "samples.npz")
    result = {"dataset": config["dataset"], "config": str(Path(args.config)),
              "sequences": len(names), "aggregation": "sample-weighted valid rows, full training split",
              "scope": "fixed full-write replay endpoints; oracle uses target GT and is not deployable",
              "aggregate": aggregate(list(per_sequence.values())), "per_sequence": per_sequence,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({k:v for k,v in result.items() if k != "per_sequence"}, indent=2))


if __name__ == "__main__":
    main()
