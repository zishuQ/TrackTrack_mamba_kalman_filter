"""Small-array, CPU-only training-objective audit for a motion harm weight."""
import argparse
import json
import os
import time
from pathlib import Path
from analyze_motion_gate_targets import np, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.nice(10)
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    factors = (1., 2., 4., 8.)
    gates = (0., .25, .5, .75, .8, .85, .9, .95, .99, .9999, 1.)
    sums = {str(f): {"weight_sum": 0., "loss_sum": {str(g): 0. for g in gates},
                     "derivative_sum_at_05": 0., "derivative_sum_at_099": 0.} for f in factors}
    total = harmful = 0
    for name in config["sequence_names"]:
        with np.load(Path(config["data_root"]) / name / "samples.npz", allow_pickle=False) as archive:
            mask = archive["motion_gt_valid"].astype(bool)
            b0 = archive["pred_box_g0"][mask].astype(np.float64)
            b1 = archive["pred_box_g1"][mask].astype(np.float64)
            gt = archive["gt_box_t1"][mask].astype(np.float64)
        delta = b1 - b0
        curve = {str(g): losses(b0 + g*delta, gt) for g in gates}
        # Endpoint comparison has a numerical tolerance, not a tuned margin.
        harm = curve["1.0"] > curve["0.0"] + 1e-6
        total += len(harm)
        harmful += int(harm.sum())
        derivative = {}
        for g in (.5, .99):
            derivative[g] = (losses(b0+(g+1e-5)*delta, gt) -
                             losses(b0+(g-1e-5)*delta, gt)) / 2e-5
        for f in factors:
            weights = np.where(harm, f, 1.)
            target = sums[str(f)]
            target["weight_sum"] += float(weights.sum())
            for g, values in curve.items():
                target["loss_sum"][g] += float(np.dot(weights, values))
            target["derivative_sum_at_05"] += float(np.dot(weights, derivative[.5]))
            target["derivative_sum_at_099"] += float(np.dot(weights, derivative[.99]))
    result = {"valid_rows": total, "harmful_endpoint_rows": harmful,
              "harm_definition": "L(g=1) > L(g=0) + 1e-6",
              "scope": "fixed replay, training data only; no model training", "factors": {}}
    for f, values in sums.items():
        n = max(1., values["weight_sum"])
        curve = {g:v/n for g,v in values["loss_sum"].items()}
        result["factors"][f] = {
            "weighted_loss_curve": curve, "best_scanned_constant_gate": min(curve, key=curve.get),
            "weighted_derivative_at_05": values["derivative_sum_at_05"]/n,
            "weighted_derivative_at_099": values["derivative_sum_at_099"]/n,
            "harmful_weight_share": harmful*float(f)/n}
    result["seconds"] = time.perf_counter()-started
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
