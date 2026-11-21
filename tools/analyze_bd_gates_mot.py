#!/usr/bin/env python3
"""Appearance/motion gate statistics on contested vs ordinary train events.

Runs one or more checkpoints over the training-sequence shards and reports,
for column_valid (contested) and other matched events, the predicted gates.
Used to verify that column supervision acts on contention events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
for path in (
    str(WORKTREE / "tools"),
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_supervision_groups import (  # noqa: E402
    SequenceData,
    batch_tensors,
    derive_targets,
    model_inputs,
    to_device,
)
from shutrack.training.checkpoint import load_shutrack_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("MOT17", "MOT20"), required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--taus", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    taus = json.loads(args.taus)
    data_root = Path(args.data_root)
    checkpoints = []
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        model, payload = load_shutrack_checkpoint(path, map_location="cpu")
        model.eval()
        checkpoints.append((name, model, payload))

    torch.set_num_threads(4)
    results = {name: {"valid": 0, "other": 0} for name, _, _ in checkpoints}
    sums = {
        name: {
            "appearance_valid": 0.0,
            "appearance_other": 0.0,
            "motion_valid": 0.0,
            "motion_other": 0.0,
        }
        for name, _, _ in checkpoints
    }
    for sequence in args.sequences:
        summary = json.loads((data_root / sequence / "summary.json").read_text())
        split = summary.get("split")
        if split not in ("train", "val", "all"):
            split = "all"
        data = SequenceData(sequence, split, data_root, args.dataset)
        data.open_cache()
        try:
            targets = derive_targets(data.samples, taus)
            for name, model, payload in checkpoints:
                data.input_statistics = (
                    np.asarray(payload["normalization_mean"], dtype=np.float32),
                    np.asarray(payload["normalization_std"], dtype=np.float32),
                )
                for start in range(0, data.n, args.rows):
                    rows = np.arange(start, min(start + args.rows, data.n), dtype=np.int64)
                    batch = to_device(
                        batch_tensors(data, rows, targets, "appearance_row_column"),
                        torch.device("cpu"),
                    )
                    with torch.no_grad():
                        gates = torch.sigmoid(model(**model_inputs(batch))["gate_logits"]).numpy()
                    valid = batch["column_valid"].numpy().astype(bool)
                    results[name]["valid"] += int(valid.sum())
                    results[name]["other"] += int((~valid).sum())
                    sums[name]["appearance_valid"] += float(gates[valid, 1].sum())
                    sums[name]["appearance_other"] += float(gates[~valid, 1].sum())
                    sums[name]["motion_valid"] += float(gates[valid, 0].sum())
                    sums[name]["motion_other"] += float(gates[~valid, 0].sum())
        finally:
            data.close_cache()

    report = {"data_root": str(data_root), "dataset": args.dataset,
              "sequences": list(args.sequences), "groups": {}}
    for name, _, _ in checkpoints:
        valid = max(results[name]["valid"], 1)
        other = max(results[name]["other"], 1)
        report["groups"][name] = {
            "valid_events": results[name]["valid"],
            "other_events": results[name]["other"],
            "appearance_gate_valid": sums[name]["appearance_valid"] / valid,
            "appearance_gate_other": sums[name]["appearance_other"] / other,
            "motion_gate_valid": sums[name]["motion_valid"] / valid,
            "motion_gate_other": sums[name]["motion_other"] / other,
        }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
