#!/usr/bin/env python3
"""Verify B/D supervision compatibility on a shared full-write column dataset.

Checks, on real batches:
  1. B (appearance_rank) and D (appearance_row_column) see identical shared
     inputs and produce identical model outputs.
  2. D with column_weight=0 has bit-identical loss and gate gradients to B.
  3. The real column term is finite and produces a finite, non-zero appearance
     gate gradient on valid (contested) events, with no motion-gate leak.
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
    load_global_taus,
    model_inputs,
    to_device,
)
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model  # noqa: E402
from shutrack.training.loss import (  # noqa: E402
    appearance_column_loss,
    appearance_rank_loss,
    motion_box_diou_loss,
)

ALPHA = 0.95
TAU = 0.1
RANK_WEIGHT = 18.0
COLUMN_WEIGHT = 50.0
ROWS_PER_SEQUENCE = 4096


def values_equal(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def tensor_equal(left, right) -> bool:
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tensor_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            tensor_equal(a, b) for a, b in zip(left, right)
        )
    if torch.is_tensor(left):
        return left.shape == right.shape and torch.equal(left, right)
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--dataset", choices=("SportsMOT", "MOT17", "MOT20"), default="SportsMOT")
    parser.add_argument("--rows", type=int, default=ROWS_PER_SEQUENCE)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    torch.set_num_threads(4)
    data_root = Path(args.data_root)
    sequences = list(args.sequences)
    taus = load_global_taus(sequences, data_root)
    torch.manual_seed(42)
    model = ShutrackV2Model(ShutrackV2Config(reid_dim=2048))
    model.eval()

    per_sequence = {}
    all_ok = True
    for sequence in sequences:
        summary = json.loads((data_root / sequence / "summary.json").read_text())
        split = summary.get("split")
        if split not in ("train", "val", "all"):
            split = "train"
        data = SequenceData(sequence, split, data_root, args.dataset)
        data.open_cache()
        try:
            targets = derive_targets(data.samples, taus)
            rows = np.arange(min(data.n, args.rows), dtype=np.int64)
            batch_b = to_device(
                batch_tensors(data, rows, targets, "appearance_rank"),
                torch.device("cpu"),
            )
            batch_d = to_device(
                batch_tensors(data, rows, targets, "appearance_row_column"),
                torch.device("cpu"),
            )
            shared_equal = all(
                values_equal(batch_b[key], batch_d[key])
                for key in batch_b
                if key in batch_d
            )
            inputs_b = model_inputs(batch_b)
            inputs_d = model_inputs(batch_d)
            with torch.no_grad():
                outputs_b = model(**inputs_b)
                outputs_d = model(**inputs_d)
            outputs_equal = tensor_equal(outputs_b, outputs_d)

            logits_b = outputs_b["gate_logits"].clone().requires_grad_(True)
            logits_d = outputs_d["gate_logits"].clone().requires_grad_(True)
            box_b, _ = motion_box_diou_loss({"gate_logits": logits_b}, batch_b)
            rank_b, _ = appearance_rank_loss(
                {"gate_logits": logits_b}, batch_b,
                alpha=ALPHA, tau=TAU, rank_weight=RANK_WEIGHT,
            )
            loss_b = box_b + rank_b
            grad_b = torch.autograd.grad(loss_b, logits_b, retain_graph=True)[0]

            box_d, _ = motion_box_diou_loss({"gate_logits": logits_d}, batch_d)
            rank_d, _ = appearance_rank_loss(
                {"gate_logits": logits_d}, batch_d,
                alpha=ALPHA, tau=TAU, rank_weight=RANK_WEIGHT,
            )
            column = appearance_column_loss(
                {"gate_logits": logits_d}, batch_d, alpha=ALPHA, tau=TAU
            )
            loss_d0 = box_d + rank_d + 0.0 * column
            grad_d0 = torch.autograd.grad(loss_d0, logits_d, retain_graph=True)[0]
            loss_equal = torch.equal(loss_b, loss_d0)
            grad_equal = torch.equal(grad_b, grad_d0)

            grad_column = torch.autograd.grad(
                column, logits_d, retain_graph=True
            )[0]
            valid = batch_d["column_valid"].to(torch.bool)
            appearance_grad = grad_column[:, 1]
            motion_grad = grad_column[:, 0]
            valid_count = int(valid.sum())
            if valid_count:
                valid_abs = appearance_grad.abs()[valid]
                grad_mean = float(valid_abs.mean())
                nonzero_fraction = float((valid_abs > 0).float().mean())
                finite = bool(torch.isfinite(grad_column[valid]).all())
            else:
                grad_mean = None
                nonzero_fraction = None
                finite = False
            result = {
                "rows": int(rows.size),
                "column_valid": valid_count,
                "shared_inputs_equal": bool(shared_equal),
                "model_outputs_equal": bool(outputs_equal),
                "loss_b": float(loss_b),
                "loss_d_weight0": float(loss_d0),
                "loss_difference": float((loss_d0 - loss_b).abs()),
                "loss_equal": bool(loss_equal),
                "gate_grad_equal_weight0": bool(grad_equal),
                "column_loss": float(column),
                "column_loss_finite": bool(torch.isfinite(column)),
                "column_grad_appearance_mean": grad_mean,
                "column_grad_nonzero_fraction": nonzero_fraction,
                "column_grad_finite": finite,
                "column_grad_motion_max": float(motion_grad.abs().max()),
                "column_loss_weight50": float(COLUMN_WEIGHT * column),
            }
            per_sequence[sequence] = result
            all_ok &= (
                result["shared_inputs_equal"]
                and result["model_outputs_equal"]
                and result["loss_equal"]
                and result["gate_grad_equal_weight0"]
                and result["column_loss_finite"]
                and result["column_grad_finite"]
                and valid_count > 0
            )
        finally:
            data.close_cache()

    report = {
        "data_root": str(data_root),
        "dataset": args.dataset,
        "sequences": sequences,
        "taus": taus,
        "all_ok": bool(all_ok),
        "per_sequence": per_sequence,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
