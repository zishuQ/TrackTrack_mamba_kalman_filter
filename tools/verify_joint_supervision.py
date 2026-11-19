#!/usr/bin/env python3
"""Real-batch validation for the joint motion-DIoU + appearance-score group."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "tools"))
sys.path.insert(0, str(WORKTREE))

from train_supervision_groups import (  # noqa: E402
    DATA,
    INIT_CKPT,
    SequenceData,
    batch_tensors,
    derive_targets,
    load_global_taus,
    model_inputs,
    to_device,
)
from shutrack.training.checkpoint import load_shutrack_checkpoint  # noqa: E402
from shutrack.training.loss import appearance_soft_loss, motion_box_diou_loss  # noqa: E402

SEQUENCES = ("v_9MHDmAMxO5I_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")
LIMIT = 400


def main() -> None:
    torch.set_num_threads(4)
    taus = load_global_taus(list(SEQUENCES), DATA)
    model, _ = load_shutrack_checkpoint(INIT_CKPT, map_location="cpu")
    model.eval()
    report = {"taus": taus, "sequences": {}, "totals": {}}
    totals = {
        "samples": 0,
        "motion_loss_diff_max": 0.0,
        "appearance_target_mismatch": 0,
        "fallback_samples": 0,
        "diag_samples": 0,
        "motion_grad_nonzero": 0,
        "appearance_grad_nonzero": 0,
        "motion_grad_nonfinite": 0,
        "appearance_grad_nonfinite": 0,
        "motion_grad_abs_mean_sum": 0.0,
        "appearance_grad_abs_mean_sum": 0.0,
    }
    for sequence in SEQUENCES:
        summary = json.loads((DATA / sequence / "summary.json").read_text())
        split = summary.get("split")
        if split not in ("train", "val"):
            val_dir = Path(
                "/home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache/SportsMOT/val"
            )
            split = "val" if (val_dir / sequence).is_dir() else "train"
        data = SequenceData(sequence, split, DATA)
        data.open_cache()
        targets = derive_targets(data.samples, taus)
        rows = np.arange(min(LIMIT, data.n))
        batch_motion = to_device(
            batch_tensors(data, rows, targets, "motion_diou"), torch.device("cpu")
        )
        batch_joint = to_device(
            batch_tensors(data, rows, targets, "motion_diou_appearance_score"),
            torch.device("cpu"),
        )
        batch_appearance = to_device(
            batch_tensors(data, rows, targets, "appearance_score"), torch.device("cpu")
        )
        inputs = model_inputs(batch_joint)
        logits = model(**inputs)["gate_logits"]
        motion_loss_motion, _ = motion_box_diou_loss(
            {"gate_logits": logits}, batch_motion
        )
        motion_loss_joint, _ = motion_box_diou_loss(
            {"gate_logits": logits}, batch_joint
        )
        motion_diff = float((motion_loss_motion - motion_loss_joint).abs())
        joint_target = batch_joint["appearance_target"].numpy()
        appearance_target = batch_appearance["appearance_target"].numpy()
        mismatch = int(np.sum(joint_target != appearance_target))
        delta = data.samples["appearance_delta"][rows]
        diag = data.samples["appearance_diag_valid"][rows]
        fallback = (~diag).sum().item()
        diag_count = int(diag.sum())
        # Fallback samples must equal the original appearance soft label.
        safe = targets["appearance_safe"][rows]
        fallback_mismatch = int(
            np.sum(joint_target[~diag] != safe[~diag])
        ) if fallback else 0

        grad_logits = logits.clone().requires_grad_(True)
        motion_loss, _ = motion_box_diou_loss(
            {"gate_logits": grad_logits}, batch_joint
        )
        motion_grad = torch.autograd.grad(motion_loss, grad_logits, retain_graph=True)[0]
        appearance_loss, _ = appearance_soft_loss(
            {"gate_logits": grad_logits}, batch_joint
        )
        appearance_grad = torch.autograd.grad(appearance_loss, grad_logits)[0]
        finite_motion = bool(torch.isfinite(motion_grad).all())
        finite_appearance = bool(torch.isfinite(appearance_grad).all())
        motion_nonzero = float((motion_grad[:, 0].abs() > 0).float().mean())
        appearance_nonzero = float((appearance_grad[:, 1].abs() > 0).float().mean())
        entry = {
            "samples": int(len(rows)),
            "motion_loss_diff": motion_diff,
            "appearance_target_mismatch": mismatch,
            "fallback_samples": int(fallback),
            "fallback_target_mismatch": fallback_mismatch,
            "diag_samples": diag_count,
            "motion_grad_finite": finite_motion,
            "appearance_grad_finite": finite_appearance,
            "motion_grad_nonzero_fraction": motion_nonzero,
            "appearance_grad_nonzero_fraction": appearance_nonzero,
            "motion_grad_abs_mean": float(motion_grad[:, 0].abs().mean()),
            "appearance_grad_abs_mean": float(appearance_grad[:, 1].abs().mean()),
            "motion_grad_on_appearance_max": float(
                motion_grad[:, 1].abs().max()
            ),
            "appearance_grad_on_motion_max": float(
                appearance_grad[:, 0].abs().max()
            ),
        }
        report["sequences"][sequence] = entry
        totals["samples"] += entry["samples"]
        totals["motion_loss_diff_max"] = max(
            totals["motion_loss_diff_max"], motion_diff
        )
        totals["appearance_target_mismatch"] += mismatch
        totals["fallback_samples"] += int(fallback)
        totals["fallback_target_mismatch"] = totals.get("fallback_target_mismatch", 0) + fallback_mismatch
        totals["diag_samples"] += diag_count
        totals["motion_grad_nonzero"] += int(motion_nonzero * entry["samples"])
        totals["appearance_grad_nonzero"] += int(appearance_nonzero * entry["samples"])
        totals["motion_grad_nonfinite"] += 0 if finite_motion else entry["samples"]
        totals["appearance_grad_nonfinite"] += 0 if finite_appearance else entry["samples"]
        totals["motion_grad_abs_mean_sum"] += entry["motion_grad_abs_mean"] * entry["samples"]
        totals["appearance_grad_abs_mean_sum"] += entry["appearance_grad_abs_mean"] * entry["samples"]
        data.close_cache()
    totals["motion_grad_nonzero_fraction"] = totals["motion_grad_nonzero"] / totals["samples"]
    totals["appearance_grad_nonzero_fraction"] = totals["appearance_grad_nonzero"] / totals["samples"]
    totals["motion_grad_abs_mean"] = totals["motion_grad_abs_mean_sum"] / totals["samples"]
    totals["appearance_grad_abs_mean"] = totals["appearance_grad_abs_mean_sum"] / totals["samples"]
    totals["motion_loss_equal"] = totals["motion_loss_diff_max"] <= 1e-6
    totals["appearance_target_equal"] = (
        totals["appearance_target_mismatch"] == 0 and totals["fallback_target_mismatch"] == 0
    )
    totals["gradients_finite"] = (
        totals["motion_grad_nonfinite"] == 0 and totals["appearance_grad_nonfinite"] == 0
    )
    report["totals"] = totals
    out = DATA.parent / "joint_validation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("saved", out)


if __name__ == "__main__":
    main()
