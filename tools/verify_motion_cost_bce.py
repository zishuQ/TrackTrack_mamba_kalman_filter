#!/usr/bin/env python3
"""Preflight for the cost-weighted motion-gate BCE experiment.

Checks, on one real sequence/batch:
  - target sidecar fingerprint, row binding and exact gate-label agreement;
  - valid/harmful/useful/target=1/target<1 counts;
  - gradient direction at g~1 for harmful events and at over-suppressed g for
    useful events (the weight is detached);
  - frozen shared/appearance parameters, the exact trainable motion head,
    unchanged appearance output and frozen weights after one optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from train_motion_cost_bce import cost_motion_loss, arguments_from_config
from shutrack.training.motion_targets import (
    GRID, SOURCE_KEYS, TIE_TOLERANCE, generate_targets, source_fingerprint,
)
from shutrack.training.loss import _diou_xyxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lambda-cost", type=float, default=1.0)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    data_root = Path(config["data_root"])
    sequence = args.sequence or config["sequence_names"][0]
    split = config.get("split", "train")
    dataset = config.get("dataset", "SportsMOT")
    group = config["group"]

    seq = trainer.SequenceData(sequence, split, data_root, dataset)
    target_dir = Path(args.target_root) / sequence
    metadata = json.loads((target_dir / "metadata.json").read_text())
    source = {key: seq.small[key] if key in seq.small else seq.samples[key]
              for key in SOURCE_KEYS}
    fingerprint_ok = metadata["source_sha256"] == source_fingerprint(source)
    target_all = np.load(target_dir / "motion_gate_target.npy", allow_pickle=False)
    checksum_ok = metadata["target_sha256"] == hashlib.sha256(target_all.tobytes()).hexdigest()

    rows = np.arange(min(args.rows, seq.n), dtype=np.int64)
    seq.open_cache()
    taus = trainer.load_global_taus([sequence], data_root)
    targets = trainer.derive_targets(seq.samples, taus)
    batch_np = trainer.batch_tensors(seq, rows, targets, group, compact_reid=True)
    target = target_all[rows]
    batch_np["motion_gate_target"] = target
    valid_np = batch_np["motion_gt_valid"].astype(bool)
    expected = generate_targets(
        batch_np["pred_box_g0"], batch_np["pred_box_g1"], batch_np["gt_box_t1"], valid_np
    )
    binding_ok = bool(np.array_equal(expected[valid_np], target[valid_np]))

    batch = trainer.to_device(batch_np, torch.device("cpu"))
    batch["motion_gate_target"] = torch.from_numpy(
        np.asarray(batch_np["motion_gate_target"], dtype=np.float32))
    valid = batch["motion_gt_valid"].bool()
    loss0 = 1.0 - _diou_xyxy(batch["pred_box_g0"], batch["gt_box_t1"])
    loss1 = 1.0 - _diou_xyxy(batch["pred_box_g1"], batch["gt_box_t1"])
    harmful = valid & (loss1 > loss0 + 1e-6)
    useful = valid & ~harmful
    target1 = valid & (batch["motion_gate_target"] >= 1.0 - 1e-6)
    target_lt1 = valid & (batch["motion_gate_target"] < 1.0 - 1e-6)

    n = int(valid.sum())
    harmful_count = int(harmful.sum())
    useful_count = int(useful.sum())
    target1_count = int(target1.sum())
    target_lt1_count = int(target_lt1.sum())
    loss1_sum = float(loss1[valid].sum())
    loss0_sum = float(loss0[valid].sum())
    best_box = batch["pred_box_g0"] + batch["motion_gate_target"].unsqueeze(-1) * (
        batch["pred_box_g1"] - batch["pred_box_g0"])
    loss_best = 1.0 - _diou_xyxy(best_box, batch["gt_box_t1"])
    saved_harmful = float((loss1 - loss_best)[harmful].sum())
    added_useful = float((loss_best - loss1)[useful].sum()) * -1.0

    def gate_grad(logit_value):
        logits = torch.zeros(n_total, 2, requires_grad=True)
        with torch.no_grad():
            logits[:, 0] = logit_value
        outputs = {"gate_logits": logits}
        total, components = cost_motion_loss(outputs, batch, args.lambda_cost)
        total.backward()
        return logits.grad[:, 0].detach(), components

    n_total = batch["motion_gt_valid"].shape[0]
    grad_h, comp_h = gate_grad(10.0)
    grad_u, comp_u = gate_grad(-10.0)
    weight_detached = not comp_h["weight"].requires_grad
    harmful_grad_mean = float(grad_h[harmful].mean()) if harmful.any() else None
    harmful_grad_pos_fraction = float((grad_h[harmful] > 0).float().mean()) if harmful.any() else None
    useful_grad_mean = float(grad_u[useful].mean()) if useful.any() else None
    useful_grad_neg_fraction = float((grad_u[useful] < 0).float().mean()) if useful.any() else None
    harmful_weight_mean = float(comp_h["weight"][harmful].mean()) if harmful.any() else None
    useful_weight_mean = float(comp_u["weight"][useful].mean()) if useful.any() else None

    model, payload = trainer.load_shutrack_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("rsc.gate_heads.0.")
    trainable = {name: int(p.numel()) for name, p in model.named_parameters() if p.requires_grad}
    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    inputs = trainer.model_inputs(batch)
    with torch.no_grad():
        out_before = model(**inputs)
        appearance_before = out_before["gate_logits"][:, 1].clone()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    out = model(**inputs)
    total, components = cost_motion_loss(out, batch, args.lambda_cost)
    total.backward()
    optimizer.step()
    with torch.no_grad():
        out_after = model(**inputs)
        appearance_after = out_after["gate_logits"][:, 1]
        appearance_diff = float((appearance_before - appearance_after).abs().max())
    frozen_diff = max(
        (float((p.detach() - frozen_before[name]).abs().max()) for name, p in model.named_parameters()
         if not p.requires_grad),
        default=0.0,
    )

    report = {
        "sequence": sequence, "rows": len(rows),
        "fingerprint_ok": bool(fingerprint_ok), "checksum_ok": bool(checksum_ok),
        "row_binding_ok": binding_ok,
        "counts": {"valid": n, "harmful": harmful_count, "useful": useful_count,
                   "target1": target1_count, "target_lt1": target_lt1_count},
        "loss": {"full_mean": loss1_sum / max(1, n), "zero_mean": loss0_sum / max(1, n),
                 "best_mean": float(loss_best[valid].sum()) / max(1, n),
                 "saved_harmful_mean": saved_harmful / max(1, harmful_count),
                 "added_useful_mean": added_useful / max(1, useful_count)},
        "gradient": {
            "harmful_at_g1_mean": harmful_grad_mean,
            "harmful_at_g1_positive_fraction": harmful_grad_pos_fraction,
            "useful_over_suppressed_mean": useful_grad_mean,
            "useful_over_suppressed_negative_fraction": useful_grad_neg_fraction,
            "harmful_weight_mean": harmful_weight_mean,
            "useful_weight_mean": useful_weight_mean,
            "weight_detached": bool(weight_detached),
        },
        "model": {
            "trainable": trainable,
            "trainable_total": int(sum(trainable.values())),
            "frozen_count": len(frozen_before),
            "appearance_logit_max_diff_after_step": appearance_diff,
            "frozen_param_max_diff_after_step": frozen_diff,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
