#!/usr/bin/env python3
"""Fixed-val diagnostics for a motion-gate checkpoint.

Runs the gate head over the 45 SportsMOT val sequences using the cached paired
features (no tracker, no post-processing) and reports, against the sidecar
gate targets and rank labels:
  - beneficial/harmful and g_star=1 / g_star<1 gate distributions;
  - loss at the predicted gate vs full write vs best gate (saved/added/net);
  - pair count, ranking accuracy, logit gap and gate gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "3. Tracker"
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from train_motion_cost_bce import _hist
from shutrack.training.motion_targets import SOURCE_KEYS, source_fingerprint
from shutrack.training.loss import _diou_xyxy


def val_sequences():
    path = TRACKER / "trackeval/seqmap/sportsmot/val.txt"
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and line.strip() != "name"]


def diag_loss(outputs, batch):
    z = outputs["gate_logits"].float()[:, 0]
    gate = torch.sigmoid(z)
    target = batch["motion_gate_target"].float()
    valid = batch["motion_gt_valid"].bool()
    box0 = batch["pred_box_g0"].float()
    box1 = batch["pred_box_g1"].float()
    gt = batch["gt_box_t1"].float()
    box = box0 + gate.unsqueeze(-1) * (box1 - box0)
    loss_pred = 1.0 - _diou_xyxy(box, gt)
    box_star = box0 + target.unsqueeze(-1) * (box1 - box0)
    loss_best = 1.0 - _diou_xyxy(box_star, gt)
    loss1 = 1.0 - _diou_xyxy(box1, gt)
    loss0 = 1.0 - _diou_xyxy(box0, gt)
    weight = torch.clamp(loss_pred.detach() - loss_best.detach(), min=0.0)
    bce = F.binary_cross_entropy_with_logits(z, target, reduction="none")
    z_app = outputs["gate_logits"].float()[:, 1]
    app_target = batch["appearance_target"].float()
    app_valid = batch["appearance_valid"].bool()
    app_bce = F.binary_cross_entropy_with_logits(z_app, app_target, reduction="none")
    return {
        "z": z, "gate": gate, "target": target, "valid": valid,
        "loss_pred": loss_pred, "loss_best": loss_best, "loss1": loss1, "loss0": loss0,
        "weight": weight, "bce": bce,
        "app_gate": torch.sigmoid(z_app), "app_target": app_target,
        "app_valid": app_valid, "app_bce": app_bce,
    }


def accumulate(sums, components, batch, generator):
    with torch.no_grad():
        gate = components["gate"]
        valid = components["valid"]
        target = components["target"]
        pred = components["loss_pred"]
        best = components["loss_best"]
        loss1 = components["loss1"]
        loss0 = components["loss0"]
        harmful = valid & (loss1 > loss0 + 1e-6)
        useful = valid & ~harmful
        target1 = valid & (target >= 1.0 - 1e-6)
        target_lt = valid & (target < 1.0 - 1e-6)
        weight = components["weight"]
        bce = components["bce"]
        fields = {
            "valid": valid.sum(), "harmful_count": harmful.sum(),
            "useful_count": useful.sum(), "target1_count": target1.sum(),
            "target_lt1_count": target_lt.sum(),
            "gate_sum": (gate * valid).sum(),
            "harmful_gate_sum": (gate * harmful).sum(),
            "useful_gate_sum": (gate * useful).sum(),
            "target1_gate_sum": (gate * target1).sum(),
            "target_lt1_gate_sum": (gate * target_lt).sum(),
            "target_sum": (target * valid).sum(),
            "gate_abs_error_sum": ((gate - target).abs() * valid).sum(),
            "loss_full_sum": (loss1 * valid).sum(),
            "loss_pred_sum": (pred * valid).sum(),
            "loss_best_sum": (best * valid).sum(),
            "saved_harmful_sum": ((loss1 - pred) * harmful).sum(),
            "added_useful_sum": ((pred - loss1) * useful).sum(),
            "net_delta_sum": ((pred - loss1) * valid).sum(),
            "weight_sum": (weight * valid).sum(),
            "bce_weighted_sum": (weight * bce * valid).sum(),
            "gate_gt095_count": ((gate > 0.95) & valid).sum(),
            "gate_lt005_count": ((gate < 0.05) & valid).sum(),
            "harmful_gate_gt095": ((gate > 0.95) & harmful).sum(),
            "harmful_gate_lt005": ((gate < 0.05) & harmful).sum(),
            "useful_gate_gt095": ((gate > 0.95) & useful).sum(),
            "useful_gate_lt005": ((gate < 0.05) & useful).sum(),
            "target1_gate_gt095": ((gate > 0.95) & target1).sum(),
            "target1_gate_lt005": ((gate < 0.05) & target1).sum(),
            "target_lt1_gate_gt095": ((gate > 0.95) & target_lt).sum(),
            "target_lt1_gate_lt005": ((gate < 0.05) & target_lt).sum(),
        }
        for prefix, mask in (("harmful", harmful), ("useful", useful),
                             ("target1", target1), ("target_lt1", target_lt), ("all", valid)):
            hist = _hist(gate, mask).cpu().tolist()
            sums[prefix + "_gate_hist"] = [
                sums.get(prefix + "_gate_hist", [0.0] * 10)[i] + hist[i] for i in range(10)]

        z = components["z"]
        app_gate = components["app_gate"]
        app_target = components["app_target"]
        app_valid = components["app_valid"]
        fields.update({
            "appearance_valid_count": app_valid.sum(),
            "appearance_target_sum": (app_target * app_valid).sum(),
            "appearance_gate_sum": (app_gate * app_valid).sum(),
            "appearance_abs_error_sum": ((app_gate - app_target).abs() * app_valid).sum(),
            "appearance_gate_gt095_count": ((app_gate > 0.95) & app_valid).sum(),
            "appearance_gate_lt005_count": ((app_gate < 0.05) & app_valid).sum(),
            "appearance_bce_sum": (components["app_bce"] * app_valid).sum(),
        })
        for key, value in fields.items():
            sums[key] += float(value)
        labels = batch["motion_rank_label"].long()
        sums["keep_count"] += float(((labels == 1) & valid).sum())
        sums["suppress_count"] += float(((labels == 2) & valid).sum())
        bad = ((labels == 2) & valid).nonzero(as_tuple=False).flatten()
        good = ((labels == 1) & valid).nonzero(as_tuple=False).flatten()
        if bad.numel() and good.numel():
            picks = torch.randint(0, int(good.numel()), (int(bad.numel()),), generator=generator)
            good_idx = good[picks.to(good.device)]
            difference = z[good_idx] - z[bad]
            gain = batch["motion_rank_gain"].float()[bad]
            damage = batch["motion_rank_damage"].float()[good_idx]
            pair_weight = torch.sqrt(torch.clamp(gain * damage, min=0.0))
            sums["rank_pairs"] += float(bad.numel())
            sums["rank_correct"] += float((difference > 0).float().sum())
            sums["rank_logit_gap"] += float(difference.sum())
            sums["rank_gate_gap"] += float((torch.sigmoid(z[good_idx]) - torch.sigmoid(z[bad])).sum())
            sums["rank_weight_sum"] += float(pair_weight.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--rank-root", required=True)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    group = config["group"]
    dataset = config.get("dataset", "SportsMOT")
    sequence_root = Path(args.sequence_root)
    target_root = Path(args.target_root)
    rank_root = Path(args.rank_root)
    names = val_sequences()
    device = torch.device(args.device)
    model, payload = trainer.load_shutrack_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device).eval()
    normalization = (
        np.asarray(payload["normalization_mean"], dtype=np.float32),
        np.asarray(payload["normalization_std"], dtype=np.float32),
    )
    generator = torch.Generator(device="cpu").manual_seed(20261122)
    sums = defaultdict(float)
    per_sequence = {}
    for sequence in names:
        data = trainer.SequenceData(sequence, args.split, sequence_root, dataset)
        data.input_statistics = normalization
        taus = trainer.load_global_taus([sequence], sequence_root)
        targets = trainer.derive_targets(data.samples, taus)
        target = np.load(target_root / sequence / "motion_gate_target.npy", allow_pickle=False)
        rank_dir = rank_root / sequence
        label = np.load(rank_dir / "motion_rank_label.npy", allow_pickle=False)
        gain = np.load(rank_dir / "motion_rank_gain.npy", allow_pickle=False)
        damage = np.load(rank_dir / "motion_rank_damage.npy", allow_pickle=False)
        seq_sums = defaultdict(float)
        data.open_cache()
        for start in range(0, data.n, args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, data.n), dtype=np.int64)
            batch_np = trainer.batch_tensors(data, rows, targets, group, compact_reid=True)
            batch_np["motion_gate_target"] = target[rows]
            batch_np["motion_rank_label"] = label[rows]
            batch_np["motion_rank_gain"] = gain[rows]
            batch_np["motion_rank_damage"] = damage[rows]
            batch = trainer.to_device(batch_np, device)
            for key in ("motion_gate_target", "motion_rank_label",
                        "motion_rank_gain", "motion_rank_damage"):
                batch[key] = torch.from_numpy(np.asarray(batch_np[key])).to(device)
            with torch.no_grad():
                outputs = model(**trainer.model_inputs(batch))
            components = diag_loss(outputs, batch)
            accumulate(seq_sums, components, batch, generator)
        data.close_cache()
        data.samples.close()
        for key, value in seq_sums.items():
            if isinstance(value, list):
                current = sums.get(key, [0.0] * 10)
                sums[key] = [current[i] + value[i] for i in range(10)]
            else:
                sums[key] += value
        valid = max(1.0, seq_sums["valid"])
        per_sequence[sequence] = {
            "valid": seq_sums["valid"],
            "keep": seq_sums["keep_count"],
            "suppress": seq_sums["suppress_count"],
            "gate_mean": seq_sums["gate_sum"] / valid,
            "target_mean": seq_sums["target_sum"] / valid,
            "target_mae": seq_sums["gate_abs_error_sum"] / valid,
            "harmful_gate_mean": (seq_sums["harmful_gate_sum"] / seq_sums["harmful_count"]
                                  if seq_sums["harmful_count"] else None),
            "useful_gate_mean": (seq_sums["useful_gate_sum"] / seq_sums["useful_count"]
                                 if seq_sums["useful_count"] else None),
            "target1_gate_mean": (seq_sums["target1_gate_sum"] / seq_sums["target1_count"]
                                  if seq_sums["target1_count"] else None),
            "target_lt1_gate_mean": (seq_sums["target_lt1_gate_sum"] / seq_sums["target_lt1_count"]
                                     if seq_sums["target_lt1_count"] else None),
            "net_delta_mean": seq_sums["net_delta_sum"] / valid,
            "appearance_gate_mean": (seq_sums["appearance_gate_sum"]
                                     / seq_sums["appearance_valid_count"]
                                     if seq_sums["appearance_valid_count"] else None),
            "appearance_mae": (seq_sums["appearance_abs_error_sum"]
                               / seq_sums["appearance_valid_count"]
                               if seq_sums["appearance_valid_count"] else None),
        }

    valid = max(1.0, sums["valid"])
    pairs = sums["rank_pairs"]
    report = {
        "checkpoint": str(args.checkpoint), "epoch": payload.get("epoch"),
        "sequences": len(names), "split": args.split,
        "counts": {
            "valid": sums["valid"], "harmful": sums["harmful_count"],
            "useful": sums["useful_count"], "target1": sums["target1_count"],
            "target_lt1": sums["target_lt1_count"],
            "keep": sums["keep_count"], "suppress": sums["suppress_count"],
        },
        "gate": {
            "gate_mean": sums["gate_sum"] / valid,
            "target_mean": sums["target_sum"] / valid,
            "target_mae": sums["gate_abs_error_sum"] / valid,
            "gt095_fraction": sums["gate_gt095_count"] / valid,
            "lt005_fraction": sums["gate_lt005_count"] / valid,
            "harmful_mean": (sums["harmful_gate_sum"] / sums["harmful_count"]
                             if sums["harmful_count"] else None),
            "useful_mean": (sums["useful_gate_sum"] / sums["useful_count"]
                            if sums["useful_count"] else None),
            "target1_mean": (sums["target1_gate_sum"] / sums["target1_count"]
                             if sums["target1_count"] else None),
            "target_lt1_mean": (sums["target_lt1_gate_sum"] / sums["target_lt1_count"]
                                if sums["target_lt1_count"] else None),
            "harmful_gt095_fraction": (sums["harmful_gate_gt095"] / sums["harmful_count"]
                                       if sums["harmful_count"] else None),
            "harmful_lt005_fraction": (sums["harmful_gate_lt005"] / sums["harmful_count"]
                                       if sums["harmful_count"] else None),
            "useful_gt095_fraction": (sums["useful_gate_gt095"] / sums["useful_count"]
                                      if sums["useful_count"] else None),
            "useful_lt005_fraction": (sums["useful_gate_lt005"] / sums["useful_count"]
                                      if sums["useful_count"] else None),
            "target1_lt005_fraction": (sums["target1_gate_lt005"] / sums["target1_count"]
                                       if sums["target1_count"] else None),
            "target_lt1_gt095_fraction": (sums["target_lt1_gate_gt095"] / sums["target_lt1_count"]
                                          if sums["target_lt1_count"] else None),
            "histograms": {prefix: sums.get(prefix + "_gate_hist", [0.0] * 10)
                           for prefix in ("harmful", "useful", "target1", "target_lt1", "all")},
        },
        "loss": {
            "full_mean": sums["loss_full_sum"] / valid,
            "pred_mean": sums["loss_pred_sum"] / valid,
            "best_mean": sums["loss_best_sum"] / valid,
            "saved_harmful_mean": (sums["saved_harmful_sum"] / sums["harmful_count"]
                                   if sums["harmful_count"] else None),
            "added_useful_mean": (sums["added_useful_sum"] / sums["useful_count"]
                                  if sums["useful_count"] else None),
            "net_delta_mean": sums["net_delta_sum"] / valid,
        },
        "rank": {
            "pairs": pairs,
            "accuracy": (sums["rank_correct"] / pairs) if pairs else None,
            "logit_gap_mean": (sums["rank_logit_gap"] / pairs) if pairs else None,
            "gate_gap_mean": (sums["rank_gate_gap"] / pairs) if pairs else None,
            "pair_weight_mean": (sums["rank_weight_sum"] / pairs) if pairs else None,
        },
        "appearance": (lambda n: {
            "valid": sums["appearance_valid_count"],
            "target_mean": sums["appearance_target_sum"] / n,
            "gate_mean": sums["appearance_gate_sum"] / n,
            "mae": sums["appearance_abs_error_sum"] / n,
            "gt095_fraction": sums["appearance_gate_gt095_count"] / n,
            "lt005_fraction": sums["appearance_gate_lt005_count"] / n,
            "bce_mean": sums["appearance_bce_sum"] / n,
        })(max(1.0, sums["appearance_valid_count"])),
        "per_sequence": per_sequence,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("checkpoint", "epoch", "counts", "gate", "loss", "rank", "appearance")},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
