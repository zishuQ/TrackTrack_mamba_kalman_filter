#!/usr/bin/env python3
"""Cost-weighted motion-gate BCE on the fixed paired trainer.

L_motion = mean_valid(1 - DIoU(box(g), GT_t1))
         + lambda * mean_valid(w * BCEWithLogits(z, g_star))
w = stop_gradient(max(L_pred - L_best, 0))

The shared network and appearance branch are frozen; only the existing motion
gate head (``rsc.gate_heads.0``) is trained. Targets (g_star) are the cached
sidecars produced by generate_motion_gate_targets.py; L_best is recomputed
from the stored endpoints and g_star, never rescanned in the batch loop.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from shutrack.training.motion_targets import (
    GRID, SOURCE_KEYS, TIE_TOLERANCE, generate_targets, source_fingerprint,
)
from shutrack.training.loss import _diou_xyxy
from shutrack.training.paired_pipeline import DeferredMetrics


class StopAfterEpoch(Exception):
    pass


def arguments_from_config(config, groups_root, resume):
    args = dict(config)
    args.update(groups_root=str(groups_root), resume=resume,
                max_epochs=config["epochs"], limit_sequences=0, sequences=None,
                device="cuda", prefetch=1, no_compact_reid=False,
                no_pin_memory=False, motion_harm_weight=1.0)
    args["from_scratch"] = config["from_scratch"]
    args["init"] = config.get("init")
    return SimpleNamespace(**args)


def cost_motion_loss(outputs, batch, lambda_cost):
    """Original DIoU regression plus cost-weighted BCE to the cached gate target."""
    logits = outputs["gate_logits"].float()[:, 0]
    gate = torch.sigmoid(logits)
    target = batch["motion_gate_target"].float()
    valid = batch["motion_gt_valid"].bool()
    box0 = batch["pred_box_g0"].float()
    box1 = batch["pred_box_g1"].float()
    gt = batch["gt_box_t1"].float()

    box = box0 + gate.unsqueeze(-1) * (box1 - box0)
    loss_pred = 1.0 - _diou_xyxy(box, gt)
    box_star = box0 + target.unsqueeze(-1) * (box1 - box0)
    loss_best = (1.0 - _diou_xyxy(box_star, gt)).detach()
    weight = torch.clamp(loss_pred.detach() - loss_best, min=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    denominator = valid.sum().clamp_min(1.0)
    l_pred = (loss_pred * valid).sum() / denominator
    l_bce = (weight * bce * valid).sum() / denominator
    total = l_pred + float(lambda_cost) * l_bce
    components = {
        "loss": total,
        "l_pred": l_pred.detach(),
        "l_bce": l_bce.detach(),
        "gate": gate.detach(),
        "weight": weight,
        "bce": bce.detach(),
        "loss_pred": loss_pred.detach(),
        "loss_best": loss_best,
        "valid": valid,
        "target": target,
        "box0": box0.detach(),
        "box1": box1.detach(),
    }
    return total, components


def _hist(values, mask):
    selected = values[mask]
    if selected.numel() == 0:
        return torch.zeros(10, device=values.device)
    return torch.histc(selected.float(), bins=10, min=0.0, max=1.0)


@contextmanager
def target_training(target_root, group_dir, *, stop_after_epoch=0, lambda_cost=0.0,
                    freeze_motion_head=True, init_path=None):
    target_root = Path(target_root).resolve()
    group_dir = Path(group_dir)
    names = ("SequenceData", "batch_tensors", "PAIRED_TENSOR_KEYS",
             "motion_box_diou_loss", "plan_phase", "build_checkpoint_payload",
             "save_checkpoint_atomic", "ShutrackV2Model")
    original = {name: getattr(trainer, name) for name in names}
    instances = []
    epoch_metrics = None

    class TargetSequenceData(original["SequenceData"]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            directory = target_root / self.sequence
            metadata = json.loads((directory / "metadata.json").read_text())
            source = {key: self.small[key] if key in self.small else self.samples[key]
                      for key in SOURCE_KEYS}
            if metadata["source_sha256"] != source_fingerprint(source):
                raise ValueError(f"{self.sequence}: gate labels do not match source sample rows")
            if metadata["grid"] != GRID.tolist() or metadata["tie_tolerance"] != TIE_TOLERANCE:
                raise ValueError(f"{self.sequence}: gate label policy differs")
            self.motion_gate_target = np.load(directory / "motion_gate_target.npy", allow_pickle=False)
            if self.motion_gate_target.shape != (self.n,) or self.motion_gate_target.dtype != np.float32:
                raise ValueError(f"{self.sequence}: gate labels have wrong shape or dtype")
            if metadata["target_sha256"] != hashlib.sha256(self.motion_gate_target.tobytes()).hexdigest():
                raise ValueError(f"{self.sequence}: gate label checksum differs")
            instances.append(self)

    def batch_tensors(sequence, rows, *args, **kwargs):
        batch = original["batch_tensors"](sequence, rows, *args, **kwargs)
        batch["motion_gate_target"] = sequence.motion_gate_target[rows]
        return batch

    def plan_phase(*args, **kwargs):
        nonlocal epoch_metrics
        epoch_metrics = DeferredMetrics(torch.device("cuda"))
        return original["plan_phase"](*args, **kwargs)

    def motion_loss(outputs, batch, **kwargs):
        total, components = cost_motion_loss(outputs, batch, lambda_cost)
        if epoch_metrics is not None:
            with torch.no_grad():
                gate = components["gate"]
                valid = components["valid"]
                target = components["target"]
                pred = components["loss_pred"]
                best = components["loss_best"]
                loss1 = 1.0 - _diou_xyxy(components["box1"], batch["gt_box_t1"].float())
                loss0 = 1.0 - _diou_xyxy(components["box0"], batch["gt_box_t1"].float())
                harmful = valid & (loss1 > loss0 + 1e-6)
                useful = valid & ~harmful
                target1 = valid & (target >= 1.0 - 1e-6)
                target_lt = valid & (target < 1.0 - 1e-6)
                weight = components["weight"]
                bce = components["bce"]
                delta = pred - loss1
                saved_harmful = ((loss1 - pred) * harmful).sum()
                added_useful = ((pred - loss1) * useful).sum()
                net_delta = (delta * valid).sum()
                base = {
                    "valid": valid.sum(), "harmful_count": harmful.sum(),
                    "useful_count": useful.sum(), "target1_count": target1.sum(),
                    "target_lt1_count": target_lt.sum(),
                    "harmful_gate_sum": (gate * harmful).sum(),
                    "useful_gate_sum": (gate * useful).sum(),
                    "target1_gate_sum": (gate * target1).sum(),
                    "target_lt1_gate_sum": (gate * target_lt).sum(),
                    "target_sum": (target * valid).sum(),
                    "gate_abs_error_sum": ((gate - target).abs() * valid).sum(),
                    "loss_full_sum": (loss1 * valid).sum(),
                    "loss_pred_sum": (pred * valid).sum(),
                    "loss_best_sum": (best * valid).sum(),
                    "saved_harmful_sum": saved_harmful,
                    "added_useful_sum": added_useful,
                    "net_delta_sum": net_delta,
                    "weight_sum": (weight * valid).sum(),
                    "weight_pos_count": ((weight > 0) & valid).sum(),
                    "bce_weighted_sum": (weight * bce * valid).sum(),
                    "bce_unweighted_sum": (bce * valid).sum(),
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
                for key, value in base.items():
                    epoch_metrics.add(key, value)
                for prefix, mask in (("harmful", harmful), ("useful", useful),
                                     ("target1", target1), ("target_lt1", target_lt),
                                     ("all", valid)):
                    hist = _hist(gate, mask)
                    for index in range(10):
                        epoch_metrics.add(f"{prefix}_hist_{index}", hist[index])
        return total, {"loss": total.detach(), "l_pred": components["l_pred"],
                       "l_bce": components["l_bce"]}

    descriptor = {
        "kind": "gt_oracle_cost_bce", "target_root": str(target_root),
        "grid": GRID.tolist(), "tie_tolerance": TIE_TOLERANCE,
        "lambda_cost": float(lambda_cost), "freeze_motion_head": bool(freeze_motion_head),
        "init": None if init_path is None else str(init_path),
    }

    def checkpoint_payload(**kwargs):
        payload = original["build_checkpoint_payload"](**kwargs)
        payload["motion_supervision"] = descriptor
        if epoch_metrics is not None:
            sums = epoch_metrics.summary()
            n = max(1.0, float(sums["valid"]))
            record = {
                "epoch": kwargs["epoch"], "global_step": kwargs["global_step"],
                "lambda_cost": float(lambda_cost), "valid": float(sums["valid"]),
                "target_mean": float(sums["target_sum"]) / n,
                "target_mae": float(sums["gate_abs_error_sum"]) / n,
                "loss_full_mean": float(sums["loss_full_sum"]) / n,
                "loss_pred_mean": float(sums["loss_pred_sum"]) / n,
                "loss_best_mean": float(sums["loss_best_sum"]) / n,
                "net_delta_mean": float(sums["net_delta_sum"]) / n,
                "weight_mean": float(sums["weight_sum"]) / n,
                "weight_pos_fraction": float(sums["weight_pos_count"]) / n,
                "bce_weighted_mean": float(sums["bce_weighted_sum"]) / n,
                "bce_unweighted_mean": float(sums["bce_unweighted_sum"]) / n,
                "gate_gt095_fraction": float(sums["gate_gt095_count"]) / n,
                "gate_lt005_fraction": float(sums["gate_lt005_count"]) / n,
                "cuda_peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            }
            for name in ("harmful", "useful", "target1", "target_lt1"):
                count = float(sums[name + "_count"])
                record[name + "_count"] = count
                record[name + "_gate_mean"] = (float(sums[name + "_gate_sum"]) / count
                                               if count else None)
                record[name + "_saved_mean"] = None
                record[name + "_added_mean"] = None
                if count:
                    if name == "harmful":
                        record[name + "_saved_mean"] = float(sums["saved_harmful_sum"]) / count
                    if name == "useful":
                        record[name + "_added_mean"] = float(sums["added_useful_sum"]) / count
                record[name + "_gate_gt095_fraction"] = (
                    float(sums.get(name + "_gate_gt095", 0.0)) / count if count else None)
                record[name + "_gate_lt005_fraction"] = (
                    float(sums.get(name + "_gate_lt005", 0.0)) / count if count else None)
            for prefix in ("harmful", "useful", "target1", "target_lt1", "all"):
                record[prefix + "_gate_hist"] = [
                    float(sums[f"{prefix}_hist_{index}"]) for index in range(10)
                ]
            with (group_dir / "motion_cost_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("motion cost epoch: " + json.dumps(record), flush=True)
        return payload

    def save_checkpoint(path, payload):
        original["save_checkpoint_atomic"](path, payload)
        if Path(path).name == "last.pt" and stop_after_epoch and payload["epoch"] >= stop_after_epoch:
            raise StopAfterEpoch()

    if freeze_motion_head:
        class FrozenMotionModel(original["ShutrackV2Model"]):
            def __init__(self, config):
                super().__init__(config)
                for name, parameter in self.named_parameters():
                    parameter.requires_grad = name.startswith("rsc.gate_heads.0.")
                trainable = {name: int(parameter.numel())
                             for name, parameter in self.named_parameters()
                             if parameter.requires_grad}
                if not trainable:
                    raise RuntimeError("no trainable motion-head parameters")
                (group_dir / "trainable_parameters.json").write_text(
                    json.dumps(trainable, indent=2, sort_keys=True) + "\n")
                print("trainable parameters: " + json.dumps(trainable, sort_keys=True), flush=True)

        trainer.ShutrackV2Model = FrozenMotionModel

    trainer.SequenceData = TargetSequenceData
    trainer.batch_tensors = batch_tensors
    trainer.PAIRED_TENSOR_KEYS = original["PAIRED_TENSOR_KEYS"] | {"motion_gate_target"}
    trainer.motion_box_diou_loss = motion_loss
    trainer.plan_phase = plan_phase
    trainer.build_checkpoint_payload = checkpoint_payload
    trainer.save_checkpoint_atomic = save_checkpoint
    try:
        yield descriptor
    finally:
        for name, value in original.items():
            setattr(trainer, name, value)
        for data in instances:
            data.close_cache()
            data.samples.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--groups-root", required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--freeze-motion-head", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    options = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(Path(options.config).read_text())
    group_dir = Path(options.groups_root) / config["group"]
    group_dir.mkdir(parents=True, exist_ok=True)
    with target_training(
        options.target_root, group_dir,
        stop_after_epoch=options.stop_after_epoch,
        lambda_cost=options.lambda_cost,
        freeze_motion_head=options.freeze_motion_head,
        init_path=config.get("init"),
    ) as descriptor:
        if options.resume:
            payload = torch.load(group_dir / "checkpoints/last.pt", map_location="cpu", weights_only=False)
            if payload.get("motion_supervision") != descriptor:
                raise ValueError("Resume checkpoint uses different motion supervision")
        manifest = group_dir / "motion_supervision.json"
        if manifest.exists() and json.loads(manifest.read_text()) != descriptor:
            raise ValueError("Experiment directory uses different gate target settings")
        manifest.write_text(json.dumps(descriptor, indent=2) + "\n")
        try:
            trainer.train(config["group"], arguments_from_config(config, options.groups_root, options.resume))
        except StopAfterEpoch:
            print(json.dumps({"stop_after_epoch": options.stop_after_epoch,
                              "checkpoint": str(group_dir / "checkpoints/last.pt")}), flush=True)


if __name__ == "__main__":
    main()
