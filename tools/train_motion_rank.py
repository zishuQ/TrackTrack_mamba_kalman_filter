#!/usr/bin/env python3
"""Pairwise motion-gate ranking on top of the cost-weighted BCE trainer.

L_base = mean_valid(L_DIoU) + 0.25 * mean_valid(detach(max(L_DIoU-L_best,0))
                                               * BCEWithLogits(z,g_star))
L_rank = mean_pairs(detach(sqrt(gain_bad*damage_good))
                    * softplus(1 + z_bad - z_good))

Suppress events (g_star<1 and L(1)>L_best+1e-6) are paired within each batch
with random keep events (g_star=1 and L(0.5)>L(1)+1e-6) using an independent
generator. A: rank_weight=0; B: rank_weight=1.

All model parameters are trained (shared encoder, HCE, RSC and both gates);
the original appearance rank + column losses and their weights are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from train_motion_cost_bce import cost_motion_loss, _hist
from shutrack.training.loss import _diou_xyxy
from shutrack.training.paired_pipeline import DeferredMetrics
from shutrack.training.motion_targets import GRID, SOURCE_KEYS, TIE_TOLERANCE, source_fingerprint


class StopAfterEpoch(Exception):
    pass


PARAM_GROUPS = (
    "appearance_encoder", "event_fusion", "evidence_encoder", "hce",
    "rsc.base_queries.0", "rsc.base_queries.1", "rsc.condition_encoder",
    "rsc.gate_heads.0", "rsc.gate_heads.1",
    "rsc.modulations.0", "rsc.modulations.1",
)


def _param_group(name):
    for group in PARAM_GROUPS:
        if name.startswith(group + ".") or name == group:
            return group
    return "other"


def arguments_from_config(config, groups_root, resume):
    args = dict(config)
    args.update(groups_root=str(groups_root), resume=resume,
                max_epochs=config["epochs"], limit_sequences=0, sequences=None,
                device="cuda", prefetch=1, no_compact_reid=False,
                no_pin_memory=False, motion_harm_weight=1.0)
    args["from_scratch"] = config["from_scratch"]
    args["init"] = config.get("init")
    return SimpleNamespace(**args)


def pairwise_rank_loss(outputs, batch, rank_weight, generator):
    """Return (loss, stats). Stats are computed even when rank_weight is 0."""
    z = outputs["gate_logits"].float()[:, 0]
    labels = batch["motion_rank_label"].long()
    valid = batch["motion_gt_valid"].bool()
    bad = ((labels == 2) & valid).nonzero(as_tuple=False).flatten()
    good = ((labels == 1) & valid).nonzero(as_tuple=False).flatten()
    n_bad, n_good = int(bad.numel()), int(good.numel())
    stats = {"rank_pairs": 0.0, "rank_correct": 0.0, "rank_logit_gap": 0.0,
             "rank_gate_gap": 0.0, "rank_weight_sum": 0.0, "rank_loss_sum": 0.0}
    if n_bad == 0 or n_good == 0:
        return z.sum() * 0.0, stats
    picks = torch.randint(0, n_good, (n_bad,), generator=generator)
    good_idx = good[picks.to(good.device)]
    gain = batch["motion_rank_gain"].float()[bad]
    damage = batch["motion_rank_damage"].float()[good_idx]
    pair_weight = torch.sqrt(torch.clamp(gain * damage, min=0.0)).detach()
    difference = z[good_idx] - z[bad]
    pair_loss = F.softplus(1.0 + z[bad] - z[good_idx])
    loss = (pair_weight * pair_loss).sum() / max(1, n_bad)
    with torch.no_grad():
        gate = torch.sigmoid(z)
        stats.update({
            "rank_pairs": float(n_bad),
            "rank_correct": float((difference > 0).float().sum()),
            "rank_logit_gap": float(difference.sum()),
            "rank_gate_gap": float((gate[good_idx] - gate[bad]).sum()),
            "rank_weight_sum": float(pair_weight.sum()),
            "rank_loss_sum": float((pair_weight * pair_loss).sum()),
        })
    return float(rank_weight) * loss, stats


def _accumulate_cost(epoch_metrics, batch, components):
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
            "saved_harmful_sum": ((loss1 - pred) * harmful).sum(),
            "added_useful_sum": ((pred - loss1) * useful).sum(),
            "net_delta_sum": (delta * valid).sum(),
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


def _accumulate_appearance(epoch_metrics, batch, outputs):
    with torch.no_grad():
        z = outputs["gate_logits"].float()[:, 1]
        gate = torch.sigmoid(z)
        target = batch["appearance_target"].float()
        valid = batch["appearance_valid"].bool()
        bce = F.binary_cross_entropy_with_logits(z, target, reduction="none")
        values = {
            "appearance_valid_count": valid.sum(),
            "appearance_target_sum": (target * valid).sum(),
            "appearance_gate_sum": (gate * valid).sum(),
            "appearance_abs_error_sum": ((gate - target).abs() * valid).sum(),
            "appearance_gate_gt095_count": ((gate > 0.95) & valid).sum(),
            "appearance_gate_lt005_count": ((gate < 0.05) & valid).sum(),
            "appearance_bce_sum": (bce * valid).sum(),
        }
        for key, value in values.items():
            epoch_metrics.add(key, value)


@contextmanager
def target_training(target_root, rank_root, group_dir, *, stop_after_epoch=0,
                    base_lambda=0.25, rank_weight=0.0,
                    pair_seed=20261122, init_path=None, seed=None,
                    grad_every=10):
    target_root = Path(target_root).resolve()
    rank_root = Path(rank_root).resolve()
    group_dir = Path(group_dir)
    names = ("SequenceData", "batch_tensors", "PAIRED_TENSOR_KEYS",
             "motion_box_diou_loss", "plan_phase", "build_checkpoint_payload",
             "save_checkpoint_atomic", "ShutrackV2Model")
    original = {name: getattr(trainer, name) for name in names}
    instances = []
    epoch_metrics = None
    pair_generator = torch.Generator(device="cpu").manual_seed(int(pair_seed))

    class TrainingRecorder:
        def __init__(self, every):
            self.every = max(1, int(every))
            self.model = None
            self.snapshots = {}
            self.grad_sums = {}
            self.grad_samples = 0
            self.step_times = []
            self.step_last = None
            self.steps = 0

        def _flat(self, group):
            tensors = [p.detach().reshape(-1) for name, p in self.model.named_parameters()
                       if _param_group(name) == group]
            return torch.cat(tensors) if tensors else None

        def on_plan(self):
            self.snapshots = {}
            if self.model is not None:
                for group in PARAM_GROUPS:
                    flat = self._flat(group)
                    if flat is not None:
                        self.snapshots[group] = flat.clone()
            self.grad_sums = {}
            self.grad_samples = 0
            self.step_times = []
            self.step_last = None
            self.steps = 0

        def on_step(self):
            now = time.perf_counter()
            if self.step_last is not None:
                self.step_times.append(now - self.step_last)
            self.step_last = now
            self.steps += 1
            if self.model is None or self.steps % self.every:
                return
            squared = defaultdict(lambda: torch.zeros((), device=next(self.model.parameters()).device))
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                group = _param_group(name)
                squared[group] = squared[group] + parameter.grad.detach().float().pow(2).sum()
            for group, total in squared.items():
                self.grad_sums[group] = self.grad_sums.get(group, 0.0) + float(total.sqrt())
            self.grad_samples += 1

        def summary(self):
            updates = {}
            if self.model is not None:
                for group, old in self.snapshots.items():
                    new = self._flat(group)
                    if new is not None:
                        updates[group] = float(torch.linalg.vector_norm(new - old))
            grads = ({group: total / self.grad_samples
                      for group, total in self.grad_sums.items()}
                     if self.grad_samples else {})
            times = self.step_times
            return {
                "grad_norms": grads, "update_norms": updates,
                "step_seconds_mean": (sum(times) / len(times)) if times else None,
                "step_seconds_max": max(times) if times else None,
                "steps_measured": len(times), "grad_samples": self.grad_samples,
            }

    recorder = TrainingRecorder(grad_every)
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip_recording(parameters, max_norm, *args, **kwargs):
        recorder.on_step()
        return original_clip(parameters, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = clip_recording

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
            if metadata["target_sha256"] != hashlib.sha256(self.motion_gate_target.tobytes()).hexdigest():
                raise ValueError(f"{self.sequence}: gate label checksum differs")
            rank_dir = rank_root / self.sequence
            rank_meta = json.loads((rank_dir / "metadata.json").read_text())
            if rank_meta["source_sha256"] != metadata["source_sha256"] or \
                    rank_meta["target_sha256"] != metadata["target_sha256"]:
                raise ValueError(f"{self.sequence}: rank labels do not match target sidecar")
            self.motion_rank_label = np.load(rank_dir / "motion_rank_label.npy", allow_pickle=False)
            self.motion_rank_gain = np.load(rank_dir / "motion_rank_gain.npy", allow_pickle=False)
            self.motion_rank_damage = np.load(rank_dir / "motion_rank_damage.npy", allow_pickle=False)
            for key, array in (("label", self.motion_rank_label), ("gain", self.motion_rank_gain),
                               ("damage", self.motion_rank_damage)):
                if array.shape != (self.n,) or \
                        rank_meta[key + "_sha256"] != hashlib.sha256(array.tobytes()).hexdigest():
                    raise ValueError(f"{self.sequence}: rank {key} rows or checksum differ")
            instances.append(self)

    def batch_tensors(sequence, rows, *args, **kwargs):
        batch = original["batch_tensors"](sequence, rows, *args, **kwargs)
        batch["motion_gate_target"] = sequence.motion_gate_target[rows]
        batch["motion_rank_label"] = sequence.motion_rank_label[rows]
        batch["motion_rank_gain"] = sequence.motion_rank_gain[rows]
        batch["motion_rank_damage"] = sequence.motion_rank_damage[rows]
        return batch

    def plan_phase(*args, **kwargs):
        nonlocal epoch_metrics
        epoch_metrics = DeferredMetrics(torch.device("cuda"))
        recorder.on_plan()
        return original["plan_phase"](*args, **kwargs)

    def motion_loss(outputs, batch, **kwargs):
        base_total, base_components = cost_motion_loss(outputs, batch, base_lambda)
        rank_total, rank_stats = pairwise_rank_loss(outputs, batch, rank_weight, pair_generator)
        total = base_total + rank_total
        if epoch_metrics is not None:
            _accumulate_cost(epoch_metrics, batch, base_components)
            _accumulate_appearance(epoch_metrics, batch, outputs)
            for key, value in rank_stats.items():
                epoch_metrics.add(key, value)
        return total, {"loss": total.detach(), "l_pred": base_components["l_pred"],
                       "l_bce": base_components["l_bce"], "l_rank": rank_total.detach()
                       if torch.is_tensor(rank_total) else torch.tensor(float(rank_total))}

    descriptor = {
        "kind": "gt_oracle_cost_bce_rank", "target_root": str(target_root),
        "rank_root": str(rank_root), "grid": GRID.tolist(),
        "tie_tolerance": TIE_TOLERANCE, "base_lambda": float(base_lambda),
        "rank_weight": float(rank_weight), "pair_seed": int(pair_seed),
        "train_all_parameters": True, "seed": seed,
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
                "base_lambda": float(base_lambda), "rank_weight": float(rank_weight),
                "valid": float(sums["valid"]),
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
                record[name + "_gate_mean"] = float(sums[name + "_gate_sum"]) / count if count else None
                record[name + "_saved_mean"] = None
                record[name + "_added_mean"] = None
                if count and name == "harmful":
                    record[name + "_saved_mean"] = float(sums["saved_harmful_sum"]) / count
                if count and name == "useful":
                    record[name + "_added_mean"] = float(sums["added_useful_sum"]) / count
                record[name + "_gate_gt095_fraction"] = (
                    float(sums.get(name + "_gate_gt095", 0.0)) / count if count else None)
                record[name + "_gate_lt005_fraction"] = (
                    float(sums.get(name + "_gate_lt005", 0.0)) / count if count else None)
            for prefix in ("harmful", "useful", "target1", "target_lt1", "all"):
                record[prefix + "_gate_hist"] = [
                    float(sums[f"{prefix}_hist_{index}"]) for index in range(10)]
            pairs = float(sums.get("rank_pairs", 0.0))
            record.update({
                "rank_pairs": pairs, "rank_pairs_per_epoch": pairs,
                "rank_accuracy": (float(sums.get("rank_correct", 0.0)) / pairs) if pairs else None,
                "rank_logit_gap_mean": (float(sums.get("rank_logit_gap", 0.0)) / pairs) if pairs else None,
                "rank_gate_gap_mean": (float(sums.get("rank_gate_gap", 0.0)) / pairs) if pairs else None,
                "rank_pair_weight_mean": (float(sums.get("rank_weight_sum", 0.0)) / pairs) if pairs else None,
                "rank_loss_mean": (float(sums.get("rank_loss_sum", 0.0)) / pairs) if pairs else None,
            })
            appearance_n = max(1.0, float(sums.get("appearance_valid_count", 0.0)))
            record.update({
                "appearance_valid": float(sums.get("appearance_valid_count", 0.0)),
                "appearance_target_mean": float(sums.get("appearance_target_sum", 0.0)) / appearance_n,
                "appearance_gate_mean": float(sums.get("appearance_gate_sum", 0.0)) / appearance_n,
                "appearance_mae": float(sums.get("appearance_abs_error_sum", 0.0)) / appearance_n,
                "appearance_gate_gt095_fraction": float(sums.get("appearance_gate_gt095_count", 0.0)) / appearance_n,
                "appearance_gate_lt005_fraction": float(sums.get("appearance_gate_lt005_count", 0.0)) / appearance_n,
                "appearance_bce_mean": float(sums.get("appearance_bce_sum", 0.0)) / appearance_n,
            })
            monitoring = recorder.summary()
            record["grad_norm"] = monitoring["grad_norms"]
            record["update_norm"] = monitoring["update_norms"]
            record["step_seconds_mean"] = monitoring["step_seconds_mean"]
            record["step_seconds_max"] = monitoring["step_seconds_max"]
            record["steps_measured"] = monitoring["steps_measured"]
            record["grad_samples"] = monitoring["grad_samples"]
            with (group_dir / "motion_rank_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("motion rank epoch: " + json.dumps(record), flush=True)
        return payload

    def save_checkpoint(path, payload):
        original["save_checkpoint_atomic"](path, payload)
        if Path(path).name == "last.pt" and stop_after_epoch and payload["epoch"] >= stop_after_epoch:
            raise StopAfterEpoch()

    class AllParameterModel(original["ShutrackV2Model"]):
        def __init__(self, config):
            super().__init__(config)
            recorder.model = self
            parameters = {name: int(p.numel()) for name, p in self.named_parameters()}
            total = int(sum(parameters.values()))
            digest = hashlib.sha256()
            for name, parameter in sorted(self.named_parameters()):
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().numpy().tobytes())
            (group_dir / "trainable_parameters.json").write_text(
                json.dumps({"train_all_parameters": True,
                            "optimizer_scope": "model.parameters()",
                            "parameter_tensors": len(parameters), "total": total,
                            "all_parameters_requires_grad": all(
                                p.requires_grad for p in self.parameters())},
                           indent=2, sort_keys=True) + "\n")
            (group_dir / "initial_state.json").write_text(
                json.dumps({"sha256": digest.hexdigest(), "seed": seed,
                            "from_scratch": init_path is None,
                            "parameter_tensors": len(parameters), "total": total},
                           indent=2, sort_keys=True) + "\n")
            print("all parameters trainable: %d tensors, %d values; init %s"
                  % (len(parameters), total, digest.hexdigest()[:16]), flush=True)

    trainer.ShutrackV2Model = AllParameterModel
    trainer.SequenceData = TargetSequenceData
    trainer.batch_tensors = batch_tensors
    trainer.PAIRED_TENSOR_KEYS = original["PAIRED_TENSOR_KEYS"] | {
        "motion_gate_target", "motion_rank_label", "motion_rank_gain", "motion_rank_damage"}
    trainer.motion_box_diou_loss = motion_loss
    trainer.plan_phase = plan_phase
    trainer.build_checkpoint_payload = checkpoint_payload
    trainer.save_checkpoint_atomic = save_checkpoint
    try:
        yield descriptor
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        for name, value in original.items():
            setattr(trainer, name, value)
        for data in instances:
            data.close_cache()
            data.samples.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--rank-root", required=True)
    parser.add_argument("--groups-root", required=True)
    parser.add_argument("--base-lambda", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--pair-seed", type=int, default=20261122)
    parser.add_argument("--grad-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    options = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(Path(options.config).read_text())
    group_dir = Path(options.groups_root) / config["group"]
    group_dir.mkdir(parents=True, exist_ok=True)
    with target_training(
        options.target_root, options.rank_root, group_dir,
        stop_after_epoch=options.stop_after_epoch,
        base_lambda=options.base_lambda, rank_weight=options.rank_weight,
        pair_seed=options.pair_seed, init_path=config.get("init"),
        seed=config.get("seed"), grad_every=options.grad_every,
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
