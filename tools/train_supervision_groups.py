#!/usr/bin/env python3
"""Train the supervision groups on the paired SportsMOT trainval data.

Groups (same data, schedule, optimizer budget, e200 init):
  control          original motion+appearance soft labels
  motion_diou      motion loss replaced by next-frame GT-box DIoU
  appearance_score appearance label replaced by the template-discrimination
                   score (fallback to the original soft label when the
                   discrimination diagnostic is invalid)
  motion_diou_appearance_score  both of the above
  appearance_rank  motion DIoU + identity ranking on the gate-updated track
                   template (fallback to the original soft label)
  motion_diou_speed  motion DIoU + KF centre-velocity L1 + discrimination
                   appearance soft labels

Schedule: 40 shard-epochs = 4 shards x 10 cycles; per shard each sequence
contributes its contiguous fraction of samples (frame order).  Weights are
saved at epoch 20 and 40; last.pt is written every epoch for resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentguard.data.detection_cache import SequenceDetectionCache
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.training.checkpoint import (
    build_checkpoint_payload,
    load_shutrack_checkpoint,
    save_checkpoint_atomic,
)
from shutrack.training.loss import (
    appearance_rank_loss,
    appearance_soft_loss,
    gate_loss,
    motion_box_diou_loss,
    motion_speed_l1_loss,
)
from shutrack.training.train import _load_init_state_with_extension

ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
DATA = ROOT / "data"
CACHE = MAIN / "outputs/agentguard/detection_cache"
INIT_CKPT = MAIN / "outputs/agentguard/experiments/shutrack_a_submit_20261117/SportsMOT/checkpoints/last.pt"
SHARDS = 4
CYCLES = 10
EPOCHS = SHARDS * CYCLES


def tau_of(values: list[np.ndarray]) -> float:
    if not values:
        return 0.1
    array = np.concatenate(values)
    array = array[np.isfinite(array)]
    nonzero = np.abs(array[np.abs(array) > 1e-12])
    if nonzero.size == 0:
        return 0.1
    return float(np.clip(float(np.median(nonzero)), 1e-3, 0.1))


def load_global_taus(sequences: list[str], data_root: Path = DATA) -> dict:
    motion, appearance, delta = [], [], []
    for sequence in sequences:
        samples = np.load(data_root / sequence / "samples.npz", allow_pickle=False)
        motion.append(samples["motion_benefit"][samples["valid_motion"]])
        appearance.append(samples["appearance_benefit"][samples["valid_appearance"]])
        delta.append(samples["appearance_delta"][samples["appearance_diag_valid"]])
    return {
        "tau_motion": tau_of(motion),
        "tau_appearance": tau_of(appearance),
        "tau_delta": tau_of(delta),
    }


def derive_targets(samples, taus):
    motion_benefit = samples["motion_benefit"].astype(np.float64)
    appearance_benefit = samples["appearance_benefit"].astype(np.float64)
    valid_motion = samples["valid_motion"].astype(bool)
    valid_appearance = samples["valid_appearance"].astype(bool)
    coverage_gt = samples["gt_coverage"].astype(np.float64)
    coverage_oracle = samples["oracle_detection_coverage"].astype(np.float64)

    def safe(benefit, valid, coverage, tau):
        soft = 1.0 / (1.0 + np.exp(-benefit / tau))
        confidence = np.clip(
            coverage * np.minimum(np.abs(benefit) / (2.0 * tau), 1.0), 0.0, 1.0
        )
        value = confidence * soft + (1.0 - confidence) * 1.0
        value = np.where(valid, value, 1.0)
        return value.astype(np.float32)

    motion_safe = safe(motion_benefit, valid_motion, coverage_gt, taus["tau_motion"])
    appearance_safe = safe(
        appearance_benefit, valid_appearance, coverage_oracle, taus["tau_appearance"]
    )
    delta = samples["appearance_delta"].astype(np.float64)
    diag = samples["appearance_diag_valid"].astype(bool)
    appearance_new = np.where(
        diag,
        1.0 / (1.0 + np.exp(-delta / taus["tau_delta"])),
        appearance_safe,
    ).astype(np.float32)
    return {
        "motion_safe": motion_safe,
        "appearance_safe": appearance_safe,
        "appearance_new": appearance_new,
        "valid_motion": valid_motion,
        "valid_appearance": valid_appearance,
    }


class SequenceData:
    def __init__(self, sequence: str, split: str, data_root: Path = DATA):
        self.sequence = sequence
        self.dir = data_root / sequence
        self.samples = np.load(self.dir / "samples.npz", allow_pickle=False)
        self.event_evidence = np.load(self.dir / "event_evidence.npy", mmap_mode="r")
        self.event_track = np.load(self.dir / "event_track.npy", mmap_mode="r")
        self.event_det = np.load(self.dir / "event_detection_index.npy", mmap_mode="r")
        self.event_has = np.load(self.dir / "event_has_detection.npy", mmap_mode="r")
        self.split = split
        self.n = int(self.samples["frame"].shape[0])
        self.cache: SequenceDetectionCache | None = None
        self.rank_pos: np.ndarray | None = None
        if "rank_rows" in self.samples.files:
            rows = self.samples["rank_rows"]
            self.rank_pos = np.full(self.n, -1, dtype=np.int64)
            self.rank_pos[rows] = np.arange(rows.size, dtype=np.int64)

    def open_cache(self) -> None:
        if self.cache is None:
            self.cache = SequenceDetectionCache(
                CACHE / "SportsMOT" / self.split / self.sequence
            )

    def close_cache(self) -> None:
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    def load_rank_arrays(self) -> None:
        if getattr(self, "_rank_arrays", None) is not None:
            return
        if self.rank_pos is None:
            raise RuntimeError(f"{self.sequence}: rank fields not present in data")
        s = self.samples
        self._rank_arrays = (
            np.asarray(s["rank_template0"], dtype=np.float16),
            np.asarray(s["rank_detection"], dtype=np.float16),
            np.asarray(s["rank_positive"], dtype=np.float16),
            np.asarray(s["rank_competitor"], dtype=np.float16),
            np.asarray(s["rank_score"], dtype=np.float32),
        )


def batch_tensors(seq: SequenceData, rows: np.ndarray, targets, group: str) -> dict:
    samples = seq.samples
    indices = samples["window_indices"][rows]
    mask = indices < 0
    safe_indices = np.where(mask, 0, indices)
    history_evidence = np.asarray(seq.event_evidence[safe_indices], dtype=np.float32)
    history_track = np.asarray(seq.event_track[safe_indices], dtype=np.float32)
    history_evidence[mask] = 0.0
    history_track[mask] = 0.0
    history_det = np.asarray(seq.event_det[safe_indices], dtype=np.int64)
    history_det[mask] = -1
    history_has = np.asarray(seq.event_has[safe_indices], dtype=np.bool_)
    history_has[mask] = False
    current = samples["current_event_index"][rows]
    current_evidence = np.asarray(seq.event_evidence[current], dtype=np.float32)
    current_track = np.asarray(seq.event_track[current], dtype=np.float32)
    current_det = np.asarray(seq.event_det[current], dtype=np.int64)
    current_has = np.asarray(seq.event_has[current], dtype=np.bool_)
    all_indices = np.concatenate([history_det, current_det[:, None]], axis=1)
    features = np.zeros(
        (all_indices.shape[0], 6, seq.event_track.shape[1]), dtype=np.float32
    )
    valid = all_indices >= 0
    features[valid] = np.asarray(seq.cache.features[all_indices[valid]], dtype=np.float32)
    batch = {
        "history_evidence": history_evidence,
        "history_track": history_track,
        "history_detection": features[:, :5],
        "history_mask": mask,
        "history_has_detection": history_has,
        "history_reset": np.zeros(mask.shape, dtype=np.bool_),
        "current_evidence": current_evidence,
        "current_competition": samples["current_competition"][rows].astype(np.float32),
        "current_track": current_track,
        "current_detection": features[:, 5],
        "current_has_detection": current_has,
        "safe_gate_target": np.stack(
            [
                targets["motion_safe"][rows],
                targets["appearance_new"][rows]
                if group in ("appearance_score", "motion_diou_appearance_score", "motion_diou_speed")
                else targets["appearance_safe"][rows],
            ],
            axis=1,
        ).astype(np.float32),
        "valid_channels": np.stack(
            [targets["valid_motion"][rows], targets["valid_appearance"][rows]], axis=1
        ).astype(np.bool_),
        "sample_weight": np.ones(len(rows), dtype=np.float32),
        "appearance_target": (
            targets["appearance_new"][rows]
            if group in ("appearance_score", "motion_diou_appearance_score", "motion_diou_speed")
            else targets["appearance_safe"][rows]
        ).astype(np.float32),
        "appearance_valid": targets["valid_appearance"][rows].astype(np.bool_),
        "pred_box_g0": samples["pred_box_g0"][rows].astype(np.float32),
        "pred_box_g1": samples["pred_box_g1"][rows].astype(np.float32),
        "gt_box_t1": samples["gt_box_t1"][rows].astype(np.float32),
        "motion_gt_valid": samples["motion_gt_valid"][rows].astype(np.bool_),
    }
    if group == "appearance_rank":
        seq.load_rank_arrays()
        rank_position = seq.rank_pos[rows]
        hit = rank_position >= 0
        width = seq._rank_arrays[0].shape[1]

        def gather_rank(array: np.ndarray) -> np.ndarray:
            out = np.zeros((len(rows), width), dtype=np.float32)
            if hit.any():
                out[hit] = array[rank_position[hit]].astype(np.float32)
            return out

        batch["rank_template0"] = gather_rank(seq._rank_arrays[0])
        batch["rank_detection"] = gather_rank(seq._rank_arrays[1])
        batch["rank_positive"] = gather_rank(seq._rank_arrays[2])
        batch["rank_competitor"] = gather_rank(seq._rank_arrays[3])
        scores = np.zeros(len(rows), dtype=np.float32)
        if hit.any():
            scores[hit] = seq._rank_arrays[4][rank_position[hit]]
        batch["rank_score"] = scores
        batch["rank_valid"] = hit
    if group == "motion_diou_speed":
        batch["kf_velocity_g0"] = samples["kf_velocity_g0"][rows].astype(np.float32)
        batch["kf_velocity_g1"] = samples["kf_velocity_g1"][rows].astype(np.float32)
        batch["gt_velocity"] = samples["gt_velocity"][rows].astype(np.float32)
        batch["speed_valid"] = samples["speed_valid"][rows].astype(np.bool_)
    return batch


def model_inputs(batch: dict) -> dict:
    keys = (
        "history_evidence",
        "history_track",
        "history_detection",
        "history_mask",
        "history_has_detection",
        "history_reset",
        "current_evidence",
        "current_competition",
        "current_track",
        "current_detection",
        "current_has_detection",
    )
    return {key: batch[key] for key in keys}


def to_device(batch: dict, device: torch.device) -> dict:
    tensor_keys = {
        "history_evidence",
        "history_track",
        "history_detection",
        "history_mask",
        "history_has_detection",
        "history_reset",
        "current_evidence",
        "current_competition",
        "current_track",
        "current_detection",
        "current_has_detection",
        "safe_gate_target",
        "valid_channels",
        "sample_weight",
        "appearance_target",
        "appearance_valid",
        "pred_box_g0",
        "pred_box_g1",
        "gt_box_t1",
        "motion_gt_valid",
        "rank_template0",
        "rank_detection",
        "rank_positive",
        "rank_competitor",
        "rank_score",
        "rank_valid",
        "kf_velocity_g0",
        "kf_velocity_g1",
        "gt_velocity",
        "speed_valid",
    }
    return {
        key: torch.from_numpy(value).to(device) if key in tensor_keys else value
        for key, value in batch.items()
    }


def scheduler_lambda(total_steps: int, warmup_steps: int):
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return multiplier


def build_schedule(datasets: dict[str, SequenceData], batch_size: int, epochs: int = EPOCHS):
    epoch_steps = []
    for epoch in range(1, epochs + 1):
        shard = (epoch - 1) % SHARDS
        steps = 0
        for sequence, data in datasets.items():
            start = data.n * shard // SHARDS
            end = data.n * (shard + 1) // SHARDS
            steps += int(math.ceil((end - start) / batch_size))
        epoch_steps.append(steps)
    return epoch_steps


def shard_rows(data: SequenceData, shard: int) -> np.ndarray:
    start = data.n * shard // SHARDS
    end = data.n * (shard + 1) // SHARDS
    return np.arange(start, end, dtype=np.int64)


def train(group: str, args) -> None:
    data_root = Path(args.data_root)
    groups_root = Path(args.groups_root)
    group_dir = groups_root / group
    checkpoint_dir = group_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    metrics_path = group_dir / "metrics.jsonl"
    log_path = group_dir / "training.log"
    if last_path.exists() and not args.resume:
        raise SystemExit(f"{last_path} exists; pass --resume or use a new group dir")

    sequence_dirs = sorted(
        path.name for path in data_root.iterdir() if (path / "samples.npz").is_file()
    )
    if args.limit_sequences > 0:
        sequence_dirs = sequence_dirs[: args.limit_sequences]
    if not sequence_dirs:
        raise SystemExit("no generated sequences")
    epochs_total = EPOCHS if args.max_epochs <= 0 else min(EPOCHS, args.max_epochs)
    taus = load_global_taus(sequence_dirs, data_root)
    datasets: dict[str, SequenceData] = {}
    targets = {}
    for sequence in sequence_dirs:
        summary = json.loads((data_root / sequence / "summary.json").read_text())
        split = summary.get("split")
        if split not in ("train", "val"):
            split = (
                "val"
                if (CACHE / "SportsMOT" / "val" / sequence).is_dir()
                else "train"
            )
        data = SequenceData(sequence, split, data_root)
        datasets[sequence] = data
        targets[sequence] = derive_targets(data.samples, taus)
    batch_size = int(args.batch_size)
    epoch_steps = build_schedule(datasets, batch_size, epochs_total)
    total_steps = int(sum(epoch_steps))
    warmup_steps = int(epoch_steps[0])

    init_model, init_checkpoint = load_shutrack_checkpoint(args.init, map_location="cpu")
    model = ShutrackV2Model(
        ShutrackV2Config(**init_model.config.as_dict())
    )
    _load_init_state_with_extension(model, init_model)
    device = torch.device(args.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, scheduler_lambda(total_steps, warmup_steps)
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        print(f"resume from epoch {start_epoch} step {global_step}", flush=True)

    config = {
        "group": group,
        "data_root": str(data_root),
        "sequences": len(sequence_dirs),
        "batch_size": batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "shards": SHARDS,
        "cycles": CYCLES,
        "epochs": epochs_total,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "init": str(args.init),
        "taus": taus,
        "rank_tau": args.rank_tau,
        "rank_alpha": args.rank_alpha,
        "rank_weight": args.rank_weight,
        "speed_weight": args.speed_weight,
    }
    (group_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    log_handle = log_path.open("a" if args.resume else "w")
    metrics_handle = metrics_path.open("a" if args.resume else "w")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    log(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), **config}, sort_keys=True))
    for sequence in sequence_dirs:
        datasets[sequence].open_cache()
    seed_rng = np.random.RandomState(int(args.seed))
    for epoch in range(start_epoch + 1, epochs_total + 1):
        epoch_started = time.time()
        shard = (epoch - 1) % SHARDS
        rng = np.random.RandomState(int(args.seed) + epoch)
        order = list(sequence_dirs)
        rng.shuffle(order)
        totals = {
            "loss": 0.0,
            "motion_loss": 0.0,
            "appearance_loss": 0.0,
            "samples": 0,
            "batches": 0,
        }
        gate_sum = 0.0
        appearance_gate_sum = 0.0
        gate_count = 0
        for sequence in order:
            data = datasets[sequence]
            try:
                rows = shard_rows(data, shard)
                if rows.size == 0:
                    continue
                local_seed = int(
                    hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:8], 16
                ) % 100000
                local_rng = np.random.RandomState(
                    int(args.seed) + epoch * 1000 + local_seed
                )
                rows = rows[local_rng.permutation(rows.size)]
                for start in range(0, rows.size, batch_size):
                    batch_rows = rows[start : start + batch_size]
                    batch = to_device(
                        batch_tensors(data, batch_rows, targets[sequence], group), device
                    )
                    outputs = model(**model_inputs(batch))
                    extra: dict[str, float] = {}
                    if group == "appearance_rank":
                        motion_loss, motion_components = motion_box_diou_loss(outputs, batch)
                        appearance_loss, appearance_components = appearance_rank_loss(
                            outputs,
                            batch,
                            alpha=args.rank_alpha,
                            tau=args.rank_tau,
                            rank_weight=args.rank_weight,
                        )
                        loss = motion_loss + appearance_loss
                        extra["rank_mean"] = float(appearance_components["rank_mean"])
                        extra["fallback_mean"] = float(appearance_components["fallback_mean"])
                    elif group == "motion_diou_speed":
                        box_loss, motion_components = motion_box_diou_loss(outputs, batch)
                        speed_loss, speed_components = motion_speed_l1_loss(
                            outputs, batch, weight=args.speed_weight
                        )
                        motion_loss = box_loss + speed_loss
                        appearance_loss, appearance_components = appearance_soft_loss(outputs, batch)
                        loss = motion_loss + appearance_loss
                        extra["speed_l1"] = float(speed_components["speed_l1"])
                    elif group in ("motion_diou", "motion_diou_appearance_score"):
                        motion_loss, motion_components = motion_box_diou_loss(outputs, batch)
                        appearance_loss, appearance_components = appearance_soft_loss(outputs, batch)
                        loss = motion_loss + appearance_loss
                    else:
                        loss, components = gate_loss(outputs, batch)
                        motion_loss = loss
                        appearance_loss = loss
                        motion_components = components
                        appearance_components = components
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    count = int(batch_rows.size)
                    totals["loss"] += float(loss.detach()) * count
                    totals["motion_loss"] += float(motion_loss.detach()) * count
                    totals["appearance_loss"] += float(appearance_loss.detach()) * count
                    totals["samples"] += count
                    totals["batches"] += 1
                    for key, value in extra.items():
                        totals[key] = totals.get(key, 0.0) + value * count
                    with torch.no_grad():
                        gate_means = torch.sigmoid(outputs["gate_logits"]).mean(dim=0)
                    gate_sum += float(gate_means[0]) * count
                    appearance_gate_sum += float(gate_means[1]) * count
                    gate_count += count
            except Exception:
                raise
        epoch_seconds = time.time() - epoch_started
        metrics = {
            "epoch": epoch,
            "shard": shard + 1,
            "cycle": (epoch - 1) // SHARDS + 1,
            "global_step": global_step,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": epoch_seconds,
            "samples": totals["samples"],
            "batches": totals["batches"],
            "loss": totals["loss"] / max(totals["samples"], 1),
            "motion_loss": totals["motion_loss"] / max(totals["samples"], 1),
            "appearance_loss": totals["appearance_loss"] / max(totals["samples"], 1),
            "motion_gate_mean": gate_sum / max(gate_count, 1) if gate_count else None,
            "appearance_gate_mean": (
                appearance_gate_sum / max(gate_count, 1) if gate_count else None
            ),
        }
        for key in ("rank_mean", "fallback_mean", "speed_l1"):
            if key in totals:
                metrics[key] = totals[key] / max(totals["samples"], 1)
        log("epoch summary: " + json.dumps(metrics, sort_keys=True))
        metrics_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        metrics_handle.flush()

        payload = build_checkpoint_payload(
            model=model,
            training_config=config,
            dataset_descriptor={
                "dataset": "SportsMOT",
                "split": "trainval",
                "dataset_dir": str(data_root),
                "competition_enabled": True,
                "label_source": "paired_supervision_pilot",
            },
            epoch=epoch,
            global_step=global_step,
            best_metric=metrics["loss"],
            optimizer=optimizer,
            scheduler=scheduler,
            normalization_mean=init_checkpoint["normalization_mean"],
            normalization_std=init_checkpoint["normalization_std"],
            include_optimizer=True,
        )
        save_checkpoint_atomic(last_path, payload)
        if epoch in (20, 40) and epoch <= epochs_total:
            save_checkpoint_atomic(checkpoint_dir / f"epoch_{epoch:03d}.pt", payload)
            log(f"saved epoch_{epoch:03d}.pt at step {global_step}")
    log_handle.close()
    metrics_handle.close()
    for data in datasets.values():
        data.close_cache()
    (group_dir / "complete.json").write_text(
        json.dumps(
            {"group": group, "epochs": epochs_total, "global_step": global_step, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        required=True,
        choices=[
            "control",
            "motion_diou",
            "appearance_score",
            "motion_diou_appearance_score",
            "appearance_rank",
            "motion_diou_speed",
        ],
    )
    parser.add_argument("--init", default=str(INIT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-sequences", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--data-root", default=str(DATA))
    parser.add_argument("--groups-root", default=str(ROOT / "groups"))
    parser.add_argument("--rank-tau", type=float, default=0.1)
    parser.add_argument("--rank-alpha", type=float, default=0.95)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args.group, args)


if __name__ == "__main__":
    main()
