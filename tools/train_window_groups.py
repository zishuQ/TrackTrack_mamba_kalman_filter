#!/usr/bin/env python3
"""Train SHUTrack groups with continuous-window motion supervision.

Adds two mechanisms on top of ``train_supervision_groups.py``:

* ``--window-frames K``: roll the NSA-KF state through the anchor event and up
  to K-1 preceding consecutive matched events. The model's motion gate at each
  frame controls the update; the resulting state builds the next frame's
  evidence, so later box errors backpropagate into earlier gates. The window is
  truncated at frame gaps, missing warps and events without a sample row.
* ``--augment mottrack``: per-epoch MoTTrack-style observation noise / missing
  pattern on the chain, recomputing measurement-dependent evidence and KF
  states. Ground truth and ReID features are untouched.

``--window-supervision full`` supervises every transition in the chain;
``anchor`` supervises only the anchor transition (one-step target with
recurrent state). ``none`` falls back to the plain one-step loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "3. Tracker"))
sys.path.insert(0, str(ROOT / "agentguard/src"))

import train_supervision_groups as base  # noqa: E402
from shutrack.training.checkpoint import (  # noqa: E402
    build_checkpoint_payload, save_checkpoint_atomic,
)
from shutrack.training.loss import (  # noqa: E402
    appearance_column_loss, appearance_rank_loss, _diou_xyxy,
)
from shutrack.training.recurrent_motion import (  # noqa: E402
    evidence_from_state, kf_update_gated, warp_predict,
)
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model  # noqa: E402
from shutrack.training.window_replay import (  # noqa: E402
    WINDOW_IMPLEMENTATION, WindowSequenceBank, prepare_window_batch, replay_loss,
)
from shutrack.training.observation_noise import ObservationNoise  # noqa: E402
from shutrack.training.defaults import (  # noqa: E402
    MAINLINE_APPEARANCE_PROJ_DIM, MAINLINE_ARCHITECTURE_VARIANT,
)

SCORE_KEYS = ("loss", "motion_loss", "appearance_loss")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="appearance_row_column")
    parser.add_argument("--dataset", default="SportsMOT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--groups-root", required=True)
    parser.add_argument("--architecture-variant", default=MAINLINE_ARCHITECTURE_VARIANT)
    parser.add_argument("--appearance-proj-dim", type=int, default=MAINLINE_APPEARANCE_PROJ_DIM)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-weight", type=float, default=18.0)
    parser.add_argument("--rank-alpha", type=float, default=0.95)
    parser.add_argument("--rank-tau", type=float, default=0.1)
    parser.add_argument("--column-weight", type=float, default=50.0)
    parser.add_argument("--column-hinge", action="store_true")
    parser.add_argument("--motion-harm-weight", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--preload-features", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=160)
    parser.add_argument("--grad-norm-every", type=int, default=5)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--window-supervision", choices=("full", "anchor", "none"),
                        default="full")
    parser.add_argument("--augment", choices=("none", "mottrack", "error_pool"),
                        default="none")
    parser.add_argument("--error-pool", default="",
                        help="Directory containing error_pool.npz (relative residuals)")
    parser.add_argument("--augment-pos-std", type=float, default=None)
    parser.add_argument("--augment-size-std", type=float, default=None)
    parser.add_argument("--augment-miss", type=float, default=None)
    parser.add_argument("--augment-burst", type=float, default=None)
    parser.add_argument("--augment-continue", type=float, default=0.4)
    parser.add_argument("--augment-small-area", type=float, default=2500.0)
    parser.add_argument("--augment-boundary-margin", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=1)
    args = parser.parse_args()
    if args.micro_batch is None:
        args.micro_batch = args.batch_size
    profile = ObservationNoise.for_dataset(args.dataset)
    for name, field in (("augment_pos_std","position_std"),("augment_size_std","size_std"),
                        ("augment_miss","missing"),("augment_burst","burst")):
        if getattr(args,name) is None:
            setattr(args,name,getattr(profile,field))
    return args


def augment_observation(rng, measurement, score, image_w, image_h, args):
    """One MoTTrack-style observation perturbation (cxcywh)."""
    cx, cy, w, h = measurement
    area = max(w, 1.0) * max(h, 1.0)
    probability = args.augment_miss
    if area < args.augment_small_area:
        probability *= 2.0
    margin = args.augment_boundary_margin
    if cx - w / 2 < margin or cy - h / 2 < margin or \
            cx + w / 2 > image_w - margin or cy + h / 2 > image_h - margin:
        probability *= 1.5
    probability = min(probability, 0.5)
    missing = rng.random_sample() < probability
    if missing:
        return None, 0.0, True
    noise = rng.normal(0.0, [args.augment_pos_std, args.augment_pos_std,
                             args.augment_size_std, args.augment_size_std])
    noisy = np.asarray([cx + noise[0], cy + noise[1],
                        max(w + noise[2], 1.0), max(h + noise[3], 1.0)])
    return noisy.astype(np.float32), float(score), False


def chain_for_row(data, row, window_frames):
    """Return (events, rows, frames) for the consecutive suffix ending at anchor."""
    anchor = int(data.small["current_event_index"][row])
    indices = [int(value) for value in data.small["window_indices"][row] if value >= 0]
    candidates = indices + [anchor]
    chain = []
    frames = []
    for event in candidates:
        target_row = int(data.event_target_row[event]) if hasattr(data, "event_target_row") else -1
        frame = int(data.event_frame[event]) if hasattr(data, "event_frame") else -1
        if target_row < 0 or frame < 0:
            chain = []
            frames = []
            continue
        if frames and frame != frames[-1] + 1:
            chain = []
            frames = []
        chain.append(event)
        frames.append(frame)
    if not chain or chain[-1] != anchor:
        return [], [], []
    chain = chain[-window_frames:]
    frames = frames[-window_frames:]
    rows = [int(data.event_target_row[event]) for event in chain]
    return chain, rows, frames


def build_position_batch(data, rows, targets, group):
    batch = base.batch_tensors(data, np.asarray(rows, dtype=np.int64), targets, group,
                               compact_reid=True)
    return batch


def slice_tensors(values, indices, batch_size):
    """Select rows from every tensor whose leading dimension is the batch."""
    device = "cpu"
    for value in values.values():
        if torch.is_tensor(value):
            device = value.device
            break
    selected = torch.tensor(indices, dtype=torch.long, device=device)
    result = {}
    for key, value in values.items():
        if (torch.is_tensor(value) and value.dim() > 0
                and value.shape[0] == batch_size):
            result[key] = value.index_select(0, selected.to(value.device))
        else:
            result[key] = value
    return result


def validate_resume_config(previous, current):
    """A resume keeps the objective, data order and optimizer schedule intact."""
    keys = ("window_implementation", "group", "dataset", "split", "sequence_names",
            "batch_size", "micro_batch", "lr", "weight_decay", "seed", "epochs",
            "total_steps", "warmup_steps", "window_frames", "window_supervision",
            "augment", "augment_params", "error_pool", "error_pool_sha256",
            "motion_harm_weight", "dropout",
            "rank_weight", "rank_alpha", "rank_tau", "column_weight", "column_hinge",
            "architecture_variant", "appearance_proj_dim", "taus")
    changed = [key for key in keys if previous.get(key) != current.get(key)]
    if changed:
        raise ValueError("Resume configuration changed: " + ", ".join(changed))


def main():
    args = parse_args()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    group_dir = Path(args.groups_root) / args.group
    checkpoint_dir = group_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    if last_path.is_file() and not args.resume:
        raise FileExistsError("Existing training output: use --resume or a new groups root")
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume_payload.get("training_config", {}).get("window_implementation") != WINDOW_IMPLEMENTATION:
            raise ValueError("Checkpoint uses the old window input semantics. "
                             "Keep it archived and start the corrected run in a new directory.")

    sequences = sorted(path.name for path in data_root.iterdir()
                       if (path / "samples.npz").is_file())
    train_set = base.official_sequences("train", args.dataset)
    val_set = base.official_sequences("val", args.dataset)
    expected = train_set | val_set if args.split == "trainval" else train_set
    sequences = sorted(set(sequences) & expected)
    if not sequences:
        raise SystemExit("no training sequences")
    taus = base.load_global_taus(sequences, data_root)
    datasets = {}
    targets = {}
    for sequence in sequences:
        split = "val" if sequence in val_set else "train"
        data = base.SequenceData(sequence, split, data_root, args.dataset)
        datasets[sequence] = data
        targets[sequence] = base.derive_targets(data.samples, taus)
        for name in ("event_state_mean", "event_state_cov", "event_det_measurement"):
            if not hasattr(data, name):
                raise SystemExit(f"{sequence}: missing {name}; regenerate with state dumps")
    if args.preload_features:
        started = time.time()
        for data in datasets.values():
            data.preload()
        print(f"preload {time.time()-started:.1f}s", flush=True)
    args.error_pool_array = None
    args.error_pool_sha256 = None
    if args.augment == "error_pool":
        pool_file = Path(args.error_pool) / "error_pool.npz"
        if not pool_file.is_file():
            raise SystemExit(f"missing error pool: {pool_file}")
        with np.load(pool_file) as archive:
            args.error_pool_array = np.asarray(archive["relative"], np.float32)
        args.error_pool_sha256 = hashlib.sha256(pool_file.read_bytes()).hexdigest()[:16]

    base._seed_everything(args.seed)
    dims = {data.event_track.shape[1] for data in datasets.values()}
    model = ShutrackV2Model(ShutrackV2Config(
        reid_dim=int(dims.pop()), appearance_proj_dim=args.appearance_proj_dim,
        architecture_variant=args.architecture_variant, dropout=args.dropout))
    norm_mean, norm_std = base.fit_input_statistics(datasets)
    for data in datasets.values():
        data.input_statistics = (norm_mean, norm_std)
    initial_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial_hash.update(name.encode())
        initial_hash.update(value.detach().cpu().numpy().tobytes())
    initial_sha = initial_hash.hexdigest()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    epoch_steps = [sum(math.ceil(data.samples["frame"].shape[0] / args.batch_size)
                       for data in datasets.values())] * args.max_epochs
    total_steps = sum(epoch_steps)
    warmup = min(args.warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, base.scheduler_lambda(total_steps, warmup))

    config = {
        "group": args.group, "dataset": args.dataset, "data_root": str(data_root),
        "sequences": len(sequences), "batch_size": args.batch_size,
        "micro_batch": args.micro_batch, "accum_steps": math.ceil(args.batch_size / args.micro_batch),
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "shards": args.shards, "epochs": args.max_epochs,
        "total_steps": total_steps, "warmup_steps": warmup,
        "window_frames": args.window_frames,
        "window_supervision": args.window_supervision,
        "augment": args.augment,
        "error_pool": args.error_pool or None,
        "error_pool_sha256": args.error_pool_sha256,
        "augment_params": {
            "pos_std": args.augment_pos_std, "size_std": args.augment_size_std,
            "miss": args.augment_miss, "burst": args.augment_burst,
            "continue": args.augment_continue, "small_area": args.augment_small_area,
            "boundary_margin": args.augment_boundary_margin},
        "motion_harm_weight": args.motion_harm_weight,
        "dropout": args.dropout, "rank_weight": args.rank_weight,
        "rank_alpha": args.rank_alpha, "rank_tau": args.rank_tau,
        "column_weight": args.column_weight, "column_hinge": args.column_hinge,
        "architecture_variant": args.architecture_variant,
        "appearance_proj_dim": args.appearance_proj_dim,
        "loading_mode": "gpu_sequence_bank",
        "snapshot_every": args.snapshot_every, "from_scratch": True,
        "init": None, "split": args.split, "preload_features": args.preload_features,
        "grad_norm_every": args.grad_norm_every,
        "sequence_names": sequences, "taus": taus,
    }
    config["initial_parameters_sha256"] = initial_sha
    config["stop_after_epoch"] = args.stop_after_epoch
    config["window_implementation"] = WINDOW_IMPLEMENTATION
    config["association_context"] = "frozen_replay_candidates"
    config["noise_rng"] = "stateless_anchor_epoch"
    if resume_payload is not None:
        validate_resume_config(resume_payload["training_config"], config)
        endpoint = args.stop_after_epoch or args.max_epochs
        if int(resume_payload["epoch"]) >= endpoint:
            print(f"Already at epoch {resume_payload['epoch']}; requested endpoint {endpoint}")
            return
    (group_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    log_path = group_dir / "training.log"
    metrics_path = group_dir / "metrics.jsonl"
    log_handle = log_path.open("a" if args.resume else "w")
    metrics_handle = metrics_path.open("a" if args.resume else "w")

    def log(message):
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    model_params = sum(p.numel() for p in model.parameters())
    optimizer_params = sum(p.numel() for group_ in optimizer.param_groups
                           for p in group_["params"])
    if model_params != optimizer_params:
        raise RuntimeError("optimizer does not hold every parameter")
    dropout_modules = [(name, float(module.p)) for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Dropout) and module.p > 0]
    log("verification: " + json.dumps({
        "model_param_numel": model_params, "optimizer_param_numel": optimizer_params,
        "dropout_modules": dropout_modules, "model_config": model.config.as_dict(),
    }, sort_keys=True))
    log("start: " + json.dumps({**config, "time": base.time.strftime("%Y-%m-%d %H:%M:%S")},
                               sort_keys=True))

    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = resume_payload
        if payload.get("training_config", {}).get("window_implementation") != WINDOW_IMPLEMENTATION:
            raise ValueError("This checkpoint predates corrected window normalization/history. "
                             "Start the corrected experiment in a new output directory.")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        rng_state = payload.get("rng_state")
        if rng_state:
            torch.set_rng_state(rng_state["torch"])
            if rng_state.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
        log(f"resume from epoch {start_epoch} step {global_step}")

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        epoch_started = time.time()
        order, rows_by_sequence, quotas, _ = base.plan_phase(
            datasets, 0, 1, "sample-proportional", 1.0, args.seed, epoch)
        sampled = 0
        micro_batches = 0
        sums = {key: 0.0 for key in SCORE_KEYS}
        gate_motion = gate_appearance = 0.0
        gate_count = 0
        deferred = torch.zeros(5, dtype=torch.float64, device=device)
        transition_count = 0
        missing_count = 0
        simulated_count = 0
        fallback_count = 0
        grad_probe = {}
        probe_allowed = bool(args.grad_norm_every > 0
                             and (epoch == 1 or epoch % args.grad_norm_every == 0))
        for sequence in order:
            data = datasets[sequence]
            data.open_cache()
            bank = WindowSequenceBank(data, targets[sequence], device)
            rows_all = rows_by_sequence[sequence]
            try:
                builder = lambda selected: prepare_window_batch(
                    base, data, targets[sequence], selected, epoch, args,
                    pin_memory=device.type == "cuda", bank=bank)
                with base.prepared_batches(
                    rows_all, args.batch_size, builder, (),
                    prefetch=args.prefetch, pin_memory=False,
                ) as batches:
                    for chunk, prepared in batches:
                        count = len(chunk)
                        for start in range(0, count, args.micro_batch):
                            stop = min(start + args.micro_batch, count)
                            piece = slice_prepared(prepared, start, stop)
                            loss, parts, gates = replay_loss(
                                base, model, piece, args, device,
                                grad_probe, probe_allowed=probe_allowed)
                            probe_allowed = False
                            (loss * ((stop-start) / count)).backward()
                            weight = stop-start
                            deferred += torch.stack((
                                loss.detach(), parts["motion"], parts["appearance"],
                                gates[0], gates[1],
                            )).to(torch.float64) * weight
                            sampled += weight
                            micro_batches += 1
                            gate_count += weight
                            transition_count += parts["transitions"]
                            missing_count += parts["missing"]
                            simulated_count += parts.get("simulated", 0)
                            fallback_count += parts.get("fallback", 0)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                        optimizer.zero_grad(set_to_none=True)
            finally:
                data.close_cache()
        values = deferred.cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Non-finite window training loss/metrics")
        sums = dict(zip(SCORE_KEYS, values[:3]))
        gate_motion, gate_appearance = values[3:]
        metrics = {
            "epoch": epoch, "global_step": global_step,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "samples": sampled, "micro_batches": micro_batches,
            "loss": sums["loss"] / max(1, sampled),
            "motion_loss": sums["motion_loss"] / max(1, sampled),
            "appearance_loss": sums["appearance_loss"] / max(1, sampled),
            "motion_gate_mean": gate_motion / max(1, gate_count),
            "appearance_gate_mean": gate_appearance / max(1, gate_count),
            "transitions": transition_count,
            "augmented_missing": missing_count,
            "error_pool_simulated": simulated_count,
            "error_pool_fallback": fallback_count,
            "rss_gb": round(base._process_rss_gb(), 3),
            "rss_peak_gb": round(base.resource.getrusage(base.resource.RUSAGE_SELF).ru_maxrss
                                 / (1024 * 1024), 3),
        }
        if device.type == "cuda":
            metrics["cuda_peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 20, 2)
            metrics["cuda_peak_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 2 ** 20, 2)
        metrics.update(grad_probe)
        log("epoch summary: " + json.dumps(metrics, sort_keys=True))
        metrics_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        metrics_handle.flush()
        payload = build_checkpoint_payload(
            model=model, training_config=config,
            dataset_descriptor={"dataset": args.dataset, "split": args.split,
                                "dataset_dir": str(data_root),
                                "competition_enabled": True},
            epoch=epoch, global_step=global_step, best_metric=metrics["loss"],
            optimizer=optimizer, scheduler=scheduler,
            normalization_mean=norm_mean, normalization_std=norm_std,
            include_optimizer=True)
        save_checkpoint_atomic(last_path, payload)
        snapshot_every = max(1, args.snapshot_every)
        if epoch % snapshot_every == 0 or epoch == args.max_epochs:
            save_checkpoint_atomic(checkpoint_dir / f"epoch_{epoch:03d}.pt", payload)
            log(f"saved epoch_{epoch:03d}.pt at step {global_step}")
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            break
    log_handle.close()
    metrics_handle.close()
    for data in datasets.values():
        data.close_cache()
    final_state = "complete.json" if epoch == args.max_epochs else "paused.json"
    (group_dir / final_state).write_text(json.dumps({
        "group": args.group, "epochs": epoch,
        "global_step": global_step, "finished_at": base.time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, sort_keys=True) + "\n")


def slice_prepared(prepared, start, stop):
    """Slice an effective batch while preserving each anchor\'s augmentation."""
    count = prepared["count"]
    def select(value, key=None):
        if isinstance(value, dict):
            return {k: select(v, k) for k,v in value.items()}
        if (torch.is_tensor(value) and value.ndim and value.shape[0] == count
                and key not in ("norm_mean", "norm_std")):
            return value[start:stop]
        return value
    result = select(prepared)
    result["count"] = stop-start
    valid=result["valid"]
    result["missing"]=int((valid & ~result["has"]).sum())
    result["simulated"]=int(result["simulated_mask"].sum())
    result["fallback"]=int(result["fallback_mask"].sum())
    # The caller's supervision mode is encoded by the original transition count.
    result["transitions"]=(int(valid.sum()) if prepared["transitions"] != count
                           else stop-start)
    return result


def run_microbatch(model, data, target, rows, epoch, args, device, log, grad_probe,
                   probe_allowed=False):
    prepared=prepare_window_batch(base,data,target,rows,epoch,args,
                                  pin_memory=torch.device(device).type=="cuda")
    return replay_loss(base,model,prepared,args,device,grad_probe,probe_allowed)


if __name__ == "__main__":
    main()




