"""SHUTrack v2 training entry: HCE + RSC, BCE + 0.1*MSE only."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from shutrack.data.batch_loader import BatchIndexLoader
from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.data.sampling import SequenceBatchSampler, ShardBatchSampler
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.schema import ARCHITECTURE_NAME, ARCHITECTURE_VERSION
from shutrack.training.checkpoint import (
    build_checkpoint_payload,
    load_shutrack_checkpoint,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from shutrack.training.loss import gate_loss
from shutrack.training.stats import ScalarAccumulator

TRAIN_DEFAULTS: dict[str, Any] = {
    "dataset_dir": "",
    "checkpoint_dir": "",
    "device": "cpu",
    "epochs": 100,
    "batch_size": 1024,
    "num_workers": 0,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "warmup_steps": -1,
    "grad_clip": 1.0,
    "seed": 42,
    "max_steps": 0,
    "max_samples": 0,
    "log_interval": 20,
    "competition_enabled": True,
    "resume": "",
    "init_checkpoint": "",
    "memory_shards": 1,
    "shard_cycles": 1,
    "epochs_per_shard": 0,
    "sequence_sampling": "sample-proportional",
    "d_model": 128,
    "n_heads": 4,
    "hce_layers": 2,
    "ff_dim": 256,
    "dropout": 0.0,
    "appearance_proj_dim": 64,
    "sequence_cache_size": 2,
    "data_loader": "batch",
    "prefetch": 2,
    "profile": False,
}

MODEL_INPUT_KEYS = (
    "history_evidence",
    "history_track",
    "history_detection",
    "history_mask",
    "history_has_detection",
    "current_evidence",
    "current_track",
    "current_detection",
    "current_competition",
    "current_has_detection",
    "history_reset",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def build_dataset(config: Mapping[str, Any]) -> PackedShutrackDataset:
    return PackedShutrackDataset(
        config["dataset_dir"],
        max_samples=int(config.get("max_samples", 0)),
        competition_enabled=bool(config.get("competition_enabled", True)),
        sequence_cache_size=int(config.get("sequence_cache_size", 2)),
    )


def build_model(dataset: PackedShutrackDataset, config: Mapping[str, Any]) -> ShutrackV2Model:
    model_config = ShutrackV2Config(
        reid_dim=dataset.reid_dim,
        d_model=int(config.get("d_model", 128)),
        n_heads=int(config.get("n_heads", 4)),
        hce_layers=int(config.get("hce_layers", 2)),
        ff_dim=int(config.get("ff_dim", 256)),
        dropout=float(config.get("dropout", 0.0)),
        appearance_proj_dim=int(config.get("appearance_proj_dim", 64)),
        competition_enabled=bool(config.get("competition_enabled", True)),
    )
    return ShutrackV2Model(model_config)


def _scheduler_lambda(total_steps: int, warmup_steps: int):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier


def _resolved_warmup_steps(config: Mapping[str, Any], steps_per_epoch: int, total_steps: int) -> int:
    explicit = int(config.get("warmup_steps", -1))
    if explicit >= 0:
        return min(explicit, total_steps)
    warmup = int(config.get("warmup_epochs", 1)) * max(1, steps_per_epoch)
    return min(warmup, max(1, total_steps // 10))


def _total_sequence_loads(dataset: PackedShutrackDataset, counter: Any) -> int:
    """Loads made in this process plus loads made by DataLoader workers."""

    shared = int(counter.value) if counter is not None else 0
    return int(dataset.sequence_load_count) + shared


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(int(line.split()[1]) / 1024.0)
    except OSError:
        pass
    return 0.0


def _rss_peak_mb() -> float:
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(int(line.split()[1]) / 1024.0)
    except OSError:
        pass
    return 0.0


def _cuda_peak_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)),
    )


def _validate_resume_contract(
    checkpoint: Mapping[str, Any],
    model: ShutrackV2Model,
    dataset: PackedShutrackDataset,
) -> None:
    mismatches: dict[str, Any] = {}
    if dict(checkpoint["model_config"]) != model.config.as_dict():
        mismatches["model_config"] = (
            checkpoint["model_config"],
            model.config.as_dict(),
        )
    mean = np.asarray(checkpoint.get("normalization_mean", []), dtype=np.float64)
    std = np.asarray(checkpoint.get("normalization_std", []), dtype=np.float64)
    if not np.allclose(mean, dataset.scalar_mean, rtol=0.0, atol=1e-12):
        mismatches["normalization_mean"] = "different"
    if not np.allclose(std, dataset.scalar_std, rtol=0.0, atol=1e-12):
        mismatches["normalization_std"] = "different"
    margin_removed = checkpoint.get("margin", None)
    if margin_removed is not None:
        mismatches["checkpoint_schema"] = "legacy margin-based checkpoint"
    if bool(checkpoint.get("competition_enabled")) != model.competition_enabled:
        mismatches["competition_enabled"] = (
            checkpoint.get("competition_enabled"),
            model.competition_enabled,
        )
    if dataset.competition_enabled != model.competition_enabled:
        mismatches["dataset_competition_enabled"] = (
            dataset.competition_enabled,
            model.competition_enabled,
        )
    checkpoint_dataset = checkpoint.get("dataset", {})
    current = dataset.descriptor()
    for key in (
        "dataset",
        "split",
        "train_sequences",
        "context_size",
        "max_frame_gap",
        "scalar_dim",
        "reid_dim",
        "evidence_dim",
        "competition_dim",
        "competition_enabled",
        "feature_schema_version",
        "label_source",
    ):
        if checkpoint_dataset.get(key) != current.get(key):
            mismatches[f"dataset.{key}"] = (
                checkpoint_dataset.get(key),
                current.get(key),
            )
    if mismatches:
        raise ValueError(
            "SHUTrack resume contract mismatch; refusing to continue with a "
            "different model/data/branch configuration: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


def _checkpoint_payload(
    model: ShutrackV2Model,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    dataset: PackedShutrackDataset,
    config: Mapping[str, Any],
    *,
    epoch: int,
    global_step: int,
    best_metric: float,
    include_optimizer: bool,
) -> dict[str, Any]:
    return build_checkpoint_payload(
        model=model,
        training_config=dict(config),
        dataset_descriptor=dataset.descriptor(),
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        optimizer=optimizer,
        scheduler=scheduler,
        normalization_mean=dataset.scalar_mean,
        normalization_std=dataset.scalar_std,
        include_optimizer=include_optimizer,
    )


def train_shutrack(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(TRAIN_DEFAULTS)
    resolved.update({key: value for key, value in config.items() if value is not None})
    if not resolved["dataset_dir"]:
        raise ValueError("dataset_dir is required")
    if not resolved["checkpoint_dir"]:
        raise ValueError("checkpoint_dir is required")
    checkpoint_dir = Path(resolved["checkpoint_dir"])
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    resume_path = str(resolved.get("resume", "")).strip()
    init_path = str(resolved.get("init_checkpoint", "")).strip()
    if resume_path and init_path:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    epochs = int(resolved["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if last_path.exists() and not resume_path:
        raise FileExistsError(
            "SHUTrack formal training requires a new checkpoint directory; "
            f"{last_path} already exists. Choose a new experiment name or pass "
            "--resume with the previous last.pt."
        )

    _seed_everything(int(resolved["seed"]))
    device = torch.device(str(resolved["device"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = build_dataset(resolved)
    try:
        model = build_model(dataset, resolved).to(device)
        initialization = {"mode": "random"}
        if init_path:
            source_model, source = load_shutrack_checkpoint(init_path, map_location="cpu")
            if source_model.config.as_dict() != model.config.as_dict():
                raise ValueError("--init-checkpoint model_config does not match SHUTrack A")
            model.load_state_dict(source_model.state_dict(), strict=True)
            initialization = {
                "mode": "weights_only",
                "checkpoint": str(Path(init_path).resolve()),
                "source_epoch": int(source["epoch"]),
                "source_dataset": source["dataset"]["dataset"],
            }
            del source_model, source
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
        )
        schedule_keys = (
            "memory_shards", "shard_cycles", "epochs_per_shard", "sequence_sampling"
        )
        uses_shards = any(resolved[key] != TRAIN_DEFAULTS[key] for key in schedule_keys)
        if uses_shards:
            sampler = ShardBatchSampler(dataset, resolved)
            epoch_steps = sampler.steps_per_epoch
        else:
            sampler = SequenceBatchSampler(
                dataset.sample_boundaries,
                int(resolved["batch_size"]),
                total_samples=len(dataset),
                seed=int(resolved["seed"]),
            )
            epoch_steps = [len(sampler)] * epochs
        steps_per_epoch = max(1, epoch_steps[0])
        max_steps = int(resolved.get("max_steps", 0))
        total_steps = (
            max_steps if max_steps > 0 else sum(epoch_steps)
        )
        warmup_steps = _resolved_warmup_steps(resolved, steps_per_epoch, total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _scheduler_lambda(total_steps, warmup_steps)
        )

        start_epoch = 0
        global_step = 0
        best_metric = float("inf")
        if resume_path:
            resumed_model, checkpoint = load_shutrack_checkpoint(
                resume_path, map_location=device
            )
            if "optimizer_state_dict" not in checkpoint:
                raise ValueError(
                    "resume requires a last.pt checkpoint with optimizer state; "
                    f"{resume_path} does not contain one"
                )
            _validate_resume_contract(checkpoint, model, dataset)
            checkpoint_config = checkpoint.get("training_config", {})
            if any(
                resolved[key] != checkpoint_config.get(key, TRAIN_DEFAULTS[key])
                for key in schedule_keys
            ):
                raise ValueError("SHUTrack resume sampling schedule mismatch")
            model.load_state_dict(resumed_model.state_dict(), strict=True)
            validate_checkpoint(checkpoint)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            best_metric = float(checkpoint.get("best_metric", best_metric))
            initialization = {"mode": "resume", "checkpoint": resume_path}

        data_loader_mode = str(resolved.get("data_loader", "batch"))
        if data_loader_mode not in ("batch", "items"):
            raise ValueError(
                f"data_loader must be 'batch' or 'items', got {data_loader_mode!r}"
            )
        profile = bool(resolved.get("profile", False))
        worker_counter = None
        if data_loader_mode == "batch":
            loader = BatchIndexLoader(
                dataset, sampler, prefetch=int(resolved.get("prefetch", 2))
            )
        else:
            num_workers = int(resolved["num_workers"])
            worker_init = None
            if num_workers > 0:
                worker_counter = torch.multiprocessing.Value("q", 0)

                def worker_init(worker_id: int) -> None:  # noqa: ARG001
                    dataset.set_shared_load_counter(worker_counter)

            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                collate_fn=PackedShutrackDataset.collate_fn,
                pin_memory=device.type == "cuda",
                persistent_workers=uses_shards and num_workers > 0,
                worker_init_fn=worker_init,
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = checkpoint_dir / "metrics.jsonl"
        training_log_path = checkpoint_dir / "training.log"
        summary_path = checkpoint_dir / "training_summary.json"
        started = time.time()
        log_interval = max(1, int(resolved["log_interval"]))
        stop_requested = max_steps > 0 and global_step >= max_steps
        last_epoch = start_epoch

        log_mode = "a" if resume_path else "w"
        metrics_mode = "a" if resume_path else "w"
        with training_log_path.open(log_mode) as log_handle, metrics_path.open(
            metrics_mode
        ) as metrics_handle:
            def write_log(message: str) -> None:
                log_handle.write(message + "\n")
                log_handle.flush()
                print(message, flush=True)

            write_log("=" * 72)
            write_log("SHUTrack v2 (HCE + RSC) training")
            if resume_path:
                write_log(f"RESUME from {resume_path}")
            elif init_path:
                write_log("INIT " + json.dumps(initialization, sort_keys=True))
            write_log(f"architecture={ARCHITECTURE_NAME} version={ARCHITECTURE_VERSION}")
            write_log(f"dataset={dataset.dataset_dir}")
            write_log(f"device={device}")
            write_log(
                f"data_loader={data_loader_mode} prefetch={resolved.get('prefetch')} "
                f"profile={profile}"
            )
            write_log(f"parameters={json.dumps(model.parameter_report(), sort_keys=True)}")
            write_log(f"samples={len(dataset)} steps_per_epoch={steps_per_epoch}")
            write_log(f"total_steps={total_steps} warmup_steps={warmup_steps}")
            write_log(
                f"competition_enabled={str(dataset.competition_enabled).lower()}"
            )
            write_log(
                "best.pt tracks the lowest average training loss, not tracker "
                "HOTA/MOTA/IDF1"
            )
            write_log(f"config={json.dumps(dict(resolved), sort_keys=True)}")

            for epoch in range(start_epoch + 1, epochs + 1):
                if stop_requested:
                    break
                model.train()
                sampler.set_epoch(epoch)
                if isinstance(sampler, ShardBatchSampler):
                    phase = sampler.phase
                    write_log(
                        f"epoch={epoch} cycle={phase['cycle']} shard={phase['shard']} "
                        f"planned_samples={phase['samples']} planned_batches={len(sampler)}"
                    )
                stat_names = (
                    "loss",
                    "bce",
                    "mse",
                    "motion_gate_mean",
                    "appearance_gate_mean",
                    "grad_norm",
                )
                stat_sums = {
                    name: torch.zeros((), device=device) for name in stat_names
                }
                stat_weights = {
                    name: torch.zeros((), device=device) for name in stat_names
                }
                epoch_started = time.time()
                epoch_start_loads = _total_sequence_loads(dataset, worker_counter)
                epoch_samples = 0
                data_seconds = 0.0
                h2d_seconds = 0.0
                compute_seconds = 0.0
                epoch_batches = 0
                loader_iter = iter(loader)
                try:
                    while True:
                        if profile:
                            data_started = time.perf_counter()
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            try:
                                batch = next(loader_iter)
                            except StopIteration:
                                break
                            data_seconds += time.perf_counter() - data_started
                        else:
                            try:
                                batch = next(loader_iter)
                            except StopIteration:
                                break
                        h2d_started = time.perf_counter()
                        batch = _move_batch(batch, device)
                        h2d_seconds += time.perf_counter() - h2d_started
                        compute_started = time.perf_counter()
                        epoch_batches += 1
                        epoch_samples += int(batch["safe_gate_target"].shape[0])
                        outputs = model(**_model_inputs(batch))
                        loss, components = gate_loss(outputs, batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"non-finite SHUTrack loss at epoch={epoch} "
                                f"step={global_step}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(resolved["grad_clip"])
                        )
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                        if profile:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            compute_seconds += time.perf_counter() - compute_started

                        weight = components["weight_sum"].detach()
                        weight = torch.where(
                            weight > 0, weight, torch.ones_like(weight)
                        )
                        for name, value in (
                            ("loss", loss.detach()),
                            ("bce", components["bce"].detach()),
                            ("mse", components["mse"].detach()),
                        ):
                            stat_sums[name] = stat_sums[name] + value * weight
                            stat_weights[name] = stat_weights[name] + weight
                        for name, value in (
                            (
                                "motion_gate_mean",
                                components["motion_gate_mean"].detach(),
                            ),
                            (
                                "appearance_gate_mean",
                                components["appearance_gate_mean"].detach(),
                            ),
                            ("grad_norm", grad_norm.detach()),
                        ):
                            stat_sums[name] = stat_sums[name] + value
                            stat_weights[name] = stat_weights[name] + 1.0

                        if global_step % log_interval == 0 or global_step == 1:
                            write_log(
                                f"epoch={epoch} step={global_step} "
                                f"loss={float(loss.detach()):.6f} "
                                f"bce={float(components['bce'].detach()):.6f} "
                                f"mse={float(components['mse'].detach()):.6f} "
                                f"grad_norm={float(grad_norm):.4f} "
                                f"lr={scheduler.get_last_lr()[0]:.3e}"
                            )
                        if max_steps > 0 and global_step >= max_steps:
                            stop_requested = True
                            break

                finally:
                    close_loader = getattr(loader_iter, "close", None)
                    if close_loader is not None:
                        close_loader()

                metrics = {}
                for name in stat_names:
                    weight = stat_weights[name]
                    safe = torch.where(
                        weight > 0, weight, torch.ones_like(weight)
                    )
                    value = stat_sums[name] / safe
                    metrics[name] = float(
                        torch.where(weight > 0, value, value * 0.0).item()
                    )
                last_epoch = epoch
                epoch_seconds = time.time() - epoch_started
                cuda_allocated_peak, cuda_reserved_peak = _cuda_peak_mb()
                metrics.update(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "epoch_seconds": float(epoch_seconds),
                        "samples": int(epoch_samples),
                        "samples_per_second": float(
                            epoch_samples / max(epoch_seconds, 1e-9)
                        ),
                        "data_read_seconds": float(data_seconds),
                        "h2d_seconds": float(h2d_seconds),
                        "compute_seconds": float(compute_seconds),
                        "batches": int(epoch_batches),
                        "profile": bool(profile),
                        "data_loader": data_loader_mode,
                        "sequence_load_count": int(
                            _total_sequence_loads(dataset, worker_counter)
                            - epoch_start_loads
                        ),
                        "rss_mb": _rss_mb(),
                        "rss_peak_mb": _rss_peak_mb(),
                        "cuda_peak_allocated_mb": cuda_allocated_peak,
                        "cuda_peak_reserved_mb": cuda_reserved_peak,
                        "competition_enabled": bool(
                            dataset.competition_enabled
                        ),
                    }
                )
                metrics_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                metrics_handle.flush()
                write_log(
                    f"epoch {epoch} summary: "
                    + json.dumps(
                        {
                            key: round(value, 6)
                            for key, value in metrics.items()
                            if isinstance(value, (int, float))
                        },
                        sort_keys=True,
                    )
                )

                epoch_metric = float(metrics.get("loss", float("inf")))
                improved = epoch_metric < best_metric
                if improved:
                    best_metric = epoch_metric
                save_checkpoint_atomic(
                    last_path,
                    _checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        dataset,
                        resolved,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        include_optimizer=True,
                    ),
                )
                if improved:
                    save_checkpoint_atomic(
                        best_path,
                        _checkpoint_payload(
                            model,
                            None,
                            None,
                            dataset,
                            resolved,
                            epoch=epoch,
                            global_step=global_step,
                            best_metric=best_metric,
                            include_optimizer=False,
                        ),
                    )
                if stop_requested:
                    break

        summary = {
            "architecture": ARCHITECTURE_NAME,
            "architecture_version": ARCHITECTURE_VERSION,
            "dataset": dataset.dataset_name,
            "split": dataset.split,
            "dataset_dir": str(dataset.dataset_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "epochs_completed": int(last_epoch),
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "best_metric_name": "training_loss",
            "elapsed_sec": float(time.time() - started),
            "sequence_load_count": int(
                _total_sequence_loads(dataset, worker_counter)
            ),
            "data_loader": data_loader_mode,
            "profile": bool(profile),
            "parameter_counts": model.parameter_report(),
            "competition_enabled": bool(dataset.competition_enabled),
            "rss_peak_mb": _rss_peak_mb(),
            "cuda_peak_allocated_mb": _cuda_peak_mb()[0],
            "cuda_peak_reserved_mb": _cuda_peak_mb()[1],
            "training_config": dict(resolved),
            "initialization": initialization,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    finally:
        dataset.close()


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    *,
    batch_size: int = 64,
    max_batches: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Small deterministic evaluation used by the CLI contract check."""

    model, checkpoint = load_shutrack_checkpoint(checkpoint_path, map_location="cpu")
    model.to(torch.device(device))
    model.eval()
    dataset = PackedShutrackDataset(
        dataset_dir,
        competition_enabled=bool(checkpoint["competition_enabled"]),
    )
    try:
        if dataset.reid_dim != int(checkpoint["reid_dim"]):
            raise ValueError(
                f"checkpoint/dataset reid_dim mismatch: "
                f"{checkpoint['reid_dim']} != {dataset.reid_dim}"
            )
        if dataset.competition_enabled != bool(checkpoint["competition_enabled"]):
            raise ValueError(
                "checkpoint/dataset competition branch mismatch"
            )
        checkpoint_mean = np.asarray(
            checkpoint.get("normalization_mean", []), dtype=np.float64
        ).reshape(-1)
        checkpoint_std = np.asarray(
            checkpoint.get("normalization_std", []), dtype=np.float64
        ).reshape(-1)
        if checkpoint_mean.shape != (63,) or checkpoint_std.shape != (63,):
            raise ValueError(
                "SHUTrack checkpoint/data normalization mismatch: "
                f"checkpoint={checkpoint_path} dataset_dir={dataset_dir}; "
                "checkpoint does not contain 63-dim mean/std"
            )
        mean_matches = np.allclose(
            checkpoint_mean, dataset.scalar_mean, rtol=0.0, atol=1e-12
        )
        std_matches = np.allclose(
            checkpoint_std, dataset.scalar_std, rtol=0.0, atol=1e-12
        )
        if not (mean_matches and std_matches):
            raise ValueError(
                "SHUTrack checkpoint/data normalization mismatch: "
                f"checkpoint={checkpoint_path} dataset_dir={dataset_dir}; "
                "checkpoint mean/std differ from the packed metadata "
                "(mean_max_abs_diff="
                f"{float(np.max(np.abs(checkpoint_mean - dataset.scalar_mean))):.6g}, "
                "std_max_abs_diff="
                f"{float(np.max(np.abs(checkpoint_std - dataset.scalar_std))):.6g}). "
                "Use a checkpoint trained on this dataset split."
            )
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=PackedShutrackDataset.collate_fn,
        )
        totals = ScalarAccumulator()
        batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, torch.device(device))
                outputs = model(**_model_inputs(batch))
                loss, components = gate_loss(outputs, batch)
                weight = float(components["weight_sum"].detach().cpu())
                totals.add("loss", loss, weight if weight > 0 else 1.0)
                totals.add("bce", components["bce"], weight if weight > 0 else 1.0)
                totals.add("mse", components["mse"], weight if weight > 0 else 1.0)
                totals.add(
                    "motion_gate_mean", components["motion_gate_mean"], 1.0
                )
                totals.add(
                    "appearance_gate_mean",
                    components["appearance_gate_mean"],
                    1.0,
                )
                batches += 1
                if max_batches > 0 and batches >= max_batches:
                    break
        return {
            "checkpoint": str(checkpoint_path),
            "dataset_dir": str(dataset_dir),
            "batches": batches,
            "metrics": totals.summary(),
            "parameter_counts": model.parameter_report(),
            "competition_enabled": bool(dataset.competition_enabled),
            "architecture": checkpoint["architecture"],
            "architecture_version": checkpoint["architecture_version"],
            "epoch": checkpoint.get("epoch"),
        }
    finally:
        dataset.close()


__all__ = [
    "TRAIN_DEFAULTS",
    "train_shutrack",
    "evaluate_checkpoint",
    "build_dataset",
    "build_model",
]
