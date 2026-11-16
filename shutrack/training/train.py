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

from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.data.sampling import SequenceBatchSampler
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
    "margin_mode": "auto",
    "margin_stats_json": "",
    "resume": "",
    "d_model": 128,
    "n_heads": 4,
    "hce_layers": 2,
    "ff_dim": 256,
    "dropout": 0.0,
    "appearance_proj_dim": 64,
    "sequence_cache_size": 2,
}

MODEL_INPUT_KEYS = (
    "history_motion",
    "history_track",
    "history_detection",
    "history_context",
    "history_mask",
    "history_has_detection",
    "current_motion",
    "current_track",
    "current_detection",
    "current_context",
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


def _load_margin_stats(path: str) -> dict[str, Any] | None:
    if not str(path or "").strip():
        return None
    data = json.loads(Path(path).read_text())
    if "mean" not in data or "std" not in data:
        raise ValueError(f"margin stats file must contain mean/std: {path}")
    return data


def build_dataset(config: Mapping[str, Any]) -> PackedShutrackDataset:
    return PackedShutrackDataset(
        config["dataset_dir"],
        max_samples=int(config.get("max_samples", 0)),
        margin_mode=str(config.get("margin_mode", "auto")),
        margin_stats=_load_margin_stats(str(config.get("margin_stats_json", ""))),
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


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(int(line.split()[1]) / 1024.0)
    except OSError:
        pass
    return 0.0


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
    margin = checkpoint.get("margin", {})
    if bool(margin.get("enabled")) != bool(dataset.margin_enabled):
        mismatches["margin.enabled"] = (
            margin.get("enabled"),
            dataset.margin_enabled,
        )
    if str(margin.get("source")) != str(dataset.margin_source):
        mismatches["margin.source"] = (
            margin.get("source"),
            dataset.margin_source,
        )
    if json.dumps(margin.get("stats", {}), sort_keys=True) != json.dumps(
        dataset.margin_stats, sort_keys=True
    ):
        mismatches["margin.stats"] = "different"
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
        "feature_schema_version",
        "label_source",
        "margin_enabled",
    ):
        if checkpoint_dataset.get(key) != current.get(key):
            mismatches[f"dataset.{key}"] = (
                checkpoint_dataset.get(key),
                current.get(key),
            )
    if mismatches:
        raise ValueError(
            "SHUTrack resume contract mismatch; refusing to continue with a "
            "different model/data/margin configuration: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


def _margin_block(dataset: PackedShutrackDataset) -> dict[str, Any]:
    return {
        "enabled": bool(dataset.margin_enabled),
        "source": dataset.margin_source,
        "valid_rate": dataset.margin_valid_rate,
        "stats": dataset.margin_stats,
    }


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
        margin=_margin_block(dataset),
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
    if last_path.exists() and not resume_path:
        raise FileExistsError(
            "SHUTrack formal training requires a new checkpoint directory; "
            f"{last_path} already exists. Choose a new experiment name or pass "
            "--resume with the previous last.pt."
        )

    _seed_everything(int(resolved["seed"]))
    device = torch.device(str(resolved["device"]))
    dataset = build_dataset(resolved)
    try:
        model = build_model(dataset, resolved).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
        )
        sampler = SequenceBatchSampler(
            dataset.sample_boundaries,
            int(resolved["batch_size"]),
            total_samples=len(dataset),
            seed=int(resolved["seed"]),
        )
        steps_per_epoch = max(1, len(sampler))
        max_steps = int(resolved.get("max_steps", 0))
        total_steps = (
            max_steps if max_steps > 0 else int(resolved["epochs"]) * steps_per_epoch
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
            model.load_state_dict(resumed_model.state_dict(), strict=True)
            validate_checkpoint(checkpoint)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            best_metric = float(checkpoint.get("best_metric", best_metric))

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(resolved["num_workers"]),
            collate_fn=PackedShutrackDataset.collate_fn,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = checkpoint_dir / "metrics.jsonl"
        training_log_path = checkpoint_dir / "training.log"
        summary_path = checkpoint_dir / "training_summary.json"
        started = time.time()
        epochs = int(resolved["epochs"])
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
            write_log(f"architecture={ARCHITECTURE_NAME} version={ARCHITECTURE_VERSION}")
            write_log(f"dataset={dataset.dataset_dir}")
            write_log(f"device={device}")
            write_log(f"parameters={json.dumps(model.parameter_report(), sort_keys=True)}")
            write_log(f"samples={len(dataset)} steps_per_epoch={steps_per_epoch}")
            write_log(f"total_steps={total_steps} warmup_steps={warmup_steps}")
            write_log(
                "margin_enabled="
                f"{str(dataset.margin_enabled).lower()} "
                f"margin_source={dataset.margin_source} "
                f"margin_valid_rate={dataset.margin_valid_rate}"
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
                accumulator = ScalarAccumulator()
                epoch_started = time.time()
                epoch_start_loads = dataset.sequence_load_count
                epoch_samples = 0
                for batch_index, batch in enumerate(loader):
                    epoch_samples += int(batch["safe_gate_target"].shape[0])
                    batch = _move_batch(batch, device)
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

                    weight = float(components["weight_sum"].detach().cpu())
                    accumulator.add("loss", loss, weight if weight > 0 else 1.0)
                    accumulator.add(
                        "bce", components["bce"], weight if weight > 0 else 1.0
                    )
                    accumulator.add(
                        "mse", components["mse"], weight if weight > 0 else 1.0
                    )
                    accumulator.add(
                        "motion_gate_mean", components["motion_gate_mean"], 1.0
                    )
                    accumulator.add(
                        "appearance_gate_mean",
                        components["appearance_gate_mean"],
                        1.0,
                    )
                    accumulator.add("grad_norm", grad_norm, 1.0)

                    if global_step % log_interval == 0 or global_step == 1:
                        write_log(
                            f"epoch={epoch} step={global_step} "
                            f"loss={float(loss.detach().cpu()):.6f} "
                            f"bce={float(components['bce'].detach().cpu()):.6f} "
                            f"mse={float(components['mse'].detach().cpu()):.6f} "
                            f"grad_norm={float(grad_norm):.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e}"
                        )
                    if max_steps > 0 and global_step >= max_steps:
                        stop_requested = True
                        break

                metrics = accumulator.summary()
                last_epoch = epoch
                epoch_seconds = time.time() - epoch_started
                metrics.update(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "epoch_seconds": float(epoch_seconds),
                        "samples": int(epoch_samples),
                        "samples_per_second": float(
                            epoch_samples / max(epoch_seconds, 1e-9)
                        ),
                        "sequence_load_count": int(
                            dataset.sequence_load_count - epoch_start_loads
                        ),
                        "rss_mb": _rss_mb(),
                        "margin_enabled": bool(dataset.margin_enabled),
                        "margin_source": dataset.margin_source,
                        "margin_valid_rate": dataset.margin_valid_rate,
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
            "sequence_load_count": int(dataset.sequence_load_count),
            "parameter_counts": model.parameter_report(),
            "margin": _margin_block(dataset),
            "training_config": dict(resolved),
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
    margin_mode: str = "auto",
    margin_stats_json: str = "",
) -> dict[str, Any]:
    """Small deterministic evaluation used by the CLI contract check."""

    model, checkpoint = load_shutrack_checkpoint(checkpoint_path, map_location="cpu")
    model.to(torch.device(device))
    model.eval()
    dataset = PackedShutrackDataset(
        dataset_dir,
        margin_mode=margin_mode,
        margin_stats=_load_margin_stats(margin_stats_json),
    )
    try:
        if dataset.reid_dim != int(checkpoint["reid_dim"]):
            raise ValueError(
                f"checkpoint/dataset reid_dim mismatch: "
                f"{checkpoint['reid_dim']} != {dataset.reid_dim}"
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
            "margin_source": dataset.margin_source,
            "margin_valid_rate": dataset.margin_valid_rate,
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
