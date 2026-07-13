from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import ROLLOUT_LABEL_SCHEMA_SHA256, ROLLOUT_LABEL_SCHEMA_VERSION
from agentguard.datasets.joint_window_dataset import (
    JOINT_DATASET_SCHEMA_SHA256,
    CompactIWGTSRMWindowDataset,
)
from agentguard.models.iwg import IWG
from agentguard.models.iwg_tsrm import IWGTSRM, forward_iwg_with_warmup
from agentguard.training.loss_iwg_tsrm import compute_iwg_tsrm_loss
from agentguard.training.scheduler import CosineWarmupScheduler


JOINT_MODEL_SCHEMA = "agentguard_iwg_tsrm_v3"
BASE_MODEL_SCHEMA = "agentguard_iwg_base_v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def sequence_balanced_sample_weights(
    windows: list[dict[str, Any]],
) -> torch.DoubleTensor:
    """Give every sequence equal expected probability within an epoch."""
    if not windows:
        raise ValueError("sequence-balanced sampling requires non-empty windows")
    counts = Counter(str(window["sequence"]) for window in windows)
    return torch.as_tensor(
        [1.0 / counts[str(window["sequence"])] for window in windows],
        dtype=torch.double,
    )


def _model_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    training_mode: str,
) -> dict[str, torch.Tensor]:
    if training_mode == "joint":
        return model(batch)
    outputs = forward_iwg_with_warmup(model, batch)
    return IWGTSRM.apply_unmatched_sentinel(outputs, batch["has_detection_mask"])


class _MetricAccumulator:
    def __init__(self, training_mode: str) -> None:
        self.training_mode = training_mode
        self.component_total: dict[str, float] = {}
        self.num_batches = 0
        self.base_errors: list[torch.Tensor] = []
        self.final_errors: list[torch.Tensor] = []
        self.base_gates: list[torch.Tensor] = []
        self.final_gates: list[torch.Tensor] = []
        self.corrections: list[torch.Tensor] = []
        self.strengths: list[torch.Tensor] = []
        self.cue_errors: list[torch.Tensor] = []
        self.risk_errors: list[torch.Tensor] = []
        self.improvements: list[torch.Tensor] = []

    def update(
        self,
        components: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        self.num_batches += 1
        for key, value in components.items():
            self.component_total[key] = self.component_total.get(key, 0.0) + float(
                value.detach().float().cpu()
            )
        gate_mask = batch["label_mask"].unsqueeze(-1) & torch.stack(
            [batch["valid_motion"], batch["valid_appearance"]], dim=-1
        )
        detection_mask = batch["has_detection_mask"] & ~batch["padding_mask"]
        if gate_mask.any():
            self.base_errors.append(
                (outputs["base_gate"] - batch["base_gate_target"]).abs()[gate_mask].detach().cpu()
            )
        self.base_gates.append(outputs["base_gate"][detection_mask].detach().cpu())

        cue_mask = batch["label_mask"].unsqueeze(-1) & torch.stack(
            [
                batch["valid_motion"],
                batch["valid_appearance"],
                batch["valid_motion"] | batch["valid_appearance"],
            ],
            dim=-1,
        )
        risk_mask = batch["label_mask"].unsqueeze(-1) & torch.stack(
            [
                batch["valid_motion"],
                batch["valid_appearance"],
                batch["valid_motion"] & batch["valid_appearance"],
                batch["valid_motion"] & batch["valid_appearance"],
            ],
            dim=-1,
        )
        if cue_mask.any():
            self.cue_errors.append(
                (outputs["cue"] - batch["cue_target"]).abs()[cue_mask].detach().cpu()
            )
        if risk_mask.any():
            self.risk_errors.append(
                (outputs["risk"] - batch["risk_target"]).abs()[risk_mask].detach().cpu()
            )
        if self.training_mode == "joint":
            endpoint = batch["temporal_endpoint_mask"].unsqueeze(-1)
            temporal_gate_mask = gate_mask & endpoint
            endpoint_detection_mask = detection_mask & batch["temporal_endpoint_mask"]
            if temporal_gate_mask.any():
                final_error = (outputs["final_gate"] - batch["final_gate_target"]).abs()
                base_safe_error = (outputs["base_gate"] - batch["final_gate_target"]).abs()
                self.final_errors.append(
                    final_error[temporal_gate_mask].detach().cpu()
                )
                self.improvements.append(
                    (base_safe_error - final_error)[temporal_gate_mask].detach().cpu()
                )
            self.final_gates.append(
                outputs["final_gate"][endpoint_detection_mask].detach().cpu()
            )
            self.corrections.append(
                outputs["temporal_gate_correction"][endpoint_detection_mask]
                .detach()
                .cpu()
            )
            self.strengths.append(
                outputs["revision_strength"][endpoint_detection_mask].detach().cpu()
            )

    @staticmethod
    def _distribution(tensors: list[torch.Tensor]) -> dict[str, float]:
        values = torch.cat([tensor.reshape(-1) for tensor in tensors]) if tensors else torch.empty(0)
        if values.numel() == 0:
            return {key: 0.0 for key in ("mean", "std", "p05", "p50", "p95")}
        values = values.float()
        return {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "p05": float(torch.quantile(values, 0.05)),
            "p50": float(torch.quantile(values, 0.50)),
            "p95": float(torch.quantile(values, 0.95)),
        }

    def compute(self) -> dict[str, Any]:
        result = {
            key: value / max(self.num_batches, 1)
            for key, value in self.component_total.items()
        }
        result["base_gate_distribution"] = self._distribution(self.base_gates)
        result["base_gate_mae"] = self._distribution(self.base_errors)["mean"]
        result["cue_calibration_mae"] = self._distribution(self.cue_errors)["mean"]
        result["risk_calibration_mae"] = self._distribution(self.risk_errors)["mean"]
        if self.training_mode == "joint":
            result["final_gate_distribution"] = self._distribution(self.final_gates)
            result["final_gate_mae"] = self._distribution(self.final_errors)["mean"]
            result["correction_distribution"] = self._distribution(self.corrections)
            correction = torch.cat([value.reshape(-1) for value in self.corrections]) if self.corrections else torch.empty(0)
            result["correction_mean_abs"] = float(correction.abs().mean()) if correction.numel() else 0.0
            result["correction_saturation_ratio"] = (
                float((correction.abs() >= 0.1999).float().mean()) if correction.numel() else 0.0
            )
            result["revision_strength_distribution"] = self._distribution(self.strengths)
            result["base_to_final_error_improvement"] = self._distribution(self.improvements)["mean"]
        return result


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    training_mode: str,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    global_step: int = 0,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    model.train(training)
    metrics = _MetricAccumulator(training_mode)
    grad_accum = max(1, int(config.get("grad_accum_steps", 1)))
    consecutive_amp_overflows = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, cpu_batch in enumerate(loader):
        batch = _to_device(cpu_batch, device)
        amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            outputs = _model_forward(model, batch, training_mode)
        with torch.set_grad_enabled(training):
            loss, components = compute_iwg_tsrm_loss(
                outputs,
                batch,
                training_mode=training_mode,
                delta_max=config["delta_max"],
                lambda_final=config["lambda_final"],
                lambda_residual=config["lambda_residual"],
                lambda_dynamics=config["lambda_dynamics"],
                lambda_revision=config["lambda_revision"],
            )
            scaled_loss = loss / grad_accum
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {training_mode} loss at batch {batch_index}")
        if training:
            assert scaler is not None
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                gradients_finite = all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                if not gradients_finite:
                    if not scaler.is_enabled():
                        raise FloatingPointError(
                            f"non-finite unscaled gradients at batch {batch_index}"
                        )
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_amp_overflows += 1
                    print(
                        json.dumps(
                            {
                                "amp_overflow": True,
                                "batch": batch_index,
                                "old_scale": old_scale,
                                "new_scale": scaler.get_scale(),
                                "consecutive": consecutive_amp_overflows,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if consecutive_amp_overflows >= 8:
                        raise FloatingPointError(
                            "eight consecutive AMP gradient overflows; refusing to continue"
                        )
                    metrics.update(components, outputs, batch)
                    continue
                consecutive_amp_overflows = 0
                gradient_norm = clip_grad_norm_(
                    model.parameters(), float(config["grad_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
        metrics.update(components, outputs, batch)
    return metrics.compute(), global_step


def _checkpoint_contract(
    *,
    model: torch.nn.Module,
    training_mode: str,
    dataset_metadata: dict[str, Any],
    dataset_metadata_path: Path,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    norm = np.load(dataset_metadata_path.parent / dataset_metadata["normalization_file"])
    try:
        training_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True
        ).strip()
    except Exception:
        training_commit = ""
    return {
        "model_schema_version": JOINT_MODEL_SCHEMA if training_mode == "joint" else BASE_MODEL_SCHEMA,
        "training_mode": training_mode,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "joint_dataset_schema_sha256": JOINT_DATASET_SCHEMA_SHA256,
        "reid_dim": int(dataset_metadata["reid_dim"]),
        "scalar_dim": int(dataset_metadata["scalar_dim"]),
        "event_dim": int(dataset_metadata["event_dim"]),
        "window_size": int(dataset_metadata["window_size"]),
        "iwg_warmup_events": int(dataset_metadata["iwg_warmup_events"]),
        "iwg_input_size": int(dataset_metadata["iwg_input_size"]),
        "delta_max": float(config["delta_max"]),
        "normalization_mean": np.asarray(norm["mean"], dtype=np.float64),
        "normalization_std": np.asarray(norm["std"], dtype=np.float64),
        "policy_prototypes": np.asarray(POLICY_PROTOTYPE_MATRIX, dtype=np.float64),
        "training_commit": training_commit,
        "dataset_metadata_sha256": _sha256_file(dataset_metadata_path),
        "dataset_metadata": dataset_metadata,
        "config": config,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }


def validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    expected_training_mode: str | None = None,
    dataset_metadata_path: Path | None = None,
) -> None:
    training_mode = str(checkpoint.get("training_mode", ""))
    expected_schema = JOINT_MODEL_SCHEMA if training_mode == "joint" else BASE_MODEL_SCHEMA
    expected = {
        "model_schema_version": expected_schema,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "joint_dataset_schema_sha256": JOINT_DATASET_SCHEMA_SHA256,
    }
    if expected_training_mode is not None and training_mode != expected_training_mode:
        raise ValueError(
            f"training_mode mismatch: {training_mode!r} != {expected_training_mode!r}"
        )
    missing = [
        key
        for key in (
            "normalization_mean",
            "normalization_std",
            "policy_prototypes",
            "training_commit",
            "dataset_metadata_sha256",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "window_size",
            "iwg_warmup_events",
            "iwg_input_size",
            "delta_max",
            "reid_dim",
            "scalar_dim",
            "event_dim",
        )
        if key not in checkpoint
    ]
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint schema mismatch: {mismatches}")
    if np.asarray(checkpoint["normalization_mean"]).shape != (63,):
        raise ValueError("normalization_mean must have shape (63,)")
    if np.asarray(checkpoint["normalization_std"]).shape != (63,):
        raise ValueError("normalization_std must have shape (63,)")
    if np.asarray(checkpoint["policy_prototypes"]).shape != (5, 2):
        raise ValueError("policy_prototypes must have shape (5, 2)")
    if int(checkpoint["window_size"]) < 1:
        raise ValueError("window_size must be positive")
    if int(checkpoint["iwg_warmup_events"]) != 5:
        raise ValueError("iwg_warmup_events must equal 5")
    if int(checkpoint["iwg_input_size"]) != int(checkpoint["window_size"]) + 5:
        raise ValueError("iwg_input_size must equal window_size + 5")
    if dataset_metadata_path is not None:
        actual_hash = _sha256_file(dataset_metadata_path)
        if checkpoint["dataset_metadata_sha256"] != actual_hash:
            raise ValueError(
                "dataset_metadata_sha256 mismatch: "
                f"{checkpoint['dataset_metadata_sha256']} != {actual_hash}"
            )


def _load_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    training_mode: str,
    dataset_metadata_path: Path,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_contract(
        checkpoint,
        expected_training_mode=training_mode,
        dataset_metadata_path=dataset_metadata_path,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint["best_metric"]),
    )


def train_iwg_tsrm(config: dict[str, Any]) -> dict[str, Any]:
    training_mode = str(config["training_mode"])
    if training_mode not in {"base", "joint"}:
        raise ValueError("training_mode must be base or joint")
    _seed_everything(int(config["seed"]))
    dataset_dir = Path(config["dataset_dir"]).resolve()
    checkpoint_dir = Path(config["checkpoint_dir"]).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    train_dataset = CompactIWGTSRMWindowDataset(
        dataset_dir, "train", max_windows=int(config.get("max_train_windows", 0))
    )
    val_dataset = CompactIWGTSRMWindowDataset(
        dataset_dir, "val", max_windows=int(config.get("max_val_windows", 0))
    )
    device = torch.device(config["device"])
    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    sampling_policy = str(config.get("sampling_policy", "sequence_balanced"))
    if sampling_policy == "sequence_balanced":
        sampler = WeightedRandomSampler(
            sequence_balanced_sample_weights(train_dataset.windows),
            num_samples=len(train_dataset),
            replacement=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    elif sampling_policy == "shuffle":
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=loader_generator, **loader_kwargs
        )
    else:
        raise ValueError(
            "sampling_policy must be 'sequence_balanced' or 'shuffle'"
        )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    if training_mode == "joint":
        model: torch.nn.Module = IWGTSRM(
            reid_dim=int(metadata["reid_dim"]), delta_max=float(config["delta_max"])
        )
    else:
        model = IWG(reid_dim=int(metadata["reid_dim"]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(1, int(config["grad_accum_steps"])))
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=int(config["warmup_epochs"]) * updates_per_epoch,
        total_steps=max(1, int(config["epochs"]) * updates_per_epoch),
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if config.get("resume_from"):
        start_epoch, global_step, best_metric = _load_resume(
            Path(config["resume_from"]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            training_mode=training_mode,
            dataset_metadata_path=metadata_path,
            device=device,
        )

    prefix = "iwg_tsrm" if training_mode == "joint" else "iwg_base"
    last_path = checkpoint_dir / f"{prefix}_last.pt"
    best_path = checkpoint_dir / f"{prefix}_best.pt"
    history: list[dict[str, Any]] = []
    bad_epochs = 0
    metrics_path = checkpoint_dir / "metrics.jsonl"
    try:
        for epoch in range(start_epoch, int(config["epochs"])):
            started = time.time()
            train_metrics, global_step = _run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                training_mode=training_mode,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
            )
            with torch.no_grad():
                val_metrics, _ = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    training_mode=training_mode,
                    config=config,
                )
            selection_key = "final_gate_loss" if training_mode == "joint" else "base_gate_loss"
            selection = float(val_metrics[selection_key])
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - started,
                "train_sequences": metadata["train_sequences"],
                "val_sequences": metadata["val_sequences"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            improved = selection < best_metric
            if improved:
                best_metric = selection
                bad_epochs = 0
            else:
                bad_epochs += 1
            checkpoint = _checkpoint_contract(
                model=model,
                training_mode=training_mode,
                dataset_metadata=metadata,
                dataset_metadata_path=metadata_path,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                metrics=record,
            )
            torch.save(checkpoint, last_path)
            if improved:
                torch.save(checkpoint, best_path)
            patience = int(config.get("early_stop_patience", 0))
            if patience > 0 and bad_epochs >= patience:
                break
    finally:
        train_dataset.close()
        val_dataset.close()
    return {
        "training_mode": training_mode,
        "best_metric": best_metric,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epochs_completed": len(history),
        "history": history,
    }


def validate_iwg_tsrm_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    split: str,
    device: str,
    max_batches: int = 0,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    dataset_dir = Path(dataset_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_contract(
        checkpoint, dataset_metadata_path=dataset_dir / "metadata.json"
    )
    training_mode = checkpoint["training_mode"]
    if training_mode == "joint":
        model: torch.nn.Module = IWGTSRM(
            reid_dim=int(checkpoint["reid_dim"]), delta_max=float(checkpoint["delta_max"])
        )
    else:
        model = IWG(reid_dim=int(checkpoint["reid_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    dataset = CompactIWGTSRMWindowDataset(dataset_dir, split)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    if max_batches > 0:
        batches = []
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batches.append(batch)
        loader = batches  # type: ignore[assignment]
    config = dict(checkpoint["config"])
    with torch.no_grad():
        metrics, _ = _run_epoch(
            model=model,
            loader=loader,  # type: ignore[arg-type]
            device=torch.device(device),
            training_mode=training_mode,
            config=config,
        )
    dataset.close()
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_schema_version": checkpoint["model_schema_version"],
        "training_mode": training_mode,
        "split": split,
        "validation_sequences": checkpoint["dataset_metadata"][f"{split}_sequences"],
        "schema_valid": True,
        "finite": all(
            math.isfinite(value) for value in metrics.values() if isinstance(value, float)
        ),
        "metrics": metrics,
    }
