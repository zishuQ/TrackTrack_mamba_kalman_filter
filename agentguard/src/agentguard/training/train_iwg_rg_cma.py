from __future__ import annotations

import gc
import json
import hashlib
import math
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import ROLLOUT_LABEL_SCHEMA_SHA256
from agentguard.datasets.iwg_attn_dataset import (
    SUPPORTED_IWG_ATTN_DATASET_SCHEMA_SHA256,
    StreamingIWGAttnDataset,
)
from agentguard.models.iwg_rg_cma import (
    IWG_RG_CMA_MODEL_SCHEMA,
    IWG_RG_CMA_MODEL_SCHEMA_SHA256,
    IWGRGCMA,
    RG_CMA_CORRECTION_BOUND,
    model_contract,
)
from agentguard.training.loss_iwg_rg_cma import compute_iwg_rg_cma_loss


FORMAL_CHECKPOINT_EPOCHS = {
    100: {25, 50, 75, 100},
    200: {50, 100, 150, 200},
}
FORMAL_TRAIN_EPOCHS = frozenset(FORMAL_CHECKPOINT_EPOCHS)
REQUIRED_CONFIG = {
    "seed": 42,
    "num_workers": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "amp": False,
}
FINETUNE_CHECKPOINT_EPOCHS = {5, 10, 25, 50}
FINETUNE_REQUIRED_CONFIG = {
    "seed": 42,
    "num_workers": 4,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "amp": False,
}
FINETUNE_ALLOWED_EPOCHS = frozenset({25, 50})
FORMAL_BATCH_SIZES = {1024, 2048}
LOW_MEMORY_SAMPLING_POLICY = "sequential_sequence_fractions"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_seed(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed)
    random.seed(seed)


def _git_value(args: list[str], default: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def _training_source_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "data/cache_reader.py",
        "data/compact_iwg_labels.py",
        "models/event_encoder.py",
        "models/iwg.py",
        "models/iwg_rg_cma.py",
        "datasets/iwg_attn_dataset.py",
        "rollout_labels.py",
        "training/loss_iwg_rg_cma.py",
        "training/train_iwg_rg_cma.py",
        "v0_pipeline.py",
    ]
    return {
        relative: hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    low_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if low_memory:
            loader_kwargs["prefetch_factor"] = 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=StreamingIWGAttnDataset.collate_fn,
        worker_init_fn=_worker_seed,
        generator=generator,
        **loader_kwargs,
    )


def _shutdown_loader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def _resolved_schedule_settings(config: dict[str, Any]) -> tuple[int, int, int]:
    shards = int(config.get("memory_shards", 1))
    cycles = int(config.get("shard_cycles", 1))
    epochs_per_shard = int(config.get("epochs_per_shard", 0))
    epochs = int(config["epochs"])
    if shards < 1 or cycles < 1 or epochs_per_shard < 0:
        raise ValueError("memory shard schedule values must be positive")
    phase_count = shards * cycles
    if epochs_per_shard == 0:
        if epochs % phase_count:
            raise ValueError(
                "epochs must be divisible by memory_shards * shard_cycles"
            )
        epochs_per_shard = epochs // phase_count
    if phase_count * epochs_per_shard != epochs:
        raise ValueError(
            "memory_shards * shard_cycles * epochs_per_shard must equal epochs"
        )
    return shards, cycles, epochs_per_shard


def build_memory_shard_schedule(
    dataset: StreamingIWGAttnDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    shards, cycles, epochs_per_shard = _resolved_schedule_settings(config)
    sequences = list(dataset.metadata["train_sequences"])
    source_counts = {
        sequence: int(
            dataset.metadata["timeline_counts"][sequence]["labeled_endpoints"]
        )
        for sequence in sequences
    }
    counts = dict(source_counts)
    if shards == 1 and sum(counts.values()) != len(dataset):
        remaining = len(dataset)
        for sequence in sequences:
            counts[sequence] = min(counts[sequence], remaining)
            remaining -= counts[sequence]
        if remaining:
            raise ValueError("truncated dataset exceeds metadata sample counts")
    if sum(counts.values()) != len(dataset):
        raise ValueError("memory shard counts do not cover the training dataset")
    if shards > 1 and str(dataset.metadata.get("index_format")) != "compact_memmap_v1":
        raise ValueError(
            "low-memory sequence fractions require the compact memmap index"
        )

    sequence_offsets: dict[str, int] = {}
    offset = 0
    for sequence in sequences:
        sequence_offsets[sequence] = offset
        offset += counts[sequence]

    phases: list[dict[str, Any]] = []
    for cycle in range(cycles):
        for shard in range(shards):
            ranges: list[dict[str, Any]] = []
            for sequence in sequences:
                count = counts[sequence]
                local_start = count * shard // shards
                local_end = count * (shard + 1) // shards
                global_start = sequence_offsets[sequence] + local_start
                global_end = sequence_offsets[sequence] + local_end
                ranges.append(
                    {
                        "sequence": sequence,
                        "local_start": local_start,
                        "local_end": local_end,
                        "global_start": global_start,
                        "global_end": global_end,
                        "samples": global_end - global_start,
                    }
                )
            phases.append(
                {
                    "phase": len(phases) + 1,
                    "cycle": cycle + 1,
                    "shard": shard + 1,
                    "fraction_start": shard / shards,
                    "fraction_end": (shard + 1) / shards,
                    "epochs": epochs_per_shard,
                    "samples": sum(item["samples"] for item in ranges),
                    "ranges": ranges,
                }
            )
    return {
        "sampling_policy": (
            "full_shuffle" if shards == 1 else LOW_MEMORY_SAMPLING_POLICY
        ),
        "memory_shards": shards,
        "shard_cycles": cycles,
        "epochs_per_shard": epochs_per_shard,
        "nominal_epochs": int(config["epochs"]),
        "effective_full_epochs": epochs_per_shard * cycles,
        "phases": phases,
    }


def _phase_dataset(
    dataset: StreamingIWGAttnDataset,
    phase: dict[str, Any],
) -> Dataset:
    subsets = [
        Subset(dataset, range(item["global_start"], item["global_end"]))
        for item in phase["ranges"]
        if item["global_end"] > item["global_start"]
    ]
    if not subsets:
        raise ValueError("memory shard phase contains no samples")
    return subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _write_training_log(handle, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"{timestamp} | INFO     | {message}\n")
    handle.flush()


def _forward(model: IWGRGCMA, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        batch["track_feats"],
        batch["det_feats"],
        batch["scalar_feats"],
        batch["padding_mask"],
        batch["has_detection_mask"],
        batch["reset_mask"],
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _checkpoint_payload(
    *,
    model: IWGRGCMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    resolved_batch_size: int,
    training_schedule: dict[str, Any] | None = None,
    training_progress: dict[str, Any] | None = None,
    initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    motion_target_mode = str(metadata.get("motion_target_mode", "nsa"))
    payload = {
        **model_contract(),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "reid_dim": int(metadata["reid_dim"]),
        "scalar_dim": int(metadata["scalar_dim"]),
        "event_dim": int(metadata["event_dim"]),
        "context_size": int(metadata["context_size"]),
        "max_frame_gap": int(metadata["max_frame_gap"]),
        "dataset": str(metadata["dataset"]),
        "split": str(metadata["split"]),
        "train_sequences": list(metadata["train_sequences"]),
        "index_format": str(metadata.get("index_format", "jsonl_v1")),
        "correction_bound": RG_CMA_CORRECTION_BOUND,
        "policy_prototypes": np.asarray(POLICY_PROTOTYPE_MATRIX).tolist(),
        "dataset_schema_sha256": str(metadata["dataset_schema_sha256"]),
        "dataset_sha256": str(metadata["dataset_sha256"]),
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_mean": np.asarray(norm_mean, dtype=np.float64),
        "normalization_std": np.asarray(norm_std, dtype=np.float64),
        "training_commit": _git_value(["git", "rev-parse", "HEAD"], "unknown"),
        "training_git_status": _git_value(
            ["git", "status", "--porcelain"], "unavailable"
        ),
        "training_source_sha256": _training_source_hashes(),
        "training_seed": int(config["seed"]),
        "tracker_seed": 10000,
        "training_config": dict(config),
        "resolved_batch_size": int(resolved_batch_size),
        "validation_policy": (
            f"none_epoch{int(config['epochs'])}_warm_start"
            if initialization and initialization.get("mode") == "warm_start"
            else "none_epoch100"
        ),
        "initialization": initialization or {"mode": "random"},
        "training_schedule": training_schedule or {"sampling_policy": "full_shuffle"},
        "training_progress": training_progress or {},
        "motion_target_mode": motion_target_mode,
    }
    if motion_target_mode == "mamba_hybrid":
        supervision_fields = (
            "mamba_distill_label_root",
            "mamba_checkpoint_path",
            "mamba_checkpoint_sha256",
            "mamba_teacher_config",
            "mamba_teacher_config_sha256",
            "teacher_sidecar_sha256",
            "distill_label_sha256",
            "teacher_weight_cap",
            "advantage_horizon",
            "tau_adv",
            "base_nsa_cache_sha256",
            "base_nsa_label_sha256",
        )
        missing = [key for key in supervision_fields if key not in metadata]
        if missing:
            raise ValueError(
                f"mamba_hybrid dataset metadata lacks provenance: {missing}"
            )
        payload.update({key: metadata[key] for key in supervision_fields})
    elif motion_target_mode == "mamba_native":
        supervision_fields = (
            "event_source",
            "motion_label_mode",
            "mamba_checkpoint_path",
            "mamba_checkpoint_sha256",
            "mamba_teacher_config",
            "mamba_teacher_config_sha256",
            "native_event_cache_sha256",
            "native_label_sha256",
        )
        missing = [key for key in supervision_fields if key not in metadata]
        if missing:
            raise ValueError(
                f"mamba_native dataset metadata lacks provenance: {missing}"
            )
        payload.update({key: metadata[key] for key in supervision_fields})
    return payload


def validate_iwg_rg_cma_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    expected_dataset_sha256: str | None = None,
) -> None:
    required = {
        "model_schema": IWG_RG_CMA_MODEL_SCHEMA,
        "model_schema_sha256": IWG_RG_CMA_MODEL_SCHEMA_SHA256,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "correction_bound": RG_CMA_CORRECTION_BOUND,
        "context_size": 6,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in required.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"IWG RG-CMA checkpoint contract mismatch: {mismatches}")
    if checkpoint.get("dataset_schema_sha256") not in (
        SUPPORTED_IWG_ATTN_DATASET_SCHEMA_SHA256
    ):
        raise ValueError(
            "IWG RG-CMA checkpoint contract mismatch: unsupported "
            f"dataset_schema_sha256={checkpoint.get('dataset_schema_sha256')!r}"
        )
    for key in (
        "model_state_dict",
        "dataset_sha256",
        "normalization_mean",
        "normalization_std",
        "training_commit",
        "training_seed",
        "tracker_seed",
    ):
        if key not in checkpoint:
            raise ValueError(f"IWG RG-CMA checkpoint is missing {key}")
    if expected_dataset_sha256 is not None and (
        checkpoint["dataset_sha256"] != expected_dataset_sha256
    ):
        raise ValueError("checkpoint dataset hash does not match the requested dataset")
    if np.asarray(checkpoint["normalization_mean"]).shape != (63,):
        raise ValueError("checkpoint normalization_mean must have shape (63,)")
    if np.asarray(checkpoint["normalization_std"]).shape != (63,):
        raise ValueError("checkpoint normalization_std must have shape (63,)")
    if np.asarray(checkpoint.get("policy_prototypes")).shape != (5, 2):
        raise ValueError("checkpoint policy prototypes must have shape (5, 2)")
    motion_target_mode = str(checkpoint.get("motion_target_mode", "nsa"))
    if motion_target_mode not in {"nsa", "mamba_hybrid", "mamba_native"}:
        raise ValueError(f"unsupported checkpoint motion_target_mode={motion_target_mode!r}")
    if motion_target_mode == "mamba_hybrid":
        required_distill = {
            "mamba_distill_label_root",
            "mamba_checkpoint_path",
            "mamba_checkpoint_sha256",
            "mamba_teacher_config",
            "mamba_teacher_config_sha256",
            "teacher_sidecar_sha256",
            "distill_label_sha256",
            "tau_adv",
            "base_nsa_cache_sha256",
            "base_nsa_label_sha256",
        }
        missing = sorted(required_distill.difference(checkpoint))
        if missing:
            raise ValueError(f"mamba_hybrid checkpoint lacks provenance: {missing}")
        if float(checkpoint.get("teacher_weight_cap", -1.0)) != 0.5:
            raise ValueError("mamba_hybrid checkpoint teacher_weight_cap must be 0.5")
        if int(checkpoint.get("advantage_horizon", -1)) != 5:
            raise ValueError("mamba_hybrid checkpoint advantage_horizon must be 5")
    elif motion_target_mode == "mamba_native":
        required_native = {
            "event_source",
            "motion_label_mode",
            "mamba_checkpoint_path",
            "mamba_checkpoint_sha256",
            "mamba_teacher_config",
            "mamba_teacher_config_sha256",
            "native_event_cache_sha256",
            "native_label_sha256",
        }
        missing = sorted(required_native.difference(checkpoint))
        if missing:
            raise ValueError(f"mamba_native checkpoint lacks provenance: {missing}")
        if checkpoint.get("event_source") != "mamba_native":
            raise ValueError("mamba_native checkpoint has an invalid event_source")
        if checkpoint.get("motion_label_mode") != "mamba_native_current":
            raise ValueError("mamba_native checkpoint has an invalid motion_label_mode")


def load_iwg_rg_cma_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[IWGRGCMA, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    validate_iwg_rg_cma_checkpoint_contract(checkpoint)
    model = IWGRGCMA(
        reid_dim=int(checkpoint["reid_dim"]),
        scalar_dim=int(checkpoint["scalar_dim"]),
        event_dim=int(checkpoint["event_dim"]),
        correction_bound=float(checkpoint["correction_bound"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def initialize_iwg_rg_cma_model(
    model: IWGRGCMA,
    checkpoint_path: str | Path,
    *,
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_iwg_rg_cma_checkpoint_contract(checkpoint)
    expected_dims = {
        "reid_dim": int(dataset_metadata["reid_dim"]),
        "scalar_dim": int(dataset_metadata["scalar_dim"]),
        "event_dim": int(dataset_metadata["event_dim"]),
    }
    mismatches = {
        key: (int(checkpoint.get(key, -1)), expected)
        for key, expected in expected_dims.items()
        if int(checkpoint.get(key, -1)) != expected
    }
    if mismatches:
        raise ValueError(f"initial checkpoint dimension mismatch: {mismatches}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "mode": "warm_start",
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "source_dataset": str(checkpoint.get("dataset", "unknown")),
        "source_dataset_sha256": str(checkpoint["dataset_sha256"]),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_restored": False,
        "scheduler_restored": False,
    }


def _run_training_attempt(
    config: dict[str, Any],
    *,
    dataset: StreamingIWGAttnDataset,
    batch_size: int,
) -> dict[str, Any]:
    seed = int(config["seed"])
    _seed_everything(seed)
    device = torch.device(config["device"])
    training_schedule = build_memory_shard_schedule(dataset, config)
    epoch_plan = [
        (phase, phase_epoch)
        for phase in training_schedule["phases"]
        for phase_epoch in range(1, int(phase["epochs"]) + 1)
    ]
    if len(epoch_plan) != int(config["epochs"]):
        raise AssertionError("memory shard schedule does not match configured epochs")
    for phase in training_schedule["phases"]:
        phase["steps_per_epoch"] = math.ceil(int(phase["samples"]) / batch_size)
    total_steps = sum(
        int(phase["steps_per_epoch"]) * int(phase["epochs"])
        for phase in training_schedule["phases"]
    )
    warmup_steps = (
        int(training_schedule["phases"][0]["steps_per_epoch"])
        * int(config["warmup_epochs"])
    )
    model = IWGRGCMA(
        reid_dim=int(dataset.metadata["reid_dim"]),
        scalar_dim=int(dataset.metadata["scalar_dim"]),
        event_dim=int(dataset.metadata["event_dim"]),
    ).to(device)
    init_checkpoint = str(config.get("init_checkpoint", "")).strip()
    initialization = (
        initialize_iwg_rg_cma_model(
            model,
            init_checkpoint,
            dataset_metadata=dataset.metadata,
        )
        if init_checkpoint
        else {"mode": "random"}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "metrics.jsonl"
    training_log_path = checkpoint_dir / "training.log"
    start_time = time.time()
    total_samples_seen = 0
    loader: DataLoader | None = None
    phase_dataset: Dataset | None = None
    active_phase = 0
    with metrics_path.open("w") as metrics_file, training_log_path.open("w") as training_log:
        _write_training_log(training_log, "=" * 60)
        _write_training_log(training_log, "Safe-Direct IWG + RG-CMA Training - start")
        _write_training_log(training_log, f"Output directory: {checkpoint_dir}")
        _write_training_log(
            training_log,
            "Config: " + json.dumps(config, indent=2, sort_keys=True),
        )
        _write_training_log(training_log, f"Device: {device}")
        _write_training_log(
            training_log,
            "Initialization: " + json.dumps(initialization, indent=2, sort_keys=True),
        )
        _write_training_log(
            training_log,
            "Training schedule: "
            + json.dumps(training_schedule, indent=2, sort_keys=True),
        )
        global_step = 0
        try:
            for epoch, (phase, phase_epoch) in enumerate(epoch_plan, start=1):
                if int(phase["phase"]) != active_phase:
                    _shutdown_loader(loader)
                    loader = None
                    phase_dataset = None
                    gc.collect()
                    released = dataset.release_cached_pages()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    phase_dataset = (
                        dataset
                        if int(training_schedule["memory_shards"]) == 1
                        else _phase_dataset(dataset, phase)
                    )
                    loader = _loader(
                        phase_dataset,
                        batch_size=batch_size,
                        num_workers=int(config["num_workers"]),
                        seed=seed + int(phase["phase"]) - 1,
                        low_memory=int(training_schedule["memory_shards"]) > 1,
                    )
                    if len(loader) != int(phase["steps_per_epoch"]):
                        raise AssertionError("phase loader length changed unexpectedly")
                    active_phase = int(phase["phase"])
                    _write_training_log(
                        training_log,
                        (
                            f"Phase {phase['phase']}/{len(training_schedule['phases'])} "
                            f"cycle={phase['cycle']}/{training_schedule['shard_cycles']} "
                            f"shard={phase['shard']}/{training_schedule['memory_shards']} "
                            f"fraction=[{phase['fraction_start']:.1%},"
                            f"{phase['fraction_end']:.1%}) samples={phase['samples']} "
                            f"batches={len(loader)} released={released}"
                        ),
                    )
                if loader is None:
                    raise AssertionError("phase loader was not created")

                model.train()
                totals: dict[str, float] = {}
                samples = 0
                channel_abs_error = [0.0, 0.0]
                channel_correct = [0.0, 0.0]
                channel_weight = [0.0, 0.0]
                motion_positions = torch.zeros(6, dtype=torch.float64)
                appearance_positions = torch.zeros(6, dtype=torch.float64)
                cross_matrix = torch.zeros((3, 3), dtype=torch.float64)
                motion_entropy = 0.0
                appearance_entropy = 0.0
                teacher_weight_sum = 0.0
                teacher_active = 0.0
                motion_abs_error_to_nsa = 0.0
                motion_abs_error_to_hybrid = 0.0
                motion_metric_weight = 0.0
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_start = time.time()
                _write_training_log(
                    training_log,
                    (
                        f"--- Epoch {epoch}/{config['epochs']} "
                        f"phase_epoch={phase_epoch}/{phase['epochs']} "
                        f"cycle={phase['cycle']} shard={phase['shard']} ---"
                    ),
                )
                for batch_index, raw_batch in enumerate(loader):
                    batch = _move_batch(raw_batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = _forward(model, batch)
                    loss, components = compute_iwg_rg_cma_loss(
                        outputs, batch, correction_bound=RG_CMA_CORRECTION_BOUND
                    )
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["grad_clip"])
                    )
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    count = int(batch["track_feats"].shape[0])
                    samples += count
                    for key, value in components.items():
                        totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
                    totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(
                        grad_norm
                    ) * count
                    motion_positions += (
                        outputs["motion_attention_weights"]
                        .detach()
                        .double()
                        .mean(dim=1)
                        .sum(dim=0)
                        .cpu()
                    )
                    appearance_positions += (
                        outputs["appearance_attention_weights"]
                        .detach()
                        .double()
                        .mean(dim=1)
                        .sum(dim=0)
                        .cpu()
                    )
                    cross_matrix += (
                        outputs["cross_modal_attention_weights"]
                        .detach()
                        .double()
                        .mean(dim=1)
                        .sum(dim=0)
                        .cpu()
                    )
                    motion_entropy += float(
                        outputs["motion_attention_entropy"].detach().mean()
                    ) * count
                    appearance_entropy += float(
                        outputs["appearance_attention_entropy"].detach().mean()
                    ) * count
                    valid_channels = torch.stack(
                        [batch["valid_motion"], batch["valid_appearance"]], dim=-1
                    )
                    gate_weights = (
                        valid_channels.float()
                        * batch["sample_weight"].float().unsqueeze(-1)
                    )
                    gate_error = (
                        outputs["refined_gate"].detach()
                        - batch["safe_gate_target"]
                    ).abs()
                    gate_correct = (
                        (outputs["refined_gate"].detach() >= 0.5)
                        == (batch["safe_gate_target"] >= 0.5)
                    ).float()
                    for channel in range(2):
                        channel_abs_error[channel] += float(
                            (gate_error[:, channel] * gate_weights[:, channel]).sum()
                        )
                        channel_correct[channel] += float(
                            (gate_correct[:, channel] * gate_weights[:, channel]).sum()
                        )
                        channel_weight[channel] += float(
                            gate_weights[:, channel].sum()
                        )
                    motion_weights = gate_weights[:, 0]
                    refined_motion = outputs["refined_gate"].detach()[:, 0]
                    teacher_values = batch.get(
                        "teacher_weight", torch.zeros_like(refined_motion)
                    )
                    nsa_values = batch.get(
                        "nsa_motion_target", batch["safe_gate_target"][:, 0]
                    )
                    hybrid_values = batch.get(
                        "hybrid_motion_target", batch["safe_gate_target"][:, 0]
                    )
                    teacher_weight_sum += float(teacher_values.sum())
                    teacher_active += float((teacher_values > 0.0).sum())
                    motion_abs_error_to_nsa += float(
                        (
                            (refined_motion - nsa_values).abs()
                            * motion_weights
                        ).sum()
                    )
                    motion_abs_error_to_hybrid += float(
                        (
                            (refined_motion - hybrid_values).abs()
                            * motion_weights
                        ).sum()
                    )
                    motion_metric_weight += float(motion_weights.sum())
                    if batch_index % max(1, len(loader) // 5) == 0:
                        _write_training_log(
                            training_log,
                            (
                                f"Epoch {epoch:3d}  step {global_step:6d} | "
                                f"loss={float(components['loss'].detach()):.6f} | "
                                f"base_loss={float(components['base_loss'].detach()):.6f} | "
                                f"final_loss={float(components['final_loss'].detach()):.6f} | "
                                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                            ),
                        )

                del raw_batch, batch, outputs, components, loss

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_time = time.time() - epoch_start
                total_samples_seen += samples
                total_channel_weight = sum(channel_weight)
                metrics = {
                    "epoch": epoch,
                    "phase": int(phase["phase"]),
                    "phase_epoch": phase_epoch,
                    "cycle": int(phase["cycle"]),
                    "memory_shard": int(phase["shard"]),
                    "fraction_start": float(phase["fraction_start"]),
                    "fraction_end": float(phase["fraction_end"]),
                    "samples": samples,
                    "n_batches": len(loader),
                    "optimizer_steps": global_step,
                    "effective_full_epochs_completed": total_samples_seen / len(dataset),
                    **{key: value / max(samples, 1) for key, value in totals.items()},
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_time_s": epoch_time,
                    "gate_accuracy": sum(channel_correct)
                    / max(total_channel_weight, 1.0),
                    "motion_mae": channel_abs_error[0]
                    / max(channel_weight[0], 1.0),
                    "appearance_mae": channel_abs_error[1]
                    / max(channel_weight[1], 1.0),
                    "teacher_weight_mean": teacher_weight_sum / max(samples, 1),
                    "teacher_active_rate": teacher_active / max(samples, 1),
                    "motion_mae_to_nsa": motion_abs_error_to_nsa
                    / max(motion_metric_weight, 1.0),
                    "motion_mae_to_hybrid": motion_abs_error_to_hybrid
                    / max(motion_metric_weight, 1.0),
                    "motion_attention_entropy": motion_entropy / max(samples, 1),
                    "appearance_attention_entropy": appearance_entropy
                    / max(samples, 1),
                    "motion_history_attention": (
                        motion_positions / max(samples, 1)
                    ).tolist(),
                    "appearance_history_attention": (
                        appearance_positions / max(samples, 1)
                    ).tolist(),
                    "cross_modal_attention": (
                        cross_matrix / max(samples, 1)
                    ).tolist(),
                }
                metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(metrics, sort_keys=True), flush=True)
                _write_training_log(
                    training_log,
                    (
                        f"Epoch {epoch:3d} - train_loss={metrics['loss']:.4f}  "
                        f"base_loss={metrics['base_loss']:.4f}  "
                        f"final_loss={metrics['final_loss']:.4f}  "
                        f"gate_acc={metrics['gate_accuracy']:.3f}  "
                        f"motion_mae={metrics['motion_mae']:.4f}  "
                        f"app_mae={metrics['appearance_mae']:.4f}  "
                        f"teacher_weight_mean={metrics['teacher_weight_mean']:.4f}  "
                        f"teacher_active_rate={metrics['teacher_active_rate']:.4f}  "
                        f"motion_mae_to_nsa={metrics['motion_mae_to_nsa']:.4f}  "
                        f"motion_mae_to_hybrid={metrics['motion_mae_to_hybrid']:.4f}  "
                        f"lr={metrics['learning_rate']:.2e}  "
                        f"time={epoch_time:.1f}s  train_only=true"
                    ),
                )
                _write_training_log(
                    training_log,
                    (
                        f"Attention - motion_H={metrics['motion_attention_entropy']:.4f}  "
                        f"appearance_H={metrics['appearance_attention_entropy']:.4f}"
                    ),
                )
                progress = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "phase": int(phase["phase"]),
                    "phase_epoch": phase_epoch,
                    "cycle": int(phase["cycle"]),
                    "memory_shard": int(phase["shard"]),
                    "total_samples_seen": total_samples_seen,
                    "effective_full_epochs_completed": total_samples_seen / len(dataset),
                }
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    config=config,
                    metadata=dataset.metadata,
                    norm_mean=dataset.norm_stats.mean,
                    norm_std=dataset.norm_stats.std,
                    resolved_batch_size=batch_size,
                    training_schedule=training_schedule,
                    training_progress=progress,
                    initialization=initialization,
                )
                torch.save(payload, checkpoint_dir / "iwg_rg_cma_last.pt")
                checkpoint_epochs = (
                    FINETUNE_CHECKPOINT_EPOCHS
                    if initialization["mode"] == "warm_start"
                    else FORMAL_CHECKPOINT_EPOCHS.get(
                        int(config["epochs"]), {int(config["epochs"])}
                    )
                )
                if epoch in checkpoint_epochs:
                    torch.save(
                        payload,
                        checkpoint_dir / f"iwg_rg_cma_epoch{epoch:03d}.pt",
                    )
            _write_training_log(training_log, "IWG RG-CMA training complete.")
            _write_training_log(training_log, "=" * 60)
        finally:
            _shutdown_loader(loader)
            loader = None
            phase_dataset = None
            gc.collect()
            released = dataset.release_cached_pages()
            _write_training_log(training_log, f"Final mmap release: {released}")
    return {
        "status": "completed",
        "epochs": int(config["epochs"]),
        "effective_full_epochs": total_samples_seen / len(dataset),
        "optimizer_steps": global_step,
        "training_schedule": training_schedule,
        "resolved_batch_size": int(batch_size),
        "num_train_samples": len(dataset),
        "checkpoint": str(checkpoint_dir / "iwg_rg_cma_last.pt"),
        "elapsed_s": time.time() - start_time,
        "validation": "disabled",
        "selected_epoch": int(config["epochs"]),
        "initialization": initialization,
    }


def _validate_formal_config(config: dict[str, Any]) -> int:
    init_checkpoint = str(config.get("init_checkpoint", "")).strip()
    required = FINETUNE_REQUIRED_CONFIG if init_checkpoint else REQUIRED_CONFIG
    for key, expected in required.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"fixed IWG-attn config requires {key}={expected}, got {actual}")
    if not init_checkpoint and int(config.get("epochs", 0)) not in FORMAL_TRAIN_EPOCHS:
        raise ValueError(
            "formal IWG-attn training epochs must be one of "
            f"{sorted(FORMAL_TRAIN_EPOCHS)}, got {config.get('epochs')}"
        )
    batch_size = int(config.get("batch_size", 0))
    allowed_batch_sizes = {1024} if init_checkpoint else FORMAL_BATCH_SIZES
    if batch_size not in allowed_batch_sizes:
        raise ValueError(
            "formal IWG-attn batch_size must be one of "
            f"{sorted(allowed_batch_sizes)}, got {batch_size}"
        )
    if init_checkpoint:
        epochs = int(config.get("epochs", 0))
        if epochs not in FINETUNE_ALLOWED_EPOCHS:
            raise ValueError(
                "warm-start IWG-attn epochs must be one of "
                f"{sorted(FINETUNE_ALLOWED_EPOCHS)}, got {epochs}"
            )
        shards, cycles, epochs_per_shard = _resolved_schedule_settings(config)
        if (shards, cycles, epochs_per_shard) != (1, 1, epochs):
            raise ValueError(
                "warm-start IWG-attn training requires one full-data phase "
                f"for {epochs} epochs"
            )
    _resolved_schedule_settings(config)
    return batch_size


def train_iwg_rg_cma(config: dict[str, Any]) -> dict[str, Any]:
    batch_size = _validate_formal_config(config)
    if str(config.get("device", "cuda")) != "cuda":
        raise ValueError("formal IWG-attn training is fixed to CUDA")
    checkpoint_dir = Path(config["checkpoint_dir"])
    if (checkpoint_dir / "iwg_rg_cma_last.pt").exists():
        raise FileExistsError("formal IWG-attn training requires a new checkpoint directory")
    dataset = StreamingIWGAttnDataset(
        config["dataset_dir"], max_samples=int(config.get("max_train_samples", 0))
    )
    try:
        requested_mode = str(config.get("motion_target_mode", "nsa"))
        dataset_mode = str(dataset.metadata.get("motion_target_mode", "nsa"))
        if requested_mode != dataset_mode:
            raise ValueError(
                f"motion_target_mode mismatch: config={requested_mode!r}, "
                f"dataset={dataset_mode!r}"
            )
        requested_label_root = str(
            config.get("mamba_distill_label_root", "")
        ).strip()
        if dataset_mode == "mamba_hybrid":
            if str(config.get("init_checkpoint", "")).strip():
                raise ValueError("mamba_hybrid formal training must start from scratch")
            expected_root = str(dataset.metadata["mamba_distill_label_root"])
            if str(Path(requested_label_root).resolve()) != expected_root:
                raise ValueError(
                    "mamba_distill_label_root does not match dataset metadata"
                )
        elif dataset_mode == "mamba_native":
            if str(config.get("init_checkpoint", "")).strip():
                raise ValueError("mamba_native formal training must start from scratch")
            if requested_label_root:
                raise ValueError(
                    "mamba_native training does not use a shadow distill label root"
                )
        elif requested_label_root:
            raise ValueError("nsa training must not configure Mamba distill labels")
        return _run_training_attempt(config, dataset=dataset, batch_size=batch_size)
    except torch.cuda.OutOfMemoryError:
        if batch_size != 2048:
            raise
        torch.cuda.empty_cache()
        print("CUDA OOM at batch_size=2048; applying the sole fallback 1536.", flush=True)
        return _run_training_attempt(config, dataset=dataset, batch_size=1536)
    finally:
        dataset.close()


def smoke_train_iwg_rg_cma(config: dict[str, Any]) -> dict[str, Any]:
    """Small CPU-capable trainer used only by the implementation smoke test."""
    dataset = StreamingIWGAttnDataset(
        config["dataset_dir"], max_samples=int(config.get("max_train_samples", 64))
    )
    smoke_config = {
        **config,
        "seed": int(config.get("seed", 42)),
        "epochs": int(config.get("epochs", 1)),
        "num_workers": int(config.get("num_workers", 0)),
        "lr": float(config.get("lr", 1e-4)),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
        "warmup_epochs": int(config.get("warmup_epochs", 1)),
        "grad_clip": float(config.get("grad_clip", 1.0)),
        "device": str(config.get("device", "cpu")),
    }
    try:
        return _run_training_attempt(
            smoke_config,
            dataset=dataset,
            batch_size=int(config.get("batch_size", 16)),
        )
    finally:
        dataset.close()


def validate_iwg_rg_cma_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    device: str = "cpu",
    max_batches: int = 1,
) -> dict[str, Any]:
    dataset = StreamingIWGAttnDataset(dataset_dir)
    model, checkpoint = load_iwg_rg_cma_checkpoint(
        checkpoint_path, map_location=device
    )
    validate_iwg_rg_cma_checkpoint_contract(
        checkpoint, expected_dataset_sha256=dataset.metadata["dataset_sha256"]
    )
    if checkpoint["dataset_schema_sha256"] != dataset.metadata["dataset_schema_sha256"]:
        raise ValueError("checkpoint dataset schema does not match the requested dataset")
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=StreamingIWGAttnDataset.collate_fn,
    )
    totals: dict[str, float] = {}
    count = 0
    max_correction = 0.0
    try:
        with torch.inference_mode():
            for batch_index, raw_batch in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                batch = _move_batch(raw_batch, torch.device(device))
                outputs = _forward(model, batch)
                _loss, components = compute_iwg_rg_cma_loss(outputs, batch)
                size = int(batch["track_feats"].shape[0])
                count += size
                for key, value in components.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * size
                max_correction = max(
                    max_correction, float(outputs["gate_correction"].abs().max())
                )
        if count == 0:
            raise RuntimeError("checkpoint validation processed no samples")
        return {
            "status": "valid",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "epoch": int(checkpoint["epoch"]),
            "samples": count,
            "max_abs_correction": max_correction,
            "correction_bound": RG_CMA_CORRECTION_BOUND,
            "metrics": {key: value / count for key, value in totals.items()},
            "dataset_sha256": checkpoint["dataset_sha256"],
            "model_schema": checkpoint["model_schema"],
        }
    finally:
        dataset.close()
