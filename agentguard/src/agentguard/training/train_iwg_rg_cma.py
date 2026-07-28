from __future__ import annotations

import gc
import json
import hashlib
import math
import os
import random
import subprocess
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import ROLLOUT_LABEL_SCHEMA_SHA256
from agentguard.datasets.iwg_rg_cma_dataset import (
    COMPACT_INDEX_FORMAT,
    accepted_iwg_rg_cma_dataset_schema_sha256,
    SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256,
    StreamingIWGRGCMADataset,
)
from agentguard.models.iwg_rg_cma import (
    ARCHITECTURE_LEGACY,
    IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA,
    IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA_SHA256,
    IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA,
    IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA_SHA256,
    IWG_RG_CMA_LEGACY_MODEL_SCHEMA,
    IWG_RG_CMA_LEGACY_MODEL_SCHEMA_SHA256,
    IWG_RG_CMA_MODEL_SCHEMA,
    IWG_RG_CMA_MODEL_SCHEMA_SHA256,
    IWGRGCMA,
    MODEL_CONTRACT_SPECS,
    RELIABILITY_MODE_FULL,
    SUPPORTED_ARCHITECTURE_VARIANTS,
    SUPPORTED_RELIABILITY_MODES,
    RG_CMA_CORRECTION_BOUND,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    model_contract,
)
from agentguard.training.loss_iwg_rg_cma import (
    compute_iwg_rg_cma_loss,
    validate_iwg_rg_cma_loss_config,
)


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
SEQUENCE_SAMPLING_SAMPLE_PROPORTIONAL = "sample-proportional"
SEQUENCE_SAMPLING_SQRT_SIZE = "sqrt-size"
SUPPORTED_SEQUENCE_SAMPLING = frozenset(
    {SEQUENCE_SAMPLING_SAMPLE_PROPORTIONAL, SEQUENCE_SAMPLING_SQRT_SIZE}
)
SUPPORTED_RESIDUAL_TARGET_MODES = frozenset(
    {"safe", "oracle-confidence", "safe-selective"}
)


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
        "datasets/iwg_rg_cma_dataset.py",
        "data/rollout_label_builder.py",
        "rollout_labels.py",
        "training/loss_iwg_rg_cma.py",
        "training/train_iwg_rg_cma.py",
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
    sampler: Sampler[int] | None = None,
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
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=StreamingIWGRGCMADataset.collate_fn,
        worker_init_fn=_worker_seed,
        generator=generator,
        **loader_kwargs,
    )


def _resolve_sequence_sampling(config: dict[str, Any]) -> str:
    policy = str(
        config.get(
            "sequence_sampling",
            SEQUENCE_SAMPLING_SAMPLE_PROPORTIONAL,
        )
    ).strip().lower()
    if policy not in SUPPORTED_SEQUENCE_SAMPLING:
        raise ValueError(
            "unsupported sequence_sampling policy: "
            f"{policy!r}; expected one of {sorted(SUPPORTED_SEQUENCE_SAMPLING)}"
        )
    return policy


def _sequence_sampling_weights(
    counts: dict[str, int],
    *,
    power: float,
) -> dict[str, float]:
    active = [sequence for sequence, count in counts.items() if int(count) > 0]
    if not active:
        raise ValueError("sequence sampling requires at least one non-empty sequence")
    raw = {
        sequence: float(counts[sequence]) ** float(power)
        for sequence in active
    }
    total = sum(raw.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("sequence sampling weights must have a positive finite sum")
    return {sequence: value / total for sequence, value in raw.items()}


def _allocate_sequence_sample_counts(
    *,
    num_samples: int,
    sequence_counts: dict[str, int],
    active_sequences: list[str],
    power: float,
) -> dict[str, int]:
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if not active_sequences:
        if num_samples:
            raise ValueError("cannot allocate samples without active sequences")
        return {}
    weights = _sequence_sampling_weights(
        {sequence: sequence_counts[sequence] for sequence in active_sequences},
        power=power,
    )
    raw = {
        sequence: float(num_samples) * weights[sequence]
        for sequence in active_sequences
    }
    allocated = {sequence: math.floor(value) for sequence, value in raw.items()}
    remaining = int(num_samples - sum(allocated.values()))
    remainder_order = sorted(
        active_sequences,
        key=lambda sequence: (
            -(raw[sequence] - allocated[sequence]),
            active_sequences.index(sequence),
        ),
    )
    for sequence in remainder_order[:remaining]:
        allocated[sequence] += 1
    if sum(allocated.values()) != int(num_samples):
        raise AssertionError("sequence sample allocation changed the phase budget")
    return allocated


class SequenceSqrtSampler(Sampler[int]):
    """Sample phase indices with exact sqrt-size sequence quotas.

    The sampler operates on phase-local indices, so it works for both a full
    dataset and a ``ConcatDataset`` of contiguous shard subsets. Sampling within
    each sequence is with replacement because the phase budget is preserved
    while smaller sequences receive more updates than their raw sample share.
    """

    def __init__(
        self,
        ranges: list[dict[str, Any]],
        target_counts: dict[str, int],
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        self._ranges: list[tuple[str, int, int]] = []
        phase_offset = 0
        for item in ranges:
            count = int(item["samples"])
            if count < 0:
                raise ValueError("sequence phase range has a negative sample count")
            sequence = str(item["sequence"])
            if count:
                self._ranges.append((sequence, phase_offset, phase_offset + count))
            phase_offset += count
        if phase_offset != int(num_samples):
            raise ValueError(
                "sequence sampler phase budget mismatch: "
                f"ranges={phase_offset}, requested={num_samples}"
            )
        self._target_counts = {
            str(sequence): int(count) for sequence, count in target_counts.items()
        }
        if sum(self._target_counts.values()) != int(num_samples):
            raise ValueError("sequence sampler target counts do not match phase budget")
        range_sequences = {sequence for sequence, _start, _end in self._ranges}
        if set(self._target_counts) != range_sequences:
            raise ValueError(
                "sequence sampler targets do not match non-empty phase ranges"
            )
        self._num_samples = int(num_samples)
        self._seed = int(seed)
        self._epoch = 0

    @property
    def target_counts(self) -> dict[str, int]:
        return dict(self._target_counts)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self._seed + self._epoch * 1_000_003)
        indices: list[int] = []
        for sequence, start, end in self._ranges:
            indices.extend(
                rng.randrange(start, end)
                for _ in range(self._target_counts[sequence])
            )
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self._num_samples


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
    dataset: StreamingIWGRGCMADataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    shards, cycles, epochs_per_shard = _resolved_schedule_settings(config)
    sequence_sampling = _resolve_sequence_sampling(config)
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

    sampling_power = (
        0.5
        if sequence_sampling == SEQUENCE_SAMPLING_SQRT_SIZE
        else 1.0
    )
    sequence_sampling_weights = _sequence_sampling_weights(
        counts,
        power=sampling_power,
    )

    randomize_shard_order = bool(config.get("randomize_shard_order", False))
    order_rng = random.Random(int(config.get("seed", 42)))
    shard_orders: list[list[int]] = []
    phases: list[dict[str, Any]] = []
    for cycle in range(cycles):
        shard_order = list(range(shards))
        if randomize_shard_order:
            order_rng.shuffle(shard_order)
        shard_orders.append([shard + 1 for shard in shard_order])
        for shard in shard_order:
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
            phase_samples = sum(item["samples"] for item in ranges)
            phase = {
                "phase": len(phases) + 1,
                "cycle": cycle + 1,
                "shard": shard + 1,
                "fraction_start": shard / shards,
                "fraction_end": (shard + 1) / shards,
                "epochs": epochs_per_shard,
                "samples": phase_samples,
                "ranges": ranges,
            }
            if sequence_sampling == SEQUENCE_SAMPLING_SQRT_SIZE:
                active_sequences = [
                    str(item["sequence"])
                    for item in ranges
                    if int(item["samples"]) > 0
                ]
                phase["sequence_sampling_target_counts"] = (
                    _allocate_sequence_sample_counts(
                        num_samples=phase_samples,
                        sequence_counts=counts,
                        active_sequences=active_sequences,
                        power=sampling_power,
                    )
                )
            phases.append(phase)
    return {
        "sampling_policy": (
            "sequence_sqrt_size_with_replacement"
            if sequence_sampling == SEQUENCE_SAMPLING_SQRT_SIZE
            else ("full_shuffle" if shards == 1 else LOW_MEMORY_SAMPLING_POLICY)
        ),
        "sequence_sampling": sequence_sampling,
        "sequence_sampling_power": sampling_power,
        "sequence_sampling_weights": sequence_sampling_weights,
        "sequence_sampling_replacement": (
            sequence_sampling == SEQUENCE_SAMPLING_SQRT_SIZE
        ),
        "memory_shards": shards,
        "shard_cycles": cycles,
        "epochs_per_shard": epochs_per_shard,
        "nominal_epochs": int(config["epochs"]),
        "effective_full_epochs": epochs_per_shard * cycles,
        "randomize_shard_order": randomize_shard_order,
        "shard_order_per_cycle": shard_orders,
        "phases": phases,
    }


def _phase_dataset(
    dataset: StreamingIWGRGCMADataset,
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


def _phase_sampler(
    training_schedule: dict[str, Any],
    phase: dict[str, Any],
    *,
    seed: int,
) -> Sampler[int] | None:
    if training_schedule["sequence_sampling"] != SEQUENCE_SAMPLING_SQRT_SIZE:
        return None
    return SequenceSqrtSampler(
        phase["ranges"],
        phase["sequence_sampling_target_counts"],
        num_samples=int(phase["samples"]),
        seed=seed,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _resolved_loss_settings(config: dict[str, Any]) -> dict[str, float]:
    settings = {
        "residual_beta": float(config.get("residual_beta", 1.0)),
        "residual_weight": float(config.get("residual_weight", 0.5)),
        "revision_weight": float(config.get("revision_weight", 0.01)),
        "hard_example_gain": float(config.get("hard_example_gain", 0.0)),
        "no_harm_weight": float(config.get("no_harm_weight", 0.0)),
    }
    validate_iwg_rg_cma_loss_config(**settings)
    return settings


def _resolve_residual_target_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("residual_target_mode", "safe")).strip().lower()
    if mode not in SUPPORTED_RESIDUAL_TARGET_MODES:
        raise ValueError(
            "unsupported residual_target_mode: "
            f"{mode!r}; expected {sorted(SUPPORTED_RESIDUAL_TARGET_MODES)}"
        )
    return mode


def _validate_residual_target_dataset(
    dataset: StreamingIWGRGCMADataset,
    mode: str,
) -> None:
    mode = str(mode).strip().lower()
    if mode in {"oracle-confidence", "safe-selective"}:
        required = {"oracle_gate_target", "gate_confidence"}
        available = set(
            dataset.metadata.get("compact_index_array_files", {})
        )
        # v2 compact datasets store the fields explicitly. Existing v3
        # compact datasets encode the same values in risk_target/cue_target;
        # StreamingIWGRGCMADataset derives them per sample without rewriting
        # the label or index files.
        derived = {"risk_target", "cue_target"}
        if dataset.index_format == COMPACT_INDEX_FORMAT and not (
            required.issubset(available) or derived.issubset(available)
        ):
            raise ValueError(
                "oracle-confidence dataset is missing explicit residual targets "
                "and cannot derive them from risk_target/cue_target"
            )


def _resolved_optimizer_lrs(config: dict[str, Any]) -> dict[str, float]:
    """Resolve legacy shared LR and optional Base/CMA learning rates."""
    shared_lr = float(config.get("lr", 1e-4))
    base_value = config.get("base_lr")
    cma_value = config.get("cma_lr")
    base_lr = shared_lr if base_value is None else float(base_value)
    cma_lr = shared_lr if cma_value is None else float(cma_value)
    resolved = {"base_lr": base_lr, "cma_lr": cma_lr}
    for name, value in resolved.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    return resolved


def _resolve_reliability_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("reliability_mode", RELIABILITY_MODE_FULL)).strip().lower()
    if mode not in SUPPORTED_RELIABILITY_MODES:
        raise ValueError(
            "unsupported reliability_mode: "
            f"{mode!r}; expected one of {sorted(SUPPORTED_RELIABILITY_MODES)}"
        )
    return mode


def _resolve_architecture_variant(config: dict[str, Any]) -> str:
    variant = str(
        config.get("architecture_variant", ARCHITECTURE_LEGACY)
    ).strip().lower()
    if variant not in SUPPORTED_ARCHITECTURE_VARIANTS:
        raise ValueError(
            "unsupported architecture_variant: "
            f"{variant!r}; expected one of "
            f"{sorted(SUPPORTED_ARCHITECTURE_VARIANTS)}"
        )
    return variant


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


def _build_iwg_rg_cma_optimizer(
    model: IWGRGCMA,
    *,
    base_lr: float,
    cma_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the Base IWG/RG-CMA parameter groups used by the trainer."""
    base_params = list(model.iwg.parameters())
    base_param_ids = {id(parameter) for parameter in base_params}
    cma_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in base_param_ids
    ]
    all_params = list(model.parameters())
    if len(base_params) + len(cma_params) != len(all_params):
        raise AssertionError("IWG/RG-CMA optimizer parameter groups overlap or omit parameters")
    if not base_params or not cma_params:
        raise AssertionError("IWG/RG-CMA optimizer parameter groups must both be non-empty")
    return torch.optim.AdamW(
        [
            {"name": "base_iwg", "params": base_params, "lr": float(base_lr)},
            {"name": "rg_cma", "params": cma_params, "lr": float(cma_lr)},
        ],
        weight_decay=float(weight_decay),
    )


def _optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _optimizer_group_grad_norms(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Return pre-clipping L2 norms for each optimizer parameter group."""
    norms: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        group_norms = [
            parameter.grad.detach().float().norm(2)
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        name = str(group.get("name", f"group_{index}"))
        norms[name] = (
            0.0
            if not group_norms
            else float(torch.linalg.vector_norm(torch.stack(group_norms), ord=2))
        )
    return norms


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
    payload = {
        **model_contract(
            correction_bound=float(model.correction_bound),
            context_size=int(model.context_size),
            architecture_variant=str(model.architecture_variant),
        ),
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
        "correction_bound": float(model.correction_bound),
        "reliability_mode": str(model.reliability_mode),
        "architecture_variant": str(model.architecture_variant),
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
        "optimizer_learning_rates": _resolved_optimizer_lrs(config),
        "validation_policy": (
            f"none_epoch{int(config['epochs'])}_warm_start"
            if initialization and initialization.get("mode") == "warm_start"
            else "none_epoch100"
        ),
        "initialization": initialization or {"mode": "random"},
        "training_schedule": training_schedule or {"sampling_policy": "full_shuffle"},
        "training_progress": training_progress or {},
    }
    return payload


def validate_iwg_rg_cma_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    expected_dataset_sha256: str | None = None,
) -> None:
    model_contract_key = (
        str(checkpoint.get("model_schema", "")),
        str(checkpoint.get("model_schema_sha256", "")),
    )
    expected_contract = MODEL_CONTRACT_SPECS.get(model_contract_key)
    if expected_contract is None:
        raise ValueError(
            "IWG RG-CMA checkpoint contract mismatch: unsupported model contract "
            f"{model_contract_key!r}"
        )
    expected_bound = float(expected_contract["correction_bound"])
    expected_context_size = int(expected_contract["context_size"])
    expected_architecture_variant = str(
        expected_contract["architecture_variant"]
    )
    required = {
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "correction_bound": expected_bound,
        "context_size": expected_context_size,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in required.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"IWG RG-CMA checkpoint contract mismatch: {mismatches}")
    checkpoint_architecture_variant = str(
        checkpoint.get(
            "architecture_variant",
            checkpoint.get("training_config", {}).get(
                "architecture_variant", ARCHITECTURE_LEGACY
            ),
        )
    ).strip().lower()
    if checkpoint_architecture_variant != expected_architecture_variant:
        raise ValueError(
            "IWG RG-CMA checkpoint contract mismatch: architecture_variant "
            f"{checkpoint_architecture_variant!r} != "
            f"{expected_architecture_variant!r}"
        )
    reliability_mode = str(
        checkpoint.get(
            "reliability_mode",
            checkpoint.get("training_config", {}).get(
                "reliability_mode", RELIABILITY_MODE_FULL
            ),
        )
    ).strip().lower()
    if reliability_mode not in SUPPORTED_RELIABILITY_MODES:
        raise ValueError(
            "IWG RG-CMA checkpoint has unsupported reliability_mode: "
            f"{reliability_mode!r}"
        )
    checkpoint_dataset = str(checkpoint.get("dataset", ""))
    checkpoint_split = str(checkpoint.get("split", ""))
    checkpoint_context_size = int(checkpoint.get("context_size", 0))
    if checkpoint_dataset and checkpoint_split:
        accepted_dataset_schema_sha256 = accepted_iwg_rg_cma_dataset_schema_sha256(
            checkpoint_dataset,
            checkpoint_split,
            checkpoint_context_size,
        )
    else:
        accepted_dataset_schema_sha256 = SUPPORTED_IWG_RG_CMA_DATASET_SCHEMA_SHA256
    if checkpoint.get("dataset_schema_sha256") not in accepted_dataset_schema_sha256:
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
    if str(checkpoint.get("motion_target_mode", "nsa")) != "nsa":
        raise ValueError("only NSA checkpoints are supported")


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
        context_size=int(checkpoint["context_size"]),
        reliability_mode=str(
            checkpoint.get(
                "reliability_mode",
                checkpoint.get("training_config", {}).get(
                    "reliability_mode", RELIABILITY_MODE_FULL
                ),
            )
        ),
        architecture_variant=str(
            checkpoint.get(
                "architecture_variant",
                checkpoint.get("training_config", {}).get(
                    "architecture_variant", ARCHITECTURE_LEGACY
                ),
            )
        ),
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
    checkpoint_bound = float(checkpoint["correction_bound"])
    if not math.isclose(
        float(model.correction_bound), checkpoint_bound, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "initial checkpoint correction bound does not match the requested "
            f"training bound: {checkpoint_bound} != {model.correction_bound}"
        )
    checkpoint_context_size = int(checkpoint["context_size"])
    if int(model.context_size) != checkpoint_context_size:
        raise ValueError(
            "initial checkpoint context size does not match the requested "
            f"training context: {checkpoint_context_size} != {model.context_size}"
        )
    checkpoint_architecture_variant = str(
        checkpoint.get(
            "architecture_variant",
            checkpoint.get("training_config", {}).get(
                "architecture_variant", ARCHITECTURE_LEGACY
            ),
        )
    )
    if model.architecture_variant != checkpoint_architecture_variant:
        raise ValueError(
            "initial checkpoint architecture variant does not match requested "
            f"training variant: {checkpoint_architecture_variant} != "
            f"{model.architecture_variant}"
        )
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
    dataset: StreamingIWGRGCMADataset,
    batch_size: int,
) -> dict[str, Any]:
    seed = int(config["seed"])
    _seed_everything(seed)
    device = torch.device(config["device"])
    loss_settings = _resolved_loss_settings(config)
    residual_target_mode = _resolve_residual_target_mode(config)
    _validate_residual_target_dataset(dataset, residual_target_mode)
    reliability_mode = _resolve_reliability_mode(config)
    architecture_variant = _resolve_architecture_variant(config)
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
        correction_bound=float(
            config.get("correction_bound", RG_CMA_CORRECTION_BOUND)
        ),
        context_size=int(config.get("context_size", 6)),
        reliability_mode=reliability_mode,
        architecture_variant=architecture_variant,
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
    optimizer_lrs = _resolved_optimizer_lrs(config)
    optimizer = _build_iwg_rg_cma_optimizer(
        model,
        base_lr=optimizer_lrs["base_lr"],
        cma_lr=optimizer_lrs["cma_lr"],
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
    phase_sampler: Sampler[int] | None = None
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
            "Optimizer learning rates: "
            + json.dumps(_optimizer_learning_rates(optimizer), sort_keys=True),
        )
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
                    phase_sampler = None
                    gc.collect()
                    released = dataset.release_cached_pages()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    phase_dataset = (
                        dataset
                        if int(training_schedule["memory_shards"]) == 1
                        else _phase_dataset(dataset, phase)
                    )
                    phase_seed = (
                        seed
                        + (int(phase["cycle"]) - 1)
                        * int(training_schedule["memory_shards"])
                        + int(phase["shard"])
                        - 1
                    )
                    phase_sampler = _phase_sampler(
                        training_schedule,
                        phase,
                        seed=phase_seed,
                    )
                    loader = _loader(
                        phase_dataset,
                        batch_size=batch_size,
                        num_workers=int(config["num_workers"]),
                        seed=phase_seed,
                        low_memory=int(training_schedule["memory_shards"]) > 1,
                        sampler=phase_sampler,
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
                if phase_sampler is not None:
                    phase_sampler.set_epoch(epoch)

                model.train()
                totals: dict[str, float] = {}
                samples = 0
                channel_abs_error = [0.0, 0.0]
                channel_correct = [0.0, 0.0]
                channel_weight = [0.0, 0.0]
                context_size = int(config.get("context_size", model.context_size))
                motion_positions = torch.zeros(context_size, dtype=torch.float64)
                appearance_positions = torch.zeros(context_size, dtype=torch.float64)
                appearance_to_motion_positions = torch.zeros(
                    context_size, dtype=torch.float64
                )
                motion_to_appearance_positions = torch.zeros(
                    context_size, dtype=torch.float64
                )
                appearance_to_motion_6x6_matrix = torch.zeros(
                    (context_size, context_size), dtype=torch.float64
                )
                motion_to_appearance_6x6_matrix = torch.zeros(
                    (context_size, context_size), dtype=torch.float64
                )
                cross_matrix = torch.zeros(
                    (model.cross_modal_token_count, model.cross_modal_token_count),
                    dtype=torch.float64,
                )
                motion_entropy = 0.0
                appearance_entropy = 0.0
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
                        outputs,
                        batch,
                        correction_bound=model.correction_bound,
                        residual_target_mode=residual_target_mode,
                        **loss_settings,
                    )
                    loss.backward()
                    group_grad_norms = _optimizer_group_grad_norms(optimizer)
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
                    for group_name, group_grad_norm in group_grad_norms.items():
                        key = f"{group_name}_grad_norm"
                        totals[key] = totals.get(key, 0.0) + group_grad_norm * count
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
                    if "appearance_to_motion_attention_weights" in outputs:
                        appearance_to_motion_positions += (
                            outputs["appearance_to_motion_attention_weights"]
                            .detach()
                            .double()
                            .mean(dim=1)
                            .sum(dim=0)
                            .cpu()
                        )
                        motion_to_appearance_positions += (
                            outputs["motion_to_appearance_attention_weights"]
                            .detach()
                            .double()
                            .mean(dim=1)
                            .sum(dim=0)
                            .cpu()
                        )
                    if "appearance_to_motion_6x6_attention_weights" in outputs:
                        appearance_to_motion_6x6_matrix += (
                            outputs["appearance_to_motion_6x6_attention_weights"]
                            .detach()
                            .double()
                            .mean(dim=1)
                            .sum(dim=0)
                            .cpu()
                        )
                        motion_to_appearance_6x6_matrix += (
                            outputs["motion_to_appearance_6x6_attention_weights"]
                            .detach()
                            .double()
                            .mean(dim=1)
                            .sum(dim=0)
                            .cpu()
                        )
                    if "cross_modal_attention_weights" in outputs:
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
                    if batch_index % max(1, len(loader) // 5) == 0:
                        _write_training_log(
                            training_log,
                            (
                                f"Epoch {epoch:3d}  step {global_step:6d} | "
                                f"loss={float(components['loss'].detach()):.6f} | "
                                f"base_loss={float(components['base_loss'].detach()):.6f} | "
                                f"final_loss={float(components['final_loss'].detach()):.6f} | "
                                f"corr_abs={float(components['correction_abs_mean'].detach()):.6f} | "
                                f"base_lr={optimizer.param_groups[0]['lr']:.2e} | "
                                f"cma_lr={optimizer.param_groups[1]['lr']:.2e}"
                            ),
                        )

                del raw_batch, batch, outputs, components, loss

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_time = time.time() - epoch_start
                total_samples_seen += samples
                total_channel_weight = sum(channel_weight)
                current_lrs = _optimizer_learning_rates(optimizer)
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
                    # Keep learning_rate as a backward-compatible Base-IWG alias.
                    "learning_rate": current_lrs["base_iwg"],
                    "base_learning_rate": current_lrs["base_iwg"],
                    "cma_learning_rate": current_lrs["rg_cma"],
                    "optimizer_learning_rates": current_lrs,
                    "epoch_time_s": epoch_time,
                    "gate_accuracy": sum(channel_correct)
                    / max(total_channel_weight, 1.0),
                    "motion_mae": channel_abs_error[0]
                    / max(channel_weight[0], 1.0),
                    "appearance_mae": channel_abs_error[1]
                    / max(channel_weight[1], 1.0),
                    "motion_attention_entropy": motion_entropy / max(samples, 1),
                    "appearance_attention_entropy": appearance_entropy
                    / max(samples, 1),
                    "motion_history_attention": (
                        motion_positions / max(samples, 1)
                    ).tolist(),
                    "appearance_history_attention": (
                        appearance_positions / max(samples, 1)
                    ).tolist(),
                }
                if not model.uses_full_history_cma:
                    metrics["cross_modal_attention"] = (
                        cross_matrix / max(samples, 1)
                    ).tolist()
                if model.uses_bidirectional_history_cma:
                    metrics["appearance_to_motion_history_attention"] = (
                        appearance_to_motion_positions / max(samples, 1)
                    ).tolist()
                    metrics["motion_to_appearance_history_attention"] = (
                        motion_to_appearance_positions / max(samples, 1)
                    ).tolist()
                if model.uses_full_history_cma:
                    metrics["appearance_to_motion_6x6_attention"] = (
                        appearance_to_motion_6x6_matrix / max(samples, 1)
                    ).tolist()
                    metrics["motion_to_appearance_6x6_attention"] = (
                        motion_to_appearance_6x6_matrix / max(samples, 1)
                    ).tolist()
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
                        f"corr_abs={metrics['correction_abs_mean']:.4f}  "
                        f"corr_target={metrics['correction_target_abs_mean']:.4f}  "
                        f"corr_sign={metrics['correction_sign_agreement']:.3f}  "
                        f"base_lr={metrics['base_learning_rate']:.2e}  "
                        f"cma_lr={metrics['cma_learning_rate']:.2e}  "
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
        "optimizer_learning_rates": _resolved_optimizer_lrs(config),
        "architecture_variant": str(model.architecture_variant),
        "checkpoint": str(checkpoint_dir / "iwg_rg_cma_last.pt"),
        "elapsed_s": time.time() - start_time,
        "validation": "disabled",
        "selected_epoch": int(config["epochs"]),
        "initialization": initialization,
    }


def _validate_formal_config(config: dict[str, Any]) -> int:
    _resolve_sequence_sampling(config)
    _resolved_loss_settings(config)
    _resolve_residual_target_mode(config)
    _resolved_optimizer_lrs(config)
    _resolve_reliability_mode(config)
    architecture_variant = _resolve_architecture_variant(config)
    model_contract(
        correction_bound=float(
            config.get("correction_bound", RG_CMA_CORRECTION_BOUND)
        ),
        context_size=int(config.get("context_size", 6)),
        architecture_variant=architecture_variant,
    )
    init_checkpoint = str(config.get("init_checkpoint", "")).strip()
    required = FINETUNE_REQUIRED_CONFIG if init_checkpoint else REQUIRED_CONFIG
    for key, expected in required.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"fixed IWG RG-CMA config requires {key}={expected}, got {actual}")
    if not init_checkpoint and int(config.get("epochs", 0)) not in FORMAL_TRAIN_EPOCHS:
        raise ValueError(
            "formal IWG RG-CMA training epochs must be one of "
            f"{sorted(FORMAL_TRAIN_EPOCHS)}, got {config.get('epochs')}"
        )
    batch_size = int(config.get("batch_size", 0))
    allowed_batch_sizes = {1024} if init_checkpoint else FORMAL_BATCH_SIZES
    if batch_size not in allowed_batch_sizes:
        raise ValueError(
            "formal IWG RG-CMA batch_size must be one of "
            f"{sorted(allowed_batch_sizes)}, got {batch_size}"
        )
    if init_checkpoint:
        epochs = int(config.get("epochs", 0))
        if epochs not in FINETUNE_ALLOWED_EPOCHS:
            raise ValueError(
                "warm-start IWG RG-CMA epochs must be one of "
                f"{sorted(FINETUNE_ALLOWED_EPOCHS)}, got {epochs}"
            )
        shards, cycles, epochs_per_shard = _resolved_schedule_settings(config)
        if (shards, cycles, epochs_per_shard) != (1, 1, epochs):
            raise ValueError(
                "warm-start IWG RG-CMA training requires one full-data phase "
                f"for {epochs} epochs"
            )
    _resolved_schedule_settings(config)
    return batch_size


def train_iwg_rg_cma(config: dict[str, Any]) -> dict[str, Any]:
    batch_size = _validate_formal_config(config)
    if str(config.get("device", "cuda")) != "cuda":
        raise ValueError("formal IWG RG-CMA training is fixed to CUDA")
    checkpoint_dir = Path(config["checkpoint_dir"])
    if (checkpoint_dir / "iwg_rg_cma_last.pt").exists():
        raise FileExistsError("formal IWG RG-CMA training requires a new checkpoint directory")
    dataset = StreamingIWGRGCMADataset(
        config["dataset_dir"], max_samples=int(config.get("max_train_samples", 0))
    )
    try:
        dataset_context_size = int(dataset.metadata["context_size"])
        configured_context_size = int(config.get("context_size", 6))
        if dataset_context_size != configured_context_size:
            raise ValueError(
                "training context_size does not match dataset metadata: "
                f"{configured_context_size} != {dataset_context_size}"
            )
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
    dataset = StreamingIWGRGCMADataset(
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
    dataset = StreamingIWGRGCMADataset(dataset_dir)
    model, checkpoint = load_iwg_rg_cma_checkpoint(
        checkpoint_path, map_location=device
    )
    validate_iwg_rg_cma_checkpoint_contract(
        checkpoint, expected_dataset_sha256=dataset.metadata["dataset_sha256"]
    )
    if int(checkpoint["context_size"]) != int(dataset.metadata["context_size"]):
        raise ValueError("checkpoint context_size does not match dataset metadata")
    if checkpoint["dataset_schema_sha256"] != dataset.metadata["dataset_schema_sha256"]:
        raise ValueError("checkpoint dataset schema does not match the requested dataset")
    _validate_residual_target_dataset(
        dataset,
        str(
            checkpoint.get("training_config", {}).get(
                "residual_target_mode", "safe"
            )
        ),
    )
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=StreamingIWGRGCMADataset.collate_fn,
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
                _loss, components = compute_iwg_rg_cma_loss(
                    outputs,
                    batch,
                    correction_bound=float(checkpoint["correction_bound"]),
                    residual_target_mode=str(
                        checkpoint.get("training_config", {}).get(
                            "residual_target_mode", "safe"
                        )
                    ),
                    **_resolved_loss_settings(
                        checkpoint.get("training_config", {})
                    ),
                )
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
            "correction_bound": float(checkpoint["correction_bound"]),
            "metrics": {key: value / count for key, value in totals.items()},
            "dataset_sha256": checkpoint["dataset_sha256"],
            "model_schema": checkpoint["model_schema"],
            "architecture_variant": checkpoint.get(
                "architecture_variant", ARCHITECTURE_LEGACY
            ),
        }
    finally:
        dataset.close()
