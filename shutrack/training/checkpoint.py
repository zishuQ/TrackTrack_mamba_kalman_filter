"""Independent SHUTrack v3 checkpoint schema, strict loader and atomic saves.

The loader never falls back to ``strict=False`` and explicitly rejects legacy
IWG / RG-CMA checkpoints: the new architecture must be retrained.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.schema import (
    ARCHITECTURE_NAME,
    ARCHITECTURE_VERSION,
    COMPETITION_DIM,
    CONTEXT_SIZE,
    EVIDENCE_DIM,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    HISTORY_SIZE,
    LEGACY_SCALAR_DIM,
    feature_schema_descriptor,
)

CHECKPOINT_VERSION = 1
_LEGACY_MARKERS = (
    "model_schema",
    "correction_bound",
    "policy_prototypes",
    "reliability_mode",
    "architecture_variant",
    "base_gate",
    "refined_gate",
)
_REQUIRED_CHECKPOINT_KEYS = (
    "checkpoint_version",
    "architecture",
    "architecture_version",
    "feature_schema_version",
    "feature_schema_sha256",
    "context_size",
    "reid_dim",
    "model_config",
    "model_state_dict",
)


def _git_value(arguments: list[str], default: str) -> str:
    try:
        import subprocess

        result = subprocess.run(
            arguments, capture_output=True, text=True, check=False, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip() or default
    except Exception:
        pass
    return default


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to("cpu", copy=True)
        for key, value in model.state_dict().items()
    }


def model_from_config(model_config: Mapping[str, Any]) -> ShutrackV2Model:
    allowed = {
        "reid_dim",
        "d_model",
        "n_heads",
        "hce_layers",
        "ff_dim",
        "dropout",
        "appearance_proj_dim",
        "context_size",
        "competition_enabled",
    }
    unknown = sorted(set(model_config).difference(allowed))
    if unknown:
        raise ValueError(
            f"SHUTrack checkpoint model_config has unsupported keys: {unknown}"
        )
    return ShutrackV2Model(ShutrackV2Config(**dict(model_config)))


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SHUTrack checkpoint must be a mapping")
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE_NAME:
        legacy = sorted(set(checkpoint).intersection(_LEGACY_MARKERS))
        hint = (
            " This looks like a legacy IWG/RG-CMA checkpoint "
            f"(fields: {legacy})."
            if legacy
            else ""
        )
        raise ValueError(
            "not a SHUTrack v2 checkpoint: expected architecture="
            f"{ARCHITECTURE_NAME!r}, got {architecture!r}.{hint} "
            "SHUTrack v2 must be trained from scratch; old checkpoints cannot "
            "be loaded with strict=False."
        )
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "SHUTrack checkpoint architecture_version mismatch: expected "
            f"{ARCHITECTURE_VERSION!r}, got {checkpoint.get('architecture_version')!r}"
        )
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported SHUTrack checkpoint_version: "
            f"{checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "SHUTrack feature schema mismatch: expected "
            f"{FEATURE_SCHEMA_VERSION!r}, got "
            f"{checkpoint.get('feature_schema_version')!r}"
        )
    if checkpoint.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError(
            "SHUTrack feature schema hash mismatch; refusing to load a "
            "checkpoint built from a different input definition"
        )
    if int(checkpoint.get("context_size", -1)) != CONTEXT_SIZE:
        raise ValueError(
            f"SHUTrack checkpoint context_size must be {CONTEXT_SIZE}, got "
            f"{checkpoint.get('context_size')!r}"
        )
    if int(checkpoint.get("history_size", -1)) != HISTORY_SIZE:
        raise ValueError(
            f"SHUTrack checkpoint history_size must be {HISTORY_SIZE}, got "
            f"{checkpoint.get('history_size')!r}"
        )
    if int(checkpoint.get("evidence_dim", -1)) != EVIDENCE_DIM:
        raise ValueError("SHUTrack checkpoint evidence dimension mismatch")
    if int(checkpoint.get("competition_dim", -1)) != COMPETITION_DIM:
        raise ValueError("SHUTrack checkpoint competition dimension mismatch")
    if int(checkpoint.get("scalar_dim", -1)) != LEGACY_SCALAR_DIM:
        raise ValueError("SHUTrack checkpoint legacy scalar dimension mismatch")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("SHUTrack checkpoint is missing model_config")
    if int(model_config.get("context_size", -1)) != CONTEXT_SIZE:
        raise ValueError("SHUTrack checkpoint model_config context_size mismatch")
    if int(model_config.get("reid_dim", -1)) != int(checkpoint.get("reid_dim", -2)):
        raise ValueError("SHUTrack checkpoint reid_dim does not match model_config")
    competition_enabled = checkpoint.get("competition_enabled")
    if not isinstance(competition_enabled, bool):
        raise ValueError(
            "SHUTrack checkpoint competition_enabled must be an explicit boolean; "
            "retrain with the current code so the branch configuration is recorded"
        )
    if bool(model_config.get("competition_enabled")) != competition_enabled:
        raise ValueError(
            "SHUTrack checkpoint competition_enabled does not match model_config"
        )
    missing = [key for key in _REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"SHUTrack checkpoint is missing fields: {missing}")


def build_checkpoint_payload(
    *,
    model: ShutrackV2Model,
    training_config: Mapping[str, Any],
    dataset_descriptor: Mapping[str, Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    normalization_mean,
    normalization_std,
    include_optimizer: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_schema": feature_schema_descriptor(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "context_size": CONTEXT_SIZE,
        "history_size": HISTORY_SIZE,
        "evidence_dim": EVIDENCE_DIM,
        "competition_dim": COMPETITION_DIM,
        "competition_enabled": bool(model.competition_enabled),
        "scalar_dim": LEGACY_SCALAR_DIM,
        "reid_dim": int(model.reid_dim),
        "model_config": model.config.as_dict(),
        "parameter_counts": model.parameter_report(),
        "model_state_dict": cpu_state_dict(model),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_metric_name": "training_loss",
        "training_config": dict(training_config),
        "dataset": dict(dataset_descriptor),
        "normalization_mean": list(map(float, normalization_mean)),
        "normalization_std": list(map(float, normalization_std)),
        "training_commit": _git_value(["git", "rev-parse", "HEAD"], "unknown"),
        "training_git_status": _git_value(
            ["git", "status", "--porcelain"], "unavailable"
        ),
    }
    if include_optimizer:
        if optimizer is None:
            raise ValueError("optimizer is required for a resumable last checkpoint")
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = (
            scheduler.state_dict() if scheduler is not None else None
        )
    return payload


def save_checkpoint_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_shutrack_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ShutrackV2Model, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"SHUTrack checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    validate_checkpoint(checkpoint)
    model = model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(map_location if isinstance(map_location, torch.device) else map_location)
    model.eval()
    return model, dict(checkpoint)


def checkpoint_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "architecture": checkpoint.get("architecture"),
        "architecture_version": checkpoint.get("architecture_version"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_metric": checkpoint.get("best_metric"),
        "reid_dim": checkpoint.get("reid_dim"),
        "model_config": checkpoint.get("model_config"),
        "competition_enabled": checkpoint.get("competition_enabled"),
        "parameter_counts": checkpoint.get("parameter_counts"),
        "dataset": {
            key: checkpoint.get("dataset", {}).get(key)
            for key in (
                "dataset",
                "split",
                "dataset_dir",
                "competition_enabled",
                "feature_schema_version",
            )
        },
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "validate_checkpoint",
    "build_checkpoint_payload",
    "save_checkpoint_atomic",
    "load_shutrack_checkpoint",
    "model_from_config",
    "cpu_state_dict",
    "checkpoint_summary",
]
