from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    metadata: Dict[str, Any],
    path: str,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> str:
    """Save a full training checkpoint to disk.

    The checkpoint includes:

    - Model state dict
    - Optimiser state dict (if provided)
    - Scheduler state dict (if provided)
    - Current epoch
    - Arbitrary user metadata (config, metrics, prototypes, …)

    Parameters
    ----------
    model : nn.Module
        Model whose ``state_dict`` will be saved.
    optimizer : Optimizer or None
        Optimiser state.
    scheduler : LRScheduler or None
        Scheduler state.
    epoch : int
        Current epoch number (0-indexed).
    metadata : dict
        Additional data to include (config, dataset info, metrics, etc.).
    path : str
        Destination file path.

    Returns
    -------
    str
        The absolute path of the saved checkpoint.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "reid_dim": metadata.get("reid_dim", metadata.get("config", {}).get("reid_dim")),
        "scalar_dim": metadata.get("scalar_dim", metadata.get("config", {}).get("scalar_dim", 63)),
        "event_dim": metadata.get("event_dim", metadata.get("config", {}).get("event_dim", 128)),
        "policy_prototypes": metadata.get("policy_prototypes", POLICY_PROTOTYPE_MATRIX),
        "feature_schema_sha256": metadata.get(
            "feature_schema_sha256",
            metadata.get("config", {}).get("feature_schema_sha256", FEATURE_SCHEMA_SHA256),
        ),
        "cache_schema_version": metadata.get(
            "cache_schema_version",
            metadata.get("config", {}).get(
                "cache_schema_version",
                COMPACT_CACHE_SCHEMA_VERSION,
            ),
        ),
        "training_commit": metadata.get(
            "training_commit",
            metadata.get("config", {}).get("training_commit", ""),
        ),
        "dataset_metadata_path": metadata.get(
            "dataset_metadata_path",
            metadata.get("config", {}).get("dataset_metadata_path", ""),
        ),
        "dataset_metadata_sha256": metadata.get(
            "dataset_metadata_sha256",
            metadata.get("config", {}).get("dataset_metadata_sha256", ""),
        ),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if norm_mean is not None:
        checkpoint["normalization_mean"] = norm_mean
    if norm_std is not None:
        checkpoint["normalization_std"] = norm_std

    torch.save(checkpoint, path)
    return os.path.abspath(path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    model : nn.Module
        Model into which the state dict will be loaded.
    optimizer : Optimizer or None
        If provided, its state dict will be restored.
    scheduler : LRScheduler or None
        If provided, its state dict will be restored.

    Returns
    -------
    dict
        The metadata dict stored in the checkpoint, containing at least
        ``epoch`` and any keys saved in ``metadata``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If the checkpoint cannot be loaded (e.g. key mismatch).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = next(model.parameters()).device

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e

    # Load model state.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimiser state.
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state.
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return metadata (including epoch).
    metadata: Dict[str, Any] = {
        "epoch": checkpoint.get("epoch", -1),
        **checkpoint.get("metadata", {}),
    }
    return metadata
