"""Runtime checkpoint loading for SHUTrack v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from shutrack.models.shutrack import ShutrackV2Model
from shutrack.training.checkpoint import load_shutrack_checkpoint


def load_runtime_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ShutrackV2Model, dict[str, Any]]:
    model, checkpoint = load_shutrack_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    if not isinstance(checkpoint.get("competition_enabled"), bool):
        raise ValueError(
            "SHUTrack checkpoint is missing competition_enabled; retrain so "
            "online inference can obey the training branch configuration"
        )
    model.to(device)
    model.eval()
    mean = np.asarray(checkpoint.get("normalization_mean", []), dtype=np.float64)
    std = np.asarray(checkpoint.get("normalization_std", []), dtype=np.float64)
    if mean.shape != (63,) or std.shape != (63,):
        raise ValueError(
            "SHUTrack checkpoint is missing 63-dim normalization statistics; "
            "online inference cannot reproduce training features"
        )
    return model, dict(checkpoint)


__all__ = ["load_runtime_model"]
