"""Runtime checkpoint loading for SHUTrack v2."""

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
    model.to(device)
    model.eval()
    mean = np.asarray(checkpoint.get("normalization_mean", []), dtype=np.float64)
    std = np.asarray(checkpoint.get("normalization_std", []), dtype=np.float64)
    if mean.shape != (63,) or std.shape != (63,):
        raise ValueError(
            "SHUTrack checkpoint is missing 63-dim normalization statistics; "
            "online inference cannot reproduce training features"
        )
    margin = checkpoint.get("margin", {})
    if not isinstance(margin, dict) or "stats" not in margin:
        raise ValueError("SHUTrack checkpoint is missing margin metadata")
    return model, dict(checkpoint)


__all__ = ["load_runtime_model"]
