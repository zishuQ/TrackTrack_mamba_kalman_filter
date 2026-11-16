"""End-to-end SHUTrack v2 model: shared encoders + HCE + RSC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from shutrack.models.encoders import (
    AppearanceEncoder,
    ContextEncoder,
    MotionEncoder,
)
from shutrack.models.hce import HCE
from shutrack.models.rsc import RSC
from shutrack.schema import (
    ARCHITECTURE_NAME,
    ARCHITECTURE_VERSION,
    CONTEXT_DIM,
    CONTEXT_SIZE,
    HISTORY_SIZE,
    MOTION_DIM,
)


@dataclass(frozen=True)
class ShutrackV2Config:
    reid_dim: int
    d_model: int = 128
    n_heads: int = 4
    hce_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0
    appearance_proj_dim: int = 64
    context_size: int = CONTEXT_SIZE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShutrackV2Model(nn.Module):
    """HCE builds historical contexts; RSC produces the two gates."""

    def __init__(self, config: ShutrackV2Config) -> None:
        super().__init__()
        if int(config.context_size) != CONTEXT_SIZE:
            raise ValueError(
                f"SHUTrack v2 requires context_size={CONTEXT_SIZE}, got "
                f"{config.context_size}"
            )
        self.config = config
        self.motion_encoder = MotionEncoder(
            d_model=config.d_model, ff_dim=config.ff_dim
        )
        self.context_encoder = ContextEncoder(
            d_model=config.d_model, ff_dim=config.ff_dim
        )
        self.appearance_encoder = AppearanceEncoder(
            config.reid_dim,
            proj_dim=config.appearance_proj_dim,
            d_model=config.d_model,
            ff_dim=config.ff_dim,
        )
        self.event_norm = nn.LayerNorm(config.d_model)
        self.hce = HCE(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.hce_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
        self.rsc = RSC(
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )

    @property
    def architecture_name(self) -> str:
        return ARCHITECTURE_NAME

    @property
    def architecture_version(self) -> str:
        return ARCHITECTURE_VERSION

    @property
    def reid_dim(self) -> int:
        return int(self.config.reid_dim)

    def forward(
        self,
        history_motion: torch.Tensor,
        history_track: torch.Tensor,
        history_detection: torch.Tensor,
        history_context: torch.Tensor,
        history_mask: torch.Tensor,
        history_has_detection: torch.Tensor,
        current_motion: torch.Tensor,
        current_track: torch.Tensor,
        current_detection: torch.Tensor,
        current_context: torch.Tensor,
        current_has_detection: torch.Tensor,
        history_reset: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = history_motion.shape[0]
        expected = {
            "history_motion": (batch_size, HISTORY_SIZE, MOTION_DIM),
            "history_track": (batch_size, HISTORY_SIZE, self.reid_dim),
            "history_detection": (batch_size, HISTORY_SIZE, self.reid_dim),
            "history_context": (batch_size, HISTORY_SIZE, CONTEXT_DIM),
            "history_mask": (batch_size, HISTORY_SIZE),
            "history_has_detection": (batch_size, HISTORY_SIZE),
            "current_motion": (batch_size, MOTION_DIM),
            "current_track": (batch_size, self.reid_dim),
            "current_detection": (batch_size, self.reid_dim),
            "current_context": (batch_size, CONTEXT_DIM),
        }
        given = {
            "history_motion": history_motion,
            "history_track": history_track,
            "history_detection": history_detection,
            "history_context": history_context,
            "history_mask": history_mask,
            "history_has_detection": history_has_detection,
            "current_motion": current_motion,
            "current_track": current_track,
            "current_detection": current_detection,
            "current_context": current_context,
        }
        for name, shape in expected.items():
            if tuple(given[name].shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(given[name].shape)}"
                )
        if current_has_detection.numel() != batch_size:
            raise ValueError(
                "current_has_detection must contain one flag per sample"
            )

        history_tokens = self.event_norm(
            self.appearance_encoder(
                history_track.reshape(-1, self.reid_dim),
                history_detection.reshape(-1, self.reid_dim),
                history_has_detection.reshape(-1),
            ).reshape(batch_size, HISTORY_SIZE, self.config.d_model)
            + self.motion_encoder(history_motion)
            + self.context_encoder(history_context)
        )
        history_motion_context, history_appearance_context = self.hce(
            history_tokens,
            history_mask,
            history_reset,
        )

        current_motion_token = self.motion_encoder(current_motion)
        current_appearance_token = self.appearance_encoder(
            current_track,
            current_detection,
            current_has_detection.bool(),
        )
        current_context_token = self.context_encoder(current_context)
        return self.rsc(
            history_motion_context,
            history_appearance_context,
            current_motion_token,
            current_appearance_token,
            current_context_token,
            current_has_detection,
        )

    def parameter_report(self) -> dict[str, int]:
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "hce": sum(p.numel() for p in self.hce.parameters()),
            "rsc": sum(p.numel() for p in self.rsc.parameters()),
            "encoders": sum(
                p.numel()
                for module in (
                    self.motion_encoder,
                    self.context_encoder,
                    self.appearance_encoder,
                )
                for p in module.parameters()
            ),
        }
        return {key: int(value) for key, value in counts.items()}

    def contract(self) -> dict[str, Any]:
        return {
            "architecture": ARCHITECTURE_NAME,
            "architecture_version": ARCHITECTURE_VERSION,
            "model_config": self.config.as_dict(),
            "parameter_counts": self.parameter_report(),
        }


__all__ = [
    "ShutrackV2Config",
    "ShutrackV2Model",
]
