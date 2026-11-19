"""End-to-end SHUTrack v3 model: shared encoders + HCE + RSC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from shutrack.models.encoders import (
    AppearanceEncoder,
    CompetitionEncoder,
    EvidenceEncoder,
)
from shutrack.models.hce import HCE
from shutrack.models.rsc import RSC
from shutrack.schema import (
    ARCHITECTURE_NAME,
    ARCHITECTURE_VERSION,
    COMPETITION_DIM,
    CONTEXT_SIZE,
    EVIDENCE_DIM,
    HISTORY_SIZE,
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
    competition_enabled: bool = True
    appearance_match_input: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShutrackV2Model(nn.Module):
    """HCE builds historical contexts; RSC produces the two gates."""

    def __init__(self, config: ShutrackV2Config) -> None:
        super().__init__()
        if int(config.context_size) != CONTEXT_SIZE:
            raise ValueError(
                f"SHUTrack requires context_size={CONTEXT_SIZE}, got "
                f"{config.context_size}"
            )
        self.config = config
        self.evidence_encoder = EvidenceEncoder(
            d_model=config.d_model, ff_dim=config.ff_dim
        )
        self.appearance_encoder = AppearanceEncoder(
            config.reid_dim,
            proj_dim=config.appearance_proj_dim,
            d_model=config.d_model,
            ff_dim=config.ff_dim,
        )
        self.competition_encoder = (
            CompetitionEncoder(d_model=config.d_model, ff_dim=config.ff_dim)
            if config.competition_enabled
            else None
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
            competition_enabled=config.competition_enabled,
            appearance_match_input=config.appearance_match_input,
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

    @property
    def competition_enabled(self) -> bool:
        return bool(self.config.competition_enabled)

    def forward(
        self,
        history_evidence: torch.Tensor,
        history_track: torch.Tensor,
        history_detection: torch.Tensor,
        history_mask: torch.Tensor,
        history_has_detection: torch.Tensor,
        current_evidence: torch.Tensor,
        current_track: torch.Tensor,
        current_detection: torch.Tensor,
        current_competition: torch.Tensor,
        current_has_detection: torch.Tensor,
        history_reset: torch.Tensor | None = None,
        current_match_distance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = history_evidence.shape[0]
        expected = {
            "history_evidence": (batch_size, HISTORY_SIZE, EVIDENCE_DIM),
            "history_track": (batch_size, HISTORY_SIZE, self.reid_dim),
            "history_detection": (batch_size, HISTORY_SIZE, self.reid_dim),
            "history_mask": (batch_size, HISTORY_SIZE),
            "history_has_detection": (batch_size, HISTORY_SIZE),
            "current_evidence": (batch_size, EVIDENCE_DIM),
            "current_track": (batch_size, self.reid_dim),
            "current_detection": (batch_size, self.reid_dim),
            "current_competition": (batch_size, COMPETITION_DIM),
        }
        given = {
            "history_evidence": history_evidence,
            "history_track": history_track,
            "history_detection": history_detection,
            "history_mask": history_mask,
            "history_has_detection": history_has_detection,
            "current_evidence": current_evidence,
            "current_track": current_track,
            "current_detection": current_detection,
            "current_competition": current_competition,
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
            + self.evidence_encoder(history_evidence)
        )
        history_evidence_context, history_appearance_context = self.hce(
            history_tokens,
            history_mask,
            history_reset,
        )

        current_evidence_token = self.evidence_encoder(current_evidence)
        current_appearance_token = self.appearance_encoder(
            current_track,
            current_detection,
            current_has_detection.bool(),
        )
        current_competition_token = None
        if self.competition_enabled:
            assert self.competition_encoder is not None
            current_competition_token = self.competition_encoder(
                current_competition
            )
        if self.config.appearance_match_input:
            if current_match_distance is None or current_match_distance.shape != (
                batch_size,
            ):
                raise ValueError(
                    "appearance_match_input model requires current_match_distance "
                    f"with shape ({batch_size},), got "
                    f"{None if current_match_distance is None else tuple(current_match_distance.shape)}"
                )
        else:
            current_match_distance = None
        return self.rsc(
            history_evidence_context,
            history_appearance_context,
            current_evidence_token,
            current_appearance_token,
            current_competition_token,
            current_has_detection,
            current_match_distance=current_match_distance,
        )

    def parameter_report(self) -> dict[str, int]:
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "hce": sum(p.numel() for p in self.hce.parameters()),
            "rsc": sum(p.numel() for p in self.rsc.parameters()),
            "evidence_encoder": sum(
                p.numel() for p in self.evidence_encoder.parameters()
            ),
            "appearance_encoder": sum(
                p.numel() for p in self.appearance_encoder.parameters()
            ),
        }
        if self.competition_encoder is not None:
            counts["competition_encoder"] = sum(
                p.numel() for p in self.competition_encoder.parameters()
            )
        else:
            counts["competition_encoder"] = 0
        counts["encoders"] = (
            counts["evidence_encoder"]
            + counts["appearance_encoder"]
            + counts["competition_encoder"]
        )
        return {key: int(value) for key, value in counts.items()}

    def contract(self) -> dict[str, Any]:
        return {
            "architecture": ARCHITECTURE_NAME,
            "architecture_version": ARCHITECTURE_VERSION,
            "model_config": self.config.as_dict(),
            "competition_enabled": self.competition_enabled,
            "parameter_counts": self.parameter_report(),
        }


__all__ = [
    "ShutrackV2Config",
    "ShutrackV2Model",
]
