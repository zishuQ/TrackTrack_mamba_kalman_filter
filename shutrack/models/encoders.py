"""Structured event encoders shared by history (HCE) and current (RSC) tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from shutrack.schema import EVIDENCE_DIM


def _mlp(in_dim: int, d_model: int, ff_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
    )


class EvidenceEncoder(nn.Module):
    """Encode the 13-dim motion+observation evidence into a d_model token."""

    def __init__(self, d_model: int = 128, ff_dim: int = 256) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(EVIDENCE_DIM)
        self.net = _mlp(EVIDENCE_DIM, d_model, ff_dim)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(evidence.float()))


class AppearanceEncoder(nn.Module):
    """Project and interact the pre-update track ReID and detection ReID.

    The projection is shared between the two vectors, then the four-way
    interaction ``concat(track, det, |track-det|, track*det)`` is mapped to a
    d_model token.  Missing detections use a learned placeholder and remain
    explicitly masked at the caller (the zero ReID stays zero).
    """

    def __init__(
        self,
        reid_dim: int,
        *,
        proj_dim: int = 64,
        d_model: int = 128,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.reid_dim = int(reid_dim)
        self.proj_dim = int(proj_dim)
        self.reid_proj = nn.Linear(self.reid_dim, self.proj_dim, bias=False)
        self.reid_norm = nn.LayerNorm(self.proj_dim)
        self.missing_detection = nn.Parameter(torch.zeros(1, self.proj_dim))
        self.missing_track = nn.Parameter(torch.zeros(1, self.proj_dim))
        nn.init.normal_(self.missing_detection, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_track, mean=0.0, std=0.02)
        self.interaction = nn.Sequential(
            nn.Linear(4 * self.proj_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        track_feature: torch.Tensor,
        detection_feature: torch.Tensor,
        has_detection: torch.Tensor,
    ) -> torch.Tensor:
        track = self.reid_norm(self.reid_proj(track_feature.float()))
        detection = self.reid_norm(self.reid_proj(detection_feature.float()))
        has_detection = has_detection.bool().unsqueeze(-1)
        detection = torch.where(has_detection, detection, self.missing_detection)
        track_missing = track_feature.abs().sum(dim=-1, keepdim=True) <= 0
        track = torch.where(track_missing, self.missing_track, track)
        interaction = torch.cat(
            [track, detection, (track - detection).abs(), track * detection],
            dim=-1,
        )
        return self.interaction(interaction)


__all__ = [
    "EvidenceEncoder",
    "CompetitionEncoder",
    "AppearanceEncoder",
]
