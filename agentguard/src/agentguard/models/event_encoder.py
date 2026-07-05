import torch
import torch.nn as nn


class EventEncoder(nn.Module):
    """Encodes a single event into a 128-dim embedding."""

    def __init__(self, reid_dim: int):
        super().__init__()
        self.reid_dim = reid_dim

        self.reid_proj = nn.Linear(reid_dim, 64, bias=False)
        self.reid_norm = nn.LayerNorm(64)

        self.interaction_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

        self.scalar_norm = nn.LayerNorm(63)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(63, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

    def forward(self, track_feat: torch.Tensor, det_feat: torch.Tensor, scalar_feat: torch.Tensor) -> torch.Tensor:
        track_emb = self.reid_norm(self.reid_proj(track_feat))
        det_emb = self.reid_norm(self.reid_proj(det_feat))

        diff = torch.abs(track_emb - det_emb)
        prod = track_emb * det_emb
        interaction_in = torch.cat([track_emb, det_emb, diff, prod], dim=-1)
        interaction_out = self.interaction_mlp(interaction_in)

        scalar_out = self.scalar_mlp(self.scalar_norm(scalar_feat))

        out = torch.cat([interaction_out, scalar_out], dim=-1)
        out = self.fusion(out)
        return out
