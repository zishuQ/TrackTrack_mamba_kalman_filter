"""RSC — Reliability-aware Suppression Controller.

RSC uses the two HCE history contexts as queries and the current motion,
appearance and match-context tokens as key/value.  One cross-modal attention
layer produces two representation deltas; the deltas are fused residually in
representation space and decoded by two independent sigmoid heads.  There is
no softmax coupling, no base/refined split, no policy, cue, risk or correction
head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from shutrack.schema import CONTEXT_DIM, MOTION_DIM


class RSC(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.current_modality_embedding = nn.Parameter(
            torch.zeros(1, 3, self.d_model)
        )
        nn.init.normal_(self.current_modality_embedding, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            self.d_model, int(n_heads), dropout=float(dropout), batch_first=True
        )
        self.query_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.d_model),
            nn.Dropout(float(dropout)),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.delta_motion = nn.Linear(self.d_model, self.d_model)
        self.delta_appearance = nn.Linear(self.d_model, self.d_model)
        self.residual_motion = nn.Linear(self.d_model, self.d_model)
        self.residual_appearance = nn.Linear(self.d_model, self.d_model)
        for projection in (self.residual_motion, self.residual_appearance):
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(projection.bias)
        self.motion_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.appearance_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        history_motion: torch.Tensor,
        history_appearance: torch.Tensor,
        current_motion_token: torch.Tensor,
        current_appearance_token: torch.Tensor,
        current_context_token: torch.Tensor,
        current_has_detection: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_motion.shape != history_appearance.shape:
            raise ValueError(
                "history motion/appearance contexts must share a shape, got "
                f"{tuple(history_motion.shape)} and {tuple(history_appearance.shape)}"
            )
        if history_motion.ndim != 2 or history_motion.shape[1] != self.d_model:
            raise ValueError(
                f"history contexts must be [B, {self.d_model}], got "
                f"{tuple(history_motion.shape)}"
            )
        batch_size = history_motion.shape[0]
        for name, token in (
            ("current_motion_token", current_motion_token),
            ("current_appearance_token", current_appearance_token),
            ("current_context_token", current_context_token),
        ):
            if token.shape != (batch_size, self.d_model):
                raise ValueError(
                    f"{name} must be {(batch_size, self.d_model)}, got "
                    f"{tuple(token.shape)}"
                )
        has_detection = current_has_detection.bool().reshape(batch_size)

        queries = torch.stack([history_motion, history_appearance], dim=1)
        key_value = torch.stack(
            [current_motion_token, current_appearance_token, current_context_token],
            dim=1,
        ) + self.current_modality_embedding
        key_padding_mask = torch.zeros(
            (batch_size, 3), dtype=torch.bool, device=queries.device
        )
        key_padding_mask[:, 1] = ~has_detection

        attended, _ = self.cross_attention(
            queries,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.query_norm(queries + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))

        delta_motion = self.delta_motion(fused[:, 0, :])
        delta_appearance = self.delta_appearance(fused[:, 1, :])
        z_motion = history_motion + self.residual_motion(delta_motion)
        z_appearance = history_appearance + self.residual_appearance(
            delta_appearance
        )
        gate_logits = torch.cat(
            [self.motion_head(z_motion), self.appearance_head(z_appearance)],
            dim=-1,
        )
        return {"gate_logits": gate_logits, "gates": torch.sigmoid(gate_logits)}


__all__ = ["RSC"]
