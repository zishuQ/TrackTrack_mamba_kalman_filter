"""HCE — Historical Context Encoder.

HCE reads only the five events before the current one.  Each history event
fuses a motion+observation evidence token with an appearance token; a shared
temporal Transformer encodes the fused history.  Two learned queries read out
the motion/observation and appearance history contexts ``h_m`` and ``h_a``.  A
learned null token is always present in the key/value sequence, so an empty
history (cold start) or an all-padding window never produces an all-masked
attention row.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from shutrack.schema import HISTORY_SIZE


class HCE(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.history_size = HISTORY_SIZE
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.history_size + 1, self.d_model)
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.query_motion = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.query_appearance = nn.Parameter(torch.zeros(1, 1, self.d_model))
        for parameter in (
            self.position_embedding,
            self.null_token,
            self.query_motion,
            self.query_appearance,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

        self.read_motion = nn.MultiheadAttention(
            self.d_model, int(n_heads), dropout=float(dropout), batch_first=True
        )
        self.read_appearance = nn.MultiheadAttention(
            self.d_model, int(n_heads), dropout=float(dropout), batch_first=True
        )
        self.read_norm_motion = nn.LayerNorm(self.d_model)
        self.read_norm_appearance = nn.LayerNorm(self.d_model)

    def forward(
        self,
        history_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        history_reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_tokens.ndim != 3:
            raise ValueError("history_tokens must be [B, T, D]")
        batch_size, seq_len, dim = history_tokens.shape
        if seq_len != self.history_size or dim != self.d_model:
            raise ValueError(
                f"HCE expected [B, {self.history_size}, {self.d_model}], got "
                f"{tuple(history_tokens.shape)}"
            )
        mask = history_mask.bool()
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"history_mask must be {(batch_size, seq_len)}, got {tuple(mask.shape)}"
            )
        if history_reset is not None:
            reset = history_reset.bool()
            if reset.shape != (batch_size, seq_len):
                raise ValueError(
                    "history_reset must match history_mask shape, got "
                    f"{tuple(reset.shape)}"
                )
            mask = mask | reset

        null = self.null_token.expand(batch_size, 1, self.d_model)
        tokens = torch.cat([null, history_tokens], dim=1)
        tokens = tokens + self.position_embedding
        null_key_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=tokens.device
        )
        key_padding_mask = torch.cat([null_key_mask, mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        query_motion = self.query_motion.expand(batch_size, 1, self.d_model)
        query_appearance = self.query_appearance.expand(batch_size, 1, self.d_model)
        attended_motion, _ = self.read_motion(
            query_motion,
            encoded,
            encoded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_appearance, _ = self.read_appearance(
            query_appearance,
            encoded,
            encoded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h_motion = self.read_norm_motion(attended_motion + query_motion)
        h_appearance = self.read_norm_appearance(
            attended_appearance + query_appearance
        )
        return h_motion.squeeze(1), h_appearance.squeeze(1)


__all__ = ["HCE"]
