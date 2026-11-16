"""RSC — Reliability-aware Suppression Controller.

RSC uses the two HCE history contexts as queries and reads the current
motion+observation token, the appearance token and (optionally) the
association-competition token as key/value.  One cross-modal attention layer
produces two representation deltas; the deltas are fused residually in
representation space and decoded by two independent sigmoid heads.  There is
no softmax coupling, no base/refined split, no policy, cue, risk or correction
head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RSC(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
        competition_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.competition_enabled = bool(competition_enabled)
        self.token_count = 3 if self.competition_enabled else 2
        self.current_modality_embedding = nn.Parameter(
            torch.zeros(1, self.token_count, self.d_model)
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
        self.delta_evidence = nn.Linear(self.d_model, self.d_model)
        self.delta_appearance = nn.Linear(self.d_model, self.d_model)
        self.residual_evidence = nn.Linear(self.d_model, self.d_model)
        self.residual_appearance = nn.Linear(self.d_model, self.d_model)
        for projection in (self.residual_evidence, self.residual_appearance):
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(projection.bias)
        self.evidence_head = nn.Sequential(
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
        history_evidence: torch.Tensor,
        history_appearance: torch.Tensor,
        current_evidence_token: torch.Tensor,
        current_appearance_token: torch.Tensor,
        current_competition_token: torch.Tensor | None,
        current_has_detection: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_evidence.shape != history_appearance.shape:
            raise ValueError(
                "history evidence/appearance contexts must share a shape, got "
                f"{tuple(history_evidence.shape)} and "
                f"{tuple(history_appearance.shape)}"
            )
        if history_evidence.ndim != 2 or history_evidence.shape[1] != self.d_model:
            raise ValueError(
                f"history contexts must be [B, {self.d_model}], got "
                f"{tuple(history_evidence.shape)}"
            )
        batch_size = history_evidence.shape[0]
        if current_evidence_token.shape != (batch_size, self.d_model):
            raise ValueError(
                "current_evidence_token must be "
                f"{(batch_size, self.d_model)}, got "
                f"{tuple(current_evidence_token.shape)}"
            )
        if current_appearance_token.shape != (batch_size, self.d_model):
            raise ValueError(
                "current_appearance_token must be "
                f"{(batch_size, self.d_model)}, got "
                f"{tuple(current_appearance_token.shape)}"
            )
        if self.competition_enabled:
            if current_competition_token is None or current_competition_token.shape != (
                batch_size,
                self.d_model,
            ):
                raise ValueError(
                    "competition-enabled RSC requires a current competition token"
                )
        elif current_competition_token is not None:
            raise ValueError(
                "competition-disabled RSC must not receive a competition token"
            )
        has_detection = current_has_detection.bool().reshape(batch_size)

        queries = torch.stack([history_evidence, history_appearance], dim=1)
        tokens = [current_evidence_token, current_appearance_token]
        if self.competition_enabled:
            tokens.append(current_competition_token)
        key_value = torch.stack(tokens, dim=1) + self.current_modality_embedding
        key_padding_mask = torch.zeros(
            (batch_size, self.token_count),
            dtype=torch.bool,
            device=queries.device,
        )
        # Token order: evidence, appearance, [competition].
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

        delta_evidence = self.delta_evidence(fused[:, 0, :])
        delta_appearance = self.delta_appearance(fused[:, 1, :])
        z_evidence = history_evidence + self.residual_evidence(delta_evidence)
        z_appearance = history_appearance + self.residual_appearance(
            delta_appearance
        )
        gate_logits = torch.cat(
            [self.evidence_head(z_evidence), self.appearance_head(z_appearance)],
            dim=-1,
        )
        return {"gate_logits": gate_logits, "gates": torch.sigmoid(gate_logits)}


__all__ = ["RSC"]
