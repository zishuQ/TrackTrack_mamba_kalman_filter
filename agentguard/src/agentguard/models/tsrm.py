from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DYN_SCALAR_INDICES = [
    2, 6, 9, 15, 18, 21, 24, 25,
    27, 28, 29, 30,
    39, 40, 41, 42,
    43, 44, 45, 46,
    55, 56, 57, 58, 59,
]


class CausalResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.left_padding = 2 * int(dilation)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=int(dilation),
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        residual = x
        convolved = self.conv(F.pad(x.transpose(1, 2), (self.left_padding, 0)))
        update = self.dropout(F.gelu(convolved.transpose(1, 2)))
        output = self.norm(residual + update)
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class TSRM(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 128,
        scalar_dim: int = 63,
        hidden_dim: int = 128,
        delta_max: float = 0.2,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.scalar_dim = int(scalar_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_max = float(delta_max)
        self.absolute_dim = self.event_dim + self.scalar_dim + 2 + 5 + 3 + 4

        self.abs_proj = nn.Linear(self.absolute_dim, hidden_dim)
        self.delta_proj = nn.Linear(self.absolute_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.tcn_blocks = nn.ModuleList(
            [CausalResidualTCNBlock(hidden_dim, dilation) for dilation in (1, 2, 4)]
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fusion_head = nn.Linear(hidden_dim * 2, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.strength_head = nn.Linear(hidden_dim, 1)
        self.dynamics_head = nn.Linear(hidden_dim, len(DYN_SCALAR_INDICES))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _reset_aware_gru(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.BoolTensor,
        reset_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, length, _ = tokens.shape
        hidden = tokens.new_zeros((1, batch_size, self.hidden_dim))
        outputs: list[torch.Tensor] = []
        for position in range(length):
            reset = reset_mask[:, position] | padding_mask[:, position]
            hidden = hidden.masked_fill(reset.view(1, batch_size, 1), 0.0)
            current, hidden = self.gru(tokens[:, position : position + 1], hidden)
            current = current[:, 0].masked_fill(
                padding_mask[:, position].unsqueeze(-1), 0.0
            )
            hidden = hidden.masked_fill(
                padding_mask[:, position].view(1, batch_size, 1), 0.0
            )
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        *,
        event_embedding: torch.Tensor,
        scalar_feats: torch.Tensor,
        base_gate: torch.Tensor,
        policy_probs: torch.Tensor,
        cue: torch.Tensor,
        risk: torch.Tensor,
        padding_mask: torch.BoolTensor,
        has_detection_mask: torch.BoolTensor,
        reset_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, length = padding_mask.shape
        if reset_mask is None:
            reset_mask = torch.zeros_like(padding_mask)
            first_valid = (~padding_mask).float().argmax(dim=1)
            reset_mask[
                torch.arange(batch_size, device=padding_mask.device), first_valid
            ] = True
        absolute = torch.cat(
            [event_embedding, scalar_feats, base_gate, policy_probs, cue, risk], dim=-1
        )
        if absolute.shape != (batch_size, length, self.absolute_dim):
            raise ValueError(
                f"TSRM absolute input must have shape {(batch_size, length, self.absolute_dim)}, "
                f"got {tuple(absolute.shape)}"
            )
        previous = F.pad(absolute[:, :-1], (0, 0, 1, 0))
        previous_padding = F.pad(padding_mask[:, :-1], (1, 0), value=True)
        delta_valid = ~padding_mask & ~previous_padding & ~reset_mask
        delta = (absolute - previous) * delta_valid.unsqueeze(-1)
        token = self.token_norm(self.abs_proj(absolute) + self.delta_proj(delta))
        token = token.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        short = token
        for block in self.tcn_blocks:
            short = block(short, padding_mask)
        long = self._reset_aware_gru(token, padding_mask, reset_mask)
        fusion_weight = torch.sigmoid(self.fusion_head(torch.cat([short, long], dim=-1)))
        memory = fusion_weight * short + (1.0 - fusion_weight) * long
        memory = memory.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        temporal_gate_delta = self.delta_max * torch.tanh(self.delta_head(memory))
        revision_strength = torch.sigmoid(self.strength_head(memory))
        temporal_gate_correction = revision_strength * temporal_gate_delta
        final_gate = torch.clamp(base_gate + temporal_gate_correction, 0.0, 1.0)
        final_gate = final_gate * has_detection_mask.unsqueeze(-1)
        return {
            "absolute_token": absolute,
            "short_memory": short,
            "long_memory": long,
            "fusion_weight": fusion_weight,
            "memory": memory,
            "temporal_gate_delta": temporal_gate_delta,
            "revision_strength": revision_strength,
            "temporal_gate_correction": temporal_gate_correction,
            "final_gate": final_gate,
            "predicted_next_scalar_delta": self.dynamics_head(memory),
        }
