"""Query-conditioned HCE memory and competition-modulated bilinear RSC.

Inputs and outputs match the existing SHUTrack contract. Future supervision
(rank/column/GT) is never a model input. Motion and appearance keep independent
queries and sigmoid scores; attention is shared across the two queries.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from shutrack.schema import COMPETITION_DIM, HISTORY_SIZE


class QueryHCE(nn.Module):
    """Keep all temporal events, then read them using RSC's current queries."""

    def __init__(self, *, d_model=128, n_heads=4, n_layers=2,
                 ff_dim=256, dropout=0.0):
        super().__init__()
        self.d_model = int(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.position_embedding = nn.Parameter(torch.zeros(1, HISTORY_SIZE + 1, d_model))
        self.null_memory = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.null_memory, std=0.02)
        self.read_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.read_norm = nn.LayerNorm(d_model)
        self.read_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.read_ffn_norm = nn.LayerNorm(d_model)

    def encode(self, history_events, history_mask, history_reset=None):
        batch_size = history_events.shape[0]
        if tuple(history_events.shape) != (batch_size, HISTORY_SIZE, self.d_model):
            raise ValueError("history_events must be [B, 5, d_model]")
        if tuple(history_mask.shape) != (batch_size, HISTORY_SIZE):
            raise ValueError("history_mask must be [B, 5]")
        mask = history_mask.bool()
        if history_reset is not None:
            if history_reset.shape != history_mask.shape:
                raise ValueError("history_reset must match history_mask shape")
            mask = mask | history_reset.bool()
        # Erase masked content before attention as well as masking its keys.
        history_events = history_events.masked_fill(mask.unsqueeze(-1), 0.0)
        events = torch.cat(
            [self.null_memory.expand(batch_size, -1, -1), history_events], dim=1,
        ) + self.position_embedding
        null_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=events.device)
        padding_mask = torch.cat([null_mask, mask], dim=1)
        memory = self.encoder(events, src_key_padding_mask=padding_mask)
        return memory, padding_mask

    def read(self, memory, padding_mask, queries):
        attended, _ = self.read_attention(
            queries, memory, memory, key_padding_mask=padding_mask, need_weights=False,
        )
        readout = self.read_norm(queries + attended)
        return self.read_ffn_norm(readout + self.read_ffn(readout))

    def forward(self, history_events, history_mask, queries, history_reset=None):
        memory, padding_mask = self.encode(history_events, history_mask, history_reset)
        return self.read(memory, padding_mask, queries)


class BilinearGate(nn.Module):
    """Score current-query/history agreement plus their separate linear terms."""

    def __init__(self, d_model):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.history_norm = nn.LayerNorm(d_model)
        self.interaction = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.interaction)
        self.linear = nn.Linear(2 * d_model, 1)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, query, history):
        query = self.query_norm(query)
        history = self.history_norm(history)
        # No [B,D,D] outer-product tensor is materialized.
        joint = ((query @ self.interaction) * history).sum(-1, keepdim=True)
        return joint * self.scale + self.linear(torch.cat([query, history], dim=-1))


class QueryRSC(nn.Module):
    """Generate the queries, modulate them with competition, score both gates."""

    def __init__(self, *, d_model=128, competition_enabled=True):
        super().__init__()
        self.competition_enabled = bool(competition_enabled)
        self.base_queries = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
            for _ in range(2)
        ])
        if self.competition_enabled:
            # These three scalars have already been normalized by the existing
            # input pipeline. Do not normalize them a second time here.
            self.condition_encoder = nn.Sequential(nn.Linear(COMPETITION_DIM, 32), nn.GELU())
            self.modulations = nn.ModuleList([nn.Linear(32, 2 * d_model) for _ in range(2)])
            for modulation in self.modulations:
                nn.init.zeros_(modulation.weight)
                nn.init.zeros_(modulation.bias)
        else:
            self.condition_encoder = None
            self.modulations = nn.ModuleList()
        self.gate_heads = nn.ModuleList([BilinearGate(d_model) for _ in range(2)])

    def make_queries(self, current_event, current_competition):
        queries = [projection(current_event) for projection in self.base_queries]
        if self.condition_encoder is not None:
            condition = self.condition_encoder(current_competition.float())
            for index, modulation in enumerate(self.modulations):
                gamma, beta = modulation(condition).chunk(2, dim=-1)
                queries[index] = (1.0 + gamma) * queries[index] + beta
        return torch.stack(queries, dim=1)

    def forward(self, queries, history_readouts):
        logits = torch.cat([
            head(queries[:, index], history_readouts[:, index])
            for index, head in enumerate(self.gate_heads)
        ], dim=-1)
        return {"gate_logits": logits, "gates": torch.sigmoid(logits)}
