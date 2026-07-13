import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.models.event_encoder import EventEncoder


def make_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )


class IWG(nn.Module):
    """Instance-wise Gate network."""

    def __init__(self, reid_dim: int, scalar_dim: int = 63, event_dim: int = 128):
        super().__init__()
        self.encoder = EventEncoder(reid_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=event_dim, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.policy_head = make_head(event_dim, 64, 5)
        self.iwg_gate_residual_head = make_head(event_dim, 64, 2)
        self.risk_head = make_head(event_dim, 64, 4)
        self.cue_head = make_head(event_dim, 32, 3)

        self.position_embedding = nn.Parameter(torch.zeros(1, 6, event_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, track_feats: torch.Tensor, det_feats: torch.Tensor,
                scalar_feats: torch.Tensor, mask: torch.BoolTensor = None) -> dict:
        B, seq_len, _ = track_feats.shape

        track_flat = track_feats.view(-1, self.encoder.reid_dim)
        det_flat = det_feats.view(-1, self.encoder.reid_dim)
        scalar_flat = scalar_feats.view(-1, scalar_feats.shape[-1])
        event_embs = self.encoder(track_flat, det_flat, scalar_flat)
        event_embs = event_embs.view(B, seq_len, -1)

        x = event_embs + self.position_embedding[:, :seq_len, :]
        if mask is not None:
            trans_out = self.transformer(x, src_key_padding_mask=mask)
        else:
            trans_out = self.transformer(x)

        # Use position -1 (last position, guaranteed non-padded for left-padded sequences)
        last_out = trans_out[:, -1, :]

        policy_logits = self.policy_head(last_out)
        policy_probs = F.softmax(policy_logits, dim=-1)

        iwg_gate_residual = self.iwg_gate_residual_head(last_out)
        risk_logits = self.risk_head(last_out)
        cue_logits = self.cue_head(last_out)
        cue = torch.sigmoid(cue_logits)

        device = policy_probs.device
        prototype = torch.from_numpy(POLICY_PROTOTYPE_MATRIX).to(device=device, dtype=policy_probs.dtype)
        g_mix = policy_probs @ prototype
        base_gate = torch.clamp(
            g_mix + 0.15 * torch.tanh(iwg_gate_residual), 0, 1
        )

        return {
            'policy_logits': policy_logits,
            'policy_probs': policy_probs,
            'base_gate': base_gate,
            'gate': base_gate,
            'event_embedding': last_out,
            'cue_logits': cue_logits,
            'risk_logits': risk_logits,
            'risk': torch.sigmoid(risk_logits),
            'cue': cue,
            'iwg_gate_residual': iwg_gate_residual,
        }
