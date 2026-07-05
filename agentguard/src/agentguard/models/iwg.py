import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.models.event_encoder import EventEncoder


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

        self.policy_head = nn.Sequential(nn.Linear(event_dim, 64), nn.Linear(64, 5))
        self.gate_residual_head = nn.Sequential(nn.Linear(event_dim, 64), nn.Linear(64, 2))
        self.event_head = nn.Sequential(nn.Linear(event_dim, 64), nn.Linear(64, 10))
        self.cue_head = nn.Sequential(nn.Linear(event_dim, 32), nn.Linear(32, 3))

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

        gate_residual = self.gate_residual_head(last_out)
        event_logits = self.event_head(last_out)
        cue = torch.sigmoid(self.cue_head(last_out))

        device = policy_probs.device
        prototype = torch.from_numpy(POLICY_PROTOTYPE_MATRIX).to(device=device, dtype=policy_probs.dtype)
        g_mix = policy_probs @ prototype
        g_final = torch.clamp(g_mix + 0.15 * torch.tanh(gate_residual), 0, 1)

        return {
            'policy_probs': policy_probs,
            'gate': g_final,
            'event_logits': event_logits,
            'cue': cue,
            'gate_residual': gate_residual,
        }
