import torch
import torch.nn as nn

from agentguard.models.event_encoder import EventEncoder


class TGR(nn.Module):
    """Temporal Gate Reviser."""

    def __init__(self, reid_dim: int, event_dim: int = 128):
        super().__init__()
        self.encoder = EventEncoder(reid_dim)

        self.policy_proj = nn.Sequential(nn.Linear(5, 16), nn.GELU())
        self.gate_proj = nn.Sequential(nn.Linear(2, 16), nn.GELU())

        self.pre_fusion = nn.Sequential(
            nn.Linear(event_dim + 16 + 16, 128),
            nn.LayerNorm(128),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.residual_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2),
        )

        self.position_embedding = nn.Parameter(torch.zeros(1, 4, event_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, track_feats: torch.Tensor, det_feats: torch.Tensor,
                scalar_feats: torch.Tensor, iwg_policy_probs: torch.Tensor,
                iwg_gates: torch.Tensor, has_detection_mask: torch.BoolTensor,
                padding_mask: torch.BoolTensor = None):
        B, seq_len, _ = track_feats.shape

        track_flat = track_feats.view(-1, self.encoder.reid_dim)
        det_flat = det_feats.view(-1, self.encoder.reid_dim)
        scalar_flat = scalar_feats.view(-1, scalar_feats.shape[-1])
        event_embs = self.encoder(track_flat, det_flat, scalar_flat)
        event_embs = event_embs.view(B, seq_len, -1)

        if iwg_policy_probs is not None:
            policy_proj = self.policy_proj(iwg_policy_probs)
        else:
            policy_proj = torch.zeros(B, seq_len, 16, device=track_feats.device)

        gate_proj = self.gate_proj(iwg_gates)

        fused = torch.cat([event_embs, policy_proj, gate_proj], dim=-1)
        fused = self.pre_fusion(fused)
        fused = fused + self.position_embedding[:, :seq_len, :]

        if padding_mask is not None:
            trans_out = self.transformer(fused, src_key_padding_mask=padding_mask)
        else:
            trans_out = self.transformer(fused)

        gate_residual = self.residual_head(trans_out)

        g_revised = torch.clamp(iwg_gates + 0.5 * torch.tanh(gate_residual), 0, 1)
        g_revised = g_revised * has_detection_mask.unsqueeze(-1).float()

        return g_revised, gate_residual
