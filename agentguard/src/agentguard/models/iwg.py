import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _apply_heads(self, event_embedding: torch.Tensor) -> dict:
        policy_logits = self.policy_head(event_embedding)
        policy_probs = F.softmax(policy_logits, dim=-1)
        iwg_gate_residual = self.iwg_gate_residual_head(event_embedding)
        risk_logits = self.risk_head(event_embedding)
        cue_logits = self.cue_head(event_embedding)

        prototype = torch.as_tensor(
            POLICY_PROTOTYPE_MATRIX,
            device=policy_probs.device,
            dtype=policy_probs.dtype,
        )
        g_policy = policy_probs @ prototype
        base_gate = torch.clamp(
            g_policy + 0.15 * torch.tanh(iwg_gate_residual), 0.0, 1.0
        )
        return {
            "policy_logits": policy_logits,
            "policy_probs": policy_probs,
            "base_gate": base_gate,
            "gate": base_gate,
            "event_embedding": event_embedding,
            "cue_logits": cue_logits,
            "cue": torch.sigmoid(cue_logits),
            "risk_logits": risk_logits,
            "risk": torch.sigmoid(risk_logits),
            "iwg_gate_residual": iwg_gate_residual,
        }

    def _encode_inputs(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        mask: torch.BoolTensor | None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        batch_size, seq_len, _ = track_feats.shape
        if seq_len < 1:
            raise ValueError("IWG sequence length must be positive")
        if (
            det_feats.shape[:2] != (batch_size, seq_len)
            or scalar_feats.shape[:2] != (batch_size, seq_len)
        ):
            raise ValueError(
                "IWG input tensors must share batch and sequence dimensions"
            )
        if mask is None:
            mask = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=track_feats.device,
            )
        elif mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
            )

        track_flat = track_feats.reshape(-1, self.encoder.reid_dim)
        det_flat = det_feats.reshape(-1, self.encoder.reid_dim)
        scalar_flat = scalar_feats.reshape(-1, scalar_feats.shape[-1])
        event_embs = self.encoder(track_flat, det_flat, scalar_flat)
        return event_embs.reshape(batch_size, seq_len, -1), mask

    def _transform_windows(
        self,
        windows: torch.Tensor,
        window_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        flat_mask = window_mask.clone()
        # Attention returns NaN for an all-masked row. Such rows only
        # represent output positions that are themselves padding.
        all_padding = flat_mask.all(dim=-1)
        flat_mask[all_padding, -1] = False
        transformed = self.transformer(
            windows + self.position_embedding,
            src_key_padding_mask=flat_mask,
        )
        return transformed[:, -1, :]

    def forward_sequence(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> dict:
        """Evaluate every causal six-event window with one EventEncoder pass."""
        batch_size, seq_len, _ = track_feats.shape
        event_embs, mask = self._encode_inputs(
            track_feats, det_feats, scalar_feats, mask
        )

        left_embeddings = F.pad(event_embs, (0, 0, 5, 0))
        left_mask = F.pad(mask, (5, 0), value=True)
        windows = left_embeddings.unfold(1, 6, 1).permute(0, 1, 3, 2)
        window_mask = left_mask.unfold(1, 6, 1)
        flat_windows = windows.reshape(batch_size * seq_len, 6, -1)
        flat_mask = window_mask.reshape(batch_size * seq_len, 6)
        current = self._transform_windows(flat_windows, flat_mask).reshape(
            batch_size, seq_len, -1
        )
        return self._apply_heads(current)

    def forward(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> dict:
        batch_size, seq_len = track_feats.shape[:2]
        event_embs, normalized_mask = self._encode_inputs(
            track_feats, det_feats, scalar_feats, mask
        )
        if mask is None:
            last_valid = torch.full(
                (batch_size,), seq_len - 1, dtype=torch.long, device=track_feats.device
            )
        else:
            positions = torch.arange(seq_len, device=track_feats.device).expand(batch_size, -1)
            last_valid = positions.masked_fill(normalized_mask, -1).max(dim=-1).values
            if (last_valid < 0).any():
                raise ValueError("IWG forward requires at least one valid event per sample")

        left_embeddings = F.pad(event_embs, (0, 0, 5, 0))
        left_mask = F.pad(normalized_mask, (5, 0), value=True)
        offsets = torch.arange(6, device=track_feats.device).unsqueeze(0)
        gather_indices = last_valid.unsqueeze(1) + offsets
        windows = torch.gather(
            left_embeddings,
            1,
            gather_indices.unsqueeze(-1).expand(-1, -1, event_embs.shape[-1]),
        )
        window_mask = torch.gather(left_mask, 1, gather_indices)
        current = self._transform_windows(windows, window_mask)
        return self._apply_heads(current)
