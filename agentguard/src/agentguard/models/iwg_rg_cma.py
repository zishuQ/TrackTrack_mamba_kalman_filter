from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.models.iwg import IWG, make_head


IWG_CONTEXT_SIZE = 6
SUPPORTED_IWG_CONTEXT_SIZES = frozenset({6, 8})
RG_CMA_DIM = 128
RG_CMA_HEADS = 4
RG_CMA_LEGACY_CORRECTION_BOUND = 0.05
RG_CMA_CORRECTION_BOUND = 0.10
RELIABILITY_MODE_FULL = "full"
RELIABILITY_MODE_NO_SCALAR = "no-scalar"
SUPPORTED_RELIABILITY_MODES = frozenset(
    {RELIABILITY_MODE_FULL, RELIABILITY_MODE_NO_SCALAR}
)


def _model_descriptor(
    name: str,
    correction_bound: float,
    context_size: int = IWG_CONTEXT_SIZE,
) -> dict[str, Any]:
    return {
        "name": name,
        "iwg_context_size": int(context_size),
        "modal_token_dim": 64,
        "attention_dim": RG_CMA_DIM,
        "attention_heads": RG_CMA_HEADS,
        "attention_dropout": 0.1,
        "cross_modal_tokens": ["motion", "appearance", "reliability"],
        "correction_bound": correction_bound,
        "base_heads": "linear_linear",
        "gradient_contract": "safe_direct_base_detached_rg_cma",
    }


def _descriptor_sha256(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


IWG_RG_CMA_LEGACY_MODEL_SCHEMA = "agentguard_iwg_rg_cma_v1"
IWG_RG_CMA_LEGACY_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_LEGACY_MODEL_SCHEMA, RG_CMA_LEGACY_CORRECTION_BOUND
)
IWG_RG_CMA_LEGACY_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_LEGACY_MODEL_DESCRIPTOR
)

IWG_RG_CMA_MODEL_SCHEMA = "agentguard_iwg_rg_cma_v2"
IWG_RG_CMA_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_MODEL_SCHEMA, RG_CMA_CORRECTION_BOUND
)
IWG_RG_CMA_MODEL_SCHEMA_SHA256 = _descriptor_sha256(IWG_RG_CMA_MODEL_DESCRIPTOR)

IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA = "agentguard_iwg_rg_cma_v3"
IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    context_size=8,
)
IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_DESCRIPTOR
)

IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA = "agentguard_iwg_rg_cma_v4"
IWG_RG_CMA_CONTEXT8_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA,
    RG_CMA_CORRECTION_BOUND,
    context_size=8,
)
IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_CONTEXT8_MODEL_DESCRIPTOR
)


def _linear_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, out_dim))


def _validate_sequence_masks(
    padding_mask: torch.BoolTensor,
    reset_mask: torch.BoolTensor | None,
) -> torch.BoolTensor:
    if reset_mask is None:
        return torch.zeros_like(padding_mask)
    if reset_mask.shape != padding_mask.shape:
        raise ValueError(
            "reset_mask must match padding_mask: "
            f"{tuple(reset_mask.shape)} != {tuple(padding_mask.shape)}"
        )
    return reset_mask.bool() & ~padding_mask


def _mask_before_last_reset(
    window_mask: torch.BoolTensor,
    reset_windows: torch.BoolTensor,
) -> torch.BoolTensor:
    positions = torch.arange(
        window_mask.shape[-1], device=window_mask.device
    ).view(*([1] * (window_mask.ndim - 1)), -1)
    last_reset = positions.masked_fill(~reset_windows, -1).max(dim=-1).values
    return window_mask | (positions < last_reset.unsqueeze(-1))


def _causal_windows(
    values: torch.Tensor,
    padding_mask: torch.BoolTensor,
    reset_mask: torch.BoolTensor | None,
    context_size: int,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Build left-padded causal windows without crossing reset points."""
    context_size = int(context_size)
    if context_size < 1:
        raise ValueError(f"context_size must be positive, got {context_size}")
    if values.shape[:2] != padding_mask.shape:
        raise ValueError("values and padding_mask must share batch/sequence dimensions")
    reset = _validate_sequence_masks(padding_mask, reset_mask)
    left = context_size - 1
    left_values = F.pad(values, (0, 0, left, 0))
    left_padding = F.pad(
        padding_mask, (left, 0), value=True
    )
    left_reset = F.pad(reset, (left, 0), value=False)
    windows = left_values.unfold(1, context_size, 1).permute(0, 1, 3, 2)
    window_mask = left_padding.unfold(1, context_size, 1)
    reset_windows = left_reset.unfold(1, context_size, 1)
    return windows, _mask_before_last_reset(window_mask, reset_windows)


def _last_valid_indices(mask: torch.BoolTensor) -> torch.LongTensor:
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(mask, -1).max(dim=-1).values
    if (indices < 0).any():
        raise ValueError("at least one valid event is required per sample")
    return indices


def _gather_position(value: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    batch = torch.arange(value.shape[0], device=value.device)
    return value[batch, indices]


class SafeDirectIWG(IWG):
    """IWG with the verified linear gate heads and isolated auxiliary heads."""

    def __init__(
        self,
        reid_dim: int,
        scalar_dim: int = 63,
        event_dim: int = 128,
        context_size: int = IWG_CONTEXT_SIZE,
    ):
        super().__init__(
            reid_dim=reid_dim,
            scalar_dim=scalar_dim,
            event_dim=event_dim,
            context_size=context_size,
        )
        self.policy_head = _linear_head(event_dim, 64, 5)
        self.iwg_gate_residual_head = _linear_head(event_dim, 64, 2)

    def _encode_modal_inputs(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        mask: torch.BoolTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        batch_size, seq_len, _ = track_feats.shape
        if seq_len < 1:
            raise ValueError("SafeDirectIWG sequence length must be positive")
        if (
            det_feats.shape[:2] != (batch_size, seq_len)
            or scalar_feats.shape[:2] != (batch_size, seq_len)
        ):
            raise ValueError("IWG inputs must share batch and sequence dimensions")
        if mask is None:
            normalized_mask = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool, device=track_feats.device
            )
        else:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
                )
            normalized_mask = mask.bool()

        fused, appearance, motion = self.encoder.forward_with_modal_tokens(
            track_feats.reshape(-1, self.encoder.reid_dim),
            det_feats.reshape(-1, self.encoder.reid_dim),
            scalar_feats.reshape(-1, scalar_feats.shape[-1]),
        )
        return (
            fused.reshape(batch_size, seq_len, -1),
            appearance.reshape(batch_size, seq_len, -1),
            motion.reshape(batch_size, seq_len, -1),
            normalized_mask,
        )

    def _apply_safe_heads(self, event_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        policy_logits = self.policy_head(event_embedding)
        policy_probs = F.softmax(policy_logits, dim=-1)
        residual = self.iwg_gate_residual_head(event_embedding)
        detached = event_embedding.detach()
        cue_logits = self.cue_head(detached)
        risk_logits = self.risk_head(detached)
        prototype = torch.as_tensor(
            POLICY_PROTOTYPE_MATRIX,
            device=policy_probs.device,
            dtype=policy_probs.dtype,
        )
        base_gate = torch.clamp(
            policy_probs @ prototype + 0.15 * torch.tanh(residual), 0.0, 1.0
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
            "iwg_gate_residual": residual,
        }

    def forward_sequence(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        reset_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        event, appearance, motion, padding = self._encode_modal_inputs(
            track_feats, det_feats, scalar_feats, mask
        )
        windows, window_mask = _causal_windows(
            event, padding, reset_mask, self.context_size
        )
        batch_size, seq_len = padding.shape
        current = self._transform_windows(
            windows.reshape(batch_size * seq_len, self.context_size, -1),
            window_mask.reshape(batch_size * seq_len, self.context_size),
        ).reshape(batch_size, seq_len, -1)
        outputs = self._apply_safe_heads(current)
        outputs.update(
            {
                "appearance_token": appearance,
                "motion_token": motion,
                "padding_mask": padding,
            }
        )
        return outputs

    def forward(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        reset_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        event, appearance, motion, padding = self._encode_modal_inputs(
            track_feats, det_feats, scalar_feats, mask
        )
        indices = _last_valid_indices(padding)
        event_windows, window_mask = _causal_windows(
            event, padding, reset_mask, self.context_size
        )
        current = self._transform_windows(
            _gather_position(event_windows, indices),
            _gather_position(window_mask, indices),
        )
        outputs = self._apply_safe_heads(current)
        outputs["appearance_token"] = _gather_position(appearance, indices)
        outputs["motion_token"] = _gather_position(motion, indices)
        outputs["_appearance_history"] = appearance
        outputs["_motion_history"] = motion
        outputs["_history_padding_mask"] = padding
        return outputs


class RecordingTransformerEncoderLayer(nn.Module):
    """One TransformerEncoderLayer that also returns per-head attention."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=RG_CMA_DIM,
            nhead=RG_CMA_HEADS,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layer
        attended, weights = layer.self_attn(
            value,
            value,
            value,
            need_weights=True,
            average_attn_weights=False,
        )
        value = layer.norm1(value + layer.dropout1(attended))
        feed_forward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(value)))
        )
        value = layer.norm2(value + layer.dropout2(feed_forward))
        return value, weights


class IWGRGCMA(nn.Module):
    """Safe-direct IWG refined by causal reliability-guided cross-modal attention."""

    def __init__(
        self,
        reid_dim: int,
        *,
        scalar_dim: int = 63,
        event_dim: int = 128,
        correction_bound: float = RG_CMA_CORRECTION_BOUND,
        context_size: int = IWG_CONTEXT_SIZE,
        reliability_mode: str = RELIABILITY_MODE_FULL,
    ) -> None:
        super().__init__()
        correction_bound = float(correction_bound)
        context_size = int(context_size)
        if context_size not in SUPPORTED_IWG_CONTEXT_SIZES:
            raise ValueError(
                "unsupported RG-CMA context_size: "
                f"{context_size}; expected one of {sorted(SUPPORTED_IWG_CONTEXT_SIZES)}"
            )
        if not math.isfinite(correction_bound) or not 0.0 < correction_bound <= 1.0:
            raise ValueError(
                "RG-CMA correction bound must be finite and in (0, 1], got "
                f"{correction_bound}"
            )
        reliability_mode = str(reliability_mode).strip().lower()
        if reliability_mode not in SUPPORTED_RELIABILITY_MODES:
            raise ValueError(
                "unsupported RG-CMA reliability_mode: "
                f"{reliability_mode!r}; expected one of "
                f"{sorted(SUPPORTED_RELIABILITY_MODES)}"
            )
        self.scalar_dim = int(scalar_dim)
        self.correction_bound = correction_bound
        self.context_size = context_size
        self.reliability_mode = reliability_mode
        self.iwg = SafeDirectIWG(
            reid_dim=reid_dim,
            scalar_dim=scalar_dim,
            event_dim=event_dim,
            context_size=context_size,
        )
        self.motion_projection = nn.Linear(64, RG_CMA_DIM)
        self.appearance_projection = nn.Linear(64, RG_CMA_DIM)
        self.modality_embedding = nn.Parameter(torch.zeros(2, RG_CMA_DIM))
        self.relative_position_embedding = nn.Parameter(
            torch.zeros(context_size, RG_CMA_DIM)
        )
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.relative_position_embedding, mean=0.0, std=0.02)
        self.motion_temporal_attention = nn.MultiheadAttention(
            RG_CMA_DIM, RG_CMA_HEADS, dropout=0.1, batch_first=True
        )
        self.appearance_temporal_attention = nn.MultiheadAttention(
            RG_CMA_DIM, RG_CMA_HEADS, dropout=0.1, batch_first=True
        )
        reliability_dim = 2 + 5 + 3 + 4 + self.scalar_dim
        self.reliability_projection = nn.Linear(reliability_dim, RG_CMA_DIM)
        self.cross_modality_embedding = nn.Parameter(torch.zeros(3, RG_CMA_DIM))
        nn.init.normal_(self.cross_modality_embedding, mean=0.0, std=0.02)
        self.cross_modal_attention = RecordingTransformerEncoderLayer()
        self.motion_correction_head = make_head(RG_CMA_DIM, 64, 1)
        self.appearance_correction_head = make_head(RG_CMA_DIM, 64, 1)
        for head in (self.motion_correction_head, self.appearance_correction_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _apply_unmatched_sentinel(
        outputs: dict[str, torch.Tensor],
        has_detection_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        result = dict(outputs)
        no_detection = ~has_detection_mask.bool()
        result["base_gate"] = outputs["base_gate"].masked_fill(
            no_detection.unsqueeze(-1), 0.0
        )
        result["gate"] = result["base_gate"]
        hold_both = outputs["policy_probs"].new_tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0]
        )
        result["policy_probs"] = torch.where(
            no_detection.unsqueeze(-1), hold_both, outputs["policy_probs"]
        )
        result["cue"] = outputs["cue"].masked_fill(
            no_detection.unsqueeze(-1), 0.0
        )
        unmatched_risk = outputs["risk"].new_tensor([0.0, 0.0, 1.0, 0.0])
        result["risk"] = torch.where(
            no_detection.unsqueeze(-1), unmatched_risk, outputs["risk"]
        )
        return result

    @staticmethod
    def _safe_attention_mask(mask: torch.BoolTensor) -> torch.BoolTensor:
        safe = mask.clone()
        all_padding = safe.all(dim=-1)
        safe[all_padding, -1] = False
        return safe

    @staticmethod
    def _attention_entropy(weights: torch.Tensor) -> torch.Tensor:
        probabilities = weights.float().clamp(min=1e-8)
        return -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=-1)

    def _refine_flat(
        self,
        *,
        motion_windows: torch.Tensor,
        appearance_windows: torch.Tensor,
        window_mask: torch.BoolTensor,
        scalar_current: torch.Tensor,
        base_outputs: dict[str, torch.Tensor],
        has_detection: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        # Every producer outside RG-CMA is detached by construction.
        motion_windows = motion_windows.detach()
        appearance_windows = appearance_windows.detach()
        positions = self.relative_position_embedding.unsqueeze(0)
        motion_values = (
            self.motion_projection(motion_windows)
            + self.modality_embedding[0].view(1, 1, -1)
            + positions
        )
        appearance_values = (
            self.appearance_projection(appearance_windows)
            + self.modality_embedding[1].view(1, 1, -1)
            + positions
        )
        motion_query = motion_values[:, -1:, :]
        appearance_query = appearance_values[:, -1:, :]
        safe_mask = self._safe_attention_mask(window_mask)
        motion_summary, motion_weights = self.motion_temporal_attention(
            motion_query,
            motion_values,
            motion_values,
            key_padding_mask=safe_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        appearance_summary, appearance_weights = self.appearance_temporal_attention(
            appearance_query,
            appearance_values,
            appearance_values,
            key_padding_mask=safe_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        reliability_input = self._build_reliability_input(
            scalar_current=scalar_current,
            base_outputs=base_outputs,
        )
        reliability = self.reliability_projection(reliability_input).unsqueeze(1)
        cross_tokens = torch.cat(
            [motion_summary, appearance_summary, reliability], dim=1
        ) + self.cross_modality_embedding.unsqueeze(0)
        attended, cross_weights = self.cross_modal_attention(cross_tokens)
        motion_raw = self.motion_correction_head(attended[:, 0])
        appearance_raw = self.appearance_correction_head(attended[:, 1])
        correction = self.correction_bound * torch.tanh(
            torch.cat([motion_raw, appearance_raw], dim=-1)
        )
        correction = correction.masked_fill(~has_detection.unsqueeze(-1), 0.0)
        refined = torch.clamp(base_outputs["base_gate"].detach() + correction, 0.0, 1.0)
        refined = refined.masked_fill(~has_detection.unsqueeze(-1), 0.0)
        return {
            "gate_correction": correction,
            "temporal_gate_correction": correction,
            "refined_gate": refined,
            "final_gate": refined,
            "motion_attention_weights": motion_weights.squeeze(2),
            "appearance_attention_weights": appearance_weights.squeeze(2),
            "cross_modal_attention_weights": cross_weights,
            "motion_attention_entropy": self._attention_entropy(motion_weights),
            "appearance_attention_entropy": self._attention_entropy(
                appearance_weights
            ),
        }

    def _build_reliability_input(
        self,
        *,
        scalar_current: torch.Tensor,
        base_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        reliability_scalar = scalar_current.detach()
        if self.reliability_mode == RELIABILITY_MODE_NO_SCALAR:
            # Zero is the normalized feature mean, preserving projection shape
            # while removing scalar evidence from the ablation.
            reliability_scalar = torch.zeros_like(reliability_scalar)
        return torch.cat(
            [
                base_outputs["base_gate"].detach(),
                base_outputs["policy_probs"].detach(),
                base_outputs["cue"].detach(),
                base_outputs["risk"].detach(),
                reliability_scalar,
            ],
            dim=-1,
        )

    def forward(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        padding_mask: torch.BoolTensor | None = None,
        has_detection_mask: torch.BoolTensor | None = None,
        reset_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.iwg(
            track_feats, det_feats, scalar_feats, padding_mask, reset_mask
        )
        history_padding = base.pop("_history_padding_mask")
        appearance_history = base.pop("_appearance_history")
        motion_history = base.pop("_motion_history")
        indices = _last_valid_indices(history_padding)
        if has_detection_mask is None:
            detection_sequence = ~history_padding
        else:
            if has_detection_mask.shape != history_padding.shape:
                raise ValueError("has_detection_mask must match padding_mask")
            detection_sequence = has_detection_mask.bool() & ~history_padding
        has_current = _gather_position(detection_sequence, indices)
        base = self._apply_unmatched_sentinel(base, has_current)
        motion_windows, window_mask = _causal_windows(
            motion_history, history_padding, reset_mask, self.context_size
        )
        appearance_windows, appearance_mask = _causal_windows(
            appearance_history, history_padding, reset_mask, self.context_size
        )
        if not torch.equal(window_mask, appearance_mask):
            raise AssertionError("motion and appearance history masks diverged")
        refine = self._refine_flat(
            motion_windows=_gather_position(motion_windows, indices),
            appearance_windows=_gather_position(appearance_windows, indices),
            window_mask=_gather_position(window_mask, indices),
            scalar_current=_gather_position(scalar_feats, indices),
            base_outputs=base,
            has_detection=has_current,
        )
        base.update(refine)
        base["gate"] = refine["refined_gate"]
        return base

    def forward_sequence(
        self,
        track_feats: torch.Tensor,
        det_feats: torch.Tensor,
        scalar_feats: torch.Tensor,
        padding_mask: torch.BoolTensor | None = None,
        has_detection_mask: torch.BoolTensor | None = None,
        reset_mask: torch.BoolTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.iwg.forward_sequence(
            track_feats, det_feats, scalar_feats, padding_mask, reset_mask
        )
        padding = base.pop("padding_mask")
        if has_detection_mask is None:
            detection = ~padding
        else:
            if has_detection_mask.shape != padding.shape:
                raise ValueError("has_detection_mask must match padding_mask")
            detection = has_detection_mask.bool() & ~padding
        base = self._apply_unmatched_sentinel(base, detection)
        motion_windows, window_mask = _causal_windows(
            base["motion_token"], padding, reset_mask, self.context_size
        )
        appearance_windows, appearance_mask = _causal_windows(
            base["appearance_token"], padding, reset_mask, self.context_size
        )
        if not torch.equal(window_mask, appearance_mask):
            raise AssertionError("motion and appearance sequence masks diverged")
        batch_size, seq_len = padding.shape
        flat_base = {
            key: value.reshape(batch_size * seq_len, *value.shape[2:])
            for key, value in base.items()
        }
        refine = self._refine_flat(
            motion_windows=motion_windows.reshape(
                batch_size * seq_len, self.context_size, -1
            ),
            appearance_windows=appearance_windows.reshape(
                batch_size * seq_len, self.context_size, -1
            ),
            window_mask=window_mask.reshape(
                batch_size * seq_len, self.context_size
            ),
            scalar_current=scalar_feats.reshape(batch_size * seq_len, -1),
            base_outputs=flat_base,
            has_detection=detection.reshape(-1),
        )
        outputs = dict(base)
        for key, value in refine.items():
            outputs[key] = value.reshape(batch_size, seq_len, *value.shape[1:])
        outputs["gate"] = outputs["refined_gate"]
        outputs["padding_mask"] = padding
        return outputs


def model_contract(
    *,
    correction_bound: float = RG_CMA_CORRECTION_BOUND,
    context_size: int = IWG_CONTEXT_SIZE,
) -> dict[str, Any]:
    correction_bound = float(correction_bound)
    context_size = int(context_size)
    contracts = (
        (
            RG_CMA_LEGACY_CORRECTION_BOUND,
            IWG_CONTEXT_SIZE,
            IWG_RG_CMA_LEGACY_MODEL_SCHEMA,
            IWG_RG_CMA_LEGACY_MODEL_DESCRIPTOR,
            IWG_RG_CMA_LEGACY_MODEL_SCHEMA_SHA256,
        ),
        (
            RG_CMA_CORRECTION_BOUND,
            IWG_CONTEXT_SIZE,
            IWG_RG_CMA_MODEL_SCHEMA,
            IWG_RG_CMA_MODEL_DESCRIPTOR,
            IWG_RG_CMA_MODEL_SCHEMA_SHA256,
        ),
        (
            RG_CMA_LEGACY_CORRECTION_BOUND,
            8,
            IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA,
            IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_DESCRIPTOR,
            IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA_SHA256,
        ),
        (
            RG_CMA_CORRECTION_BOUND,
            8,
            IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA,
            IWG_RG_CMA_CONTEXT8_MODEL_DESCRIPTOR,
            IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA_SHA256,
        ),
    )
    for bound, context, schema, descriptor, schema_sha256 in contracts:
        if context == context_size and math.isclose(
            correction_bound, bound, rel_tol=0.0, abs_tol=1e-12
        ):
            return {
                "model_schema": schema,
                "model_schema_sha256": schema_sha256,
                "model_schema_descriptor": descriptor,
            }
    if context_size not in SUPPORTED_IWG_CONTEXT_SIZES:
        raise ValueError(
            "unsupported RG-CMA context_size for checkpoint contract: "
            f"{context_size}"
        )
    raise ValueError(
        "unsupported RG-CMA correction bound for checkpoint contract: "
        f"{correction_bound} at context_size={context_size}"
    )
