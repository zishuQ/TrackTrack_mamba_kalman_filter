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
ARCHITECTURE_LEGACY = "legacy"
ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL = "legacy-clean-cross-modal"
ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED = (
    "legacy-clean-cross-modal-base-conditioned"
)
ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA = (
    "legacy-clean-bidirectional-cma"
)
ARCHITECTURE_LEGACY_CLEAN_6X6_CMA = "legacy-clean-6x6-cma"
ARCHITECTURE_DIRECT_BASE = "direct-base"
ARCHITECTURE_CLEAN_CROSS_MODAL = "clean-cross-modal"
ARCHITECTURE_SELECTIVE_CORRECTION = "selective-correction"
SUPPORTED_ARCHITECTURE_VARIANTS = frozenset(
    {
        ARCHITECTURE_LEGACY,
        ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
        ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
        ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
        ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
        ARCHITECTURE_DIRECT_BASE,
        ARCHITECTURE_CLEAN_CROSS_MODAL,
        ARCHITECTURE_SELECTIVE_CORRECTION,
    }
)

# Scalar63 indices 27:55 are geometry/KF state. IoU and angle are the only
# association features retained in the clean motion stream; the complement is
# treated as matching reliability/context evidence.
CLEAN_MOTION_SCALAR_INDICES = (0, 1, 4, *range(27, 55))
CLEAN_RELIABILITY_SCALAR_INDICES = tuple(
    index for index in range(63) if index not in CLEAN_MOTION_SCALAR_INDICES
)


def _model_descriptor(
    name: str,
    correction_bound: float,
    context_size: int = IWG_CONTEXT_SIZE,
    architecture_variant: str = ARCHITECTURE_LEGACY,
) -> dict[str, Any]:
    if architecture_variant == ARCHITECTURE_LEGACY:
        # This byte-for-byte descriptor shape preserves every existing model
        # schema hash and keeps old checkpoints loadable.
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
    clean_cross_modal = architecture_variant in {
        ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
        ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
        ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
        ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
        ARCHITECTURE_CLEAN_CROSS_MODAL,
        ARCHITECTURE_SELECTIVE_CORRECTION,
    }
    base_conditioned = (
        architecture_variant
        == ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED
    )
    direct_base = architecture_variant in {
        ARCHITECTURE_DIRECT_BASE,
        ARCHITECTURE_CLEAN_CROSS_MODAL,
        ARCHITECTURE_SELECTIVE_CORRECTION,
    }
    descriptor = {
        "name": name,
        "architecture_variant": architecture_variant,
        "iwg_context_size": int(context_size),
        "modal_token_dim": 64,
        "attention_dim": RG_CMA_DIM,
        "attention_heads": RG_CMA_HEADS,
        "attention_dropout": 0.1,
        "cross_modal_tokens": (
            ["motion", "appearance"]
            if clean_cross_modal
            else ["motion", "appearance", "reliability"]
        ),
        "correction_bound": correction_bound,
        "base_gate": (
            "direct_continuous_aux_policy_detached"
            if direct_base
            else "policy_prototype_plus_bounded_residual"
        ),
        "modal_split": (
            "reid_appearance_vs_geometry_kf_motion"
            if clean_cross_modal
            else "reid_appearance_vs_scalar63"
        ),
        "reliability_control": (
            "separate_two_channel_scale"
            if architecture_variant == ARCHITECTURE_SELECTIVE_CORRECTION
            else "base_conditioned_correction_head"
            if base_conditioned
            else "none" if clean_cross_modal else "cross_attention_token"
        ),
        "gradient_contract": "safe_direct_base_detached_rg_cma",
    }
    if architecture_variant == ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA:
        descriptor["cross_modal_operator"] = (
            "bidirectional_history_attention_plus_two_token_fusion"
        )
    if architecture_variant == ARCHITECTURE_LEGACY_CLEAN_6X6_CMA:
        descriptor["cross_modal_operator"] = (
            "shared_bidirectional_full_history_6x6_attention"
        )
        descriptor["cross_modal_readout"] = "endpoint_query_temporal_attention"
        descriptor["cross_modal_fusion"] = "none"
    return descriptor


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

IWG_RG_CMA_DIRECT_BASE_MODEL_SCHEMA = "agentguard_iwg_rg_cma_direct_base_v1"
IWG_RG_CMA_DIRECT_BASE_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_DIRECT_BASE_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    architecture_variant=ARCHITECTURE_DIRECT_BASE,
)
IWG_RG_CMA_DIRECT_BASE_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_DIRECT_BASE_MODEL_DESCRIPTOR
)

IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_SCHEMA = (
    "agentguard_iwg_rg_cma_legacy_clean_cross_modal_v1"
)
IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    architecture_variant=ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
)
IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_DESCRIPTOR
)

IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_SCHEMA = (
    "agentguard_iwg_rg_cma_legacy_clean_cross_modal_base_conditioned_v1"
)
IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_DESCRIPTOR = (
    _model_descriptor(
        IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_SCHEMA,
        RG_CMA_LEGACY_CORRECTION_BOUND,
        architecture_variant=ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
    )
)
IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_SCHEMA_SHA256 = (
    _descriptor_sha256(
        IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_DESCRIPTOR
    )
)

IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_SCHEMA = (
    "agentguard_iwg_rg_cma_legacy_clean_bidirectional_cma_v1"
)
IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    architecture_variant=ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
)
IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_SCHEMA_SHA256 = (
    _descriptor_sha256(
        IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_DESCRIPTOR
    )
)

IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_SCHEMA = (
    "agentguard_iwg_rg_cma_legacy_clean_6x6_cma_v1"
)
IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    architecture_variant=ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
)
IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_DESCRIPTOR
)

IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_SCHEMA = (
    "agentguard_iwg_rg_cma_clean_cross_modal_v1"
)
IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    architecture_variant=ARCHITECTURE_CLEAN_CROSS_MODAL,
)
IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_DESCRIPTOR
)

IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_SCHEMA = (
    "agentguard_iwg_rg_cma_selective_correction_v1"
)
IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_DESCRIPTOR = _model_descriptor(
    IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_SCHEMA,
    RG_CMA_LEGACY_CORRECTION_BOUND,
    architecture_variant=ARCHITECTURE_SELECTIVE_CORRECTION,
)
IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_SCHEMA_SHA256 = _descriptor_sha256(
    IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_DESCRIPTOR
)

MODEL_CONTRACT_SPECS = {
    (IWG_RG_CMA_LEGACY_MODEL_SCHEMA, IWG_RG_CMA_LEGACY_MODEL_SCHEMA_SHA256): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_LEGACY,
        "descriptor": IWG_RG_CMA_LEGACY_MODEL_DESCRIPTOR,
    },
    (IWG_RG_CMA_MODEL_SCHEMA, IWG_RG_CMA_MODEL_SCHEMA_SHA256): {
        "correction_bound": RG_CMA_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_LEGACY,
        "descriptor": IWG_RG_CMA_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA,
        IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 8,
        "architecture_variant": ARCHITECTURE_LEGACY,
        "descriptor": IWG_RG_CMA_CONTEXT8_LEGACY_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA,
        IWG_RG_CMA_CONTEXT8_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_CORRECTION_BOUND,
        "context_size": 8,
        "architecture_variant": ARCHITECTURE_LEGACY,
        "descriptor": IWG_RG_CMA_CONTEXT8_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_DIRECT_BASE_MODEL_SCHEMA,
        IWG_RG_CMA_DIRECT_BASE_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_DIRECT_BASE,
        "descriptor": IWG_RG_CMA_DIRECT_BASE_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_SCHEMA,
        IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
        "descriptor": IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_SCHEMA,
        IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": (
            ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED
        ),
        "descriptor": (
            IWG_RG_CMA_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED_MODEL_DESCRIPTOR
        ),
    },
    (
        IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_SCHEMA,
        IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
        "descriptor": IWG_RG_CMA_LEGACY_CLEAN_BIDIRECTIONAL_CMA_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_SCHEMA,
        IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
        "descriptor": IWG_RG_CMA_LEGACY_CLEAN_6X6_CMA_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_SCHEMA,
        IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_CLEAN_CROSS_MODAL,
        "descriptor": IWG_RG_CMA_CLEAN_CROSS_MODAL_MODEL_DESCRIPTOR,
    },
    (
        IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_SCHEMA,
        IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_SCHEMA_SHA256,
    ): {
        "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        "context_size": 6,
        "architecture_variant": ARCHITECTURE_SELECTIVE_CORRECTION,
        "descriptor": IWG_RG_CMA_SELECTIVE_CORRECTION_MODEL_DESCRIPTOR,
    },
}


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


def _current_detection_mask(
    padding_mask: torch.BoolTensor,
    has_detection_mask: torch.BoolTensor | None,
    indices: torch.LongTensor,
) -> torch.BoolTensor:
    """Resolve the current-event detection flag.

    Online inference may pass a per-sample endpoint mask of shape ``(B,)``
    instead of a full ``(B, T)`` sequence. Padding positions cannot be
    marked as detections.
    """
    batch = torch.arange(padding_mask.shape[0], device=padding_mask.device)
    valid_endpoint = ~padding_mask[batch, indices]
    if has_detection_mask is None:
        return valid_endpoint
    mask = has_detection_mask.bool()
    if mask.shape == padding_mask.shape:
        return mask[batch, indices] & valid_endpoint
    if mask.ndim == 1 and mask.shape == (padding_mask.shape[0],):
        return mask & valid_endpoint
    raise ValueError(
        "has_detection_mask must match padding_mask or be a per-sample "
        "endpoint mask"
    )


class SafeDirectIWG(IWG):
    """IWG with the verified linear gate heads and isolated auxiliary heads."""

    def __init__(
        self,
        reid_dim: int,
        scalar_dim: int = 63,
        event_dim: int = 128,
        context_size: int = IWG_CONTEXT_SIZE,
        architecture_variant: str = ARCHITECTURE_LEGACY,
    ):
        super().__init__(
            reid_dim=reid_dim,
            scalar_dim=scalar_dim,
            event_dim=event_dim,
            context_size=context_size,
        )
        self.architecture_variant = str(architecture_variant)
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
        direct_base = self.architecture_variant in {
            ARCHITECTURE_DIRECT_BASE,
            ARCHITECTURE_CLEAN_CROSS_MODAL,
            ARCHITECTURE_SELECTIVE_CORRECTION,
        }
        policy_input = event_embedding.detach() if direct_base else event_embedding
        policy_logits = self.policy_head(policy_input)
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
        if direct_base:
            base_gate = torch.sigmoid(residual)
        else:
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

    def forward(
        self,
        value: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layer = self.layer
        attended, weights = layer.self_attn(
            value,
            value,
            value,
            need_weights=return_diagnostics,
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
        architecture_variant: str = ARCHITECTURE_LEGACY,
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
        architecture_variant = str(architecture_variant).strip().lower()
        if architecture_variant not in SUPPORTED_ARCHITECTURE_VARIANTS:
            raise ValueError(
                "unsupported RG-CMA architecture_variant: "
                f"{architecture_variant!r}; expected one of "
                f"{sorted(SUPPORTED_ARCHITECTURE_VARIANTS)}"
            )
        if architecture_variant != ARCHITECTURE_LEGACY and (
            context_size != IWG_CONTEXT_SIZE
            or not math.isclose(
                correction_bound,
                RG_CMA_LEGACY_CORRECTION_BOUND,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "structural ablations require context_size=6 and "
                "correction_bound=0.05"
            )
        if (
            architecture_variant
            in {
                ARCHITECTURE_CLEAN_CROSS_MODAL,
                ARCHITECTURE_SELECTIVE_CORRECTION,
                ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
                ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
                ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
                ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
            }
            and reliability_mode != RELIABILITY_MODE_FULL
        ):
            raise ValueError(
                "clean cross-modal ablations require reliability_mode='full'"
            )
        self.scalar_dim = int(scalar_dim)
        self.correction_bound = correction_bound
        self.context_size = context_size
        self.reliability_mode = reliability_mode
        self.architecture_variant = architecture_variant
        self.uses_clean_modalities = architecture_variant in {
            ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
            ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
            ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
            ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
            ARCHITECTURE_CLEAN_CROSS_MODAL,
            ARCHITECTURE_SELECTIVE_CORRECTION,
        }
        self.uses_selective_correction = (
            architecture_variant == ARCHITECTURE_SELECTIVE_CORRECTION
        )
        self.uses_base_conditioned_correction = (
            architecture_variant
            == ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED
        )
        self.uses_bidirectional_history_cma = (
            architecture_variant == ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA
        )
        self.uses_full_history_cma = (
            architecture_variant == ARCHITECTURE_LEGACY_CLEAN_6X6_CMA
        )
        self.cross_modal_token_count = 2 if self.uses_clean_modalities else 3
        self.iwg = SafeDirectIWG(
            reid_dim=reid_dim,
            scalar_dim=scalar_dim,
            event_dim=event_dim,
            context_size=context_size,
            architecture_variant=architecture_variant,
        )
        if self.uses_clean_modalities:
            self.clean_motion_encoder = nn.Sequential(
                nn.LayerNorm(len(CLEAN_MOTION_SCALAR_INDICES)),
                nn.Linear(len(CLEAN_MOTION_SCALAR_INDICES), 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
            )
            self.register_buffer(
                "_clean_motion_indices",
                torch.tensor(CLEAN_MOTION_SCALAR_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_clean_reliability_indices",
                torch.tensor(CLEAN_RELIABILITY_SCALAR_INDICES, dtype=torch.long),
                persistent=False,
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
        if self.uses_bidirectional_history_cma:
            self.appearance_to_motion_cross_attention = nn.MultiheadAttention(
                RG_CMA_DIM, RG_CMA_HEADS, dropout=0.1, batch_first=True
            )
            self.motion_to_appearance_cross_attention = nn.MultiheadAttention(
                RG_CMA_DIM, RG_CMA_HEADS, dropout=0.1, batch_first=True
            )
            self.motion_cross_modal_norm = nn.LayerNorm(RG_CMA_DIM)
            self.appearance_cross_modal_norm = nn.LayerNorm(RG_CMA_DIM)
            self.motion_cross_modal_dropout = nn.Dropout(0.1)
            self.appearance_cross_modal_dropout = nn.Dropout(0.1)
        if self.uses_full_history_cma:
            self.bidirectional_history_cross_attention = nn.MultiheadAttention(
                RG_CMA_DIM, RG_CMA_HEADS, dropout=0.1, batch_first=True
            )
            self.motion_history_cross_norm = nn.LayerNorm(RG_CMA_DIM)
            self.appearance_history_cross_norm = nn.LayerNorm(RG_CMA_DIM)
            self.history_cross_dropout = nn.Dropout(0.1)
        if not self.uses_clean_modalities:
            reliability_dim = 2 + 5 + 3 + 4 + self.scalar_dim
            self.reliability_projection = nn.Linear(reliability_dim, RG_CMA_DIM)
        if not self.uses_full_history_cma:
            self.cross_modality_embedding = nn.Parameter(
                torch.zeros(self.cross_modal_token_count, RG_CMA_DIM)
            )
            nn.init.normal_(self.cross_modality_embedding, mean=0.0, std=0.02)
            self.cross_modal_attention = RecordingTransformerEncoderLayer()
        correction_head_input_dim = RG_CMA_DIM + (
            1 if self.uses_base_conditioned_correction else 0
        )
        self.motion_correction_head = make_head(correction_head_input_dim, 64, 1)
        self.appearance_correction_head = make_head(
            correction_head_input_dim, 64, 1
        )
        for head in (self.motion_correction_head, self.appearance_correction_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if self.uses_selective_correction:
            selective_reliability_dim = (
                2 + 3 + 4 + len(CLEAN_RELIABILITY_SCALAR_INDICES)
            )
            self.reliability_gate_head = make_head(
                selective_reliability_dim, 64, 2
            )

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

    def _encode_clean_motion(self, scalar_feats: torch.Tensor) -> torch.Tensor:
        if not self.uses_clean_modalities:
            raise RuntimeError("clean motion encoding requires a clean-modal variant")
        motion_scalars = scalar_feats.index_select(
            -1, self._clean_motion_indices.to(device=scalar_feats.device)
        )
        return self.clean_motion_encoder(motion_scalars)

    def _refine_flat(
        self,
        *,
        motion_windows: torch.Tensor,
        appearance_windows: torch.Tensor,
        window_mask: torch.BoolTensor,
        scalar_current: torch.Tensor,
        base_outputs: dict[str, torch.Tensor],
        has_detection: torch.BoolTensor,
        return_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        # Base-IWG producers are detached. The clean motion encoder belongs to
        # RG-CMA and must retain gradients in the structural ablations.
        if not self.uses_clean_modalities:
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
        safe_mask = self._safe_attention_mask(window_mask)
        appearance_to_motion_weights = None
        motion_to_appearance_weights = None
        appearance_to_motion_6x6_weights = None
        motion_to_appearance_6x6_weights = None
        if self.uses_full_history_cma:
            # Shared weights keep the two modality directions symmetric. Each
            # history token queries the complete opposite-modality history.
            appearance_to_motion, appearance_to_motion_6x6_weights = (
                self.bidirectional_history_cross_attention(
                    motion_values,
                    appearance_values,
                    appearance_values,
                    key_padding_mask=safe_mask,
                    need_weights=return_diagnostics,
                    average_attn_weights=False,
                )
            )
            motion_to_appearance, motion_to_appearance_6x6_weights = (
                self.bidirectional_history_cross_attention(
                    appearance_values,
                    motion_values,
                    motion_values,
                    key_padding_mask=safe_mask,
                    need_weights=return_diagnostics,
                    average_attn_weights=False,
                )
            )
            motion_values = self.motion_history_cross_norm(
                motion_values + self.history_cross_dropout(appearance_to_motion)
            )
            appearance_values = self.appearance_history_cross_norm(
                appearance_values
                + self.history_cross_dropout(motion_to_appearance)
            )
        motion_query = motion_values[:, -1:, :]
        appearance_query = appearance_values[:, -1:, :]
        motion_summary, motion_weights = self.motion_temporal_attention(
            motion_query,
            motion_values,
            motion_values,
            key_padding_mask=safe_mask,
            need_weights=return_diagnostics,
            average_attn_weights=False,
        )
        appearance_summary, appearance_weights = self.appearance_temporal_attention(
            appearance_query,
            appearance_values,
            appearance_values,
            key_padding_mask=safe_mask,
            need_weights=return_diagnostics,
            average_attn_weights=False,
        )
        if self.uses_bidirectional_history_cma:
            appearance_to_motion, appearance_to_motion_weights = (
                self.appearance_to_motion_cross_attention(
                    motion_summary,
                    appearance_values,
                    appearance_values,
                    key_padding_mask=safe_mask,
                    need_weights=return_diagnostics,
                    average_attn_weights=False,
                )
            )
            motion_to_appearance, motion_to_appearance_weights = (
                self.motion_to_appearance_cross_attention(
                    appearance_summary,
                    motion_values,
                    motion_values,
                    key_padding_mask=safe_mask,
                    need_weights=return_diagnostics,
                    average_attn_weights=False,
                )
            )
            motion_summary = self.motion_cross_modal_norm(
                motion_summary
                + self.motion_cross_modal_dropout(appearance_to_motion)
            )
            appearance_summary = self.appearance_cross_modal_norm(
                appearance_summary
                + self.appearance_cross_modal_dropout(motion_to_appearance)
            )
        cross_weights = None
        if self.uses_full_history_cma:
            # The endpoint temporal summaries already include cross-modal
            # history, so this variant intentionally omits 2-token fusion.
            motion_head_input = motion_summary[:, 0]
            appearance_head_input = appearance_summary[:, 0]
        else:
            if self.uses_clean_modalities:
                cross_tokens = torch.cat(
                    [motion_summary, appearance_summary], dim=1
                )
            else:
                reliability_input = self._build_reliability_input(
                    scalar_current=scalar_current,
                    base_outputs=base_outputs,
                )
                reliability = self.reliability_projection(reliability_input).unsqueeze(1)
                cross_tokens = torch.cat(
                    [motion_summary, appearance_summary, reliability], dim=1
                )
            cross_tokens = cross_tokens + self.cross_modality_embedding.unsqueeze(0)
            attended, cross_weights = self.cross_modal_attention(
                cross_tokens, return_diagnostics=return_diagnostics
            )
            motion_head_input = attended[:, 0]
            appearance_head_input = attended[:, 1]
        if self.uses_base_conditioned_correction:
            base_gate = base_outputs["base_gate"].detach()
            motion_head_input = torch.cat(
                [motion_head_input, base_gate[:, 0:1]], dim=-1
            )
            appearance_head_input = torch.cat(
                [appearance_head_input, base_gate[:, 1:2]], dim=-1
            )
        motion_raw = self.motion_correction_head(motion_head_input)
        appearance_raw = self.appearance_correction_head(appearance_head_input)
        raw_correction = self.correction_bound * torch.tanh(
            torch.cat([motion_raw, appearance_raw], dim=-1)
        )
        if self.uses_selective_correction:
            selective_input = self._build_selective_reliability_input(
                scalar_current=scalar_current,
                base_outputs=base_outputs,
            )
            correction_scale = torch.sigmoid(
                self.reliability_gate_head(selective_input)
            )
        else:
            correction_scale = torch.ones_like(raw_correction)
        correction = raw_correction * correction_scale
        correction = correction.masked_fill(~has_detection.unsqueeze(-1), 0.0)
        refined = torch.clamp(base_outputs["base_gate"].detach() + correction, 0.0, 1.0)
        refined = refined.masked_fill(~has_detection.unsqueeze(-1), 0.0)
        result = {
            "raw_gate_correction": raw_correction,
            "correction_scale": correction_scale,
            "gate_correction": correction,
            "temporal_gate_correction": correction,
            "refined_gate": refined,
            "final_gate": refined,
        }
        if return_diagnostics:
            result["motion_attention_weights"] = motion_weights.squeeze(2)
            result["appearance_attention_weights"] = appearance_weights.squeeze(2)
            result["motion_attention_entropy"] = self._attention_entropy(
                motion_weights
            )
            result["appearance_attention_entropy"] = self._attention_entropy(
                appearance_weights
            )
            if appearance_to_motion_weights is not None:
                result["appearance_to_motion_attention_weights"] = (
                    appearance_to_motion_weights.squeeze(2)
                )
                result["motion_to_appearance_attention_weights"] = (
                    motion_to_appearance_weights.squeeze(2)
                )
            if appearance_to_motion_6x6_weights is not None:
                result["appearance_to_motion_6x6_attention_weights"] = (
                    appearance_to_motion_6x6_weights
                )
                result["motion_to_appearance_6x6_attention_weights"] = (
                    motion_to_appearance_6x6_weights
                )
            if cross_weights is not None:
                result["cross_modal_attention_weights"] = cross_weights
        return result

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

    def _build_selective_reliability_input(
        self,
        *,
        scalar_current: torch.Tensor,
        base_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.uses_selective_correction:
            raise RuntimeError(
                "selective reliability input requires selective-correction"
            )
        reliability_scalar = scalar_current.detach().index_select(
            -1, self._clean_reliability_indices.to(device=scalar_current.device)
        )
        return torch.cat(
            [
                base_outputs["base_gate"].detach(),
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
        return_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        base = self.iwg(
            track_feats, det_feats, scalar_feats, padding_mask, reset_mask
        )
        history_padding = base.pop("_history_padding_mask")
        appearance_history = base.pop("_appearance_history")
        motion_history = base.pop("_motion_history")
        if self.uses_clean_modalities:
            motion_history = self._encode_clean_motion(scalar_feats)
        indices = _last_valid_indices(history_padding)
        has_current = _current_detection_mask(
            history_padding, has_detection_mask, indices
        )
        base = self._apply_unmatched_sentinel(base, has_current)
        if self.uses_clean_modalities:
            base["motion_token"] = _gather_position(motion_history, indices)
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
            return_diagnostics=return_diagnostics,
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
        return_diagnostics: bool = True,
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
        if self.uses_clean_modalities:
            base["motion_token"] = self._encode_clean_motion(scalar_feats)
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
            return_diagnostics=return_diagnostics,
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
    architecture_variant: str = ARCHITECTURE_LEGACY,
) -> dict[str, Any]:
    correction_bound = float(correction_bound)
    context_size = int(context_size)
    architecture_variant = str(architecture_variant).strip().lower()
    for (schema, _schema_fingerprint), spec in MODEL_CONTRACT_SPECS.items():
        if (
            int(spec["context_size"]) == context_size
            and str(spec["architecture_variant"]) == architecture_variant
            and math.isclose(
                correction_bound,
                float(spec["correction_bound"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return {
                "model_schema": schema,
                "model_schema_descriptor": spec["descriptor"],
            }
    if architecture_variant not in SUPPORTED_ARCHITECTURE_VARIANTS:
        raise ValueError(
            "unsupported RG-CMA architecture_variant for checkpoint contract: "
            f"{architecture_variant!r}"
        )
    if context_size not in SUPPORTED_IWG_CONTEXT_SIZES:
        raise ValueError(
            "unsupported RG-CMA context_size for checkpoint contract: "
            f"{context_size}"
        )
    raise ValueError(
        "unsupported RG-CMA model contract: "
        f"architecture_variant={architecture_variant}, "
        f"correction_bound={correction_bound}, context_size={context_size}"
    )
