from __future__ import annotations

import torch
import torch.nn as nn

from agentguard.models.iwg import IWG
from agentguard.models.tsrm import TSRM


IWG_WARMUP_EVENTS = 5


def forward_iwg_with_warmup(
    iwg: IWG,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Encode warm-up plus formal events and return formal-window outputs."""
    required = (
        "iwg_track_feats",
        "iwg_det_feats",
        "iwg_scalar_feats",
        "iwg_padding_mask",
        "padding_mask",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"IWG warm-up batch missing required fields: {missing}")
    formal_length = int(batch["padding_mask"].shape[1])
    iwg_length = int(batch["iwg_padding_mask"].shape[1])
    if iwg_length != formal_length + IWG_WARMUP_EVENTS:
        raise ValueError(
            "IWG input must contain exactly five warm-up positions: "
            f"iwg_length={iwg_length}, formal_length={formal_length}"
        )
    sequence_outputs = iwg.forward_sequence(
        batch["iwg_track_feats"],
        batch["iwg_det_feats"],
        batch["iwg_scalar_feats"],
        batch["iwg_padding_mask"],
    )
    return {
        key: value[:, -formal_length:]
        for key, value in sequence_outputs.items()
    }


class IWGTSRM(nn.Module):
    def __init__(
        self,
        reid_dim: int,
        *,
        scalar_dim: int = 63,
        event_dim: int = 128,
        hidden_dim: int = 128,
        delta_max: float = 0.2,
    ) -> None:
        super().__init__()
        self.iwg = IWG(reid_dim=reid_dim, scalar_dim=scalar_dim, event_dim=event_dim)
        self.tsrm = TSRM(
            event_dim=event_dim,
            scalar_dim=scalar_dim,
            hidden_dim=hidden_dim,
            delta_max=delta_max,
        )

    @staticmethod
    def apply_unmatched_sentinel(
        outputs: dict[str, torch.Tensor],
        has_detection_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        sentinel = {key: value for key, value in outputs.items()}
        no_detection = ~has_detection_mask
        sentinel["base_gate"] = outputs["base_gate"].masked_fill(
            no_detection.unsqueeze(-1), 0.0
        )
        sentinel["gate"] = sentinel["base_gate"]
        hold_both = outputs["policy_probs"].new_tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        sentinel["policy_probs"] = torch.where(
            no_detection.unsqueeze(-1), hold_both, outputs["policy_probs"]
        )
        sentinel["cue"] = outputs["cue"].masked_fill(no_detection.unsqueeze(-1), 0.0)
        unmatched_risk = outputs["risk"].new_tensor([0.0, 0.0, 1.0, 0.0])
        sentinel["risk"] = torch.where(
            no_detection.unsqueeze(-1), unmatched_risk, outputs["risk"]
        )
        return sentinel

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        padding_mask = batch.get("padding_mask", batch.get("mask"))
        if padding_mask is None:
            raise KeyError("IWGTSRM batch requires padding_mask")
        iwg_outputs = forward_iwg_with_warmup(self.iwg, batch)
        outputs = self.apply_unmatched_sentinel(
            iwg_outputs, batch["has_detection_mask"]
        )
        tsrm_outputs = self.tsrm(
            event_embedding=outputs["event_embedding"],
            scalar_feats=batch["scalar_feats"],
            base_gate=outputs["base_gate"],
            policy_probs=outputs["policy_probs"],
            cue=outputs["cue"],
            risk=outputs["risk"],
            padding_mask=padding_mask,
            has_detection_mask=batch["has_detection_mask"],
            reset_mask=batch.get("reset_mask"),
        )
        outputs.update(tsrm_outputs)
        return outputs
