from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateDecision:
    """Output of the gating mechanism — motion and appearance gates plus policy distribution."""

    motion_gate: float
    appearance_gate: float
    policy_probs: np.ndarray  # (5,) policy distribution
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "motion_gate",
            float(np.clip(self.motion_gate, 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "appearance_gate",
            float(np.clip(self.appearance_gate, 0.0, 1.0)),
        )
