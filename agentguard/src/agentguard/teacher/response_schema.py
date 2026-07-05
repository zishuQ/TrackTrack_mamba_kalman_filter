from __future__ import annotations

from typing import List

from pydantic import BaseModel, field_validator


class CueReliability(BaseModel):
    """Reliability scores for tracking cues in [0, 1]."""

    motion: float = 0.0
    appearance: float = 0.0
    box: float = 0.0


class PolicyPrior(BaseModel):
    """Prior distribution over write policies (must sum to 1.0)."""

    FULL_WRITE: float = 0.2
    MOTION_ONLY: float = 0.2
    APPEARANCE_ONLY: float = 0.2
    HOLD_BOTH: float = 0.2
    SOFT_CAUTION: float = 0.2


class TeacherResponse(BaseModel):
    """Structured response from the Teacher LLM.

    This is the parsed output of the multimodal LLM, validated via
    pydantic.
    """

    event_types: List[str] = []
    cue_reliability: CueReliability = CueReliability()
    policy_prior: PolicyPrior = PolicyPrior()
    request_temporal_revision: bool = False
    confidence: float = 0.0
    abstain: bool = False
    evidence_frames: List[int] = []

    # ------------------------------------------------------------------
    #  Validators
    # ------------------------------------------------------------------

    @field_validator("event_types")
    @classmethod
    def validate_event_types(cls, v: List[str]) -> List[str]:
        """Ensure each event type is one of the recognised values."""
        valid = {
            "CLEAN_OBSERVATION",
            "BOX_JITTER",
            "PARTIAL_BOX",
            "MOTION_OUTLIER",
            "APPEARANCE_CONTAMINATION",
            "HEAVY_OCCLUSION",
            "MOTION_APPEARANCE_CONFLICT",
            "IDENTITY_AMBIGUITY",
            "LOST_REAPPEARANCE",
            "INSUFFICIENT_EVIDENCE",
        }
        for et in v:
            if et not in valid:
                raise ValueError(f"Invalid event type: {et}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp confidence to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))

    @field_validator("evidence_frames")
    @classmethod
    def validate_frames(cls, v: List[int]) -> List[int]:
        """Validate evidence frames.

        Future-frame checks are performed at the call site because the
        current frame context is not available here.
        """
        return v


def validate_policy_sum(response: TeacherResponse) -> bool:
    """Check that the 5 policy probabilities sum to between 0.99 and 1.01.

    Parameters
    ----------
    response : TeacherResponse
        Parsed teacher response.

    Returns
    -------
    bool
        ``True`` if the sum is within [0.99, 1.01], ``False`` otherwise.
    """
    probs = [
        response.policy_prior.FULL_WRITE,
        response.policy_prior.MOTION_ONLY,
        response.policy_prior.APPEARANCE_ONLY,
        response.policy_prior.HOLD_BOTH,
        response.policy_prior.SOFT_CAUTION,
    ]
    total = sum(probs)
    return 0.99 <= total <= 1.01


def should_upgrade_to_plus(
    response: TeacherResponse,
    rollout_gate: float,
) -> bool:
    """Check whether a Flash response needs a Plus upgrade.

    An upgrade is required when any of the following hold:

    - JSON validation failure (indicated by a malformed or empty response;
      at the schema level we check ``confidence < 0.65`` as a proxy).
    - ``confidence < 0.65``.
    - L1 difference between the response's policy centre-of-mass and the
      rollout gate exceeds 0.4.
    - Event type contains ``INSUFFICIENT_EVIDENCE``.

    Parameters
    ----------
    response : TeacherResponse
        The parsed teacher response from the Flash model.
    rollout_gate : float
        The rollout gate value (mean of motion and appearance gates).

    Returns
    -------
    bool
        ``True`` if the response should be upgraded to Plus.
    """
    # Confidence below threshold
    if response.confidence < 0.65:
        return True

    # Insufficient evidence
    if "INSUFFICIENT_EVIDENCE" in response.event_types:
        return True

    # L1 difference between policy centre-of-mass and rollout gate
    probs = [
        response.policy_prior.FULL_WRITE,
        response.policy_prior.MOTION_ONLY,
        response.policy_prior.APPEARANCE_ONLY,
        response.policy_prior.HOLD_BOTH,
        response.policy_prior.SOFT_CAUTION,
    ]
    # Prototype matrix: [[1,1],[1,0],[0,1],[0,0],[0.5,0.5]]
    prototypes = [
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]
    motion_com = sum(p * proto[0] for p, proto in zip(probs, prototypes))
    appearance_com = sum(p * proto[1] for p, proto in zip(probs, prototypes))
    policy_gate = (motion_com + appearance_com) / 2.0
    l1_diff = abs(policy_gate - rollout_gate)
    if l1_diff > 0.4:
        return True

    return False
