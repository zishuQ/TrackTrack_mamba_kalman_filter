"""Test that TeacherResponse Pydantic validation works correctly.

- Rejects invalid event types.
- Requires policy sum in [0.99, 1.01] via validate_policy_sum.
- Clamps confidence to [0, 1].
"""
from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, "src")

from agentguard.teacher.response_schema import (
    CueReliability,
    PolicyPrior,
    TeacherResponse,
    should_upgrade_to_plus,
    validate_policy_sum,
)


class TestTeacherResponseValidation:
    """Tests for TeacherResponse Pydantic model validation."""

    def test_valid_response(self):
        """A fully valid response should not raise."""
        response = TeacherResponse(
            event_types=["CLEAN_OBSERVATION"],
            cue_reliability=CueReliability(motion=0.9, appearance=0.8, box=0.95),
            policy_prior=PolicyPrior(
                FULL_WRITE=0.5, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.1, HOLD_BOTH=0.1, SOFT_CAUTION=0.1,
            ),
            request_temporal_revision=False,
            confidence=0.85,
            abstain=False,
            evidence_frames=[100, 101],
        )
        assert response.event_types == ["CLEAN_OBSERVATION"]
        assert response.confidence == 0.85
        assert response.abstain is False

    def test_rejects_invalid_event_type(self):
        """An invalid event type should raise ValidationError."""
        with pytest.raises(ValidationError):
            TeacherResponse(
                event_types=["INVALID_EVENT_TYPE"],
                confidence=0.5,
            )

    def test_rejects_partially_invalid_event_types(self):
        """Mixing valid and invalid event types should raise."""
        with pytest.raises(ValidationError):
            TeacherResponse(
                event_types=["CLEAN_OBSERVATION", "FAKE_EVENT"],
                confidence=0.5,
            )

    def test_accepts_all_valid_event_types(self):
        """All 10 valid event types should be accepted."""
        valid_types = [
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
        ]
        response = TeacherResponse(event_types=valid_types, confidence=0.9)
        assert len(response.event_types) == 10

    def test_empty_event_types(self):
        """Empty event types list should be valid."""
        response = TeacherResponse(event_types=[], confidence=0.5)
        assert response.event_types == []

    def test_clamps_confidence_above_1(self):
        """Confidence > 1 should be clamped to 1."""
        response = TeacherResponse(confidence=1.5)
        assert response.confidence == 1.0

    def test_clamps_confidence_below_0(self):
        """Confidence < 0 should be clamped to 0."""
        response = TeacherResponse(confidence=-0.5)
        assert response.confidence == 0.0

    def test_confidence_default(self):
        """Default confidence should be 0.0."""
        response = TeacherResponse()
        assert response.confidence == 0.0

    def test_default_values(self):
        """All fields should have sensible defaults."""
        response = TeacherResponse()
        assert response.event_types == []
        assert response.cue_reliability.motion == 0.0
        assert response.cue_reliability.appearance == 0.0
        assert response.cue_reliability.box == 0.0
        assert response.policy_prior.FULL_WRITE == 0.2
        assert response.policy_prior.MOTION_ONLY == 0.2
        assert response.policy_prior.APPEARANCE_ONLY == 0.2
        assert response.policy_prior.HOLD_BOTH == 0.2
        assert response.policy_prior.SOFT_CAUTION == 0.2
        assert response.request_temporal_revision is False
        assert response.confidence == 0.0
        assert response.abstain is False
        assert response.evidence_frames == []


class TestValidatePolicySum:
    """Tests for validate_policy_sum."""

    def test_policy_sum_within_range(self):
        """Sum of 1.0 should be valid."""
        response = TeacherResponse(
            policy_prior=PolicyPrior(
                FULL_WRITE=0.3, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.2, HOLD_BOTH=0.15, SOFT_CAUTION=0.15,
            ),
            confidence=0.5,
        )
        assert validate_policy_sum(response) is True

    def test_policy_sum_at_upper_bound(self):
        """Sum of 1.01 should be valid."""
        response = TeacherResponse(
            policy_prior=PolicyPrior(
                FULL_WRITE=0.21, MOTION_ONLY=0.20,
                APPEARANCE_ONLY=0.20, HOLD_BOTH=0.20, SOFT_CAUTION=0.20,
            ),
            confidence=0.5,
        )
        assert validate_policy_sum(response) is True

    def test_policy_sum_at_lower_bound(self):
        """Sum of 0.99 should be valid."""
        response = TeacherResponse(
            policy_prior=PolicyPrior(
                FULL_WRITE=0.19, MOTION_ONLY=0.20,
                APPEARANCE_ONLY=0.20, HOLD_BOTH=0.20, SOFT_CAUTION=0.20,
            ),
            confidence=0.5,
        )
        assert validate_policy_sum(response) is True

    def test_policy_sum_too_low(self):
        """Sum < 0.99 should be invalid."""
        response = TeacherResponse(
            policy_prior=PolicyPrior(
                FULL_WRITE=0.1, MOTION_ONLY=0.1,
                APPEARANCE_ONLY=0.1, HOLD_BOTH=0.1, SOFT_CAUTION=0.1,
            ),
            confidence=0.5,
        )
        assert validate_policy_sum(response) is False

    def test_policy_sum_too_high(self):
        """Sum > 1.01 should be invalid."""
        response = TeacherResponse(
            policy_prior=PolicyPrior(
                FULL_WRITE=0.5, MOTION_ONLY=0.5,
                APPEARANCE_ONLY=0.5, HOLD_BOTH=0.5, SOFT_CAUTION=0.5,
            ),
            confidence=0.5,
        )
        assert validate_policy_sum(response) is False

    def test_default_policy_sum(self):
        """Default policy (all 0.2) sums to 1.0 -> valid."""
        response = TeacherResponse(confidence=0.5)
        assert validate_policy_sum(response) is True


class TestShouldUpgradeToPlus:
    """Tests for should_upgrade_to_plus."""

    def test_low_confidence_upgrades(self):
        """Confidence < 0.65 should trigger upgrade."""
        response = TeacherResponse(confidence=0.5)
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is True

    def test_high_confidence_no_upgrade(self):
        """Confidence >= 0.65 should not trigger upgrade by itself."""
        response = TeacherResponse(
            confidence=0.8,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.3, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.2, HOLD_BOTH=0.15, SOFT_CAUTION=0.15,
            ),
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is False

    def test_insufficient_evidence_upgrades(self):
        """INSUFFICIENT_EVIDENCE in event types should trigger upgrade."""
        response = TeacherResponse(
            event_types=["INSUFFICIENT_EVIDENCE"],
            confidence=0.9,
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is True

    def test_large_l1_diff_upgrades(self):
        """L1 diff > 0.4 should trigger upgrade."""
        # Policy far from [0.5, 0.5] gate
        response = TeacherResponse(
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.0, MOTION_ONLY=0.0,
                APPEARANCE_ONLY=0.0, HOLD_BOTH=1.0, SOFT_CAUTION=0.0,
            ),
        )
        # HOLD_BOTH = [0, 0], rollout_gate=0.5
        # policy_com = 0*1 + 0*1 + 0*0 + 1*0 + 0*0.5  ... for motion
        # Let's compute: prototypes = [(1,1), (1,0), (0,1), (0,0), (0.5,0.5)]
        # motion_com = 0*1 + 0*1 + 0*0 + 1*0 + 0*0.5 = 0
        # appearance_com = 0*1 + 0*0 + 0*1 + 1*0 + 0*0.5 = 0
        # policy_gate = (0 + 0) / 2 = 0
        # l1_diff = |0 - 0.5| = 0.5 > 0.4
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is True

    def test_boundary_l1_no_upgrade(self):
        """L1 diff exactly 0.4 should not trigger upgrade."""
        # MOTION_ONLY = [1, 0] gives policy_gate of (1+0)/2 = 0.5
        # With rollout_gate=0.1: l1_diff = |0.5 - 0.1| = 0.4
        response = TeacherResponse(
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.0, MOTION_ONLY=1.0,
                APPEARANCE_ONLY=0.0, HOLD_BOTH=0.0, SOFT_CAUTION=0.0,
            ),
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.1) is False

    def test_multiple_reasons_upgrade(self):
        """Multiple upgrade reasons should still return True."""
        response = TeacherResponse(
            event_types=["INSUFFICIENT_EVIDENCE"],
            confidence=0.3,
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is True
