"""Test should_upgrade_to_plus logic for Teacher Flash->Plus fallback."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from agentguard.teacher.response_schema import (
    PolicyPrior,
    TeacherResponse,
    should_upgrade_to_plus,
)


class TestShouldUpgradeToPlus:
    """Tests for the Flash-to-Plus upgrade decision."""

    def test_low_confidence_upgrades(self):
        """Confidence below 0.65 should always upgrade."""
        response = TeacherResponse(
            event_types=["CLEAN_OBSERVATION"],
            confidence=0.64,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.3, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.2, HOLD_BOTH=0.15, SOFT_CAUTION=0.15,
            ),
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is True

    def test_exactly_065_no_upgrade(self):
        """Confidence exactly 0.65 should NOT trigger upgrade (on its own)."""
        response = TeacherResponse(
            event_types=["CLEAN_OBSERVATION"],
            confidence=0.65,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.3, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.2, HOLD_BOTH=0.15, SOFT_CAUTION=0.15,
            ),
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is False

    def test_insufficient_evidence_upgrades_high_confidence(self):
        """INSUFFICIENT_EVIDENCE triggers upgrade even with high confidence."""
        response = TeacherResponse(
            event_types=["INSUFFICIENT_EVIDENCE"],
            confidence=0.95,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.2, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.2, HOLD_BOTH=0.2, SOFT_CAUTION=0.2,
            ),
        )
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is True

    def test_l1_diff_above_threshold_upgrades(self):
        """L1 diff > 0.4 between policy gate and rollout gate should upgrade."""
        # Use HOLD_BOTH [0, 0] -> policy_gate = 0
        response = TeacherResponse(
            event_types=["CLEAN_OBSERVATION"],
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.0, MOTION_ONLY=0.0,
                APPEARANCE_ONLY=0.0, HOLD_BOTH=1.0, SOFT_CAUTION=0.0,
            ),
        )
        # rollout_gate = 0.9 -> l1_diff = |0 - 0.9| = 0.9 > 0.4
        assert should_upgrade_to_plus(response, rollout_gate=0.9) is True

    def test_l1_diff_at_threshold_no_upgrade(self):
        """L1 diff exactly 0.4 should NOT trigger upgrade."""
        # FULL_WRITE [1, 1] -> policy_gate = 1.0
        response = TeacherResponse(
            event_types=["CLEAN_OBSERVATION"],
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=1.0, MOTION_ONLY=0.0,
                APPEARANCE_ONLY=0.0, HOLD_BOTH=0.0, SOFT_CAUTION=0.0,
            ),
        )
        # policy_gate = (1+1)/2 = 1.0, rollout_gate = 0.6
        # l1_diff = |1.0 - 0.6| = 0.4 -> not > 0.4
        assert should_upgrade_to_plus(response, rollout_gate=0.6) is False

    def test_no_upgrade_when_all_conditions_met(self):
        """When confidence >= 0.65, no insufficient evidence, L1 diff <= 0.4 -> no upgrade."""
        response = TeacherResponse(
            event_types=["CLEAN_OBSERVATION"],
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.3, MOTION_ONLY=0.2,
                APPEARANCE_ONLY=0.2, HOLD_BOTH=0.15, SOFT_CAUTION=0.15,
            ),
        )
        # FULL_WRITE 0.3 -> contributes (1,1), MOTION_ONLY 0.2 -> (1,0),
        # APPEARANCE_ONLY 0.2 -> (0,1), HOLD_BOTH 0.15 -> (0,0),
        # SOFT_CAUTION 0.15 -> (0.5,0.5)
        # motion_com = 0.3*1 + 0.2*1 + 0.2*0 + 0.15*0 + 0.15*0.5 = 0.575
        # app_com = 0.3*1 + 0.2*0 + 0.2*1 + 0.15*0 + 0.15*0.5 = 0.575
        # policy_gate = (0.575+0.575)/2 = 0.575
        # l1_diff = |0.575 - 0.5| = 0.075 <= 0.4
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is False

    def test_policy_computation_correctness(self):
        """Verify the policy centre-of-mass computation manually."""
        # ALL policies set to 0 except SOFT_CAUTION at 1.0
        response = TeacherResponse(
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.0, MOTION_ONLY=0.0,
                APPEARANCE_ONLY=0.0, HOLD_BOTH=0.0, SOFT_CAUTION=1.0,
            ),
        )
        # SOFT_CAUTION = [0.5, 0.5]
        # policy_gate = (0.5 + 0.5) / 2 = 0.5
        # With rollout_gate = 0.5, l1_diff = 0 -> no upgrade
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is False

    def test_mixed_policy_l1_diff(self):
        """Mixing MOTION_ONLY and APPEARANCE_ONLY should give correct gate."""
        # MOTION_ONLY [1,0] weight 0.5 + APPEARANCE_ONLY [0,1] weight 0.5
        response = TeacherResponse(
            confidence=0.9,
            policy_prior=PolicyPrior(
                FULL_WRITE=0.0, MOTION_ONLY=0.5,
                APPEARANCE_ONLY=0.5, HOLD_BOTH=0.0, SOFT_CAUTION=0.0,
            ),
        )
        # motion_com = 0.5*1 + 0.5*0 = 0.5
        # app_com = 0.5*0 + 0.5*1 = 0.5
        # policy_gate = 0.5
        assert should_upgrade_to_plus(response, rollout_gate=0.5) is False
        # With rollout_gate = 0.0: l1_diff = |0.5 - 0| = 0.5 > 0.4 -> upgrade
        assert should_upgrade_to_plus(response, rollout_gate=0.0) is True
