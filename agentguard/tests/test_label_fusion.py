"""Test fuse_labels with both abstained and normal cases."""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.teacher.response_schema import TeacherResponse
from agentguard.verifier.label_fusion import (
    compute_rollout_reliability,
    compute_teacher_confidence,
    fuse_labels,
)


class TestFuseLabels:
    """Tests for fuse_labels function."""

    def test_teacher_abstained_uses_rollout(self):
        """When teacher abstains, fused gate should equal rollout gate."""
        rollout_gate = np.array([0.8, 0.6], dtype=np.float64)
        agent_gate = np.array([0.2, 0.3], dtype=np.float64)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=0.5,
            teacher_confidence=0.7,
            teacher_abstained=True,
        )
        np.testing.assert_array_equal(fused, rollout_gate)

    def test_teacher_not_abstained_weighted_average(self):
        """When teacher does not abstain, weighted average of rollout and agent."""
        rollout_gate = np.array([0.8, 0.6], dtype=np.float64)
        agent_gate = np.array([0.2, 0.3], dtype=np.float64)
        r = 0.4  # rollout reliability
        c = 0.6  # teacher confidence

        # fused = (0.4 * [0.8, 0.6] + 0.6 * [0.2, 0.3]) / (0.4 + 0.6)
        expected = (r * rollout_gate + c * agent_gate) / (r + c)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=r,
            teacher_confidence=c,
            teacher_abstained=False,
        )
        np.testing.assert_array_almost_equal(fused, expected, decimal=10)

    def test_equal_weights(self):
        """When r == c, result should be simple average."""
        rollout_gate = np.array([1.0, 0.0], dtype=np.float64)
        agent_gate = np.array([0.0, 1.0], dtype=np.float64)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=0.5,
            teacher_confidence=0.5,
            teacher_abstained=False,
        )
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(fused, expected, decimal=10)

    def test_zero_reliability_uses_agent_only(self):
        """When rollout_reliability is 0 and teacher not abstained, use agent."""
        rollout_gate = np.array([0.8, 0.6], dtype=np.float64)
        agent_gate = np.array([0.2, 0.3], dtype=np.float64)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=0.0,
            teacher_confidence=0.7,
            teacher_abstained=False,
        )
        # fused = (0 * rollout + 0.7 * agent) / 0.7 = agent
        np.testing.assert_array_almost_equal(fused, agent_gate, decimal=10)

    def test_zero_confidence_uses_rollout_only(self):
        """When teacher_confidence is 0 and teacher not abstained, use rollout."""
        rollout_gate = np.array([0.8, 0.6], dtype=np.float64)
        agent_gate = np.array([0.2, 0.3], dtype=np.float64)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=0.5,
            teacher_confidence=0.0,
            teacher_abstained=False,
        )
        # fused = (0.5 * rollout + 0 * agent) / 0.5 = rollout
        np.testing.assert_array_almost_equal(fused, rollout_gate, decimal=10)

    def test_both_weights_zero_falls_back_to_rollout(self):
        """When both weights are near zero, falls back to rollout."""
        rollout_gate = np.array([0.8, 0.6], dtype=np.float64)
        agent_gate = np.array([0.2, 0.3], dtype=np.float64)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=1e-15,
            teacher_confidence=1e-15,
            teacher_abstained=False,
        )
        # total_weight < 1e-12 -> fallback to rollout
        np.testing.assert_array_equal(fused, rollout_gate)

    def test_result_clipped_to_0_1(self):
        """Fused gate should be clipped to [0, 1]."""
        rollout_gate = np.array([1.2, -0.2], dtype=np.float64)  # intentionally out of bounds
        agent_gate = np.array([0.5, 0.5], dtype=np.float64)
        fused = fuse_labels(
            rollout_gate=rollout_gate,
            agent_gate=agent_gate,
            rollout_reliability=0.5,
            teacher_confidence=0.5,
            teacher_abstained=False,
        )
        assert fused[0] <= 1.0
        assert fused[1] >= 0.0


class TestComputeRolloutReliability:
    """Tests for compute_rollout_reliability."""

    def test_default_values(self):
        """Default values should produce a low reliability (c_horizon drives floor)."""
        event = object()  # has no attributes -> c_gt=0, c_oracle=0, c_benefit=0, c_horizon=1.0
        r = compute_rollout_reliability(event, motion_benefit=0.0, appearance_benefit=0.0)
        # r = 0.4*0 + 0.3*0 + 0.2*0 + 0.1*1.0 = 0.1, clamped to [0.05, 1.0]
        assert r == pytest.approx(0.1)

    def test_perfect_reliability(self):
        """Perfect oracle coverage and benefits should give high reliability."""
        class MockEvent:
            future_oracle_coverage = 1.0
            oracle_iou = 1.0
            num_future_frames = 5

        r = compute_rollout_reliability(
            MockEvent(), motion_benefit=1.0, appearance_benefit=1.0
        )
        # c_gt=1.0, c_oracle=1.0, c_benefit=1.0, c_horizon=1.0
        # r = 0.4*1.0 + 0.3*1.0 + 0.2*1.0 + 0.1*1.0 = 1.0
        assert r == pytest.approx(1.0)

    def test_partial_reliability(self):
        """Partial oracle coverage should reduce reliability."""
        class MockEvent:
            future_oracle_coverage = 0.5
            oracle_iou = 0.7
            num_future_frames = 3

        r = compute_rollout_reliability(
            MockEvent(), motion_benefit=0.2, appearance_benefit=0.0
        )
        # c_gt=0.5, c_oracle=0.7, c_benefit=0.1, c_horizon=3/5=0.6
        # r = 0.4*0.5 + 0.3*0.7 + 0.2*0.1 + 0.1*0.6 = 0.2 + 0.21 + 0.02 + 0.06 = 0.49
        assert r == pytest.approx(0.49)

    def test_negative_benefit_zero_clamped(self):
        """Negative benefit should be clamped to 0 for c_benefit."""
        class MockEvent:
            future_oracle_coverage = 0.0
            oracle_iou = 0.0
            num_future_frames = 0

        r = compute_rollout_reliability(
            MockEvent(), motion_benefit=-0.5, appearance_benefit=-0.3
        )
        # c_benefit = max(0, (-0.5 + -0.3)/2) = 0
        assert r >= 0.05  # clamped to minimum


class TestComputeTeacherConfidence:
    """Tests for compute_teacher_confidence."""

    def test_no_harm(self):
        """With zero harm rate, confidence should be unchanged."""
        response = TeacherResponse(confidence=0.85)
        adjusted = compute_teacher_confidence(response, harm_rate=0.0)
        assert adjusted == pytest.approx(0.85)

    def test_partial_harm(self):
        """With harm rate 0.3, confidence should be reduced by 30%."""
        response = TeacherResponse(confidence=0.8)
        adjusted = compute_teacher_confidence(response, harm_rate=0.3)
        assert adjusted == pytest.approx(0.8 * 0.7)

    def test_full_harm(self):
        """With harm rate 1.0, confidence should be 0."""
        response = TeacherResponse(confidence=0.9)
        adjusted = compute_teacher_confidence(response, harm_rate=1.0)
        assert adjusted == 0.0

    def test_clipped_to_0_1(self):
        """Result should always be within [0, 1]."""
        response = TeacherResponse(confidence=1.5)  # clamped to 1.0 by validator
        adjusted = compute_teacher_confidence(response, harm_rate=2.0)
        assert 0.0 <= adjusted <= 1.0
