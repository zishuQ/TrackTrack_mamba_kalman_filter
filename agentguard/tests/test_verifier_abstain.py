"""Test that the Verifier abstains when the cross-event group is insufficient.

When there are too few similar events, the verifier should return
empty scores or abstain from making recommendations.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.verifier.cross_event import compute_cross_event_scores


# A mock verifier that can be made to fail / return empty
class MockVerifier:
    """Mock verifier that returns configurable results."""

    def __init__(self, policy_deltas: Optional[Dict[str, float]] = None):
        self._policy_deltas = policy_deltas

    def verify_event(self, event: object, oracle_data: dict) -> Dict[str, Any]:
        if self._policy_deltas is None:
            return {"policy_deltas": {}}
        return {"policy_deltas": self._policy_deltas}


class EmptyVerifier:
    """Verifier that always returns empty deltas (simulating abstention)."""

    def verify_event(self, event: object, oracle_data: dict) -> Dict[str, Any]:
        return {"policy_deltas": {}}


class FailingVerifier:
    """Verifier that always raises (simulating replay failure)."""

    def verify_event(self, event: object, oracle_data: dict) -> Dict[str, Any]:
        msg = "Replay failed"
        raise RuntimeError(msg)


class TestVerifierAbstain:
    """Tests for verifier abstention behavior."""

    def test_empty_similar_events_returns_empty(self):
        """With no similar events, scores should be empty."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        verifier = MockVerifier()
        scores = compute_cross_event_scores(event, [], verifier)
        assert scores == {}

    def test_no_deltas_returns_inf_for_non_full(self):
        """When similar events produce no deltas, non-FULL_WRITE scores
        should be -inf."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        similar = [
            (event, np.array([1.0, 0.0]), 0.8),
        ]
        # EmptyVerifier returns empty deltas
        scores = compute_cross_event_scores(event, similar, EmptyVerifier())
        # FULL_WRITE always gets 0.0
        assert scores.get("FULL_WRITE") == 0.0
        # Others get -inf because len(deltas) == 0
        for policy in ["MOTION_ONLY", "APPEARANCE_ONLY", "HOLD_BOTH", "SOFT_CAUTION"]:
            assert scores.get(policy) == -float("inf")

    def test_all_replays_fail_returns_empty(self):
        """When all replays fail, only FULL_WRITE has 0.0, others -inf."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        similar = [
            (event, np.array([1.0, 0.0]), 0.8),
            (event, np.array([1.0, 0.0]), 0.7),
        ]
        scores = compute_cross_event_scores(event, similar, FailingVerifier())
        assert scores.get("FULL_WRITE") == 0.0
        for policy in ["MOTION_ONLY", "APPEARANCE_ONLY", "HOLD_BOTH", "SOFT_CAUTION"]:
            assert scores.get(policy) == -float("inf")

    def test_single_similar_event_with_deltas(self):
        """A single similar event with positive deltas should produce finite scores."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        similar = [
            (event, np.array([1.0, 0.0]), 0.8),
        ]
        deltas = {
            "MOTION_ONLY": 0.05,
            "APPEARANCE_ONLY": 0.03,
            "HOLD_BOTH": -0.02,
            "SOFT_CAUTION": 0.01,
        }
        verifier = MockVerifier(policy_deltas=deltas)
        scores = compute_cross_event_scores(event, similar, verifier)

        assert "FULL_WRITE" in scores
        assert scores["FULL_WRITE"] == 0.0
        # With 1 element: mean = delta, std = 0, harm_rate = 1 if delta < 0 else 0
        # MOTION_ONLY: s = 0.05 - 0.5*0 - 0.5*0 = 0.05
        assert scores["MOTION_ONLY"] == pytest.approx(0.05)
        # HOLD_BOTH: s = -0.02 - 0.5*0 - 0.5*1 = -0.52
        assert scores["HOLD_BOTH"] == pytest.approx(-0.52)

    def test_score_formula_correctness(self):
        """Verify S_m = mean - 0.5*std - 0.5*harm_rate against known values."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        # Two similar events with the same policy deltas
        similar = [
            (event, np.array([1.0, 0.0]), 0.8),
            (event, np.array([1.0, 0.0]), 0.7),
        ]
        # Deltas: [0.1, -0.05]
        # mean = (0.1 + (-0.05)) / 2 = 0.025
        # std = sqrt(((0.1-0.025)^2 + (-0.05-0.025)^2) / 2)
        #      = sqrt((0.005625 + 0.005625) / 2) = sqrt(0.005625) = 0.075
        # harm_rate = 1/2 = 0.5 (one negative)
        # s = 0.025 - 0.5*0.075 - 0.5*0.5 = 0.025 - 0.0375 - 0.25 = -0.2625
        deltas1 = {"MOTION_ONLY": 0.1, "APPEARANCE_ONLY": 0.05, "HOLD_BOTH": 0.0, "SOFT_CAUTION": -0.02}
        deltas2 = {"MOTION_ONLY": -0.05, "APPEARANCE_ONLY": 0.01, "HOLD_BOTH": -0.1, "SOFT_CAUTION": 0.03}

        class MultiMockVerifier:
            def __init__(self, delta_list):
                self._idx = 0
                self._delta_list = delta_list

            def verify_event(self, evt, oracle):
                d = self._delta_list[self._idx % len(self._delta_list)]
                self._idx += 1
                return {"policy_deltas": d}

        verifier = MultiMockVerifier([deltas1, deltas2])
        scores = compute_cross_event_scores(event, similar, verifier)

        # MOTION_ONLY: deltas = [0.1, -0.05]
        expected_motion = 0.025 - 0.5 * 0.075 - 0.5 * 0.5  # = -0.2625
        assert scores["MOTION_ONLY"] == pytest.approx(expected_motion, abs=1e-10)

    def test_all_similar_replays_fail_skipped(self):
        """When all replays fail (exception), FULL_WRITE gets 0.0 and others get -inf."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        similar = [
            (event, np.array([1.0, 0.0]), 0.8),
            (event, np.array([1.0, 0.0]), 0.7),
        ]
        scores = compute_cross_event_scores(event, similar, FailingVerifier())
        assert scores["FULL_WRITE"] == 0.0
        # Other policies have no deltas -> -inf
        for policy in ["MOTION_ONLY", "APPEARANCE_ONLY", "HOLD_BOTH", "SOFT_CAUTION"]:
            assert scores[policy] == -float("inf")

    def test_mixed_replay_success_failure(self):
        """When some replays fail, the successful ones still contribute."""
        event = TrackEvent(
            event_id="test",
            dataset="test", sequence="seq1", frame_id=0, track_id=0,
            image_width=640, image_height=480, has_detection=False,
        )
        similar = [
            (event, np.array([1.0, 0.0]), 0.8),
            (event, np.array([1.0, 0.0]), 0.7),
        ]

        class MixedVerifier:
            def __init__(self):
                self.call_count = 0

            def verify_event(self, evt, oracle):
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("First replay failed")
                return {"policy_deltas": {"MOTION_ONLY": 0.1}}

        verifier = MixedVerifier()
        scores = compute_cross_event_scores(event, similar, verifier)
        # Only 1 successful replay -> MOTION_ONLY with single element
        assert scores["MOTION_ONLY"] != -float("inf")
