"""Tests for the rollout and labels modules."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.appearance import (
    compute_appearance_benefit,
    ema_update,
)
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.losses import (
    appearance_loss,
    iou_loss,
    l1_normalized_loss,
    motion_frame_loss,
    sigmoid,
)
from agentguard.rollout.motion import compute_motion_benefit, compute_motion_rollout
from agentguard.rollout.window import (
    compute_tgr_window_labels,
    generate_window_augmentations,
)
from agentguard.rollout_labels import (
    compute_dataset_stats,
    compute_policy_soft_target,
    compute_soft_target,
    build_rollout_labels,
)


# ---------------------------------------------------------------------------
#  Helper: build a RolloutContext from old-style (event, oracle_data)
# ---------------------------------------------------------------------------


def _build_rollout_context(
    event: TrackEvent,
    oracle_data: dict,
    *,
    identity_prototype: Optional[np.ndarray] = None,
    appearance_alpha: float = 0.95,
) -> RolloutContext:
    """Convert old-style (event, oracle_data) arguments to a RolloutContext."""
    # Use explicit future_gt_boxes if provided; otherwise fall back to
    # future_oracle_detections (old-style) as the GT boxes for loss computation.
    future_gt = oracle_data.get("future_gt_boxes")
    if future_gt is None:
        future_gt = oracle_data.get("future_oracle_detections", [None] * 5)

    raw_dets = oracle_data.get("future_oracle_detections", [None] * 5)
    raw_feats = oracle_data.get("future_oracle_features", [None] * 5)
    raw_scores = oracle_data.get("future_oracle_scores", [0.0] * 5)
    future_warps = oracle_data.get("future_warp_matrices", [np.eye(2, 3)] * 5)
    current_gt = oracle_data.get("current_gt_box")
    if current_gt is None:
        current_gt = np.array([0, 0, 0, 0], dtype=np.float64)

    oracle_dets: List[Optional[DetectionObservation]] = []
    for i in range(len(raw_dets)):
        box = raw_dets[i]
        if box is None:
            oracle_dets.append(None)
        else:
            feat = raw_feats[i] if i < len(raw_feats) and raw_feats[i] is not None else np.zeros((1, 64), dtype=np.float64)
            score = raw_scores[i] if i < len(raw_scores) else 0.0
            oracle_dets.append(
                DetectionObservation(
                    detection_index=i,
                    box=box,
                    score=score,
                    feature=feat,
                    source=0,
                    class_id=1,
                )
            )

    return RolloutContext(
        frame_id=event.frame_id,
        target_gt_id=getattr(event, "target_gt_id", -1),
        pre_update_state=event.pre_update_state,
        current_candidate=event.detection if event.has_detection else None,
        current_gt_box=current_gt,
        future_gt_boxes=future_gt,
        future_oracle_detections=oracle_dets,
        future_warp_matrices=future_warps,
        identity_prototype=identity_prototype,
        appearance_alpha=appearance_alpha,
    )


# =========================================================================
#  Fixtures
# =========================================================================


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def kf():
    return NSAKalmanFilter()


@pytest.fixture
def identity_prototype():
    """A random unit-norm prototype of dimension 64."""
    v = np.random.RandomState(0).randn(64).astype(np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def make_event(rng):
    """Factory for a minimal TrackEvent with a pre_update_state."""

    def _make(
        track_id=1,
        frame_id=100,
        has_detection=True,
        det_box=None,
    ):
        # Build a pre-update KF state
        mean = np.array([100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(8, dtype=np.float64) * 0.1
        box = np.array([75.0, 75.0, 125.0, 125.0], dtype=np.float64)  # x1y1x2y2
        feature = np.ones((1, 64), dtype=np.float64) * 0.5

        state = TrackStateSnapshot(
            track_id=track_id,
            box=box,
            score=0.9,
            mean=mean,
            covariance=cov,
            velocity=np.zeros((4, 2), dtype=np.float64),
            feature=feature,
            history={},
            end_frame_id=frame_id - 1,
            state=1,  # TRACKED
        )

        if det_box is None:
            det_box = np.array([76.0, 76.0, 126.0, 126.0], dtype=np.float64)

        det = DetectionObservation(
            detection_index=0,
            box=det_box,
            score=0.85,
            feature=np.ones((1, 64), dtype=np.float64) * 0.6,
            source=0,
            class_id=1,
        )

        assoc = AssociationPairFeatures(
            iou_similarity=0.8,
            iou_distance=0.2,
            cosine_distance=0.1,
            confidence_distance=0.05,
            angle_distance=0.01,
            raw_cost=0.3,
            final_cost=0.25,
            assignment_round=0,
            assignment_threshold=-1.0,
            detection_source=0,
        )

        event = TrackEvent(
            event_id=f"test_{track_id}_{frame_id}",
            dataset="test_dataset",
            sequence="test_seq",
            frame_id=frame_id,
            track_id=track_id,
            image_width=1920,
            image_height=1080,
            has_detection=has_detection,
            frame_start_state=state,
            pre_update_state=state,
            detection=det if has_detection else None,
            association=assoc,
            warp_matrix=np.eye(2, 3, dtype=np.float64),
            scalar_features=np.zeros(63, dtype=np.float64),
            track_feature=np.ones(64, dtype=np.float64) * 0.5,
            detection_feature=np.ones(64, dtype=np.float64) * 0.6 if has_detection else np.zeros(0, dtype=np.float64),
        )
        return event

    return _make


@pytest.fixture
def future_oracle_data():
    """Standard future oracle data for 5 frames."""
    boxes = []
    features = []
    scores = []
    warp_mats = []
    for i in range(5):
        shift = (i + 1) * 2.0
        boxes.append(np.array([75.0 + shift, 75.0 + shift, 125.0 + shift, 125.0 + shift], dtype=np.float64))
        features.append(np.ones((1, 64), dtype=np.float64) * (0.5 + i * 0.05))
        scores.append(0.9 - i * 0.02)
        warp_mats.append(np.eye(2, 3, dtype=np.float64))

    return {
        "future_oracle_detections": boxes,
        "future_oracle_features": features,
        "future_oracle_scores": scores,
        "future_warp_matrices": warp_mats,
    }


# =========================================================================
#  Tests: Losses
# =========================================================================


class TestLosses:
    def test_iou_loss_perfect(self):
        box = np.array([10.0, 10.0, 50.0, 50.0])
        loss = iou_loss(box, box)
        assert loss == 0.0

    def test_iou_loss_no_overlap(self):
        box_a = np.array([0.0, 0.0, 10.0, 10.0])
        box_b = np.array([20.0, 20.0, 30.0, 30.0])
        loss = iou_loss(box_a, box_b)
        assert loss == 1.0

    def test_iou_loss_partial(self):
        box_a = np.array([0.0, 0.0, 10.0, 10.0])
        box_b = np.array([5.0, 5.0, 15.0, 15.0])
        loss = iou_loss(box_a, box_b)
        # area_a=100, area_b=100, inter=25, union=175, iou=25/175≈0.1429
        expected = 1.0 - 25.0 / 175.0
        assert abs(loss - expected) < 1e-6

    def test_l1_normalized_loss(self):
        box_a = np.array([10.0, 10.0, 30.0, 30.0])
        box_b = np.array([12.0, 14.0, 32.0, 34.0])
        loss = l1_normalized_loss(box_a, box_b)
        # w=20, h=20
        # |10-12|/20 + |10-14|/20 + |30-32|/20 + |30-34|/20 = 0.1+0.2+0.1+0.2 = 0.6
        # /4 = 0.15
        assert abs(loss - 0.15) < 1e-6

    def test_motion_frame_loss(self):
        box_a = np.array([0.0, 0.0, 10.0, 10.0])
        box_b = np.array([0.0, 0.0, 10.0, 10.0])
        assert motion_frame_loss(box_a, box_b) == 0.0

    def test_appearance_loss_identical(self):
        feat = np.array([1.0, 0.0, 0.0])
        proto = np.array([1.0, 0.0, 0.0])
        assert appearance_loss(feat, proto) == 0.0

    def test_appearance_loss_orthogonal(self):
        feat = np.array([1.0, 0.0, 0.0])
        proto = np.array([0.0, 1.0, 0.0])
        assert abs(appearance_loss(feat, proto) - 1.0) < 1e-6

    def test_appearance_loss_zero_norm(self):
        feat = np.zeros(64)
        proto = np.ones(64) / np.linalg.norm(np.ones(64))
        assert appearance_loss(feat, proto) == 1.0

    def test_sigmoid(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-6
        assert sigmoid(100.0) == 1.0
        assert sigmoid(-100.0) < 1e-15  # effectively zero
        assert sigmoid(1.0) == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


# =========================================================================
#  Tests: Motion rollout
# =========================================================================


class TestMotionRollout:
    def test_compute_motion_benefit_basic(self, make_event, future_oracle_data, kf):
        event = make_event()
        ctx = _build_rollout_context(event, future_oracle_data)
        B_m, write_losses, skip_losses, valid_mask = (
            compute_motion_benefit(ctx, kf, future_frames=3)
        )
        assert len(write_losses) == 3
        assert len(skip_losses) == 3
        assert len(valid_mask) == 3
        assert valid_mask.dtype == bool
        # All losses should be non-negative
        for wl in write_losses:
            assert wl >= 0.0
        for sl in skip_losses:
            assert sl >= 0.0
        # B_m should be a float (could be positive or negative)
        assert isinstance(B_m, float)

    def test_motion_benefit_no_detection(self, make_event, future_oracle_data, kf):
        event = make_event(has_detection=False)
        ctx = _build_rollout_context(event, future_oracle_data)
        B_m, write_losses, skip_losses, valid_mask = (
            compute_motion_benefit(ctx, kf, future_frames=3)
        )
        # Without a detection, write and skip branches start identically,
        # so losses should be equal (benefit ~ 0)
        assert abs(B_m) < 1e-10
        assert write_losses == skip_losses

    def test_motion_benefit_no_oracle(self, make_event, kf):
        event = make_event()
        empty_oracle = {
            "future_oracle_detections": [],
            "future_warp_matrices": [],
        }
        ctx = _build_rollout_context(event, empty_oracle)
        B_m, write_losses, skip_losses, valid_mask = (
            compute_motion_benefit(ctx, kf, future_frames=5)
        )
        # No oracle => all losses are 0
        assert B_m == 0.0
        assert all(wl == 0.0 for wl in write_losses)
        assert all(sl == 0.0 for sl in skip_losses)
        assert len(valid_mask) == 0

    def test_compute_motion_rollout_dict(self, make_event, future_oracle_data, kf):
        event = make_event()
        ctx = _build_rollout_context(event, future_oracle_data)
        result = compute_motion_rollout(ctx, kf, future_frames=3)
        assert "benefit" in result
        assert "write_losses" in result
        assert "skip_losses" in result
        assert "valid_mask" in result
        assert isinstance(result["benefit"], float)
        assert len(result["valid_mask"]) == 3


# =========================================================================
#  Tests: Appearance rollout
# =========================================================================


class TestAppearanceRollout:
    def test_compute_appearance_benefit_basic(self, make_event, future_oracle_data, identity_prototype):
        event = make_event()
        ctx = _build_rollout_context(event, future_oracle_data, identity_prototype=identity_prototype)
        B_a, write_losses, skip_losses, write_feats, skip_feats, valid_mask = (
            compute_appearance_benefit(ctx, future_frames=3)
        )
        assert len(write_losses) == 3
        assert len(skip_losses) == 3
        assert len(write_feats) == 3
        assert len(skip_feats) == 3
        assert len(valid_mask) == 3
        assert valid_mask.dtype == bool
        assert isinstance(B_a, float)

    def test_ema_update(self):
        old = np.array([1.0, 0.0, 0.0])
        new = np.array([0.0, 1.0, 0.0])
        result = ema_update(old, new, score=0.9, alpha=0.95)
        # beta = 0.95 + 0.05 * 0.1 = 0.955
        # feat = 0.955 * [1,0,0] + 0.045 * [0,1,0] = [0.955, 0.045, 0]
        # norm = sqrt(0.955^2 + 0.045^2) ≈ 0.95607
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        assert result[0] > result[1]

    def test_ema_update_low_score(self):
        old = np.array([1.0, 0.0, 0.0])
        new = np.array([0.0, 1.0, 0.0])
        result = ema_update(old, new, score=0.0, alpha=0.95)
        # beta = 0.95 + 0.05 * 1.0 = 1.0  => old fully preserved
        assert np.allclose(result, old / np.linalg.norm(old))

    def test_ema_update_high_score(self):
        old = np.array([1.0, 0.0, 0.0])
        new = np.array([0.0, 1.0, 0.0])
        result = ema_update(old, new, score=1.0, alpha=0.95)
        # beta = 0.95 + 0.05 * 0.0 = 0.95
        expected = 0.95 * old + 0.05 * new
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(result, expected)

    def test_appearance_no_detection(self, make_event, future_oracle_data, identity_prototype):
        event = make_event(has_detection=False)
        ctx = _build_rollout_context(event, future_oracle_data, identity_prototype=identity_prototype)
        B_a, write_losses, skip_losses, write_feats, skip_feats, valid_mask = (
            compute_appearance_benefit(ctx, future_frames=3)
        )
        # Without detection, write and skip start identically
        assert abs(B_a) < 1e-10
        np.testing.assert_allclose(write_feats[0], skip_feats[0])


# =========================================================================
#  Tests: Window / TGR
# =========================================================================


class TestWindowLabels:
    def test_compute_tgr_window_labels(self, make_event, kf, identity_prototype):
        # Create 4 events with slight box drift
        events = []
        boxes = []
        for i in range(4):
            box = np.array([75.0 + i * 2, 75.0 + i * 2, 125.0 + i * 2, 125.0 + i * 2],
                           dtype=np.float64)
            boxes.append(box)
            evt = make_event(track_id=1, frame_id=100 + i, det_box=box)
            events.append(evt)

        future_data = {
            "future_oracle_detections": [
                np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64),
                np.array([87.0, 87.0, 137.0, 137.0], dtype=np.float64),
                np.array([89.0, 89.0, 139.0, 139.0], dtype=np.float64),
            ],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)] * 3,
            "future_oracle_features": [
                np.ones((1, 64), dtype=np.float64) * 0.7,
                np.ones((1, 64), dtype=np.float64) * 0.75,
                np.ones((1, 64), dtype=np.float64) * 0.8,
            ],
            "future_oracle_scores": [0.9, 0.88, 0.85],
        }

        result = compute_tgr_window_labels(
            events, future_data, kf, identity_prototype, future_frames=3
        )

        assert "motion_sequence" in result
        assert "appearance_sequence" in result
        assert "valid_mask" in result
        assert result["motion_sequence"].shape == (4,)
        assert result["appearance_sequence"].shape == (4,)
        assert result["valid_mask"].shape == (4,)
        assert result["motion_losses"].shape == (16,)
        assert result["appearance_losses"].shape == (16,)

        # Valid mask should be all True (all events have detections)
        assert result["valid_mask"].all()

    def test_window_no_detection_positions(self, make_event, kf, identity_prototype):
        events = []
        for i in range(4):
            # Event at index 1 has no detection
            has_det = i != 1
            box = np.array([75.0, 75.0, 125.0, 125.0], dtype=np.float64) if has_det else None
            evt = make_event(track_id=1, frame_id=100 + i, has_detection=has_det, det_box=box)
            events.append(evt)

        future_data = {
            "future_oracle_detections": [np.array([85.0, 85.0, 135.0, 135.0], dtype=np.float64)],
            "future_warp_matrices": [np.eye(2, 3, dtype=np.float64)],
            "future_oracle_features": [np.ones((1, 64), dtype=np.float64) * 0.7],
            "future_oracle_scores": [0.9],
        }

        result = compute_tgr_window_labels(
            events, future_data, kf, identity_prototype, future_frames=1
        )
        assert not result["valid_mask"][1]
        assert result["valid_mask"][0]
        assert result["valid_mask"][2]
        assert result["valid_mask"][3]

    def test_generate_window_augmentations_returns_list(self, make_event):
        events = [make_event() for _ in range(4)]
        # Pass a minimal object that mimics CandidateBuilder
        class MockBuilder:
            def build_candidates(self, event, cm, g, t, n, l, h):
                return {"b": None}  # No B candidates available

        variants = generate_window_augmentations(events, MockBuilder())
        assert len(variants) >= 1
        assert all(len(v) == 4 for v in variants)
        # Only original since no B candidates
        assert len(variants) == 1


# =========================================================================
#  Tests: Labels module
# =========================================================================


class TestLabels:
    def test_compute_dataset_stats(self):
        all_benefits = {
            "motion_benefits": [-0.05, 0.02, 0.0, -0.08, 0.01, 0.12],
            "appearance_benefits": [0.03, -0.01, 0.0, 0.06, -0.04],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AG_GUARD_DATASET"] = "test_ds"
            # Override save path via mocking save_dir would be cleaner,
            # but we can just check the returned values
            stats = compute_dataset_stats(all_benefits)

        assert "tau_motion" in stats
        assert "tau_appearance" in stats
        assert 1e-3 <= stats["tau_motion"] <= 0.1
        assert 1e-3 <= stats["tau_appearance"] <= 0.1

    def test_compute_dataset_stats_all_zero(self):
        all_benefits = {
            "motion_benefits": [0.0, 0.0],
            "appearance_benefits": [0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AG_GUARD_DATASET"] = "test_ds"
            stats = compute_dataset_stats(all_benefits)
        # Should fall back to 0.1 when no non-zero benefits
        assert stats["tau_motion"] == 0.1
        assert stats["tau_appearance"] == 0.1

    def test_compute_soft_target(self):
        # Large positive benefit => near 1 (write is better)
        assert compute_soft_target(0.1, 0.01) > 0.999
        # Large negative benefit => near 0 (skip is better)
        assert compute_soft_target(-0.1, 0.01) < 0.001
        # Zero benefit => 0.5
        assert abs(compute_soft_target(0.0, 0.01) - 0.5) < 1e-6

    def test_compute_policy_soft_target(self):
        gate = np.array([1.0, 0.0], dtype=np.float64)  # MOTION_ONLY
        probs = compute_policy_soft_target(gate)
        assert probs.shape == (5,)
        assert abs(np.sum(probs) - 1.0) < 1e-6
        # The closest prototype should be MOTION_ONLY at [1.0, 0.0]
        motion_only_idx = 1
        assert probs[motion_only_idx] > 0.5

    def test_compute_policy_soft_target_full_write(self):
        gate = np.array([1.0, 1.0], dtype=np.float64)  # FULL_WRITE
        probs = compute_policy_soft_target(gate)
        full_write_idx = 0
        assert probs[full_write_idx] > 0.5

    def test_compute_policy_soft_target_hold_both(self):
        gate = np.array([0.0, 0.0], dtype=np.float64)  # HOLD_BOTH
        probs = compute_policy_soft_target(gate)
        hold_both_idx = 3
        assert probs[hold_both_idx] > 0.5

    def test_build_rollout_labels(self, make_event, identity_prototype):
        events = [make_event(track_id=1, frame_id=100),
                  make_event(track_id=1, frame_id=101),
                  make_event(track_id=2, frame_id=200)]
        motion_benefits = [-0.05, 0.02, 0.0]
        appearance_benefits = [0.03, -0.01, 0.01]
        dataset_stats = {"tau_motion": 0.02, "tau_appearance": 0.015}
        prototypes = {1: identity_prototype, 2: identity_prototype}

        labels = build_rollout_labels(
            events, motion_benefits, appearance_benefits,
            dataset_stats, prototypes,
        )

        assert len(labels) == 3
        for label in labels:
            assert "motion_target" in label
            assert "appearance_target" in label
            assert "policy_soft_target" in label
            assert "sample_type" in label
            assert "valid_motion" in label
            assert "valid_appearance" in label
            assert "sample_weight" in label
            assert label["label_schema_version"] == 3
            assert len(label["cue_target"]) == 3
            assert len(label["risk_targets"]) == 4
            assert label["valid_motion"] is True
            assert label["valid_appearance"] is True
            assert label["sample_weight"] == 1.0
            assert label["policy_soft_target"].shape == (5,)
            assert abs(np.sum(label["policy_soft_target"]) - 1.0) < 1e-6

    def test_build_rollout_labels_sample_types(self, make_event, identity_prototype):
        events = [
            make_event(track_id=1, frame_id=100, has_detection=True),
            make_event(track_id=1, frame_id=101, has_detection=False),
        ]
        motion_benefits = [0.01, -0.02]
        appearance_benefits = [0.01, 0.02]
        dataset_stats = {"tau_motion": 0.02, "tau_appearance": 0.02}
        prototypes = {1: identity_prototype}

        labels = build_rollout_labels(
            events, motion_benefits, appearance_benefits,
            dataset_stats, prototypes,
        )
        assert labels[0]["sample_type"] == "matched"
        assert labels[1]["sample_type"] == "unmatched"


# =========================================================================
#  Integration test: motion + appearance + window
# =========================================================================


class TestIntegration:
    def test_end_to_end_rollout(self, make_event, future_oracle_data, kf, identity_prototype):
        """Run motion and appearance benefit computation, collect stats, build labels."""
        events = [make_event(track_id=1, frame_id=100 + i) for i in range(5)]

        all_motion = []
        all_appearance = []
        for evt in events:
            ctx_m = _build_rollout_context(evt, future_oracle_data)
            B_m, *_ = compute_motion_benefit(ctx_m, kf, future_frames=3)
            ctx_a = _build_rollout_context(
                evt, future_oracle_data, identity_prototype=identity_prototype,
            )
            B_a, *_ = compute_appearance_benefit(ctx_a, future_frames=3)
            all_motion.append(B_m)
            all_appearance.append(B_a)

        all_benefits = {
            "motion_benefits": all_motion,
            "appearance_benefits": all_appearance,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AG_GUARD_DATASET"] = "integration_test"
            stats = compute_dataset_stats(all_benefits)

        prototypes = {1: identity_prototype}
        labels = build_rollout_labels(events, all_motion, all_appearance, stats, prototypes)

        assert len(labels) == 5
        for lbl in labels:
            assert 0.0 <= lbl["motion_target"] <= 1.0
            assert 0.0 <= lbl["appearance_target"] <= 1.0
