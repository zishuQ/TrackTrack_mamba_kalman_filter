"""Test that compute_appearance_benefit produces non-zero differences."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import DetectionObservation, TrackStateSnapshot
from agentguard.rollout.appearance import compute_appearance_benefit, ema_update
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.losses import appearance_loss


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
    """Convert old-style (event, oracle_data) arguments to a RolloutContext.

    The *oracle_data* dict may contain:
        future_gt_boxes         — list of (4,) boxes or None (default [None]*5)
        future_oracle_detections — list of (4,) raw boxes or None
        future_oracle_features   — list of (1, D) features or None
        future_oracle_scores     — list of float scores
        future_warp_matrices     — list of (2, 3) warp matrices
        current_gt_box           — (4,) current GT box

    Raw oracle boxes are converted to ``DetectionObservation`` objects with
    the corresponding feature and score (or defaults).
    """
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

    # Convert raw boxes → DetectionObservation objects
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


@pytest.fixture
def identity_prototype() -> np.ndarray:
    """Random unit-norm prototype of dimension 64."""
    rng = np.random.RandomState(0)
    v = rng.randn(64).astype(np.float64)
    return v / np.linalg.norm(v)


def test_appearance_benefit_nonzero_with_gt(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    identity_prototype: np.ndarray,
):
    """Appearance benefit should be non-zero when detection and oracle data exist."""
    ctx = _build_rollout_context(
        mock_event, mock_future_oracle_data, identity_prototype=identity_prototype,
    )
    B_a, write_losses, skip_losses, write_feats, skip_feats, valid_mask = (
        compute_appearance_benefit(ctx, future_frames=5)
    )
    # With a detection that has different features from the track feature,
    # the write and skip branches should diverge.
    assert not np.allclose(sum(write_losses), sum(skip_losses)), \
        "Write and skip branch appearance losses should differ"
    assert B_a != pytest.approx(0.0, abs=1e-10), \
        "Appearance benefit should be non-zero with detection and oracle GT"
    assert valid_mask.dtype == bool


def test_appearance_benefit_shapes(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    identity_prototype: np.ndarray,
):
    """Output shapes should be consistent with future_frames."""
    ctx = _build_rollout_context(
        mock_event, mock_future_oracle_data, identity_prototype=identity_prototype,
    )
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


def test_appearance_benefit_no_detection(
    mock_event_no_detection: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    identity_prototype: np.ndarray,
):
    """Without detection, write and skip branches should be identical -> benefit ~0."""
    ctx = _build_rollout_context(
        mock_event_no_detection, mock_future_oracle_data, identity_prototype=identity_prototype,
    )
    B_a, write_losses, skip_losses, write_feats, skip_feats, valid_mask = (
        compute_appearance_benefit(ctx, future_frames=3)
    )
    assert abs(B_a) < 1e-10
    np.testing.assert_allclose(write_feats[0], skip_feats[0])


def test_appearance_benefit_no_oracle(
    mock_event: TrackEvent,
    identity_prototype: np.ndarray,
):
    """Without oracle data, appearance losses should be based on current features
    and should differ because write branch already consumed detection feature."""
    empty_oracle: Dict[str, Any] = {
        "future_oracle_features": [],
        "future_oracle_scores": [],
    }
    ctx = _build_rollout_context(
        mock_event, empty_oracle, identity_prototype=identity_prototype,
    )
    B_a, write_losses, skip_losses, write_feats, skip_feats, valid_mask = (
        compute_appearance_benefit(ctx, future_frames=0)
    )
    # With 0 future frames, benefit should be 0
    assert B_a == 0.0
    assert len(write_losses) == 0
    assert len(skip_losses) == 0
    assert len(valid_mask) == 0


def test_appearance_benefit_feature_divergence(
    mock_event: TrackEvent,
    identity_prototype: np.ndarray,
):
    """After writing detection feature, write branch feature should differ from skip."""
    oracle_data: Dict[str, Any] = {
        "future_oracle_features": [],
        "future_oracle_scores": [],
    }
    ctx = _build_rollout_context(
        mock_event, oracle_data, identity_prototype=identity_prototype,
    )
    B_a, write_losses, skip_losses, write_feats, skip_feats, valid_mask = (
        compute_appearance_benefit(ctx, future_frames=0)
    )
    # With future_frames=0, no future rollout happens, so B_a=0 and no features
    assert B_a == 0.0
    assert len(valid_mask) == 0


def test_ema_update_formula():
    """Verify the EMA update formula is correct."""
    old = np.array([1.0, 0.0, 0.0])
    new = np.array([0.0, 1.0, 0.0])

    # score=0.9, alpha=0.95
    # beta = 0.95 + 0.05 * (1 - 0.9) = 0.95 + 0.005 = 0.955
    result = ema_update(old, new, score=0.9, alpha=0.95)
    expected_beta = 0.95 + 0.05 * 0.1  # = 0.955
    expected = expected_beta * old + (1.0 - expected_beta) * new
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_array_almost_equal(result, expected, decimal=10)


def test_ema_update_low_score():
    """Score=0 should fully preserve old feature."""
    old = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    new = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    result = ema_update(old, new, score=0.0, alpha=0.95)
    # beta = 0.95 + 0.05 * 1.0 = 1.0 => full preservation
    expected = old / np.linalg.norm(old)
    np.testing.assert_array_almost_equal(result, expected, decimal=10)


def test_ema_update_high_score():
    """Score=1.0 applies the standard alpha blending."""
    old = np.array([1.0, 0.0, 0.0])
    new = np.array([0.0, 1.0, 0.0])
    result = ema_update(old, new, score=1.0, alpha=0.95)
    # beta = 0.95 + 0.05 * 0.0 = 0.95
    expected = 0.95 * old + 0.05 * new
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_array_almost_equal(result, expected, decimal=10)


def test_appearance_loss_zero_for_identical():
    """Loss should be 0 when feature and prototype are identical."""
    feat = np.array([1.0, 0.0, 0.0])
    proto = np.array([1.0, 0.0, 0.0])
    assert appearance_loss(feat, proto) == 0.0


def test_appearance_loss_one_for_orthogonal():
    """Loss should be 1 when feature and prototype are orthogonal."""
    feat = np.array([1.0, 0.0, 0.0])
    proto = np.array([0.0, 1.0, 0.0])
    assert abs(appearance_loss(feat, proto) - 1.0) < 1e-10


def test_appearance_loss_zero_norm():
    """Loss should be 1.0 when feature is zero-norm."""
    feat = np.zeros(64)
    proto = np.ones(64) / np.linalg.norm(np.ones(64))
    assert appearance_loss(feat, proto) == 1.0


def test_appearance_benefit_pre_state_none_raises(
    identity_prototype: np.ndarray,
):
    """Event with no pre_update_state should raise ValueError."""
    ev = TrackEvent(
        event_id="no_state",
        dataset="test", sequence="test", frame_id=0, track_id=0,
        image_width=640, image_height=480, has_detection=False,
    )
    ctx = _build_rollout_context(ev, {}, identity_prototype=identity_prototype)
    with pytest.raises(ValueError, match="no pre_update_state"):
        compute_appearance_benefit(ctx, future_frames=5)


def test_appearance_benefit_different_alpha(
    mock_event: TrackEvent,
    mock_future_oracle_data: Dict[str, Any],
    identity_prototype: np.ndarray,
):
    """Different alpha values should lead to different benefits."""
    ctx_high = _build_rollout_context(
        mock_event, mock_future_oracle_data, identity_prototype=identity_prototype, appearance_alpha=0.99,
    )
    ctx_low = _build_rollout_context(
        mock_event, mock_future_oracle_data, identity_prototype=identity_prototype, appearance_alpha=0.50,
    )
    B_a_high, *_ = compute_appearance_benefit(ctx_high, future_frames=3)
    B_a_low, *_ = compute_appearance_benefit(ctx_low, future_frames=3)
    # Different alpha values should produce different benefits
    assert not np.allclose(B_a_high, B_a_low), \
        "Different alpha values should produce different benefits"
