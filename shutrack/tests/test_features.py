"""Shared feature and window contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from shutrack.tests.conftest import make_evidence
from shutrack.data.evidence import EventEvidence
from shutrack.features.core import (
    build_context,
    build_core_features,
    extract_motion,
    normalize_scalar63,
    sanitize_float32,
)
from shutrack.features.margins import MarginResult
from shutrack.features.windows import (
    HISTORY_SIZE,
    build_window,
    select_history,
    window_from_sequence,
)
from shutrack.schema import CONTEXT_DIM, MOTION_DIM, MOTION_SCALAR_INDICES


def _raw_scalar() -> np.ndarray:
    raw = np.arange(63, dtype=np.float64)
    return raw


def test_motion_indices_and_context_order():
    raw = _raw_scalar()
    mean = np.zeros(63)
    std = np.ones(63)
    normalized = normalize_scalar63(raw, mean, std)
    motion = extract_motion(normalized)
    assert motion.shape == (MOTION_DIM,)
    assert motion.tolist() == [raw[index] for index in MOTION_SCALAR_INDICES]

    margins = MarginResult(1.5, -0.5, True, False)
    context = build_context(normalized, margins, {"mean": [0, 0], "std": [1, 1]})
    assert context.shape == (CONTEXT_DIM,)
    assert context[0] == pytest.approx(raw[9])
    assert context[1] == pytest.approx(1.5)
    assert context[2] == pytest.approx(0.0)  # invalid margin forced to zero
    assert context[3] == pytest.approx(raw[58])
    assert context[4] == pytest.approx(raw[59])
    assert context[5] == pytest.approx(raw[26])
    assert context[6] == pytest.approx(1.0)
    assert context[7] == pytest.approx(0.0)


def test_normalization_is_applied_once_by_construction():
    raw = _raw_scalar()
    mean = np.linspace(-0.2, 0.2, 63)
    std = np.linspace(0.5, 1.5, 63)
    normalized = normalize_scalar63(raw, mean, std)
    motion_packed_style, context_packed_style = build_core_features(
        normalized, MarginResult(0, 0, False, False)
    )
    motion_online_style = extract_motion(normalize_scalar63(raw, mean, std))
    np.testing.assert_allclose(motion_packed_style, motion_online_style)
    assert context_packed_style.shape == (CONTEXT_DIM,)


def test_evidence_rejects_non_finite():
    with pytest.raises(ValueError):
        EventEvidence(
            motion=np.full(12, np.nan, dtype=np.float32),
            context=np.zeros(8, dtype=np.float32),
            track_feature=np.ones(8, dtype=np.float32),
            detection_feature=np.zeros(8, dtype=np.float32),
            has_detection=True,
            frame_id=1,
            track_id=1,
            segment_id=0,
            observation_count=1,
        )
    assert np.isfinite(sanitize_float32(np.array([np.nan, np.inf, 1.0]))).all()


def test_window_left_pads_and_masks():
    current = make_evidence(frame_id=10, seed=10)
    history = [make_evidence(frame_id=8, seed=8), make_evidence(frame_id=9, seed=9)]
    window = build_window(history, current)
    assert window.history_motion.shape == (HISTORY_SIZE, MOTION_DIM)
    assert window.history_mask.tolist() == [True, True, True, False, False]
    assert window.history_has_detection.tolist() == [False, False, False, True, True]
    np.testing.assert_allclose(window.history_motion[-1], history[-1].motion)
    np.testing.assert_allclose(window.current_motion, current.motion)


def test_reset_drops_history_and_reset_mask_is_combined():
    current = make_evidence(frame_id=10, seed=10)
    history = [make_evidence(frame_id=8, seed=8), make_evidence(frame_id=9, seed=9)]
    reset_window = build_window(history, current, reset=True)
    assert reset_window.history_mask.tolist() == [True] * HISTORY_SIZE
    mask_window = build_window(
        history,
        current,
        reset_mask=np.array([False, False, False, True, False]),
    )
    assert mask_window.history_reset.tolist() == [False, False, False, True, False]
    assert mask_window.history_mask.tolist() == [True, True, True, True, False]


def test_select_history_stops_at_track_segment_and_future():
    events = [
        make_evidence(track_id=1, segment_id=0, frame_id=1, seed=1),
        make_evidence(track_id=1, segment_id=0, frame_id=2, seed=2),
        make_evidence(track_id=1, segment_id=1, frame_id=3, seed=3),
        make_evidence(track_id=2, segment_id=1, frame_id=4, seed=4),
        make_evidence(track_id=1, segment_id=1, frame_id=5, seed=5),
    ]
    current = make_evidence(track_id=1, segment_id=1, frame_id=6, seed=6)
    selected = select_history(events, current)
    # The walk stops at the first foreign track; only the contiguous
    # same-track suffix is usable.
    assert [event.frame_id for event in selected] == [5]

    contiguous = [
        make_evidence(track_id=1, segment_id=1, frame_id=3, seed=3),
        make_evidence(track_id=1, segment_id=1, frame_id=5, seed=5),
    ]
    selected = select_history(contiguous, current)
    assert [event.frame_id for event in selected] == [3, 5]

    future = make_evidence(track_id=1, segment_id=1, frame_id=7, seed=7)
    selected = select_history(contiguous + [future], current)
    assert [event.frame_id for event in selected] == [3, 5]


def test_online_evidence_matches_offline_core_selection():
    from shutrack.data.legacy_adapter import (
        build_online_evidence,
        legacy_shutrack_scalar63,
    )
    from shutrack.features.margins import INVALID_MARGIN

    params = {
        "has_detection": True,
        "image_width": 640,
        "image_height": 480,
        "frame_id": 7,
        "detection": {
            "box": np.array([95.0, 95.0, 115.0, 125.0]),
            "score": 0.9,
            "source": 0,
        },
        "association": {
            "iou_similarity": 0.7,
            "iou_distance": 0.3,
            "cosine_distance": 0.2,
            "confidence_distance": 0.1,
            "angle_distance": 0.05,
            "raw_cost": 0.25,
            "final_cost": 0.25,
            "assignment_round": 0,
            "assignment_threshold": 0.5,
            "detection_source": 0,
        },
        "pre_update_state": {
            "mean": np.array([100.0, 100.0, 40.0, 80.0, 1.0, 0.5, 0.0, 0.0]),
            "covariance": np.eye(8) * 2.0,
            "velocity": np.ones((4, 2)) * 0.3,
            "score": 0.9,
            "history": {1: [np.array([90.0, 90.0, 110.0, 120.0]), 0.9]},
            "end_frame_id": 6,
            "state": 1,
        },
    }
    mean = np.linspace(-0.1, 0.1, 63)
    std = np.linspace(0.5, 1.5, 63)
    raw = legacy_shutrack_scalar63(params)
    expected_motion, expected_context = build_core_features(
        normalize_scalar63(raw, mean, std), INVALID_MARGIN
    )
    evidence = build_online_evidence(
        scalar_params=params,
        track_feature=np.ones(8, dtype=np.float32),
        detection_feature=np.ones(8, dtype=np.float32),
        has_detection=True,
        margins=INVALID_MARGIN,
        margin_stats={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        scalar_mean=mean,
        scalar_std=std,
        frame_id=7,
        track_id=3,
        segment_id=0,
        observation_count=2,
    )
    np.testing.assert_allclose(evidence.motion, expected_motion, atol=1e-6)
    np.testing.assert_allclose(evidence.context, expected_context, atol=1e-6)


def test_window_from_sequence_is_causal_and_respects_reset():
    events = [
        make_evidence(track_id=1, segment_id=0, frame_id=i, seed=i) for i in range(1, 7)
    ] + [
        make_evidence(track_id=1, segment_id=1, frame_id=i, seed=100 + i)
        for i in range(7, 13)
    ]
    window = window_from_sequence(events, 9)
    # Current frame 10 in segment 1; only frames 7-9 are in the same segment.
    assert window.history_mask.tolist() == [True, True, False, False, False]
    assert window.history_motion[3].tolist() == pytest.approx(
        events[7].motion.tolist()
    )
