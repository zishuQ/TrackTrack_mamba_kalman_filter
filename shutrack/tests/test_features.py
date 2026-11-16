"""Shared evidence, competition and window contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from shutrack.data.evidence import EventEvidence
from shutrack.features.core import (
    build_competition,
    build_evidence,
    normalize_scalar63,
    sanitize_float32,
)
from shutrack.features.windows import (
    HISTORY_SIZE,
    build_window,
    select_history,
    window_from_sequence,
)
from shutrack.schema import COMPETITION_DIM, EVIDENCE_DIM, EVIDENCE_SCALAR_INDICES
from shutrack.tests.conftest import make_evidence


def _raw_scalar() -> np.ndarray:
    return np.arange(63, dtype=np.float64)


def test_evidence_indices_order_and_has_detection_flag():
    raw = _raw_scalar()
    mean = np.zeros(63)
    std = np.ones(63)
    normalized = normalize_scalar63(raw, mean, std)
    evidence = build_evidence(normalized, has_detection=True)
    assert evidence.shape == (EVIDENCE_DIM,)
    assert evidence.tolist() == [raw[index] for index in EVIDENCE_SCALAR_INDICES] + [1.0]
    unmatched = build_evidence(normalized, has_detection=False)
    assert unmatched[-1] == 0.0


def test_competition_indices_are_legacy_columns_15_21_25():
    raw = _raw_scalar()
    competition = build_competition(raw)
    assert competition.shape == (COMPETITION_DIM,)
    assert competition.tolist() == [raw[15], raw[21], raw[25]]


def test_normalization_is_applied_once_by_construction():
    raw = _raw_scalar()
    mean = np.linspace(-0.2, 0.2, 63)
    std = np.linspace(0.5, 1.5, 63)
    normalized = normalize_scalar63(raw, mean, std)
    packed_style = build_evidence(normalized, True)
    online_style = build_evidence(normalize_scalar63(raw, mean, std), True)
    np.testing.assert_allclose(packed_style, online_style)


def test_evidence_rejects_non_finite():
    with pytest.raises(ValueError):
        EventEvidence(
            evidence=np.full(13, np.nan, dtype=np.float32),
            competition=np.zeros(3, dtype=np.float32),
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
    assert window.history_evidence.shape == (HISTORY_SIZE, EVIDENCE_DIM)
    assert window.history_mask.tolist() == [True, True, True, False, False]
    assert window.history_has_detection.tolist() == [False, False, False, True, True]
    np.testing.assert_allclose(window.history_evidence[-1], history[-1].evidence)
    np.testing.assert_allclose(window.current_evidence, current.evidence)
    np.testing.assert_allclose(window.current_competition, current.competition)


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


def test_window_from_sequence_is_causal_and_respects_reset():
    events = [
        make_evidence(track_id=1, segment_id=0, frame_id=i, seed=i) for i in range(1, 7)
    ] + [
        make_evidence(track_id=1, segment_id=1, frame_id=i, seed=100 + i)
        for i in range(7, 13)
    ]
    window = window_from_sequence(events, 9)
    assert window.history_mask.tolist() == [True, True, False, False, False]
    np.testing.assert_allclose(
        window.history_evidence[3], events[7].evidence, atol=1e-6
    )
