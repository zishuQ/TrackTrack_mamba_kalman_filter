"""Margin definition tests (recorded candidate pool semantics)."""

from __future__ import annotations

import numpy as np
import pytest

from shutrack.features.margins import (
    INVALID_MARGIN,
    MarginResult,
    compute_margins_from_association_record,
    compute_selected_margins,
    fit_margin_statistics,
    normalize_margins,
)


def test_selected_is_best_in_row_and_column():
    raw = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.6, 0.1, 0.85],
            [0.75, 0.95, 0.9],
        ]
    )
    iou = np.ones_like(raw)
    result = compute_selected_margins(raw, iou, row_index=1, col_index=1)
    assert result.row_valid and result.col_valid
    assert result.row_margin == pytest.approx(0.5)
    # Column candidates are 0.8 and 0.95; the selected 0.1 is best.
    assert result.col_margin == pytest.approx(0.7)


def test_selected_is_not_row_or_column_optimal():
    raw = np.array(
        [
            [0.2, 0.1, 0.9],
            [0.3, 0.7, 0.8],
            [0.4, 0.5, 0.6],
        ]
    )
    iou = np.ones_like(raw)
    # Selected (1, 1) costs 0.7; a cheaper row competitor is 0.3,
    # a cheaper column competitor is 0.1.
    result = compute_selected_margins(raw, iou, row_index=1, col_index=1)
    assert result.row_valid and result.col_valid
    assert result.row_margin == pytest.approx(0.3 - 0.7)
    assert result.col_margin == pytest.approx(0.1 - 0.7)


def test_no_other_candidates_is_explicitly_invalid():
    result = compute_selected_margins(
        np.array([[0.4]]), np.ones((1, 1)), row_index=0, col_index=0
    )
    assert result == INVALID_MARGIN
    assert result.row_margin == 0.0 and not result.row_valid
    assert result.col_margin == 0.0 and not result.col_valid


def test_infeasible_and_non_finite_candidates_are_excluded():
    # The selected pair is feasible, but every alternative is either
    # infeasible (IoU <= 0.10) or non-finite, so both margins are invalid.
    raw = np.array([[0.8, 0.1], [0.2, np.nan]])
    iou = np.array([[0.05, 1.0], [1.0, 0.05]])
    result = compute_selected_margins(raw, iou, row_index=0, col_index=1)
    assert not result.row_valid
    assert not result.col_valid
    assert result.row_margin == 0.0 and result.col_margin == 0.0


def test_infeasible_candidates_do_not_count_as_competition():
    raw = np.array([[0.8, 0.2, 0.6], [0.3, 0.9, 0.4]])
    iou = np.array([[1.0, 1.0, 1.0], [0.05, 1.0, 1.0]])
    result = compute_selected_margins(raw, iou, row_index=0, col_index=1)
    # Row candidates: (0,0)=0.8, (0,2)=0.6 -> min 0.6, margin 0.4.
    # Column candidates: only (0,1) feasible? (1,1) is feasible (iou 1.0)
    # and costs 0.9 -> margin 0.7.
    assert result.row_margin == pytest.approx(0.4)
    assert result.col_margin == pytest.approx(0.7)


def test_invalid_indices_return_invalid():
    raw = np.zeros((2, 2))
    iou = np.ones((2, 2))
    assert compute_selected_margins(raw, iou, -1, 0) == INVALID_MARGIN
    assert compute_selected_margins(raw, iou, 0, -1) == INVALID_MARGIN
    assert compute_selected_margins(raw, iou, 5, 0) == INVALID_MARGIN


def test_association_record_resolves_accepted_detection_column():
    association = {
        "raw_cost": np.array([[0.9, 0.2], [0.1, 0.8]]),
        "iou_similarity": np.ones((2, 2)),
        "detection_indices": np.array([101, 202]),
    }
    result = compute_margins_from_association_record(
        association, track_row=0, accepted_detection_index=202
    )
    assert result.row_valid and result.col_valid
    assert result.row_margin == pytest.approx(0.7)
    # Column candidates: row 1 costs 0.8, so the selected 0.2 wins by 0.6.
    assert result.col_margin == pytest.approx(0.6)

    missing = compute_margins_from_association_record(
        association, track_row=0, accepted_detection_index=999
    )
    assert missing == INVALID_MARGIN

    ambiguous = {
        **association,
        "detection_indices": np.array([202, 202]),
    }
    assert (
        compute_margins_from_association_record(
            ambiguous, track_row=0, accepted_detection_index=202
        )
        == INVALID_MARGIN
    )


def test_margin_statistics_only_use_valid_entries_and_invalid_is_zero():
    observations = [
        MarginResult(2.0, 10.0, True, True),
        MarginResult(4.0, 20.0, True, True),
        MarginResult(0.0, 0.0, False, False),
        MarginResult(-2.0, 5.0, True, True),
    ]
    stats = fit_margin_statistics(observations)
    assert stats["count"] == [3, 3]
    mean = np.asarray(stats["mean"])
    assert mean[0] == pytest.approx((2.0 + 4.0 - 2.0) / 3.0)
    row, col = normalize_margins(MarginResult(0.0, 0.0, False, False), stats)
    assert row == 0.0 and col == 0.0
    row, col = normalize_margins(MarginResult(4.0, 20.0, True, True), stats)
    assert row == pytest.approx((4.0 - mean[0]) / float(np.std([2.0, 4.0, -2.0])))
