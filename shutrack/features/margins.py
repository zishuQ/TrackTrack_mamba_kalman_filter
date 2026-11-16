"""Selected-match margin definition and bounded normalization statistics.

The margin is computed from the *recorded candidate pool* of one association
event.  It is deliberately not a re-run of the association algorithm and it
never uses future information.

For the accepted pair ``(row_index, col_index)``:

    row_margin = min(raw_cost[row, other feasible detection cols]) - raw_cost[row, col]
    col_margin = min(raw_cost[other feasible track rows, col]) - raw_cost[row, col]

A positive margin means the accepted match was cheaper than every other
feasible candidate.  A negative margin means a cheaper competitor exists in
the recorded pool.  When no other feasible candidate exists, the margin is 0
and its validity bit is 0 (this is distinct from "no competition").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from shutrack.schema import MARGIN_INFEASIBLE_IOU


@dataclass(frozen=True)
class MarginResult:
    row_margin: float
    col_margin: float
    row_valid: bool
    col_valid: bool

    def as_array(self) -> np.ndarray:
        return np.asarray([self.row_margin, self.col_margin], dtype=np.float32)

    def validity_array(self) -> np.ndarray:
        return np.asarray([self.row_valid, self.col_valid], dtype=np.bool_)


INVALID_MARGIN = MarginResult(0.0, 0.0, False, False)


def _as_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {array.shape}")
    return array


def compute_selected_margins(
    raw_cost: np.ndarray,
    iou_similarity: np.ndarray,
    row_index: int,
    col_index: int,
    *,
    infeasible_iou: float = MARGIN_INFEASIBLE_IOU,
) -> MarginResult:
    """Compute row/column margins for the accepted pair.

    ``raw_cost`` is the association cost *before* penalties, clipping and the
    infeasible-column sentinel are applied (the same matrix recorded by the
    compact event cache).  ``iou_similarity`` identifies infeasible entries:
    the legacy association marks pairs with ``iou <= 0.10`` as infeasible.
    Entries that are not finite are also excluded.  The accepted pair itself is
    never a candidate.
    """

    raw = _as_matrix(raw_cost, "raw_cost")
    iou = _as_matrix(iou_similarity, "iou_similarity")
    if raw.shape != iou.shape:
        raise ValueError(
            "raw_cost and iou_similarity must share a shape, got "
            f"{raw.shape} and {iou.shape}"
        )
    row = int(row_index)
    col = int(col_index)
    if row < 0 or col < 0 or row >= raw.shape[0] or col >= raw.shape[1]:
        return INVALID_MARGIN

    selected = raw[row, col]
    if not np.isfinite(selected):
        return INVALID_MARGIN

    finite = np.isfinite(raw) & np.isfinite(iou)
    feasible = finite & (iou > float(infeasible_iou))

    row_candidates = feasible[row, :].copy()
    row_candidates[col] = False
    if row_candidates.any():
        row_margin = float(np.min(raw[row, row_candidates]) - selected)
        row_valid = True
    else:
        row_margin = 0.0
        row_valid = False

    col_candidates = feasible[:, col].copy()
    col_candidates[row] = False
    if col_candidates.any():
        col_margin = float(np.min(raw[col_candidates, col]) - selected)
        col_valid = True
    else:
        col_margin = 0.0
        col_valid = False

    return MarginResult(row_margin, col_margin, row_valid, col_valid)


def compute_margins_from_association_record(
    association: Mapping[str, object],
    *,
    track_row: int,
    accepted_detection_index: int,
) -> MarginResult:
    """Compute margins from a compact association record.

    The record stores ``raw_cost``/``iou_similarity`` as ``(num_tracks,
    num_detections)`` matrices and ``detection_indices`` in matrix-column
    order.  The accepted detection is resolved exactly as the cache reader
    does; if it is absent or ambiguous the result is explicitly invalid.
    """

    if int(track_row) < 0 or int(accepted_detection_index) < 0:
        return INVALID_MARGIN
    detection_indices = np.asarray(
        association.get("detection_indices", []), dtype=np.int64
    ).reshape(-1)
    matches = np.flatnonzero(detection_indices == int(accepted_detection_index))
    if matches.size != 1:
        return INVALID_MARGIN
    raw_cost = association.get("raw_cost")
    iou_similarity = association.get("iou_similarity")
    if raw_cost is None or iou_similarity is None:
        return INVALID_MARGIN
    try:
        return compute_selected_margins(
            np.asarray(raw_cost),
            np.asarray(iou_similarity),
            int(track_row),
            int(matches[0]),
        )
    except ValueError:
        return INVALID_MARGIN


def fit_margin_statistics(
    observations: Iterable[MarginResult],
    *,
    min_std: float = 1e-6,
) -> dict[str, object]:
    """Stream valid margin observations into small normalization stats.

    Only valid margins contribute to mean/std.  The caller may persist this
    tiny dictionary; no per-event margin array is ever materialized.
    """

    count = np.zeros(2, dtype=np.int64)
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    for result in observations:
        values = result.as_array().astype(np.float64)
        valid = result.validity_array()
        if not valid.any():
            continue
        count += valid.astype(np.int64)
        total += np.where(valid, values, 0.0)
        total_sq += np.where(valid, values * values, 0.0)

    mean = np.zeros(2, dtype=np.float64)
    std = np.ones(2, dtype=np.float64)
    for index in range(2):
        if count[index] > 0:
            mean[index] = total[index] / count[index]
            variance = max(total_sq[index] / count[index] - mean[index] ** 2, 0.0)
            std[index] = max(float(np.sqrt(variance)), float(min_std))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count.tolist(),
        "valid_count": int(count.sum()),
        "min_std": float(min_std),
    }


DEFAULT_MARGIN_STATS: dict[str, object] = {
    "mean": [0.0, 0.0],
    "std": [1.0, 1.0],
    "count": [0, 0],
    "valid_count": 0,
    "min_std": 1e-6,
}


def normalize_margins(
    result: MarginResult,
    stats: Mapping[str, object] | None,
) -> tuple[float, float]:
    """Return z-scored margins; invalid entries become exactly zero."""

    mean = np.asarray(
        (stats or DEFAULT_MARGIN_STATS).get("mean", [0.0, 0.0]), dtype=np.float64
    ).reshape(2)
    std = np.asarray(
        (stats or DEFAULT_MARGIN_STATS).get("std", [1.0, 1.0]), dtype=np.float64
    ).reshape(2)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    values = result.as_array().astype(np.float64)
    valid = result.validity_array()
    normalized = np.where(valid, (values - mean) / std, 0.0)
    return float(normalized[0]), float(normalized[1])


__all__ = [
    "MarginResult",
    "INVALID_MARGIN",
    "compute_selected_margins",
    "compute_margins_from_association_record",
    "fit_margin_statistics",
    "normalize_margins",
    "DEFAULT_MARGIN_STATS",
]
