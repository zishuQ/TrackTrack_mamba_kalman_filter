"""Shared structured-evidence construction for training and online inference.

Both sides feed the same 18 scalar values into the same positions:

* motion: 12 values from legacy 63-dim indices (27:31, 39:43, 43:47);
* context: 8 values = det_score, row_margin, col_margin, history_length,
  observation_gap, has_detection, row_margin_valid, col_margin_valid.

The legacy packed training tensors already contain z-scored scalar values.
Online inference must apply the same 63-dim statistics first; this module is
the single place where that convention is encoded.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shutrack.features.margins import MarginResult, normalize_margins
from shutrack.schema import (
    CONTEXT_DIM,
    LEGACY_SCALAR_DIM,
    MOTION_DIM,
    MOTION_SCALAR_INDICES,
)


def normalize_scalar63(
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply the legacy train-split z-score to one raw 63-dim vector."""

    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.shape != (LEGACY_SCALAR_DIM,):
        raise ValueError(
            f"raw scalar features must have shape ({LEGACY_SCALAR_DIM},), "
            f"got {values.shape}"
        )
    mean_array = np.asarray(mean, dtype=np.float64).reshape(-1)
    std_array = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean_array.shape != (LEGACY_SCALAR_DIM,) or std_array.shape != (
        LEGACY_SCALAR_DIM,
    ):
        raise ValueError("scalar normalization stats must be 63-dimensional")
    safe_std = np.where(np.isfinite(std_array) & (std_array > 0), std_array, 1.0)
    return (values - mean_array) / safe_std


def extract_motion(normalized_scalar63: np.ndarray) -> np.ndarray:
    """Select the 12 motion values from an already-normalized scalar vector."""

    values = np.asarray(normalized_scalar63, dtype=np.float64).reshape(-1)
    if values.shape != (LEGACY_SCALAR_DIM,):
        raise ValueError(
            f"normalized scalar features must have shape ({LEGACY_SCALAR_DIM},), "
            f"got {values.shape}"
        )
    motion = values[list(MOTION_SCALAR_INDICES)].astype(np.float32)
    if motion.shape != (MOTION_DIM,):
        raise AssertionError("motion extraction produced the wrong dimension")
    return motion


def build_context(
    normalized_scalar63: np.ndarray,
    margins: MarginResult,
    margin_stats: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Build the 8-dim context vector in the fixed schema order."""

    values = np.asarray(normalized_scalar63, dtype=np.float64).reshape(-1)
    if values.shape != (LEGACY_SCALAR_DIM,):
        raise ValueError(
            f"normalized scalar features must have shape ({LEGACY_SCALAR_DIM},), "
            f"got {values.shape}"
        )
    row_margin, col_margin = normalize_margins(margins, margin_stats)
    context = np.asarray(
        [
            values[9],    # detection score
            row_margin,
            col_margin,
            values[58],   # normalized history length
            values[59],   # normalized observation gap
            values[26],   # has_detection (legacy scalar value)
            1.0 if margins.row_valid else 0.0,
            1.0 if margins.col_valid else 0.0,
        ],
        dtype=np.float32,
    )
    if context.shape != (CONTEXT_DIM,):
        raise AssertionError("context construction produced the wrong dimension")
    return context


def build_core_features(
    normalized_scalar63: np.ndarray,
    margins: MarginResult,
    margin_stats: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(motion12, context8)`` from one already-normalized scalar."""

    return (
        extract_motion(normalized_scalar63),
        build_context(normalized_scalar63, margins, margin_stats),
    )


def sanitize_float32(array: np.ndarray) -> np.ndarray:
    """Replace non-finite values with zero and force float32.

    Training data is validated strictly elsewhere; the online path uses this
    helper so a single corrupted live feature cannot produce NaN gates.
    """

    return np.nan_to_num(
        np.asarray(array, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


__all__ = [
    "normalize_scalar63",
    "extract_motion",
    "build_context",
    "build_core_features",
    "sanitize_float32",
]
