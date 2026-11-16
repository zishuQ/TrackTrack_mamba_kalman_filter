"""Shared evidence construction for training and online inference.

Both sides feed the same values into the same positions:

* evidence (13): 12 normalized legacy scalar columns (27:31, 39:43, 43:45,
  9, 59) plus a raw 0/1 has-detection flag;
* competition (3, optional branch): normalized legacy columns 15, 21, 25.

The legacy packed training tensors already contain z-scored scalar values.
Online inference must apply the same 63-dim statistics first; this module is
the single place where that convention is encoded.
"""

from __future__ import annotations

import numpy as np

from shutrack.schema import (
    COMPETITION_DIM,
    COMPETITION_SCALAR_INDICES,
    EVIDENCE_DIM,
    EVIDENCE_SCALAR_INDICES,
    LEGACY_SCALAR_DIM,
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


def build_evidence(
    normalized_scalar63: np.ndarray,
    has_detection: bool,
) -> np.ndarray:
    """Build the 13-dim evidence vector from a normalized 63-dim scalar."""

    values = np.asarray(normalized_scalar63, dtype=np.float64).reshape(-1)
    if values.shape != (LEGACY_SCALAR_DIM,):
        raise ValueError(
            f"normalized scalar features must have shape ({LEGACY_SCALAR_DIM},), "
            f"got {values.shape}"
        )
    evidence = np.asarray(
        [values[index] for index in EVIDENCE_SCALAR_INDICES]
        + [1.0 if has_detection else 0.0],
        dtype=np.float32,
    )
    if evidence.shape != (EVIDENCE_DIM,):
        raise AssertionError("evidence construction produced the wrong dimension")
    return evidence


def build_competition(normalized_scalar63: np.ndarray) -> np.ndarray:
    """Build the 3-dim competition vector from a normalized 63-dim scalar."""

    values = np.asarray(normalized_scalar63, dtype=np.float64).reshape(-1)
    if values.shape != (LEGACY_SCALAR_DIM,):
        raise ValueError(
            f"normalized scalar features must have shape ({LEGACY_SCALAR_DIM},), "
            f"got {values.shape}"
        )
    competition = np.asarray(
        [values[index] for index in COMPETITION_SCALAR_INDICES],
        dtype=np.float32,
    )
    if competition.shape != (COMPETITION_DIM,):
        raise AssertionError("competition construction produced the wrong dimension")
    return competition


def sanitize_float32(array: np.ndarray) -> np.ndarray:
    """Replace non-finite values with zero and force float32."""

    return np.nan_to_num(
        np.asarray(array, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


__all__ = [
    "normalize_scalar63",
    "build_evidence",
    "build_competition",
    "sanitize_float32",
]
