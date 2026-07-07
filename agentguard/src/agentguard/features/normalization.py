from __future__ import annotations

from typing import List, Optional

import numpy as np


class NormalizationStats:
    """Tracks mean and standard deviation for the 63-dim scalar feature vector.

    Usage
    -----
    >>> stats = NormalizationStats()
    >>> stats.fit(list_of_feature_arrays)   # each array shape (63,)
    >>> normalized = stats.transform(features)
    >>> restored = stats.inverse_transform(normalized)
    >>> stats.save("norm.npz")
    >>> stats = NormalizationStats.load("norm.npz")
    """

    def __init__(self) -> None:
        self.mean: np.ndarray = np.zeros(63, dtype=np.float64)
        self.std: np.ndarray = np.ones(63, dtype=np.float64)

    # ------------------------------------------------------------------
    def fit(self, features_list: List[np.ndarray]) -> None:
        """Compute mean and standard deviation from a list of feature arrays.

        Parameters
        ----------
        features_list : list[np.ndarray]
            Each element has shape ``(63,)``.
        """
        if len(features_list) == 0:
            return  # keep default (zero mean, unit std)

        clean = [
            np.asarray(feat, dtype=np.float64).reshape(-1)
            for feat in features_list
            if np.asarray(feat).reshape(-1).size == 63
        ]
        if len(clean) == 0:
            return

        features = np.stack(clean, axis=0)  # (N, 63)
        self.mean = np.mean(features, axis=0).astype(np.float64)
        self.std = np.std(features, axis=0).astype(np.float64)
        # Prevent division by zero for constant features.
        self.std[self.std < 1e-8] = 1.0

    # ------------------------------------------------------------------
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation.

        Parameters
        ----------
        features : np.ndarray
            Array of shape ``(63,)`` or ``(N, 63)``.

        Returns
        -------
        np.ndarray
            Normalised array, same shape as input.
        """
        return (features - self.mean) / self.std

    # ------------------------------------------------------------------
    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse z-score normalisation.

        Parameters
        ----------
        normalized : np.ndarray
            Array of shape ``(63,)`` or ``(N, 63)``.

        Returns
        -------
        np.ndarray
            Original-scale array, same shape as input.
        """
        return normalized * self.std + self.mean

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist normalisation statistics to a ``.npz`` file.

        Parameters
        ----------
        path : str
            Destination file path (typically ending in ``.npz``).
        """
        np.savez(path, mean=self.mean, std=self.std)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "NormalizationStats":
        """Load normalisation statistics from a ``.npz`` file.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` file written by :meth:`save`.

        Returns
        -------
        NormalizationStats
        """
        data = np.load(path)
        inst = cls()
        inst.mean = data["mean"].astype(np.float64)
        inst.std = data["std"].astype(np.float64)
        return inst
