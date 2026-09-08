from __future__ import annotations

from typing import List

import numpy as np

_MISSING = object()


class PreparedIdentityCandidates:
    """L2-normalised unique-detection features for repeated leave-one-out queries.

    Candidate order matches insertion order of the source mapping.  Features are
    normalised once; each query excludes at most one detection, then repeats the
    original mean → top-70% → mean procedure.  Final prototypes are cached only
    for an identical excluded detection on this candidate set.
    """

    def __init__(
        self,
        detection_indices: np.ndarray,
        features_normed: np.ndarray,
    ) -> None:
        self.detection_indices = np.asarray(detection_indices, dtype=np.int64).reshape(-1)
        if self.detection_indices.size == 0:
            self.features_normed = np.zeros((0, 0), dtype=np.float64)
        else:
            self.features_normed = np.asarray(features_normed, dtype=np.float64)
        if self.features_normed.ndim != 2 or self.features_normed.shape[0] != self.detection_indices.shape[0]:
            raise ValueError("prepared identity features must be (N, D) aligned with detection indices")
        self.features_normed.flags.writeable = False
        self.detection_indices.flags.writeable = False
        self._row = {int(index): row for row, index in enumerate(self.detection_indices.tolist())}
        self._loo: dict[int, np.ndarray | None] = {}
        self._full: object = _MISSING

    def __len__(self) -> int:
        return int(self.detection_indices.shape[0])

    @classmethod
    def from_features_by_detection(
        cls,
        features_by_detection: dict[int, np.ndarray],
    ) -> "PreparedIdentityCandidates":
        if not features_by_detection:
            empty_idx = np.zeros((0,), dtype=np.int64)
            empty_idx.flags.writeable = False
            return cls(empty_idx, np.zeros((0, 0), dtype=np.float64))
        indices = np.fromiter(
            (int(index) for index in features_by_detection),
            dtype=np.int64,
            count=len(features_by_detection),
        )
        stacked = np.stack(
            [
                np.asarray(feature, dtype=np.float64).reshape(-1)
                for feature in features_by_detection.values()
            ],
            axis=0,
        )
        return cls(indices, IdentityPrototypeBuilder.l2_normalize_rows(stacked))

    def leave_one_out(self, excluded_detection_index: int) -> np.ndarray | None:
        excluded = int(excluded_detection_index)
        row = self._row.get(excluded)
        if row is None:
            if self._full is _MISSING:
                self._full = self._finish(self.features_normed)
            return self._full  # type: ignore[return-value]
        cached = self._loo.get(excluded, _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        remaining = np.concatenate(
            (self.features_normed[:row], self.features_normed[row + 1:]),
            axis=0,
        )
        prototype = self._finish(remaining)
        self._loo[excluded] = prototype
        return prototype

    @staticmethod
    def _finish(features_normed: np.ndarray) -> np.ndarray | None:
        if int(features_normed.shape[0]) < 3:
            return None
        prototype = IdentityPrototypeBuilder.build_prototype_from_normalized(features_normed)
        prototype.flags.writeable = False
        return prototype


class IdentityPrototypeBuilder:
    """Builds identity prototypes from ReID features.

    Process
    -------
    1. Collect features where detection score >= 0.6 AND detection-GT IoU >= 0.7.
    2. L2-normalise each feature.
    3. Compute initial mean vector and L2-normalise it.
    4. Keep top 70% of features by cosine similarity to the initial mean.
    5. Average the kept features and L2-normalise again → prototype.
    """

    _DET_SCORE_THRESHOLD: float = 0.6
    _IOU_THRESHOLD: float = 0.7
    _TOP_KEEP_RATIO: float = 0.7

    @staticmethod
    def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("features must be a 2-D array")
        if feats.shape[0] == 0:
            return np.zeros((0, feats.shape[1]), dtype=np.float64)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return feats / norms

    @staticmethod
    def build_leave_one_out(
        features_by_detection: dict[int, np.ndarray],
        excluded_detection_index: int,
    ) -> np.ndarray | None:
        """Exclude a unique detection before recomputing mean and top-70% trim."""
        return PreparedIdentityCandidates.from_features_by_detection(
            features_by_detection
        ).leave_one_out(excluded_detection_index)

    @staticmethod
    def build_prototype_from_normalized(features_normed: np.ndarray) -> np.ndarray:
        """Mean → top-70% by cosine → mean, on already L2-normalised rows."""
        feats_normed = np.asarray(features_normed, dtype=np.float64)
        if feats_normed.ndim != 2 or feats_normed.shape[0] == 0:
            raise ValueError("Cannot build prototype from an empty feature list.")

        init_mean = np.mean(feats_normed, axis=0)
        init_mean_norm = np.linalg.norm(init_mean)
        if init_mean_norm > 0:
            init_mean = init_mean / init_mean_norm

        similarities = feats_normed @ init_mean
        k = max(1, int(np.ceil(feats_normed.shape[0] * IdentityPrototypeBuilder._TOP_KEEP_RATIO)))
        top_indices = np.argsort(similarities)[::-1][:k]
        kept = feats_normed[top_indices]

        prototype = np.mean(kept, axis=0)
        proto_norm = np.linalg.norm(prototype)
        if proto_norm > 0:
            prototype = prototype / proto_norm
        return prototype

    @staticmethod
    def build_prototype(features: List[np.ndarray]) -> np.ndarray:
        """Build a single prototype from a list of ReID feature vectors.

        Parameters
        ----------
        features : list of ndarray
            Each element is a 1-D ReID feature vector (``(D,)``).

        Returns
        -------
        ndarray
            Prototype vector of shape ``(D,)``, L2-normalised.

        Raises
        ------
        ValueError
            If the input list is empty.
        """
        if not features:
            raise ValueError("Cannot build prototype from an empty feature list.")

        feats = np.asarray(features, dtype=np.float64)
        return IdentityPrototypeBuilder.build_prototype_from_normalized(
            IdentityPrototypeBuilder.l2_normalize_rows(feats)
        )

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors.

        Parameters
        ----------
        a : ndarray, shape ``(D,)``
        b : ndarray, shape ``(D,)``

        Returns
        -------
        float
            The dot product (cosine similarity when inputs are unit vectors).
        """
        return float(np.dot(a, b))
