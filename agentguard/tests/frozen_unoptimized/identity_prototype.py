from __future__ import annotations

from typing import List

import numpy as np


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
    def build_leave_one_out(
        features_by_detection: dict[int, np.ndarray],
        excluded_detection_index: int,
    ) -> np.ndarray | None:
        """Exclude a unique detection before recomputing mean and top-70% trim."""
        features = [
            feature for index, feature in features_by_detection.items()
            if index != excluded_detection_index
        ]
        if len(features) < 3:
            return None
        return IdentityPrototypeBuilder.build_prototype(features)

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

        # Stack into (N, D)
        feats = np.asarray(features, dtype=np.float64)

        # Step 2: L2-normalise each feature
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        feats_normed = feats / norms

        # Step 3: compute initial mean and normalise
        init_mean = np.mean(feats_normed, axis=0)
        init_mean_norm = np.linalg.norm(init_mean)
        if init_mean_norm > 0:
            init_mean = init_mean / init_mean_norm

        # Step 4: keep top 70% by cosine similarity to initial mean
        similarities = feats_normed @ init_mean  # (N,) cosine similarities
        k = max(1, int(np.ceil(len(features) * IdentityPrototypeBuilder._TOP_KEEP_RATIO)))
        # Sort descending by similarity
        top_indices = np.argsort(similarities)[::-1][:k]
        kept = feats_normed[top_indices]

        # Step 5: average and normalise again
        prototype = np.mean(kept, axis=0)
        proto_norm = np.linalg.norm(prototype)
        if proto_norm > 0:
            prototype = prototype / proto_norm

        return prototype

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
