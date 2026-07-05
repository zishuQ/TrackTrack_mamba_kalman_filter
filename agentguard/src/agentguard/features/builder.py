from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.features.normalization import NormalizationStats
from agentguard.features.scalar import compute_scalar_features


class EventFeatureBuilder:
    """Builds batched feature tensors from sequences of ``TrackEvent`` objects
    for IWG and TGR model inference.

    Parameters
    ----------
    reid_dim : int
        Dimensionality of the ReID feature vectors (track and detection).
    norm_stats : NormalizationStats or None
        Optional pre-computed normalisation statistics for the 63-d scalar
        feature vector.  When provided, every scalar feature block is
        z-score normalised before being passed to the models.
    """

    def __init__(
        self,
        reid_dim: int,
        norm_stats: Optional[NormalizationStats] = None,
    ) -> None:
        self.reid_dim = reid_dim
        self.normalizer = norm_stats

        # The scalar feature vector is known to be 63-dimensional.
        self._scalar_dim: int = 63

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def scalar_dim(self) -> int:
        return self._scalar_dim

    def normalize_scalars(self, scalars: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation to scalar features if stats are
        available; otherwise return the input unchanged.

        Parameters
        ----------
        scalars : ndarray, shape ``(..., 63)``

        Returns
        -------
        ndarray, same shape as input.
        """
        if self.normalizer is not None:
            return self.normalizer.transform(scalars)
        return scalars

    # ------------------------------------------------------------------
    # IWG input building
    # ------------------------------------------------------------------

    def build_iwg_input(
        self,
        events_sequence: List[Optional[TrackEvent]],
    ) -> Dict[str, torch.Tensor]:
        """Build batched input tensors for the IWG model.

        Parameters
        ----------
        events_sequence : list of TrackEvent or None
            Fixed-length sequence (typically 6) where ``None`` entries
            indicate padded (pre-buffer) slots.

        Returns
        -------
        dict with keys ``track_feats``, ``det_feats``, ``scalar_feats``,
        ``mask`` — each is a ``(1, seq_len, D)`` or ``(1, seq_len)``
        float / bool tensor on CPU.
        """
        seq_len = len(events_sequence)
        track_feats = np.zeros((seq_len, self.reid_dim), dtype=np.float64)
        det_feats = np.zeros((seq_len, self.reid_dim), dtype=np.float64)
        scalar_feats = np.zeros((seq_len, self._scalar_dim), dtype=np.float64)
        mask = np.zeros(seq_len, dtype=np.bool_)

        for i, evt in enumerate(events_sequence):
            if evt is None:
                mask[i] = True
                continue

            track_feats[i] = evt.track_feature.ravel()

            if evt.has_detection and evt.detection_feature.size > 0:
                det_feats[i] = evt.detection_feature.ravel()
            # else stays zero

            scalar_feats[i] = evt.scalar_features.ravel()

        # Optional normalisation
        scalar_feats = self.normalize_scalars(scalar_feats)

        return {
            "track_feats": torch.from_numpy(track_feats).unsqueeze(0).float(),
            "det_feats": torch.from_numpy(det_feats).unsqueeze(0).float(),
            "scalar_feats": torch.from_numpy(scalar_feats).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0),
        }

    # ------------------------------------------------------------------
    # TGR input building
    # ------------------------------------------------------------------

    def build_tgr_input(
        self,
        window_events: List[TrackEvent],
    ) -> Dict[str, torch.Tensor]:
        """Build batched input tensors for the TGR model.

        Parameters
        ----------
        window_events : list of TrackEvent
            Exactly 4 events (no ``None`` entries).  Ordered oldest first.

        Returns
        -------
        dict with keys ``track_feats``, ``det_feats``, ``scalar_feats``,
        ``iwg_policy_probs``, ``iwg_gates``, ``has_detection_mask`` —
        each is a ``(1, seq_len, D)`` or ``(1, seq_len)`` tensor.
        """
        seq_len = len(window_events)
        reid_dim = self.reid_dim

        track_feats = np.zeros((seq_len, reid_dim), dtype=np.float64)
        det_feats = np.zeros((seq_len, reid_dim), dtype=np.float64)
        scalar_feats = np.zeros((seq_len, self._scalar_dim), dtype=np.float64)
        iwg_policy_probs = np.full((seq_len, 5), 0.2, dtype=np.float64)
        iwg_gates = np.ones((seq_len, 2), dtype=np.float64)
        has_detection_mask = np.zeros(seq_len, dtype=np.bool_)

        for i, evt in enumerate(window_events):
            track_feats[i] = evt.track_feature.ravel()

            if evt.has_detection and evt.detection_feature.size > 0:
                det_feats[i] = evt.detection_feature.ravel()
                has_detection_mask[i] = True

            scalar_feats[i] = evt.scalar_features.ravel()

            if evt.iwg_policy_probs is not None:
                iwg_policy_probs[i] = evt.iwg_policy_probs.ravel()

            if evt.iwg_gate is not None:
                iwg_gates[i] = evt.iwg_gate.ravel()

        # Optional normalisation
        scalar_feats = self.normalize_scalars(scalar_feats)

        # No padding mask for TGR — all 4 positions are valid
        return {
            "track_feats": torch.from_numpy(track_feats).unsqueeze(0).float(),
            "det_feats": torch.from_numpy(det_feats).unsqueeze(0).float(),
            "scalar_feats": torch.from_numpy(scalar_feats).unsqueeze(0).float(),
            "iwg_policy_probs": torch.from_numpy(iwg_policy_probs).unsqueeze(0).float(),
            "iwg_gates": torch.from_numpy(iwg_gates).unsqueeze(0).float(),
            "has_detection_mask": torch.from_numpy(has_detection_mask).unsqueeze(0),
        }

    # ------------------------------------------------------------------
    # Single-event scalar computation
    # ------------------------------------------------------------------

    def compute_scalar(
        self,
        event_or_params: Any,
    ) -> np.ndarray:
        """Compute or retrieve the 63-d scalar feature vector.

        If the input is a ``TrackEvent`` with pre-computed ``scalar_features``
        those are returned directly.  Otherwise the scalar is computed from
        scratch via :func:`~agentguard.features.scalar.compute_scalar_features`.

        Parameters
        ----------
        event_or_params : TrackEvent or dict
            Event data.

        Returns
        -------
        ndarray, shape ``(63,)``.
        """
        if isinstance(event_or_params, TrackEvent):
            if event_or_params.scalar_features is not None and event_or_params.scalar_features.shape[0] >= self._scalar_dim:
                return event_or_params.scalar_features.copy()
            return compute_scalar_features(event_or_params)

        return compute_scalar_features(event_or_params)
