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

    def _fit_reid(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == self.reid_dim:
            return arr
        if arr.size == 0:
            return np.zeros(self.reid_dim, dtype=np.float64)
        if arr.size > self.reid_dim:
            return arr[: self.reid_dim]
        return np.pad(arr, (0, self.reid_dim - arr.size))

    def _fit_scalar(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == self._scalar_dim:
            return arr
        if arr.size == 0:
            return np.zeros(self._scalar_dim, dtype=np.float64)
        if arr.size > self._scalar_dim:
            return arr[: self._scalar_dim]
        return np.pad(arr, (0, self._scalar_dim - arr.size))

    def _event_scalar(self, event: TrackEvent) -> np.ndarray:
        if event.scalar_features is None:
            event.scalar_features = self.compute_scalar(event)
        return self._fit_scalar(event.scalar_features)

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

            track_feats[i] = self._fit_reid(evt.track_feature)

            if evt.has_detection and evt.detection_feature.size > 0:
                det_feats[i] = self._fit_reid(evt.detection_feature)
            # else stays zero

            scalar_feats[i] = self._event_scalar(evt)

        # Optional normalisation
        scalar_feats = self.normalize_scalars(scalar_feats)

        return {
            "track_feats": torch.from_numpy(track_feats).unsqueeze(0).float(),
            "det_feats": torch.from_numpy(det_feats).unsqueeze(0).float(),
            "scalar_feats": torch.from_numpy(scalar_feats).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0),
        }

    def build_iwg_batch_input(
        self,
        event_sequences: List[List[Optional[TrackEvent]]],
    ) -> Dict[str, torch.Tensor]:
        """Build IWG tensors for many per-track sequences in one pass.

        This is equivalent to calling :meth:`build_iwg_input` for each
        sequence and concatenating on the batch dimension, but avoids creating
        many small tensors in the online tracker path.
        """
        if not event_sequences:
            return {
                "track_feats": torch.empty((0, 0, self.reid_dim), dtype=torch.float32),
                "det_feats": torch.empty((0, 0, self.reid_dim), dtype=torch.float32),
                "scalar_feats": torch.empty((0, 0, self._scalar_dim), dtype=torch.float32),
                "mask": torch.empty((0, 0), dtype=torch.bool),
            }

        batch_size = len(event_sequences)
        seq_len = len(event_sequences[0])
        track_feats = np.zeros((batch_size, seq_len, self.reid_dim), dtype=np.float32)
        det_feats = np.zeros((batch_size, seq_len, self.reid_dim), dtype=np.float32)
        scalar_feats = np.zeros((batch_size, seq_len, self._scalar_dim), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.bool_)

        for b, events_sequence in enumerate(event_sequences):
            if len(events_sequence) != seq_len:
                raise ValueError("All IWG event sequences in a batch must have the same length")
            for i, evt in enumerate(events_sequence):
                if evt is None:
                    mask[b, i] = True
                    continue

                track_feats[b, i] = self._fit_reid(evt.track_feature)

                if evt.has_detection and evt.detection_feature.size > 0:
                    det_feats[b, i] = self._fit_reid(evt.detection_feature)

                scalar_feats[b, i] = self._event_scalar(evt)

        scalar_feats = self.normalize_scalars(scalar_feats).astype(np.float32, copy=False)

        return {
            "track_feats": torch.from_numpy(track_feats),
            "det_feats": torch.from_numpy(det_feats),
            "scalar_feats": torch.from_numpy(scalar_feats),
            "mask": torch.from_numpy(mask),
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
            track_feats[i] = self._fit_reid(evt.track_feature)

            if evt.has_detection:
                has_detection_mask[i] = True
            if evt.has_detection and evt.detection_feature.size > 0:
                det_feats[i] = self._fit_reid(evt.detection_feature)

            scalar_feats[i] = self._event_scalar(evt)

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

    def build_tgr_batch_input(
        self,
        windows: List[List[TrackEvent]],
    ) -> Dict[str, torch.Tensor]:
        """Build TGR tensors for many track windows in one pass."""
        if not windows:
            return {
                "track_feats": torch.empty((0, 0, self.reid_dim), dtype=torch.float32),
                "det_feats": torch.empty((0, 0, self.reid_dim), dtype=torch.float32),
                "scalar_feats": torch.empty((0, 0, self._scalar_dim), dtype=torch.float32),
                "iwg_policy_probs": torch.empty((0, 0, 5), dtype=torch.float32),
                "iwg_gates": torch.empty((0, 0, 2), dtype=torch.float32),
                "has_detection_mask": torch.empty((0, 0), dtype=torch.bool),
            }

        batch_size = len(windows)
        seq_len = len(windows[0])
        track_feats = np.zeros((batch_size, seq_len, self.reid_dim), dtype=np.float32)
        det_feats = np.zeros((batch_size, seq_len, self.reid_dim), dtype=np.float32)
        scalar_feats = np.zeros((batch_size, seq_len, self._scalar_dim), dtype=np.float32)
        iwg_policy_probs = np.full((batch_size, seq_len, 5), 0.2, dtype=np.float32)
        iwg_gates = np.ones((batch_size, seq_len, 2), dtype=np.float32)
        has_detection_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)

        for b, window_events in enumerate(windows):
            if len(window_events) != seq_len:
                raise ValueError("All TGR windows in a batch must have the same length")
            for i, evt in enumerate(window_events):
                track_feats[b, i] = self._fit_reid(evt.track_feature)

                if evt.has_detection:
                    has_detection_mask[b, i] = True
                if evt.has_detection and evt.detection_feature.size > 0:
                    det_feats[b, i] = self._fit_reid(evt.detection_feature)

                scalar_feats[b, i] = self._event_scalar(evt)

                if evt.iwg_policy_probs is not None:
                    iwg_policy_probs[b, i] = np.asarray(evt.iwg_policy_probs).reshape(-1)

                if evt.iwg_gate is not None:
                    iwg_gates[b, i] = np.asarray(evt.iwg_gate).reshape(-1)

        scalar_feats = self.normalize_scalars(scalar_feats).astype(np.float32, copy=False)

        return {
            "track_feats": torch.from_numpy(track_feats),
            "det_feats": torch.from_numpy(det_feats),
            "scalar_feats": torch.from_numpy(scalar_feats),
            "iwg_policy_probs": torch.from_numpy(iwg_policy_probs),
            "iwg_gates": torch.from_numpy(iwg_gates),
            "has_detection_mask": torch.from_numpy(has_detection_mask),
        }

    # ------------------------------------------------------------------
    # Single-event feature building
    # ------------------------------------------------------------------

    def build(self, event, association_context=None):
        """Build 63-dim scalar features for an event.

        Parameters
        ----------
        event : TrackEvent
            The event to build features for.
        association_context : optional
            Unused, reserved for future use.

        Returns
        -------
        tuple of (scalar_feats, track_feats, det_feats)
        """
        scalar = self.compute_scalar(event)

        # Track feature: prefer pre-computed track_feature, else from state snapshot.
        if event.track_feature is not None and event.track_feature.size > 0:
            track_raw = event.track_feature.ravel()
        else:
            src_state = event.pre_update_state or event.frame_start_state
            if src_state is not None and src_state.feature is not None and src_state.feature.size > 0:
                track_raw = src_state.feature.ravel()
            else:
                track_raw = np.zeros(self.reid_dim)
        # Pad or truncate to reid_dim
        if track_raw.size < self.reid_dim:
            track_f = np.pad(track_raw, (0, self.reid_dim - track_raw.size))
        elif track_raw.size > self.reid_dim:
            track_f = track_raw[:self.reid_dim]
        else:
            track_f = track_raw

        # Detection feature: prefer pre-computed detection_feature, else from detection observation.
        if event.has_detection:
            if event.detection_feature is not None and event.detection_feature.size > 0:
                det_raw = event.detection_feature.ravel()
            elif event.detection is not None and event.detection.feature is not None and event.detection.feature.size > 0:
                det_raw = event.detection.feature.ravel()
            else:
                det_raw = np.zeros(self.reid_dim)
            if det_raw.size < self.reid_dim:
                det_f = np.pad(det_raw, (0, self.reid_dim - det_raw.size))
            elif det_raw.size > self.reid_dim:
                det_f = det_raw[:self.reid_dim]
            else:
                det_f = det_raw
        else:
            det_f = np.zeros(self.reid_dim)

        return scalar, track_f, det_f

    def build_normalized(self, event, association_context=None):
        """Build and normalize 63-dim scalar features for an event.

        Parameters
        ----------
        event : TrackEvent
            The event to build features for.
        association_context : optional
            Unused, reserved for future use.

        Returns
        -------
        tuple of (scalar_feats, track_feats, det_feats)
            Scalar features are z-score normalised if stats are available.
        """
        scalar, t, d = self.build(event)
        if self.normalizer is not None:
            scalar = self.normalizer.transform(scalar.reshape(1, -1)).ravel()
        return scalar, t, d

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
