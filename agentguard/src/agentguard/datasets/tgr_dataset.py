from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.features.builder import EventFeatureBuilder


class TGRDataset(torch.utils.data.Dataset):
    """TGR training dataset.

    Each sample is a fixed-length window of exactly ``window_size`` consecutive
    events for the same track.  The TGR model processes the window and predicts
    a revised gate for every time-step.

    Parameters
    ----------
    events : list of TrackEvent
        All events belonging to this split (train or val).
    labels : list of dict
        Parallel to ``events``; each element is a dict with:

        - ``target_gate`` (np.ndarray, shape ``(2,)``): the ground-truth gate
          at this time-step ``[motion_gate, appearance_gate]``.
        - ``sample_weight`` (float, optional): per-event weight (default 1.0).

    feature_builder : EventFeatureBuilder
        Builder that converts windows of ``TrackEvent`` objects into batched
        feature tensors expected by the TGR model.
    window_size : int
        Number of consecutive events per window (default 4).
    """

    def __init__(
        self,
        events: List[TrackEvent],
        labels: List[Dict[str, Any]],
        feature_builder: EventFeatureBuilder,
        window_size: int = 4,
    ) -> None:
        if len(events) != len(labels):
            raise ValueError(
                f"events ({len(events)}) and labels ({len(labels)}) must have the same length."
            )
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}.")

        self.events = events
        self.labels = labels
        self.feature_builder = feature_builder
        self.window_size = window_size
        self._windows = self._build_windows()

    # ------------------------------------------------------------------
    def _build_windows(self) -> List[Dict[str, Any]]:
        """Build windows from events grouped by ``(sequence, track_id)``.

        Events within each group are sorted by ``frame_id``; any contiguous
        block of exactly ``window_size`` events forms a window.

        Returns
        -------
        list of dict, each with keys:
            - ``event_indices``: list of ``len(window_size)`` indices into
              ``self.events``.
            - ``sequence``: the video sequence name.
            - ``track_id``: the track identifier.
        """
        # Group events by (sequence, track_id).
        groups: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
        for idx, evt in enumerate(self.events):
            groups[(evt.sequence, evt.track_id)].append((evt.frame_id, idx))

        windows: List[Dict[str, Any]] = []
        for (seq, tid), entries in groups.items():
            entries.sort(key=lambda x: x[0])
            # Slide over all contiguous blocks of length window_size.
            # If there are gaps > 1 frame, we break windows at the gap.
            i = 0
            while i <= len(entries) - self.window_size:
                # Check that the window_size events are consecutive (no gap).
                gap_ok = True
                for j in range(i, i + self.window_size - 1):
                    frame_gap = entries[j + 1][0] - entries[j][0]
                    if frame_gap > 1:
                        gap_ok = False
                        break
                if not gap_ok:
                    # Move past the break.
                    i += 1
                    continue

                window_indices = [entries[j][1] for j in range(i, i + self.window_size)]
                windows.append({
                    "event_indices": window_indices,
                    "sequence": seq,
                    "track_id": tid,
                })
                i += 1

        if not windows:
            import warnings
            warnings.warn(
                f"No valid windows of size {self.window_size} could be built "
                f"from {len(self.events)} events. Check that tracks have at "
                f"least {self.window_size} consecutive events.",
                RuntimeWarning,
            )

        return windows

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a training sample.

        Returns
        -------
        dict with keys:
            track_feats        : ``(window_size, reid_dim)`` float tensor
            det_feats          : ``(window_size, reid_dim)`` float tensor
            scalar_feats       : ``(window_size, 63)`` float tensor
            iwg_policy_probs   : ``(window_size, 5)`` float tensor
            iwg_gates          : ``(window_size, 2)`` float tensor
            has_detection_mask : ``(window_size,)`` bool tensor
            target_gate        : ``(window_size, 2)`` float tensor
            sample_weight      : ``(window_size,)`` float tensor
        """
        window_info = self._windows[idx]
        window_events = [self.events[i] for i in window_info["event_indices"]]

        # Build TGR input tensors via the feature builder.
        inputs = self.feature_builder.build_tgr_input(window_events)

        # Squeeze batch dimension (feature builder adds leading 1).
        sample = {
            "track_feats": inputs["track_feats"].squeeze(0),
            "det_feats": inputs["det_feats"].squeeze(0),
            "scalar_feats": inputs["scalar_feats"].squeeze(0),
            "iwg_policy_probs": inputs["iwg_policy_probs"].squeeze(0),
            "iwg_gates": inputs["iwg_gates"].squeeze(0),
            "has_detection_mask": inputs["has_detection_mask"].squeeze(0),
        }

        # Gather target gates for each event in the window.
        target_gates = []
        sample_weights = []
        for evt_idx in window_info["event_indices"]:
            label = self.labels[evt_idx]
            if "target_gate" not in label:
                raise KeyError(f"Missing 'target_gate' in label for window")
            gate = np.asarray(label["target_gate"], dtype=np.float64)
            target_gates.append(gate)
            sample_weights.append(label.get("sample_weight", 1.0))

        sample["target_gate"] = torch.from_numpy(np.stack(target_gates)).float()
        sample["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float)

        return sample

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of samples into a padded batch.

        All sequences already have the same length (``window_size``), so
        this simply stacks.
        """
        batch_size = len(batch)
        window_size = batch[0]["track_feats"].shape[0]
        reid_dim = batch[0]["track_feats"].shape[-1]
        scalar_dim = batch[0]["scalar_feats"].shape[-1]
        device = batch[0]["track_feats"].device

        stacked = {
            "track_feats": torch.zeros(batch_size, window_size, reid_dim, device=device),
            "det_feats": torch.zeros(batch_size, window_size, reid_dim, device=device),
            "scalar_feats": torch.zeros(batch_size, window_size, scalar_dim, device=device),
            "iwg_policy_probs": torch.zeros(batch_size, window_size, 5, device=device),
            "iwg_gates": torch.zeros(batch_size, window_size, 2, device=device),
            "has_detection_mask": torch.zeros(batch_size, window_size, dtype=torch.bool, device=device),
            "target_gate": torch.zeros(batch_size, window_size, 2, device=device),
            "sample_weight": torch.zeros(batch_size, window_size, device=device),
        }

        for i, s in enumerate(batch):
            for key in stacked:
                stacked[key][i] = s[key]

        return stacked
