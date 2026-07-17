from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.data.label_schema import validate_rollout_label
from agentguard.features.builder import EventFeatureBuilder


class IWGDataset(torch.utils.data.Dataset):
    """IWG training dataset.

    Each sample comprises a current event plus up to ``max_history`` previous
    real TrackTrack events for the *same* track, forming a variable-length
    sequence fed into the IWG model.

    Train/val split **must** be performed by complete video sequences to
    prevent cross-sequence information leakage — see
    :func:`~agentguard.datasets.split_validation.validate_splits`.

    Parameters
    ----------
    events : list of TrackEvent
        All events belonging to this split (train or val).
        Events **must** be pre-filtered to a single split.
    labels : list of dict
        Parallel to ``events``; each element is a dict containing:

        - ``motion_target`` (float): ground-truth motion gate in [0, 1]
        - ``appearance_target`` (float): ground-truth appearance gate in [0, 1]
        - ``policy_soft_target`` (np.ndarray, shape ``(5,)``): soft target
          distribution over the 5 write policies.
        - ``sample_type`` (str): ``matched`` or ``unmatched`` event category.
        - ``valid_motion`` (bool): whether the motion target is reliable.
        - ``valid_appearance`` (bool): whether the appearance target is reliable.
        - ``sample_weight`` (float): importance weight for this sample.
    feature_builder : EventFeatureBuilder
        Builder that converts sequences of ``TrackEvent`` objects into the
        batched feature tensors expected by the IWG model.
    max_history : int
        Maximum number of **previous** events to include for the same track
        (default 5, yielding sequences of up to 6 elements).
    """

    def __init__(
        self,
        events: List[TrackEvent],
        labels: List[Dict[str, Any]],
        feature_builder: EventFeatureBuilder,
        max_history: int = 5,
    ) -> None:
        if len(events) != len(labels):
            raise ValueError(
                f"events ({len(events)}) and labels ({len(labels)}) must have the same length."
            )
        self.events = events
        self.labels = labels
        for index, label in enumerate(labels):
            try:
                validate_rollout_label(label)
            except ValueError as exc:
                raise ValueError(f"invalid rollout label at index {index}: {exc}") from exc
        self.feature_builder = feature_builder
        self.max_history = max_history

        # Build index: (sequence, track_id) -> list of (frame_id, index) sorted
        # by frame_id for efficient history look-up.
        self._track_index: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
        for idx, evt in enumerate(events):
            key = (evt.sequence, evt.track_id)
            self._track_index[key].append((evt.frame_id, idx))

        for key in self._track_index:
            self._track_index[key].sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.events)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a training sample.

        Returns
        -------
        dict with keys:
            track_feats  : ``(seq_len, reid_dim)`` float tensor
            det_feats    : ``(seq_len, reid_dim)`` float tensor
            scalar_feats : ``(seq_len, 63)`` float tensor
            mask         : ``(seq_len,)`` bool tensor (``True`` = padded)
            targets      : dict with label fields (see class docstring).
        """
        event = self.events[idx]
        label = self.labels[idx]

        # Gather history for the same track (older events before this one).
        key = (event.sequence, event.track_id)
        history_entries = self._track_index.get(key, [])
        # history_entries is sorted by frame_id ascending.
        # Find position of the current event.
        pos = -1
        for i, (fid, _) in enumerate(history_entries):
            if fid == event.frame_id and history_entries[i][1] == idx:
                pos = i
                break

        # Collect previous events (up to max_history).
        prev_indices: List[int] = []
        if pos >= 0:
            start = max(0, pos - self.max_history)
            for j in range(start, pos):
                prev_indices.append(history_entries[j][1])

        # Build the sequence: oldest to newest.
        seq_indices = prev_indices + [idx]
        seq_events: List[Optional[TrackEvent]] = [self.events[i] for i in seq_indices]

        # Left-pad with None to reach max_history + 1 length.
        seq_len = self.max_history + 1
        if len(seq_events) < seq_len:
            pad_count = seq_len - len(seq_events)
            seq_events = [None] * pad_count + seq_events

        # Build feature tensors via the feature builder.
        inputs = self.feature_builder.build_iwg_input(seq_events)

        # Squeeze the batch dimension (feature builder always adds a leading 1).
        sample = {
            "track_feats": inputs["track_feats"].squeeze(0),   # (seq_len, reid_dim)
            "det_feats": inputs["det_feats"].squeeze(0),       # (seq_len, reid_dim)
            "scalar_feats": inputs["scalar_feats"].squeeze(0), # (seq_len, 63)
            "mask": inputs["mask"].squeeze(0),                 # (seq_len,)
        }

        # Targets.
        targets = {
            "motion_target": torch.tensor(label["motion_soft_target"], dtype=torch.float),
            "appearance_target": torch.tensor(label["appearance_soft_target"], dtype=torch.float),
            "motion_soft_target": torch.tensor(label["motion_soft_target"], dtype=torch.float),
            "appearance_soft_target": torch.tensor(label["appearance_soft_target"], dtype=torch.float),
            "motion_safe_target": torch.tensor(label["motion_safe_target"], dtype=torch.float),
            "appearance_safe_target": torch.tensor(label["appearance_safe_target"], dtype=torch.float),
            "motion_label_confidence": torch.tensor(label["motion_label_confidence"], dtype=torch.float),
            "appearance_label_confidence": torch.tensor(label["appearance_label_confidence"], dtype=torch.float),
            "cue_target": torch.tensor(label["cue_target"], dtype=torch.float),
            "risk_target": torch.tensor(label["risk_targets"], dtype=torch.float),
            "policy_soft_target": torch.from_numpy(
                np.asarray(label["policy_soft_target"], dtype=np.float64)
            ).float(),
            "policy_safe_soft_target": torch.from_numpy(
                np.asarray(label["policy_safe_soft_target"], dtype=np.float64)
            ).float(),
            "sample_type": torch.tensor(label["sample_type"], dtype=torch.long),
            "valid_motion": torch.tensor(label["valid_motion"], dtype=torch.bool),
            "valid_appearance": torch.tensor(label["valid_appearance"], dtype=torch.bool),
            "sample_weight": torch.tensor(label["sample_weight"], dtype=torch.float),
        }
        sample["targets"] = targets
        return sample

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of samples into a padded batch.

        Sequences are padded to the maximum length found in the batch.
        """
        # Determine max sequence length across the batch.
        max_seq_len = max(s["track_feats"].shape[0] for s in batch)
        reid_dim = batch[0]["track_feats"].shape[-1]
        scalar_dim = batch[0]["scalar_feats"].shape[-1]
        batch_size = len(batch)
        device = batch[0]["track_feats"].device

        # Allocate padded tensors.
        track_feats = torch.zeros(batch_size, max_seq_len, reid_dim, device=device)
        det_feats = torch.zeros(batch_size, max_seq_len, reid_dim, device=device)
        scalar_feats = torch.zeros(batch_size, max_seq_len, scalar_dim, device=device)
        mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=device)

        targets_list: List[Dict[str, torch.Tensor]] = []

        for i, s in enumerate(batch):
            slen = s["track_feats"].shape[0]
            # Left-pad to match max_seq_len.
            offset = max_seq_len - slen
            track_feats[i, offset:] = s["track_feats"]
            det_feats[i, offset:] = s["det_feats"]
            scalar_feats[i, offset:] = s["scalar_feats"]
            mask[i, offset:] = s["mask"]
            targets_list.append(s["targets"])

        # Stack scalar targets into batched tensors.
        targets = {
            "motion_target": torch.stack([t["motion_target"] for t in targets_list]),
            "appearance_target": torch.stack([t["appearance_target"] for t in targets_list]),
            "policy_soft_target": torch.stack([t["policy_soft_target"] for t in targets_list]),
            "policy_safe_soft_target": torch.stack([t["policy_safe_soft_target"] for t in targets_list]),
            "motion_soft_target": torch.stack([t["motion_soft_target"] for t in targets_list]),
            "appearance_soft_target": torch.stack([t["appearance_soft_target"] for t in targets_list]),
            "motion_safe_target": torch.stack([t["motion_safe_target"] for t in targets_list]),
            "appearance_safe_target": torch.stack([t["appearance_safe_target"] for t in targets_list]),
            "motion_label_confidence": torch.stack([t["motion_label_confidence"] for t in targets_list]),
            "appearance_label_confidence": torch.stack([t["appearance_label_confidence"] for t in targets_list]),
            "cue_target": torch.stack([t["cue_target"] for t in targets_list]),
            "risk_target": torch.stack([t["risk_target"] for t in targets_list]),
            "sample_type": torch.stack([t["sample_type"] for t in targets_list]),
            "valid_motion": torch.stack([t["valid_motion"] for t in targets_list]),
            "valid_appearance": torch.stack([t["valid_appearance"] for t in targets_list]),
            "sample_weight": torch.stack([t["sample_weight"] for t in targets_list]),
        }

        return {
            "track_feats": track_feats,
            "det_feats": det_feats,
            "scalar_feats": scalar_feats,
            "mask": mask,
            "targets": targets,
        }
