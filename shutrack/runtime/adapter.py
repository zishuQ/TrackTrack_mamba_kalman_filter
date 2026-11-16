"""Online SHUTrack gate provider.

The runtime maintains a bounded per-track buffer of evidence captured strictly
before each write.  It never reads future events, never reuses another track's
history, resets on gaps/observation-count rollbacks/track removal, and forces
the applied gates to ``[0, 0]`` for events without a detection.  Track state
mutation remains in TrackTrack's existing ``update_with_gates``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from shutrack.data.evidence import EventEvidence
from shutrack.data.legacy_adapter import (
    build_online_evidence,
    normalizer_from_metadata,
    online_margins_from_association_meta,
)
from shutrack.features.margins import INVALID_MARGIN, MarginResult
from shutrack.features.windows import build_window, stack_windows
from shutrack.models.shutrack import ShutrackV2Model
from shutrack.runtime.statistics import ShutrackRuntimeStatistics
from shutrack.schema import HISTORY_SIZE, MAX_FRAME_GAP


@dataclass(frozen=True)
class ShutrackRuntimeConfig:
    max_frame_gap: int = MAX_FRAME_GAP
    min_observations: int = 1
    disable_kf_gate: bool = False
    disable_ema_gate: bool = False


_PAIR_FIELDS = (
    "iou_similarity",
    "iou_distance",
    "cosine_distance",
    "confidence_distance",
    "angle_distance",
    "raw_cost",
    "final_cost",
    "assignment_round",
    "assignment_threshold",
    "detection_source",
)


class ShutrackRuntime:
    def __init__(
        self,
        model: ShutrackV2Model,
        checkpoint: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        config: ShutrackRuntimeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.checkpoint = dict(checkpoint)
        self.scalar_mean, self.scalar_std = normalizer_from_metadata(
            {
                "normalization_mean": checkpoint.get("normalization_mean", []),
                "normalization_std": checkpoint.get("normalization_std", []),
            }
        )
        margin = checkpoint.get("margin", {})
        self.margin_stats = margin.get("stats") if isinstance(margin, Mapping) else None
        self.margin_source = (
            margin.get("source") if isinstance(margin, Mapping) else "unknown"
        )
        dataset_meta = checkpoint.get("dataset", {})
        checkpoint_gap = int(dataset_meta.get("max_frame_gap", MAX_FRAME_GAP))
        if config is None:
            config = ShutrackRuntimeConfig(max_frame_gap=checkpoint_gap)
        elif isinstance(config, Mapping):
            config = ShutrackRuntimeConfig(
                max_frame_gap=int(config.get("max_frame_gap", checkpoint_gap)),
                min_observations=int(config.get("min_observations", 1)),
                disable_kf_gate=bool(config.get("disable_kf_gate", False)),
                disable_ema_gate=bool(config.get("disable_ema_gate", False)),
            )
        if int(config.max_frame_gap) != checkpoint_gap:
            # A different reset horizon is a deliberate live-tracking choice,
            # but the checkpoint's own horizon is the training-time contract.
            raise ValueError(
                "SHUTrack runtime max_frame_gap does not match the checkpoint: "
                f"{config.max_frame_gap} != {checkpoint_gap}"
            )
        self.config = config
        self.stats = ShutrackRuntimeStatistics()
        # The Tracker's unmatched-event path checks this attribute before
        # populating legacy TrackEvent features. SHUTrack builds its own
        # evidence and intentionally has no legacy feature builder.
        self.feature_builder = None
        self.event_buffers: dict[int, deque[EventEvidence]] = {}
        self._last_frame: dict[int, int] = {}
        self._last_observation: dict[int, int] = {}
        self._segment: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Track lifecycle helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _observation_count(track: Any) -> int:
        value = getattr(track, "observation_count", None)
        if value is None:
            value = len(getattr(track, "history", {}) or {})
        return int(value)

    def is_mature_track(self, track: Any) -> bool:
        state = int(getattr(track, "state", -1))
        if state not in (1, 2):
            return False
        return self._observation_count(track) >= int(self.config.min_observations)

    def _maybe_reset(self, track_id: int, frame_id: int, observation_count: int) -> None:
        track_id = int(track_id)
        frame_id = int(frame_id)
        previous_frame = self._last_frame.get(track_id)
        previous_observation = self._last_observation.get(track_id)
        reset = False
        if previous_frame is not None:
            gap = frame_id - previous_frame
            reset = (
                gap <= 0
                or gap > int(self.config.max_frame_gap)
                or int(observation_count) < int(previous_observation or 0)
            )
        if reset:
            self.event_buffers.pop(track_id, None)
            self._segment[track_id] = self._segment.get(track_id, 0) + 1

    def _buffer(self, track_id: int) -> deque[EventEvidence]:
        buffer = self.event_buffers.get(int(track_id))
        if buffer is None:
            buffer = deque(maxlen=HISTORY_SIZE)
            self.event_buffers[int(track_id)] = buffer
        return buffer

    def cleanup_track(self, track_id: int) -> None:
        track_id = int(track_id)
        self.event_buffers.pop(track_id, None)
        self._last_frame.pop(track_id, None)
        self._last_observation.pop(track_id, None)
        self._segment.pop(track_id, None)

    # ------------------------------------------------------------------
    # Evidence construction
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_from_association_meta(
        association_meta: Mapping[str, Any] | None,
        track_row: int,
        detection_col: int,
    ) -> dict[str, float] | None:
        if association_meta is None or track_row < 0 or detection_col < 0:
            return None
        pair: dict[str, float] = {}
        for name in _PAIR_FIELDS:
            value = association_meta.get(name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim != 2:
                continue
            if track_row >= array.shape[0] or detection_col >= array.shape[1]:
                continue
            pair[name] = float(array[track_row, detection_col])
        return pair or None

    def make_evidence(
        self,
        *,
        track_id: int,
        frame_id: int,
        image_width: int,
        image_height: int,
        has_detection: bool,
        pre_update_state: Any,
        track_feature: np.ndarray,
        detection_feature: np.ndarray | None = None,
        detection_box: np.ndarray | None = None,
        detection_score: float | None = None,
        detection_source: int = 0,
        association_meta: Mapping[str, Any] | None = None,
        track_row: int = -1,
        detection_col: int = -1,
        observation_count: int = 0,
    ) -> EventEvidence:
        track_id = int(track_id)
        frame_id = int(frame_id)
        self._maybe_reset(track_id, frame_id, observation_count)
        segment_id = int(self._segment.get(track_id, 0))

        association_pair = (
            self._pair_from_association_meta(
                association_meta, int(track_row), int(detection_col)
            )
            if has_detection
            else None
        )
        if has_detection and association_pair is not None:
            source = int(association_pair.get("detection_source", detection_source))
        else:
            source = int(detection_source)
        scalar_params: dict[str, Any] = {
            "has_detection": bool(has_detection),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "frame_id": frame_id,
            "pre_update_state": {
                "mean": (
                    None
                    if pre_update_state is None
                    or getattr(pre_update_state, "mean", None) is None
                    else np.asarray(pre_update_state.mean, dtype=np.float64)
                ),
                "covariance": (
                    None
                    if pre_update_state is None
                    or getattr(pre_update_state, "covariance", None) is None
                    else np.asarray(pre_update_state.covariance, dtype=np.float64)
                ),
                "velocity": np.asarray(
                    getattr(pre_update_state, "velocity", np.zeros((4, 2)))
                    if pre_update_state is not None
                    else np.zeros((4, 2)),
                    dtype=np.float64,
                ),
                "score": float(
                    getattr(pre_update_state, "score", 0.0)
                    if pre_update_state is not None
                    else 0.0
                ),
                "history": (
                    getattr(pre_update_state, "history", {}) or {}
                    if pre_update_state is not None
                    else {}
                ),
                "end_frame_id": int(
                    getattr(pre_update_state, "end_frame_id", 0)
                    if pre_update_state is not None
                    else 0
                ),
                "state": int(
                    getattr(pre_update_state, "state", 0)
                    if pre_update_state is not None
                    else 0
                ),
            },
        }
        if has_detection:
            if detection_box is not None:
                scalar_params["detection"] = {
                    "box": np.asarray(detection_box, dtype=np.float64),
                    "score": float(detection_score or 0.0),
                    "source": source,
                }
            if association_pair is not None:
                scalar_params["association"] = association_pair

        margins: MarginResult
        if has_detection:
            margins = online_margins_from_association_meta(
                association_meta,
                track_row=int(track_row),
                detection_col=int(detection_col),
            )
        else:
            margins = INVALID_MARGIN

        evidence = build_online_evidence(
            scalar_params=scalar_params,
            track_feature=np.asarray(track_feature, dtype=np.float32).reshape(-1),
            detection_feature=(
                np.asarray(detection_feature, dtype=np.float32).reshape(-1)
                if detection_feature is not None
                else np.zeros_like(np.asarray(track_feature).reshape(-1), dtype=np.float32)
            ),
            has_detection=bool(has_detection),
            margins=margins,
            margin_stats=self.margin_stats,
            scalar_mean=self.scalar_mean,
            scalar_std=self.scalar_std,
            frame_id=frame_id,
            track_id=track_id,
            segment_id=segment_id,
            observation_count=int(observation_count),
        )
        return evidence

    # ------------------------------------------------------------------
    # Inference / buffering
    # ------------------------------------------------------------------
    def _windows_for(
        self, items: Sequence[tuple[int, EventEvidence]]
    ) -> list[Any]:
        windows = []
        for track_id, evidence in items:
            self._maybe_reset(
                int(track_id), int(evidence.frame_id), int(evidence.observation_count)
            )
            buffer = self.event_buffers.get(int(track_id))
            history = list(buffer) if buffer is not None else []
            windows.append(build_window(history, evidence))
        return windows

    def infer_batch(
        self, items: Sequence[tuple[int, EventEvidence]]
    ) -> list[np.ndarray]:
        if not items:
            return []
        windows = self._windows_for(items)
        batch = stack_windows(windows)
        tensors = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in batch.items()
        }
        with torch.inference_mode():
            outputs = self.model(**tensors)
        gates = outputs["gates"].float().cpu().numpy()
        if not np.isfinite(gates).all():
            raise FloatingPointError("SHUTrack produced non-finite gates")
        results: list[np.ndarray] = []
        for index, (track_id, evidence) in enumerate(items):
            applied = np.array(gates[index], dtype=np.float64)
            if self.config.disable_kf_gate:
                applied[0] = 1.0
            if self.config.disable_ema_gate:
                applied[1] = 1.0
            applied = np.clip(applied, 0.0, 1.0)
            self.stats.record_event()
            self.stats.record_gates(applied, matched=True, used_model=True)
            results.append(applied)
        return results

    def push_evidence(self, track_id: int, evidence: EventEvidence) -> None:
        track_id = int(track_id)
        self._maybe_reset(
            track_id, int(evidence.frame_id), int(evidence.observation_count)
        )
        buffer = self._buffer(track_id)
        buffer.append(evidence)
        self._last_frame[track_id] = int(evidence.frame_id)
        self._last_observation[track_id] = int(evidence.observation_count)

    def record_unmatched_evidence(
        self, track_id: int, evidence: EventEvidence
    ) -> np.ndarray:
        self.push_evidence(track_id, evidence)
        gates = np.zeros(2, dtype=np.float64)
        self.stats.record_event()
        self.stats.record_gates(gates, matched=False, used_model=False)
        return gates


__all__ = [
    "ShutrackRuntimeConfig",
    "ShutrackRuntime",
]
