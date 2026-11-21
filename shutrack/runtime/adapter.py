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
    detection_overlap_row,
    normalizer_from_metadata,
)
from shutrack.features.windows import WindowArrays, build_window, stack_windows
from shutrack.models.shutrack import ShutrackV2Model
from shutrack.runtime.statistics import ShutrackRuntimeStatistics
from shutrack.schema import CONTEXT_SIZE, HISTORY_SIZE, MAX_FRAME_GAP


@dataclass(frozen=True)
class ShutrackRuntimeConfig:
    max_frame_gap: int = MAX_FRAME_GAP
    min_observations: int = CONTEXT_SIZE
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
        if not isinstance(checkpoint.get("competition_enabled"), bool):
            raise ValueError(
                "SHUTrack checkpoint is missing competition_enabled; retrain with "
                "the current code so training and online inputs share the same "
                "branch configuration"
            )
        self.competition_enabled = bool(checkpoint["competition_enabled"])
        if self.competition_enabled != model.competition_enabled:
            raise ValueError(
                "checkpoint/model competition branch mismatch: "
                f"{self.competition_enabled} != {model.competition_enabled}"
            )
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
        dataset_meta = checkpoint.get("dataset", {})
        if bool(dataset_meta.get("competition_enabled", self.competition_enabled)) != (
            self.competition_enabled
        ):
            raise ValueError(
                "checkpoint dataset competition_enabled does not match the model"
            )
        checkpoint_gap = int(dataset_meta.get("max_frame_gap", MAX_FRAME_GAP))
        if config is None:
            config = ShutrackRuntimeConfig(max_frame_gap=checkpoint_gap)
        elif isinstance(config, Mapping):
            config = ShutrackRuntimeConfig(
                max_frame_gap=int(config.get("max_frame_gap", checkpoint_gap)),
                min_observations=int(config.get("min_observations", CONTEXT_SIZE)),
                disable_kf_gate=bool(config.get("disable_kf_gate", False)),
                disable_ema_gate=bool(config.get("disable_ema_gate", False)),
            )
        if int(config.max_frame_gap) != checkpoint_gap:
            raise ValueError(
                "SHUTrack runtime max_frame_gap does not match the checkpoint: "
                f"{config.max_frame_gap} != {checkpoint_gap}"
            )
        self.config = config
        self.stats = ShutrackRuntimeStatistics()
        self.event_buffers: dict[int, deque[EventEvidence]] = {}
        self._last_frame: dict[int, int] = {}
        self._last_observation: dict[int, int] = {}
        self._segment: dict[int, int] = {}
        self._last_reset_frame: dict[int, int] = {}
        self._lost_tracks: set[int] = set()
        self._detection_boxes: dict[int, np.ndarray] = {}
        self._pool_indices: np.ndarray | None = None
        self._pool_boxes: np.ndarray | None = None
        self._overlap_cache: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        # The Tracker's unmatched-event path checks this attribute before
        # populating legacy TrackEvent features. SHUTrack builds its own
        # evidence and intentionally has no legacy feature builder.
        self.feature_builder = None

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

    def _begin_event(
        self, track_id: int, frame_id: int, observation_count: int
    ) -> int:
        """Advance the per-track event clock exactly once per event.

        The same event never resets twice: ``make_evidence`` and
        ``push_evidence`` both call this helper, but the second call sees the
        same frame id and is a no-op.  The returned segment id is the one the
        evidence and the buffer must share.
        """

        track_id = int(track_id)
        frame_id = int(frame_id)
        observation_count = int(observation_count)
        previous_frame = self._last_frame.get(track_id)
        previous_observation = self._last_observation.get(track_id)
        reset = False
        if previous_frame is not None:
            if frame_id < previous_frame:
                reset = True
            elif frame_id > previous_frame:
                gap = frame_id - previous_frame
                reset = (
                    gap > int(self.config.max_frame_gap)
                    or observation_count < int(previous_observation or 0)
                )
        if reset:
            self.event_buffers.pop(track_id, None)
            self._segment[track_id] = self._segment.get(track_id, 0) + 1
            if self._last_reset_frame.get(track_id) != frame_id:
                self.stats.record_buffer_reset()
                self._last_reset_frame[track_id] = frame_id
        self._last_frame[track_id] = frame_id
        self._last_observation[track_id] = observation_count
        return int(self._segment.get(track_id, 0))

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
        self._last_reset_frame.pop(track_id, None)
        self._lost_tracks.discard(track_id)

    # ------------------------------------------------------------------
    # Frame-level detection pool (competition overlaps)
    # ------------------------------------------------------------------
    def set_frame_detection_pool(self, pool: Sequence[Any] | None) -> None:
        self._detection_boxes = {}
        self._pool_indices = None
        self._pool_boxes = None
        self._overlap_cache = {}
        if pool is None:
            return
        indices: list[int] = []
        boxes: list[np.ndarray] = []
        for fallback_index, detection in enumerate(pool):
            index = int(getattr(detection, "frame_detection_index", fallback_index))
            box = getattr(detection, "x1y1x2y2", None)
            if box is None:
                box = getattr(detection, "box", None)
            if box is None:
                continue
            box = np.asarray(box, dtype=np.float64).reshape(4).copy()
            self._detection_boxes[index] = box
            indices.append(index)
            boxes.append(box)
        if boxes:
            self._pool_indices = np.asarray(indices, dtype=np.int64)
            self._pool_boxes = np.stack(boxes, axis=0)

    def _overlap_row(
        self, association_meta: Mapping[str, Any], detection_col: int
    ) -> np.ndarray | None:
        indices = association_meta.get("detection_indices")
        if indices is None:
            return None
        detection_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        column = int(detection_col)
        if detection_indices.size == 0 or column < 0 or column >= detection_indices.size:
            return None
        # The candidate pool may change within one frame (mature-track
        # association vs the early-enable New path), so the cache key must
        # include the ordered detection indices, not just the column.
        cache_key = (
            tuple(int(index) for index in detection_indices.tolist()),
            column,
        )
        cached = self._overlap_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._pool_boxes is not None and np.array_equal(
            self._pool_indices, detection_indices
        ):
            pool = self._pool_boxes
        else:
            boxes = []
            for index in detection_indices:
                box = self._detection_boxes.get(int(index))
                if box is None:
                    return None
                boxes.append(box)
            pool = np.stack(boxes, axis=0)
        row = detection_overlap_row(pool[column], pool, column)
        self._overlap_cache[cache_key] = row
        return row

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

    def scalar_params_for(
        self,
        *,
        frame_id: int,
        image_width: int,
        image_height: int,
        has_detection: bool,
        pre_update_state: Any,
        detection_box: np.ndarray | None = None,
        detection_score: float | None = None,
        detection_source: int = 0,
        association_meta: Mapping[str, Any] | None = None,
        track_row: int = -1,
        detection_col: int = -1,
    ) -> dict[str, Any]:
        """Build the exact legacy scalar parameters used online.

        The recorded candidate pool is passed through as ``track_cost_row``,
        ``detection_cost_col`` and ``max_iou_with_other`` so columns 15/21/25
        reproduce the cached values exactly (same pre-iteration ``final_cost``,
        same single-candidate fallback, self excluded from the IoU row).
        """

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
            "frame_id": int(frame_id),
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
            if association_meta is not None and track_row >= 0 and detection_col >= 0:
                final_cost = association_meta.get("final_cost")
                if final_cost is not None:
                    matrix = np.asarray(final_cost)
                    if (
                        matrix.ndim == 2
                        and track_row < matrix.shape[0]
                        and detection_col < matrix.shape[1]
                    ):
                        scalar_params["track_cost_row"] = matrix[track_row, :]
                        scalar_params["detection_cost_col"] = matrix[:, detection_col]
                overlap = self._overlap_row(association_meta, detection_col)
                if overlap is not None:
                    scalar_params["max_iou_with_other"] = (
                        float(np.max(overlap)) if overlap.size > 0 else 0.0
                    )
        return scalar_params

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
        segment_id = self._begin_event(track_id, frame_id, observation_count)

        scalar_params = self.scalar_params_for(
            frame_id=frame_id,
            image_width=image_width,
            image_height=image_height,
            has_detection=has_detection,
            pre_update_state=pre_update_state,
            detection_box=detection_box,
            detection_score=detection_score,
            detection_source=detection_source,
            association_meta=association_meta,
            track_row=track_row,
            detection_col=detection_col,
        )
        return build_online_evidence(
            scalar_params=scalar_params,
            track_feature=np.asarray(track_feature, dtype=np.float32).reshape(-1),
            detection_feature=(
                np.asarray(detection_feature, dtype=np.float32).reshape(-1)
                if detection_feature is not None
                else np.zeros_like(np.asarray(track_feature).reshape(-1), dtype=np.float32)
            ),
            has_detection=bool(has_detection),
            competition_enabled=self.competition_enabled,
            scalar_mean=self.scalar_mean,
            scalar_std=self.scalar_std,
            frame_id=frame_id,
            track_id=track_id,
            segment_id=segment_id,
            observation_count=int(observation_count),
        )

    # ------------------------------------------------------------------
    # Inference / buffering
    # ------------------------------------------------------------------
    def window_for(
        self, track_id: int, evidence: EventEvidence
    ) -> WindowArrays:
        """Window the model sees for this event (used by tests/diagnostics)."""

        self._begin_event(
            int(track_id), int(evidence.frame_id), int(evidence.observation_count)
        )
        buffer = self.event_buffers.get(int(track_id))
        history = list(buffer) if buffer is not None else []
        return build_window(history, evidence)

    def infer_batch(
        self, items: Sequence[tuple[int, EventEvidence]]
    ) -> list[np.ndarray]:
        if not items:
            return []
        if self.config.disable_kf_gate and self.config.disable_ema_gate:
            # Full-write data collection forces both gates to 1.0, so the
            # model output is never used. Keep building windows (the pilot's
            # hook captures them for training) but skip the forward and the
            # tensor stacking entirely.
            for track_id, evidence in items:
                self.window_for(track_id, evidence)
            gates = np.ones((len(items), 2), dtype=np.float32)
        else:
            windows = [self.window_for(track_id, evidence) for track_id, evidence in items]
            batch = stack_windows(windows)
            tensors = {
                key: torch.from_numpy(value).to(self.device)
                for key, value in batch.items()
            }
            if not bool(getattr(self.model.config, "appearance_match_input", False)):
                tensors.pop("current_match_distance", None)
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
            track_id = int(track_id)
            if track_id in self._lost_tracks:
                self.stats.record_recovery()
                self._lost_tracks.discard(track_id)
            self.stats.record_event()
            self.stats.record_gates(applied, matched=True, used_model=True)
            results.append(applied)
        return results

    def push_evidence(self, track_id: int, evidence: EventEvidence) -> None:
        track_id = int(track_id)
        self._begin_event(
            track_id, int(evidence.frame_id), int(evidence.observation_count)
        )
        buffer = self._buffer(track_id)
        buffer.append(evidence)

    def record_unmatched_evidence(
        self, track_id: int, evidence: EventEvidence
    ) -> np.ndarray:
        self.push_evidence(track_id, evidence)
        self._lost_tracks.add(int(track_id))
        gates = np.zeros(2, dtype=np.float64)
        self.stats.record_event()
        self.stats.record_gates(gates, matched=False, used_model=False)
        return gates


__all__ = [
    "ShutrackRuntimeConfig",
    "ShutrackRuntime",
]
