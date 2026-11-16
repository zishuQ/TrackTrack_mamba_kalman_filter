"""TrackTrack adapter for SHUTrack v2.

The adapter exposes the same small duck-typed surface the Tracker already
uses for the legacy AgentGuard adapter, but it returns only the two SHUTrack
gates and never fabricates policy/cue/risk/correction fields.  Track mutation
is delegated to the existing ``Track.update_with_gates``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from shutrack.data.evidence import EventEvidence
from shutrack.runtime.adapter import ShutrackRuntime


@dataclass(frozen=True)
class ShutrackGateDecision:
    motion_gate: float
    appearance_gate: float


@dataclass(frozen=True)
class ShutrackCurrentEvent:
    track_id: int
    frame_id: int
    evidence: EventEvidence
    has_detection: bool


class ShutrackTrackerAdapter:
    capture_only = False
    event_sink = None
    stats_prefix = "shutrack"

    def __init__(self, args: Any, vid_name: str, runtime: ShutrackRuntime) -> None:
        self.args = args
        self.vid_name = vid_name
        self.runtime = runtime
        self.early_enable = bool(getattr(args, "shutrack_early_enable", False))
        self.frame_id = -1
        self.img_width = int(getattr(args, "img_w", 1920))
        self.img_height = int(getattr(args, "img_h", 1080))
        self._warp = np.eye(2, 3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------
    def begin_frame(
        self,
        frame_id: int,
        img_width: int,
        img_height: int,
        detection_pool: Optional[List[Any]] = None,
        detection_sources: Optional[List[int]] = None,
    ) -> None:
        self.frame_id = int(frame_id)
        self.img_width = int(img_width or self.img_width)
        self.img_height = int(img_height or self.img_height)
        self.runtime.set_frame_detection_pool(detection_pool)

    def set_frame_warp(self, warp_matrix: np.ndarray) -> None:
        self._warp = np.asarray(warp_matrix, dtype=np.float64).copy()

    def set_association_record(self, *args: Any, **kwargs: Any) -> None:
        # The live association metadata is passed directly to the per-event
        # builder by the Tracker; nothing needs to be retained here.
        return None

    # ------------------------------------------------------------------
    # Event construction
    # ------------------------------------------------------------------
    @staticmethod
    def _observation_count(track: Any) -> int:
        value = getattr(track, "observation_count", None)
        if value is None:
            value = len(getattr(track, "history", {}) or {})
        return int(value)

    @staticmethod
    def _track_feature(track: Any, snapshot: Any) -> np.ndarray:
        if snapshot is not None:
            feature = getattr(snapshot, "feature", None)
            if feature is not None and np.asarray(feature).size > 0:
                return np.asarray(feature, dtype=np.float32).reshape(-1)
        return np.asarray(getattr(track, "feat"), dtype=np.float32).reshape(-1)

    @staticmethod
    def _detection_source(
        association_meta: Any, track_row: int, detection_col: int
    ) -> int:
        if association_meta is None:
            return 0
        matrix = association_meta.get("detection_source")
        if matrix is None:
            return 0
        array = np.asarray(matrix)
        if (
            array.ndim != 2
            or track_row < 0
            or detection_col < 0
            or track_row >= array.shape[0]
            or detection_col >= array.shape[1]
        ):
            return 0
        return int(array[track_row, detection_col])

    def build_matched_event(
        self,
        track: Any,
        detection: Any,
        track_idx: int,
        det_idx: int,
        association_meta: Any,
        warp_matrix: np.ndarray,
        frame_start_state_snapshot: Any,
        pre_update_state_snapshot: Any,
        num_tracks: int = 0,
        num_detections: int = 0,
        no_reid: bool = False,
    ) -> ShutrackCurrentEvent:
        observation_count = self._observation_count(track)
        evidence = self.runtime.make_evidence(
            track_id=int(track.track_id),
            frame_id=self.frame_id,
            image_width=self.img_width,
            image_height=self.img_height,
            has_detection=True,
            pre_update_state=pre_update_state_snapshot,
            track_feature=self._track_feature(track, pre_update_state_snapshot),
            detection_feature=np.asarray(detection.feat, dtype=np.float32).reshape(-1),
            detection_box=np.asarray(detection.box, dtype=np.float64),
            detection_score=float(detection.score),
            detection_source=self._detection_source(
                association_meta, int(track_idx), int(det_idx)
            ),
            association_meta=association_meta,
            track_row=int(track_idx),
            detection_col=int(det_idx),
            observation_count=observation_count,
        )
        return ShutrackCurrentEvent(
            track_id=int(track.track_id),
            frame_id=int(self.frame_id),
            evidence=evidence,
            has_detection=True,
        )

    def build_unmatched_event(
        self,
        track: Any,
        warp_matrix: np.ndarray,
        frame_start_state_snapshot: Any,
        pre_update_state_snapshot: Any,
    ) -> ShutrackCurrentEvent:
        observation_count = self._observation_count(track)
        track_feature = self._track_feature(track, pre_update_state_snapshot)
        evidence = self.runtime.make_evidence(
            track_id=int(track.track_id),
            frame_id=self.frame_id,
            image_width=self.img_width,
            image_height=self.img_height,
            has_detection=False,
            pre_update_state=pre_update_state_snapshot,
            track_feature=track_feature,
            detection_feature=None,
            detection_box=None,
            detection_score=None,
            association_meta=None,
            observation_count=observation_count,
        )
        return ShutrackCurrentEvent(
            track_id=int(track.track_id),
            frame_id=int(self.frame_id),
            evidence=evidence,
            has_detection=False,
        )

    def process_new_track_match(
        self,
        track: Any,
        detection: Any,
        track_idx: int,
        det_idx: int,
        association_meta: Any,
        warp_matrix: np.ndarray,
        frame_id: int,
    ) -> bool:
        """Optional early-enable path for a matched ``TrackState.New`` track.

        Match 1 initializes the track (handled by the Tracker's init path).
        Match 2 updates with the original, ungated ``Track.update`` and records
        its pre-update evidence.  Match 3 and later apply the SHUTrack gate
        inside the New association path.  TrackTrack's own confirmation rule
        (``observation_count >= min_len``) is untouched.

        Returns ``True`` when this helper handled the match; ``False`` lets the
        Tracker fall back to the original update.
        """

        if not self.early_enable:
            return False
        observation_count = self._observation_count(track)
        if observation_count < 1:
            return False
        pre_update = track.snapshot_state(compact_history=True)
        event = self.build_matched_event(
            track,
            detection,
            track_idx,
            det_idx,
            association_meta,
            warp_matrix,
            pre_update,
            pre_update,
        )
        if observation_count >= 2:
            decision = self.get_iwg_decisions_batch(
                [(int(track.track_id), event)]
            )[0]
            self.apply_gate(track, detection, decision)
            self.runtime.stats.record_early_gate_update()
            self.record_event(int(track.track_id), event, decision)
        else:
            track.update(int(frame_id), detection)
            self.record_event(
                int(track.track_id),
                event,
                ShutrackGateDecision(motion_gate=1.0, appearance_gate=1.0),
            )
        return True

    # ------------------------------------------------------------------
    # Decisions / updates
    # ------------------------------------------------------------------
    def get_iwg_decisions_batch(
        self, items: List[Tuple[int, ShutrackCurrentEvent]]
    ) -> List[ShutrackGateDecision]:
        if not items:
            return []
        gates = self.runtime.infer_batch(
            [(int(track_id), event.evidence) for track_id, event in items]
        )
        return [
            ShutrackGateDecision(
                motion_gate=float(value[0]),
                appearance_gate=float(value[1]),
            )
            for value in gates
        ]

    def get_iwg_decision(
        self, track_id: int, event: ShutrackCurrentEvent
    ) -> ShutrackGateDecision:
        return self.get_iwg_decisions_batch([(track_id, event)])[0]

    def apply_gate(
        self,
        track: Any,
        detection: Any,
        gate_decision: ShutrackGateDecision,
    ) -> None:
        track.update_with_gates(
            self.frame_id,
            detection,
            gate_decision.motion_gate,
            gate_decision.appearance_gate,
        )
        self.runtime.stats.record_gate_update()

    def record_event(
        self,
        track_id: int,
        event: ShutrackCurrentEvent,
        gate_decision: ShutrackGateDecision,
    ) -> None:
        self.runtime.push_evidence(int(track_id), event.evidence)

    def record_unmatched_event(
        self,
        track_id: int,
        event: ShutrackCurrentEvent,
    ) -> None:
        self.runtime.record_unmatched_evidence(int(track_id), event.evidence)

    def end_frame(self) -> None:
        return None

    def remove_track(self, track_id: int) -> None:
        self.runtime.cleanup_track(int(track_id))


__all__ = [
    "ShutrackGateDecision",
    "ShutrackCurrentEvent",
    "ShutrackTrackerAdapter",
]
