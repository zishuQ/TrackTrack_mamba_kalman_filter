from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.features.geometry import pairwise_iou_xyxy
from agentguard.contracts.states import (
    DetectionObservation,
    TrackStateSnapshot,
)

from trackers.track import Track

from .converters import (
    build_association_context,
    export_association_pair,
    export_detection,
)


class AgentGuardTrackerAdapter:
    """Adapter that bridges TrackTrack's Tracker with the AgentGuard runtime.

    Only this directory (``integrations/agentguard/``) is allowed to know both
    TrackTrack internals and AgentGuard protocol data structures.
    """

    def __init__(
        self,
        args,
        vid_name: str,
        agentguard_runtime=None,
    ):
        self.args = args
        self.vid_name = vid_name
        self.runtime = agentguard_runtime

        # Frame-level bookkeeping
        self.frame_id: int = -1
        self.img_width: int = -1
        self.img_height: int = -1

        # Optional dataset name extracted from args.
        self._dataset: str = getattr(args, "dataset", "unknown")

        # EventSink from runtime (for cache recording)
        self.event_sink = getattr(agentguard_runtime, "event_sink", None) if agentguard_runtime else None
        self.capture_only = getattr(args, "agentguard_mode", "off") == "capture"

        # Pending frame-level data for EventSink
        self._pending_frame_record: Optional[dict] = None
        self._pending_frame_events: List[dict] = []

        # Per-frame detection pool for stable indexing
        self._frame_detections: List[Track] = []
        self._frame_detection_sources: List[int] = []
        self._frame_detection_overlap = np.zeros((0, 0), dtype=np.float64)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(
        self,
        frame_id: int,
        img_width: int,
        img_height: int,
        detection_pool: Optional[List[Track]] = None,
        detection_sources: Optional[List[int]] = None,
    ) -> None:
        """Called at the start of each frame.

        Stores the current frame id, image dimensions, and detection pool
        for stable detection indexing.
        """
        self.frame_id = frame_id
        self.img_width = img_width
        self.img_height = img_height

        self._frame_detections = detection_pool if detection_pool is not None else []
        self._frame_detection_sources = detection_sources if detection_sources is not None else []

        # Keep detection overlap separate from the track-to-detection
        # association matrices.  The former describes ambiguity in the
        # current detection pool and is stable for every accepted track.
        boxes = np.asarray(
            [det.x1y1x2y2 for det in self._frame_detections],
            dtype=np.float64,
        ).reshape((-1, 4))
        self._frame_detection_overlap = pairwise_iou_xyxy(boxes, boxes)
        if boxes.shape[0] > 0:
            np.fill_diagonal(self._frame_detection_overlap, 0.0)

        # Serialize detection pool with stable indices
        detections_serialized = []
        for idx, det in enumerate(self._frame_detections):
            source = (
                self._frame_detection_sources[idx]
                if idx < len(self._frame_detection_sources)
                else 0
            )
            detection_index = int(getattr(det, "frame_detection_index", idx))
            detections_serialized.append({
                "index": idx,
                "detection_index": detection_index,
                "score": float(det.score),
                "source": int(source),
                "class_id": int(getattr(det, "class_id", 0)),
            })

        # Accumulate frame record for EventSink
        self._pending_frame_record = {
            "frame_id": frame_id,
            "image_width": img_width,
            "image_height": img_height,
            "warp_matrix": None,  # Set when warp is known
            "detections": detections_serialized,
            "association": None,
        }
        self._pending_frame_events = []

    def set_frame_warp(self, warp_matrix: np.ndarray) -> None:
        """Attach the effective per-frame warp to the pending frame record."""
        if self._pending_frame_record is None:
            return
        self._pending_frame_record["warp_matrix"] = np.asarray(
            warp_matrix,
            dtype=np.float64,
        ).copy()

    def set_association_record(
        self,
        track_ids: List[int],
        detection_pool: List[Track],
        association_meta: Optional[dict],
        no_reid: bool = False,
    ) -> None:
        """Attach one frame-level association record to the pending frame.

        Detection features are deliberately not stored here.  Detections are
        represented by their sequence-global detection-cache indices.
        """
        if self._pending_frame_record is None or association_meta is None:
            return

        detection_indices = np.asarray(
            [
                int(getattr(det, "frame_detection_index", idx))
                for idx, det in enumerate(detection_pool)
            ],
            dtype=np.int64,
        )
        self._pending_frame_record["association"] = {
            "track_ids": np.asarray(track_ids, dtype=np.int64),
            "detection_indices": detection_indices,
            "raw_cost": np.asarray(association_meta["raw_cost"], dtype=np.float32).copy(),
            "final_cost": np.asarray(association_meta["final_cost"], dtype=np.float32).copy(),
            "iou_similarity": np.asarray(association_meta["iou_similarity"], dtype=np.float32).copy(),
            "iou_distance": np.asarray(association_meta["iou_distance"], dtype=np.float32).copy(),
            "cosine_distance": (
                np.zeros_like(association_meta["final_cost"], dtype=np.float32)
                if no_reid
                else np.asarray(association_meta["cosine_distance"], dtype=np.float32).copy()
            ),
            "confidence_distance": np.asarray(association_meta["confidence_distance"], dtype=np.float32).copy(),
            "angle_distance": np.asarray(association_meta["angle_distance"], dtype=np.float32).copy(),
            "assignment_round": np.asarray(association_meta["assignment_round"], dtype=np.int16).copy(),
            "assignment_threshold": np.asarray(association_meta["assignment_threshold"], dtype=np.float32).copy(),
            "detection_source": np.asarray(association_meta["detection_source"], dtype=np.int8).copy(),
            "reid_available": not bool(no_reid),
        }

    # ------------------------------------------------------------------
    # Event builders
    # ------------------------------------------------------------------

    def _make_event_id(self, track_id: int) -> str:
        """Generate a deterministic event identifier."""
        return f"{self._dataset}/{self.vid_name}/{self.frame_id:06d}/{track_id:06d}"

    def _make_candidate_id(self, event_id: str, candidate_type: str) -> str:
        """Generate a deterministic candidate identifier."""
        return f"{event_id}/{candidate_type}"

    def build_matched_event(
        self,
        track: Track,
        detection: Track,
        track_idx: int,
        det_idx: int,
        association_meta: dict,
        warp_matrix: np.ndarray,
        frame_start_state_snapshot: TrackStateSnapshot,
        pre_update_state_snapshot: TrackStateSnapshot,
        num_tracks: int = 0,
        num_detections: int = 0,
        no_reid: bool = False,
    ) -> TrackEvent:
        """Build a ``TrackEvent`` for an accepted match.

        Parameters
        ----------
        track : Track
            The matched track.
        detection : Track
            The matched detection (also a Track instance).
        track_idx : int
            Index of the track in the association matrix.
        det_idx : int
            Index of the detection in the association matrix.
        association_meta : dict
            Meta dict containing *pre-mutation copies* of cost matrices.
        warp_matrix : np.ndarray
            (2, 3) warp matrix from global motion compensation.
        frame_start_state_snapshot : TrackStateSnapshot
            Snapshot taken at the start of the frame (before prediction).
        pre_update_state_snapshot : TrackStateSnapshot
            Snapshot taken after KF predict but before the update step.
        num_tracks : int
            Total number of tracks in the association.
        num_detections : int
            Total number of detections in the association pool.
        no_reid : bool
            If True, cosine distance is unavailable.
        """
        # Detection source is read from the meta dict.
        det_source = int(
            association_meta["detection_source"][track_idx, det_idx]
        )

        # Use stable detection index from the merged frame pool
        stable_det_idx = self._resolve_stable_detection_index(detection)

        # Build AssociationContext from pre-mutation matrix copies
        assoc_ctx = build_association_context(
            association_meta,
            track_idx,
            det_idx,
            num_tracks=num_tracks if num_tracks > 0 else association_meta.get("num_tracks", 0),
            num_detections=num_detections if num_detections > 0 else association_meta.get("num_detections", 0),
            no_reid=no_reid,
            detection_overlap_row=(
                self._frame_detection_overlap[det_idx].copy()
                if 0 <= det_idx < len(self._frame_detection_overlap)
                else None
            ),
        )

        return TrackEvent(
            event_id=self._make_event_id(track.track_id),
            dataset=self._dataset,
            sequence=self.vid_name,
            frame_id=self.frame_id,
            track_id=track.track_id,
            image_width=self.img_width,
            image_height=self.img_height,
            has_detection=True,
            frame_start_state=frame_start_state_snapshot,
            pre_update_state=pre_update_state_snapshot,
            detection=export_detection(detection, det_source, stable_det_idx),
            association=export_association_pair(
                association_meta, track_idx, det_idx
            ),
            association_context=assoc_ctx,
            warp_matrix=warp_matrix.copy(),
        )

    def build_unmatched_event(
        self,
        track: Track,
        warp_matrix: np.ndarray,
        frame_start_state_snapshot: TrackStateSnapshot,
        pre_update_state_snapshot: TrackStateSnapshot,
    ) -> TrackEvent:
        """Build a ``TrackEvent`` for an unmatched track.

        The returned event has ``has_detection=False`` and ``detection`` /
        ``association`` set to ``None``.
        """
        return TrackEvent(
            event_id=self._make_event_id(track.track_id),
            dataset=self._dataset,
            sequence=self.vid_name,
            frame_id=self.frame_id,
            track_id=track.track_id,
            image_width=self.img_width,
            image_height=self.img_height,
            has_detection=False,
            frame_start_state=frame_start_state_snapshot,
            pre_update_state=pre_update_state_snapshot,
            detection=None,
            association=None,
            warp_matrix=warp_matrix.copy(),
        )

    def _resolve_stable_detection_index(self, detection: Track) -> int:
        """Resolve the stable frame_detection_index for a detection.

        Uses the frame's merged detection pool (HIGH + LOW + NMS_DELETED_HIGH)
        to find the stable index. Falls back to ``detection.track_id`` if
        detection is not found in the pool.
        """
        if hasattr(detection, "frame_detection_index"):
            return int(detection.frame_detection_index)
        for idx, det in enumerate(self._frame_detections):
            if det is detection:
                return int(getattr(det, "frame_detection_index", idx))
        return getattr(detection, 'track_id', -1)

    @property
    def reid_dim(self) -> int:
        """Derive ReID dimension from the runtime feature builder or fallback."""
        if self.runtime is not None and hasattr(self.runtime, 'feature_builder') and self.runtime.feature_builder is not None:
            return self.runtime.feature_builder.reid_dim
        return 2048

    def apply_gate(
        self,
        track: Track,
        detection: Track,
        gate_decision: GateDecision,
    ) -> None:
        """Apply a ``GateDecision`` to update a track.

        Delegates to ``track.update_with_gates(frame_id, detection,
        gate_decision.motion_gate, gate_decision.appearance_gate)``.

        .. note::

           ``update_with_gates`` is expected to be added to the ``Track``
           class as part of the AgentGuard integration.  If it is not yet
           available the call is silently skipped.
        """
        if not hasattr(track, "update_with_gates"):
            return

        track.update_with_gates(
            self.frame_id,
            detection,
            gate_decision.motion_gate,
            gate_decision.appearance_gate,
        )

    # ------------------------------------------------------------------
    # Stage lifecycle callbacks
    # ------------------------------------------------------------------

    def end_frame(self) -> None:
        """Flush the current frame to the compact event sink."""
        if self.event_sink is not None and hasattr(self.event_sink, 'on_frame') and self._pending_frame_record is not None:
            frame_events = list(self._pending_frame_events)
            self.event_sink.on_frame(self._pending_frame_record, frame_events)
            self._pending_frame_events = []
            self._pending_frame_record = None

    # ------------------------------------------------------------------
    # IWG decision
    # ------------------------------------------------------------------

    def get_iwg_decision(
        self,
        track_id: int,
        event: TrackEvent,
    ) -> GateDecision:
        """Run IWG inference for the given track event and return the gate
        decision.

        Populates feature vectors on the event (track_feature,
        detection_feature, scalar_features) before running inference.
        """
        if self.runtime is None or self.runtime.rg_cma_model is None:
            return GateDecision(1.0, 1.0, np.ones(5, dtype=np.float64) / 5.0)

        fb = self.runtime.feature_builder
        rd = self.reid_dim

        # Track feature: use the pre-update (post-predict) snapshot if available
        src_state = event.pre_update_state or event.frame_start_state
        if src_state is not None and src_state.feature.size > 0:
            event.track_feature = np.asarray(src_state.feature, dtype=np.float64).reshape(-1)
        else:
            event.track_feature = np.zeros(rd, dtype=np.float64)

        # Detection feature
        if event.has_detection and event.detection is not None:
            event.detection_feature = np.asarray(event.detection.feature, dtype=np.float64).reshape(-1)
        else:
            event.detection_feature = np.zeros(rd, dtype=np.float64)

        # Scalar features
        event.scalar_features = fb.compute_scalar(event)

        # Build inference sequence from buffer + current event
        seq_len = self.runtime.iwg_context_size
        buffer = self.runtime.event_buffers.get(track_id)
        if buffer is None:
            seq = [None] * (seq_len - 1) + [event]
        else:
            hist = buffer.get_sequence()
            history_len = seq_len - 1
            history_tail = hist[-history_len:] if history_len else []
            seq = history_tail + [event]

        result = self.runtime.run_iwg_rg_cma_inference(
            track_id,
            seq,
            frame_id=event.frame_id,
            has_detection=event.has_detection,
        )

        return GateDecision(
            motion_gate=float(result["gate"][0]),
            appearance_gate=float(result["gate"][1]),
            policy_probs=result["policy_probs"],
            base_gate=result.get("base_gate"),
            final_gate=result.get("final_gate"),
            gate_correction=result.get("gate_correction"),
        )

    def get_iwg_decisions_batch(
        self,
        items: List[Tuple[int, TrackEvent]],
    ) -> List[GateDecision]:
        """Run IWG once for many matched events from the same frame."""
        if not items:
            return []
        if self.runtime is None or self.runtime.rg_cma_model is None:
            return [
                GateDecision(1.0, 1.0, np.ones(5, dtype=np.float64) / 5.0)
                for _ in items
            ]

        fb = self.runtime.feature_builder
        rd = self.reid_dim
        sequences = []
        for track_id, event in items:
            src_state = event.pre_update_state or event.frame_start_state
            if src_state is not None and src_state.feature.size > 0:
                event.track_feature = np.asarray(
                    src_state.feature, dtype=np.float64
                ).reshape(-1)
            else:
                event.track_feature = np.zeros(rd, dtype=np.float64)

            if event.has_detection and event.detection is not None:
                event.detection_feature = np.asarray(
                    event.detection.feature, dtype=np.float64
                ).reshape(-1)
            else:
                event.detection_feature = np.zeros(rd, dtype=np.float64)

            event.scalar_features = fb.compute_scalar(event)

            seq_len = self.runtime.iwg_context_size
            buffer = self.runtime.event_buffers.get(track_id)
            if buffer is None:
                seq = [None] * (seq_len - 1) + [event]
            else:
                hist = buffer.get_sequence()
                history_len = seq_len - 1
                history_tail = hist[-history_len:] if history_len else []
                seq = history_tail + [event]
            sequences.append(seq)

        results = self.runtime.run_iwg_rg_cma_batch_inference(
            [track_id for track_id, _event in items],
            sequences,
            frame_ids=[event.frame_id for _track_id, event in items],
            has_detection=[event.has_detection for _track_id, event in items],
        )
        return [
            GateDecision(
                motion_gate=float(result["gate"][0]),
                appearance_gate=float(result["gate"][1]),
                policy_probs=result["policy_probs"],
                base_gate=result.get("base_gate"),
                final_gate=result.get("final_gate"),
                gate_correction=result.get("gate_correction"),
            )
            for result in results
        ]

    # ------------------------------------------------------------------
    # Event recording (buffer management)
    # ------------------------------------------------------------------

    def record_event(
        self,
        track_id: int,
        event: TrackEvent,
        gate_decision: GateDecision,
    ) -> None:
        """Record a matched event in Runtime context and the compact sink."""
        if self.runtime is None:
            self._accumulate_event_for_sink(event)
            return

        is_capture = self.capture_only or self.runtime.mode == "capture"

        self.runtime.stats.record_event()
        event_buffer = None
        if not is_capture:
            event_buffer = self.runtime.get_or_create_event_buffer(track_id)

        # Attach IWG outputs
        event.iwg_policy_probs = gate_decision.policy_probs.copy()
        event.iwg_gate = np.array(
            [gate_decision.motion_gate, gate_decision.appearance_gate],
            dtype=np.float64,
        )

        if not is_capture:
            assert event_buffer is not None
            event_buffer.push(self._lightweight_runtime_event(event))

        # Accumulate for EventSink
        self._accumulate_event_for_sink(event)

    def record_unmatched_event(
        self,
        track_id: int,
        event: TrackEvent,
    ) -> None:
        """Record an unmatched (lost) event into the runtime buffers.

        The gate is forced to ``[0.0, 0.0]`` (no detection).
        Unmatched events use the no-detection protocol: detection=None,
        association=None, track_feature from pre_update_state.feature,
        detection_feature = zero vector.
        """
        if self.runtime is None:
            self._accumulate_event_for_sink(event)
            return

        is_capture = self.capture_only or self.runtime.mode == "capture"

        self.runtime.stats.record_event()
        event_buffer = None
        if not is_capture:
            event_buffer = self.runtime.get_or_create_event_buffer(track_id)

        if not is_capture:
            assert event_buffer is not None
            fb = self.runtime.feature_builder
            rd = self.reid_dim
            src_state = event.pre_update_state or event.frame_start_state
            if src_state is not None and src_state.feature.size > 0:
                event.track_feature = np.asarray(
                    src_state.feature, dtype=np.float64
                ).reshape(-1)
            else:
                event.track_feature = np.zeros(rd, dtype=np.float64)
            event.detection_feature = np.zeros(rd, dtype=np.float64)
            event.scalar_features = fb.compute_scalar(event)
            history = event_buffer.get_sequence()
            seq_len = self.runtime.iwg_context_size
            history_len = seq_len - 1
            history_tail = history[-history_len:] if history_len else []
            sequence = history_tail + [event]
            result = self.runtime.run_iwg_rg_cma_inference(
                track_id,
                sequence,
                frame_id=event.frame_id,
                has_detection=False,
            )
            event.iwg_policy_probs = result["policy_probs"].copy()

        event.iwg_gate = np.array([0.0, 0.0], dtype=np.float64)

        if not is_capture:
            assert event_buffer is not None
            event_buffer.push(self._lightweight_runtime_event(event))

        # Accumulate for EventSink
        self._accumulate_event_for_sink(event)

    # ------------------------------------------------------------------
    # EventSink accumulation helper
    # ------------------------------------------------------------------

    def _lightweight_runtime_event(
        self,
        event: TrackEvent,
    ) -> TrackEvent:
        """Return an event containing only the six-event model context."""
        return TrackEvent(
            event_id=event.event_id,
            dataset=event.dataset,
            sequence=event.sequence,
            frame_id=event.frame_id,
            track_id=event.track_id,
            image_width=event.image_width,
            image_height=event.image_height,
            has_detection=event.has_detection,
            frame_start_state=None,
            pre_update_state=None,
            detection=None,
            association=None,
            association_context=None,
            warp_matrix=np.asarray(event.warp_matrix, dtype=np.float64).copy(),
            scalar_features=(
                None
                if event.scalar_features is None
                else np.asarray(event.scalar_features, dtype=np.float32).copy()
            ),
            track_feature=np.asarray(event.track_feature, dtype=np.float32).reshape(-1).copy(),
            detection_feature=np.asarray(event.detection_feature, dtype=np.float32).reshape(-1).copy(),
            iwg_policy_probs=(
                None
                if event.iwg_policy_probs is None
                else np.asarray(event.iwg_policy_probs, dtype=np.float32).copy()
            ),
            iwg_gate=(
                None
                if event.iwg_gate is None
                else np.asarray(event.iwg_gate, dtype=np.float32).copy()
            ),
        )

    def _accumulate_event_for_sink(self, event: TrackEvent) -> None:
        """Serialise *event* and append to the pending frame events list."""
        if self.event_sink is None:
            return
        if self._pending_frame_record is None:
            return  # begin_frame was not called

        if not getattr(self.event_sink, "compact", False):
            raise TypeError("AgentGuard capture requires a compact event sink")
        event_dict = self._compact_event_dict(event)
        self._pending_frame_events.append(event_dict)

    def _compact_snapshot_dict(self, snapshot: Optional[TrackStateSnapshot]) -> Optional[dict]:
        if snapshot is None:
            return None
        history = {}
        for frame_id, item in snapshot.history.items():
            if item is None or len(item) == 0:
                continue
            history[int(frame_id)] = [
                np.asarray(item[0], dtype=np.float32).copy(),
                float(item[1]) if len(item) > 1 else float(snapshot.score),
            ]
        return {
            "track_id": int(snapshot.track_id),
            "box": np.asarray(snapshot.box, dtype=np.float32).copy(),
            "score": float(snapshot.score),
            "mean": (
                np.asarray(snapshot.mean, dtype=np.float32).copy()
                if snapshot.mean is not None
                else None
            ),
            "covariance": (
                np.asarray(snapshot.covariance, dtype=np.float32).copy()
                if snapshot.covariance is not None
                else None
            ),
            "velocity": np.asarray(snapshot.velocity, dtype=np.float32).copy(),
            "history": history,
            "end_frame_id": int(snapshot.end_frame_id),
            "state": int(snapshot.state),
        }

    def _compact_event_dict(self, event: TrackEvent) -> dict:
        return {
            "event_id": event.event_id,
            "frame_id": int(event.frame_id),
            "frame_index": int(event.frame_id) - 1,
            "track_id": int(event.track_id),
            "has_detection": bool(event.has_detection),
            "accepted_detection_index": (
                int(event.detection.detection_index)
                if event.has_detection and event.detection is not None
                else -1
            ),
            "frame_start_state": self._compact_snapshot_dict(event.frame_start_state),
            "pre_update_state": self._compact_snapshot_dict(event.pre_update_state),
            "scalar_features": np.asarray(event.scalar_features, dtype=np.float32).copy(),
            "track_feature": np.asarray(event.track_feature, dtype=np.float32).reshape(-1).copy(),
        }

    # ------------------------------------------------------------------
    # Track removal cleanup
    # ------------------------------------------------------------------

    def remove_track(self, track_id: int) -> None:
        """Remove all runtime state associated with *track_id*."""
        if self.runtime is not None:
            self.runtime.cleanup_track(track_id)
