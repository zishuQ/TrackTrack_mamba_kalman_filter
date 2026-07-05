from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.enums import DetectionSource
from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.contracts.states import (
    TrackStateSnapshot,
)

from trackers.track import Track

from .converters import (
    export_association_pair,
    export_detection,
    export_track_state,
)

_REID_DIM = 2048  # default ReID feature dimension


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
        self.event_sink = agentguard_runtime.event_sink if agentguard_runtime else None

        # Pending frame-level data for EventSink
        self._pending_frame_record = None
        self._pending_frame_events: List[dict] = []

        # Initialise runtime sub-components when available
        if self.runtime is not None:
            reid_dim = getattr(args, "reid_dim", _REID_DIM)
            self.runtime.init_feature_builder(reid_dim=reid_dim)
            self.runtime.init_motion_model()

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(
        self, frame_id: int, img_width: int, img_height: int
    ) -> None:
        """Called at the start of each frame.

        Stores the current frame id and image dimensions so they are available
        when building events later in the same frame.
        """
        self.frame_id = frame_id
        self.img_width = img_width
        self.img_height = img_height

        # Accumulate frame record for EventSink
        self._pending_frame_record = {
            "frame_id": frame_id,
            "image_width": img_width,
            "image_height": img_height,
            "warp_matrix": None,  # Set when warp is known
            "detections": [],
        }
        self._pending_frame_events = []

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
            Meta dict containing association cost matrices (see
            :func:`.converters.export_association_pair`).
        warp_matrix : np.ndarray
            (2, 3) warp matrix from global motion compensation.
        frame_start_state_snapshot : TrackStateSnapshot
            Snapshot taken at the start of the frame (before prediction).
        pre_update_state_snapshot : TrackStateSnapshot
            Snapshot taken after KF predict but before the update step.
        """
        # Detection source is read from the meta dict.
        det_source = int(
            association_meta["detection_source"][track_idx, det_idx]
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
            detection=export_detection(detection, det_source, det_idx),
            association=export_association_pair(
                association_meta, track_idx, det_idx
            ),
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

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

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

    def finalize_first_stage(
        self,
        tracked_lost_tracks: List[Track],
    ) -> None:
        """Called after first-stage association is complete.

        In ``'full'`` mode:
          1. Asks the runtime to build ``ReplayPlan`` objects for every
             track whose TGR window has reached capacity.
          2. Executes each plan on the corresponding live ``Track`` via
             ``TrackTrackReplayBackend.replay_into_live_track()``.

        This actually modifies the live Track state, applying the revised
        gates determined by TGR inference.
        """
        if self.runtime is not None:
            plans = self.runtime.finalize_first_stage()
            if plans:
                from integrations.agentguard.replay_backend import (
                    TrackTrackReplayBackend,
                )

                backend = TrackTrackReplayBackend()
                track_by_id = {t.track_id: t for t in tracked_lost_tracks}

                for track_id, plan in plans.items():
                    live_track = track_by_id.get(track_id)
                    if live_track is None:
                        continue
                    backend.replay_into_live_track(live_track, plan)

        # Flush pending events to EventSink
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
        if self.runtime is None or self.runtime.iwg is None:
            return GateDecision(1.0, 1.0, np.ones(5, dtype=np.float64) / 5.0, 1.0)

        # --- Populate feature vectors on the event ---
        fb = self.runtime.feature_builder

        # Track feature: use the pre-update (post-predict) snapshot if available,
        # otherwise fall back to the frame-start snapshot.
        src_state = event.pre_update_state or event.frame_start_state
        if src_state is not None and src_state.feature.size > 0:
            event.track_feature = src_state.feature.copy()
        else:
            event.track_feature = np.zeros(fb.reid_dim, dtype=np.float64)

        # Detection feature
        if event.has_detection and event.detection is not None:
            event.detection_feature = event.detection.feature.copy()
        else:
            event.detection_feature = np.zeros(fb.reid_dim, dtype=np.float64)

        # Scalar features (computed from the event's raw data)
        event.scalar_features = fb.compute_scalar(event)

        # --- Build inference sequence from buffer + current event ---
        seq_len = 6
        buffer = self.runtime.event_buffers.get(track_id)
        if buffer is None:
            seq = [None] * (seq_len - 1) + [event]
        else:
            hist = buffer.get_sequence()
            seq = hist[-(seq_len - 1):] + [event]

        result = self.runtime.run_iwg_inference(seq)

        return GateDecision(
            motion_gate=float(result["gate"][0]),
            appearance_gate=float(result["gate"][1]),
            policy_probs=result["policy_probs"],
            confidence=float(np.mean(result["cue"])),
        )

    # ------------------------------------------------------------------
    # Event recording (buffer management)
    # ------------------------------------------------------------------

    def record_event(
        self,
        track_id: int,
        event: TrackEvent,
        gate_decision: GateDecision,
    ) -> None:
        """Record a matched event into the runtime buffers for later TGR replay.

        Attaches IWG outputs to the event and saves a checkpoint for TGR
        when in ``'full'`` mode.  Does **not** call ``track.update_with_gates``
        (the gate has already been applied via :meth:`apply_gate`).
        """
        if self.runtime is None:
            # Still accumulate for EventSink even without runtime
            self._accumulate_event_for_sink(event)
            return

        self.runtime.stats.record_event()
        event_buffer, window_buffer = self.runtime.get_or_create_buffer(track_id)
        event_buffer.push(event)
        window_buffer.push(event)

        # Attach IWG outputs
        event.iwg_policy_probs = gate_decision.policy_probs.copy()
        event.iwg_gate = np.array(
            [gate_decision.motion_gate, gate_decision.appearance_gate],
            dtype=np.float64,
        )

        # Save checkpoint for TGR (full mode)
        if self.runtime.mode == "full" and event.frame_start_state is not None:
            self.runtime.checkpoints.save_checkpoint(
                track_id, event.event_id, event.frame_start_state
            )

        # Accumulate for EventSink
        self._accumulate_event_for_sink(event)

    def record_unmatched_event(
        self,
        track_id: int,
        event: TrackEvent,
    ) -> None:
        """Record an unmatched (lost) event into the runtime buffers.

        The gate is forced to ``[0.0, 0.0]`` (no detection).
        """
        if self.runtime is None:
            self._accumulate_event_for_sink(event)
            return

        self.runtime.stats.record_event()
        event_buffer, window_buffer = self.runtime.get_or_create_buffer(track_id)
        event_buffer.push(event)
        window_buffer.push(event)

        event.iwg_gate = np.array([0.0, 0.0], dtype=np.float64)

        if self.runtime.mode == "full" and event.frame_start_state is not None:
            self.runtime.checkpoints.save_checkpoint(
                track_id, event.event_id, event.frame_start_state
            )

        # Accumulate for EventSink
        self._accumulate_event_for_sink(event)

    # ------------------------------------------------------------------
    # EventSink accumulation helper
    # ------------------------------------------------------------------

    def _accumulate_event_for_sink(self, event: TrackEvent) -> None:
        """Serialise *event* and append to the pending frame events list."""
        if self.event_sink is None:
            return
        if self._pending_frame_record is None:
            return  # begin_frame was not called

        from agentguard.contracts.serialization import serialize_event

        event_dict = serialize_event(event)
        event_dict["target_gt_id"] = None
        event_dict["detection_gt_id"] = None
        event_dict["accepted_detection_index"] = (
            event.detection.detection_index
            if event.has_detection and event.detection is not None
            else None
        )
        event_dict["association_row"] = (
            event.association.final_cost
            if event.association is not None
            else None
        )
        self._pending_frame_events.append(event_dict)

    # ------------------------------------------------------------------
    # Track removal cleanup
    # ------------------------------------------------------------------

    def remove_track(self, track_id: int) -> None:
        """Remove all runtime state associated with *track_id*."""
        if self.runtime is not None:
            self.runtime.cleanup_track(track_id)
