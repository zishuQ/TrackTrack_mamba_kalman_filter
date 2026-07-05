from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.runtime.buffers import EventBuffer, WindowBuffer
from agentguard.runtime.checkpoint import CheckpointManager
from agentguard.runtime.replay import ReplayEngine
from agentguard.runtime.statistics import RuntimeStatistics


class AgentGuardRuntime:
    """
    Central runtime manager for AgentGuard online inference.

    Tracks
    ------
    - Per-track event buffers (IWG history)
    - Per-track window buffers (TGR windows)
    - Checkpoint manager
    - Replay engine

    Modes
    -----
    - ``'off'`` : No AgentGuard, all tracks follow original TrackTrack.
    - ``'iwg'`` : Load IWG only, apply per-frame gating.
    - ``'full'`` : Load IWG + TGR, apply gating + temporal revision + replay.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        iwg_model: Optional[torch.nn.Module] = None,
        tgr_model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.mode = config.get("mode", "off")
        self.device = device
        self.iwg = iwg_model
        self.tgr = tgr_model
        if self.iwg is not None:
            self.iwg.to(device)
            self.iwg.eval()
        if self.tgr is not None:
            self.tgr.to(device)
            self.tgr.eval()

        # Per-track state
        self.event_buffers: Dict[int, EventBuffer] = {}
        self.window_buffers: Dict[int, WindowBuffer] = {}
        self.checkpoints = CheckpointManager()
        self.replay = ReplayEngine(motion_model=None)  # will be set

        # Feature builder (lazy init)
        self.feature_builder: Any = None

        # Statistics
        self.stats = RuntimeStatistics()

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def init_feature_builder(
        self,
        reid_dim: int,
        norm_stats: Any = None,
    ) -> None:
        """Initialise the feature builder once the ReID dimension is known.

        Parameters
        ----------
        reid_dim : int
            Dimensionality of the ReID feature vectors.
        norm_stats : NormalizationStats or None
            Optional pre-computed normalisation statistics for scalar features.
        """
        from agentguard.features.builder import EventFeatureBuilder

        self.feature_builder = EventFeatureBuilder(reid_dim, norm_stats)

    def init_motion_model(self) -> None:
        """Initialise the Kalman filter used by the replay engine."""
        from agentguard.motion.nsa_numpy import NSAKalmanFilter

        self.replay.motion = NSAKalmanFilter()

    # ------------------------------------------------------------------
    # Track maturity
    # ------------------------------------------------------------------

    def is_mature_track(self, track) -> bool:
        """A track is mature if its state is Tracked or Lost and its history
        contains at least 6 entries."""
        return track.state in (1, 2) and len(track.history) >= 6

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def get_or_create_buffer(
        self,
        track_id: int,
    ) -> tuple[EventBuffer, WindowBuffer]:
        """Return the event and window buffers for *track_id*, creating them
        on first access."""
        if track_id not in self.event_buffers:
            self.event_buffers[track_id] = EventBuffer()
            self.window_buffers[track_id] = WindowBuffer()
        return self.event_buffers[track_id], self.window_buffers[track_id]

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_matched_event(
        self,
        track,
        detection,
        event: TrackEvent,
        gate_decision: GateDecision,
    ) -> None:
        """Process an accepted match for a mature track.

        Stages
        ------
        1. Save the event into both the IWG event buffer and TGR window buffer.
        2. Attach IWG outputs (policy distribution and gates) to the event.
        3. Call ``track.update_with_gates(...)`` with the decision's gates.
        """
        self.stats.record_event()

        if self.mode == "off":
            return

        event_buffer, window_buffer = self.get_or_create_buffer(track.track_id)
        event_buffer.push(event)
        window_buffer.push(event)

        # Attach IWG outputs to the event for later TGR use
        event.iwg_policy_probs = gate_decision.policy_probs.copy()
        event.iwg_gate = np.array(
            [gate_decision.motion_gate, gate_decision.appearance_gate],
            dtype=np.float64,
        )

        # Update the live track with the gated decision
        track.update_with_gates(
            event.frame_id,
            detection,
            gate_decision.motion_gate,
            gate_decision.appearance_gate,
        )

        self.stats.record_iwg(event.iwg_gate)

    def process_unmatched_event(
        self,
        track,
        event: TrackEvent,
    ) -> None:
        """Process an unmatched track (mark lost).

        The detection is absent, so the gate is forced to ``[0.0, 0.0]``.
        Non-mature tracks continue with their original ``mark_lost`` logic;
        mature tracks are also marked lost (no additional gated update needed).
        """
        self.stats.record_event()

        if self.mode == "off":
            return

        event_buffer, window_buffer = self.get_or_create_buffer(track.track_id)
        event_buffer.push(event)
        window_buffer.push(event)

        # No detection → gate forced to [0, 0]
        event.iwg_gate = np.array([0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Stage finalisation (TGR)
    # ------------------------------------------------------------------

    def finalize_first_stage(
        self,
        all_tracks: List[Any],
        all_matches: List[Any],
    ) -> None:
        """Called after first-stage association is complete.

        In ``'full'`` mode this runs TGR inference on every track whose
        window buffer has reached capacity, replays the window with revised
        gates, and updates the checkpoint chain.

        Parameters
        ----------
        all_tracks : list
            List of all active Track objects (provided by the tracker).
        all_matches : list
            List of all accepted match tuples from the first association stage.
        """
        if self.mode != "full":
            return

        # For each track with a FULL window
        for track_id, window_buffer in list(self.window_buffers.items()):
            if not window_buffer.is_full:
                continue

            # Get the 4 window events (oldest first)
            window_events = window_buffer.get_window()
            if len(window_events) < window_buffer.window_size:
                continue

            # Run TGR inference → revised gates for all 4 events
            revised_gates = self._run_tgr_inference(window_events)
            self.stats.record_tgr()

            # Determine the checkpoint snapshot (state before oldest event)
            oldest_event = window_events[0]
            checkpoint_snapshot = self.checkpoints.get_checkpoint(
                track_id, oldest_event.event_id
            )
            if checkpoint_snapshot is None:
                continue

            # Assign revised gates to the events (for offline use / logging)
            for evt, gate in zip(window_events, revised_gates):
                evt.revised_gate = gate.copy()

            # ------------------------------------------------------------------
            # Full replay implementation:
            # 1. Replay the entire window from the checkpoint with revised gates.
            # 2. Save a new checkpoint for the state after the (now-replayed)
            #    oldest event — this becomes the checkpoint for the new window
            #    start (the old second-oldest event).
            # 3. Pop the oldest event from the window buffer.
            # 4. Remove the old checkpoint.
            #
            # The live track state is **not** modified; only the checkpoints
            # and events are updated.
            # ------------------------------------------------------------------
            final_snapshot = self.replay.replay_window(
                checkpoint_snapshot,
                window_events,
                revised_gates,
            )
            self.stats.record_replay()

            # Save new checkpoint: replay only the oldest event to produce the
            # state that should be checkpointed before the new oldest event.
            if len(window_events) > 1:
                new_checkpoint = self.replay.replay_event(
                    checkpoint_snapshot,
                    window_events[0],
                    revised_gates[0],
                )
                self.checkpoints.save_checkpoint(
                    track_id,
                    window_events[1].event_id,
                    new_checkpoint,
                )

            # Pop oldest from window and remove its checkpoint
            window_buffer.pop_oldest()
            self.checkpoints.remove_checkpoint(track_id, oldest_event.event_id)

    def _run_tgr_inference(
        self,
        window_events: List[TrackEvent],
    ) -> np.ndarray:
        """Run TGR on a window of events and return revised gates.

        Parameters
        ----------
        window_events : list of TrackEvent
            Exactly 4 events ordered from oldest to newest.

        Returns
        -------
        ndarray, shape ``(4, 2)``
            Revised ``[motion_gate, appearance_gate]`` for each event.
        """
        if self.tgr is None or self.feature_builder is None:
            return np.zeros((len(window_events), 2), dtype=np.float64)

        device = next(self.tgr.parameters()).device
        seq_len = len(window_events)

        # Collect per-event data
        track_feats: List[np.ndarray] = []
        det_feats: List[np.ndarray] = []
        scalar_feats: List[np.ndarray] = []
        iwg_policy_probs: List[np.ndarray] = []
        iwg_gates: List[np.ndarray] = []
        has_detection: List[float] = []

        for evt in window_events:
            track_feats.append(evt.track_feature)

            if evt.has_detection and evt.detection_feature.size > 0:
                det_feats.append(evt.detection_feature)
            else:
                det_feats.append(np.zeros(self.feature_builder.reid_dim))

            scalar_feats.append(evt.scalar_features)

            if evt.iwg_policy_probs is not None:
                iwg_policy_probs.append(evt.iwg_policy_probs)
            else:
                iwg_policy_probs.append(np.ones(5, dtype=np.float64) / 5.0)

            if evt.iwg_gate is not None:
                iwg_gates.append(evt.iwg_gate)
            else:
                iwg_gates.append(np.ones(2, dtype=np.float64))

            has_detection.append(float(evt.has_detection))

        # Stack into (1, seq_len, D) tensors
        track_t = (
            torch.from_numpy(np.stack(track_feats, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        det_t = (
            torch.from_numpy(np.stack(det_feats, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        scalar_t = (
            torch.from_numpy(np.stack(scalar_feats, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        policy_t = (
            torch.from_numpy(np.stack(iwg_policy_probs, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        gate_t = (
            torch.from_numpy(np.stack(iwg_gates, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        has_det_t = (
            torch.from_numpy(np.array(has_detection, dtype=np.bool_))
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            g_revised: torch.Tensor = self.tgr(
                track_t, det_t, scalar_t, policy_t, gate_t, has_det_t
            )[0]

        return g_revised.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # IWG inference
    # ------------------------------------------------------------------

    def run_iwg_inference(
        self,
        events_sequence: List[Optional[TrackEvent]],
    ) -> Dict[str, np.ndarray]:
        """Run IWG on a (possibly padded) sequence of events.

        Parameters
        ----------
        events_sequence : list of TrackEvent or None
            Output of ``EventBuffer.get_sequence()`` — a fixed-length list
            with ``None`` padding on the left for positions before the
            buffer filled up.

        Returns
        -------
        dict with keys:
            - ``policy_probs`` : (5,) ndarray  — policy distribution.
            - ``gate`` : (2,) ndarray — ``[motion_gate, appearance_gate]``.
            - ``event_logits`` : (10,) ndarray — event-type logits.
            - ``cue`` : (3,) ndarray — predicted cues (sigmoid outputs).
            - ``gate_residual`` : (2,) ndarray — residual before tanh mixing.
        """
        # Fallback when IWG is unavailable
        if self.iwg is None or self.feature_builder is None:
            self.stats.record_skipped_immature()
            return {
                "policy_probs": np.ones(5, dtype=np.float64) / 5.0,
                "gate": np.ones(2, dtype=np.float64),
                "event_logits": np.zeros(10, dtype=np.float64),
                "cue": np.ones(3, dtype=np.float64),
                "gate_residual": np.zeros(2, dtype=np.float64),
            }

        device = next(self.iwg.parameters()).device
        seq_len = len(events_sequence)

        # Collect per-event data; padded (None) events use zero features.
        track_feats: List[np.ndarray] = []
        det_feats: List[np.ndarray] = []
        scalar_feats: List[np.ndarray] = []

        for evt in events_sequence:
            if evt is None:
                # Padded slot — use zeros
                track_feats.append(np.zeros(self.feature_builder.reid_dim))
                det_feats.append(np.zeros(self.feature_builder.reid_dim))
                scalar_feats.append(np.zeros(63, dtype=np.float64))
            else:
                track_feats.append(evt.track_feature)
                if evt.has_detection and evt.detection_feature.size > 0:
                    det_feats.append(evt.detection_feature)
                else:
                    det_feats.append(np.zeros(self.feature_builder.reid_dim))
                scalar_feats.append(evt.scalar_features)

        # Build padding mask: True = padded / masked-out position
        mask = np.zeros(seq_len, dtype=np.bool_)
        for i, evt in enumerate(events_sequence):
            if evt is None:
                mask[i] = True

        # Convert to tensors with batch dimension
        track_t = (
            torch.from_numpy(np.stack(track_feats, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        det_t = (
            torch.from_numpy(np.stack(det_feats, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        scalar_t = (
            torch.from_numpy(np.stack(scalar_feats, axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        mask_t = (
            torch.from_numpy(mask).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            outputs = self.iwg(track_t, det_t, scalar_t, mask_t)

        # Squeeze batch dimension
        return {
            "policy_probs": outputs["policy_probs"].squeeze(0).cpu().numpy(),
            "gate": outputs["gate"].squeeze(0).cpu().numpy(),
            "event_logits": outputs["event_logits"].squeeze(0).cpu().numpy(),
            "cue": outputs["cue"].squeeze(0).cpu().numpy(),
            "gate_residual": outputs["gate_residual"].squeeze(0).cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_track(self, track_id: int) -> None:
        """Remove all runtime state associated with *track_id*."""
        self.event_buffers.pop(track_id, None)
        self.window_buffers.pop(track_id, None)
        self.checkpoints.cleanup_track(track_id)
