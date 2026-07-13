from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.runtime.buffers import EventBuffer, WindowBuffer
from agentguard.runtime.checkpoint import CheckpointManager
from agentguard.runtime.statistics import RuntimeStatistics


class AgentGuardRuntime:
    """
    Central runtime manager for AgentGuard online inference.

    Tracks
    ------
    - Per-track event buffers (IWG history)
    - Per-track window buffers (TGR windows)
    - Checkpoint manager

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

        # Feature builder (initialised once via init_feature_builder)
        self.feature_builder: Any = None
        self._feature_builder_initialised: bool = False

        # EventSink (set by external caller if needed)
        self.event_sink: Any = None
        self.pending_checkpoint_rolls: Dict[int, Any] = {}
        self.gate_revision_skip_threshold = float(
            config.get("replay_diff_threshold", 0.0) or 0.0
        )
        self.tgr_frame_stride = max(int(config.get("tgr_frame_stride", 1) or 1), 1)
        self._finalize_frame_index = 0

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
        if self._feature_builder_initialised:
            return
        from agentguard.features.builder import EventFeatureBuilder

        self.feature_builder = EventFeatureBuilder(reid_dim, norm_stats)
        self._feature_builder_initialised = True

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
        all_tracks: Optional[List[Any]] = None,
        all_matches: Optional[List[Any]] = None,
    ) -> Dict[int, Any]:
        """Build replay plans for all tracks with full TGR windows.

        Does **not** modify live track state.  The caller receives
        ``ReplayPlan`` objects and must execute them on live ``Track``
        objects via ``TrackTrackReplayBackend.replay_into_live_track()``.

        In full mode, missing TGR model or checkpoint are raised as
        ``RuntimeError`` rather than silently skipped.

        Parameters
        ----------
        all_tracks : list or None
            Unused, reserved for API compatibility.
        all_matches : list or None
            Unused, reserved for API compatibility.

        Returns
        -------
        dict[int, ReplayPlan]
            ``track_id`` → ``ReplayPlan`` for every track whose TGR window
            reached capacity.
        """
        from .replay import ReplayPlan, ReplayStep

        if self.mode != "full":
            return {}
        if self.tgr is None:
            raise RuntimeError("TGR model is None in full mode - cannot build replay plans")
        if self.feature_builder is None:
            raise RuntimeError("Feature builder is None in full mode - cannot build replay plans")

        plans: Dict[int, Any] = {}
        self.pending_checkpoint_rolls = {}
        self._finalize_frame_index += 1

        ready_windows = []
        for track_id, window_buffer in list(self.window_buffers.items()):
            if not window_buffer.is_full:
                continue
            ready_windows.append((track_id, window_buffer.get_window()))

        if not ready_windows:
            return plans

        if self.tgr_frame_stride > 1 and self._finalize_frame_index % self.tgr_frame_stride != 0:
            skipped = 0
            for track_id, window_events in ready_windows:
                oldest_event = window_events[0]
                self.window_buffers[track_id].pop_oldest()
                self.checkpoints.remove_checkpoint(track_id, oldest_event.event_id)
                skipped += 1
            self.stats.record_skipped_tgr(skipped)
            return plans

        revised_by_track = self._run_tgr_batch_inference(
            [window_events for _, window_events in ready_windows]
        )
        self.stats.record_tgr(len(ready_windows))

        for (track_id, window_events), revised_gates in zip(ready_windows, revised_by_track):

            for evt, gate in zip(window_events, revised_gates):
                evt.revised_gate = gate.copy()

            oldest_event = window_events[0]
            original_gates = np.stack(
                [
                    np.asarray(
                        evt.iwg_gate if evt.iwg_gate is not None else np.zeros(2),
                        dtype=np.float64,
                    )
                    for evt in window_events
                ],
                axis=0,
            )
            max_abs_diff = float(np.max(np.abs(revised_gates - original_gates)))
            self.stats.record_revision_diff(max_abs_diff)
            if max_abs_diff <= self.gate_revision_skip_threshold:
                self.window_buffers[track_id].pop_oldest()
                self.checkpoints.remove_checkpoint(track_id, oldest_event.event_id)
                self.stats.record_skipped_replay()
                continue

            checkpoint_snapshot = self.checkpoints.get_checkpoint(
                track_id, oldest_event.event_id
            )
            if checkpoint_snapshot is None:
                raise RuntimeError(
                    f"Missing checkpoint for track {track_id} event {oldest_event.event_id} "
                    f"in full mode - cannot build replay plan"
                )

            steps = []
            for evt, gate in zip(window_events, revised_gates):
                has_det = evt.has_detection
                det = evt.detection if has_det else None
                steps.append(ReplayStep(
                    frame_id=evt.frame_id,
                    warp_matrix=evt.warp_matrix.copy(),
                    has_detection=has_det,
                    detection=det,
                    motion_gate=float(np.clip(gate[0], 0.0, 1.0)),
                    appearance_gate=float(np.clip(gate[1], 0.0, 1.0)),
                ))

            plans[track_id] = ReplayPlan(
                track_id=track_id,
                checkpoint=checkpoint_snapshot,
                steps=steps,
                oldest_event_id=oldest_event.event_id,
                next_event_id=window_events[1].event_id if len(window_events) > 1 else None,
            )
            self.stats.record_replay(len(steps))
            self.pending_checkpoint_rolls[track_id] = {
                "oldest_event_id": oldest_event.event_id,
                "next_event_id": window_events[1].event_id if len(window_events) > 1 else None,
                "oldest_step": steps[0],
                "checkpoint": checkpoint_snapshot,
            }

        return plans

    def _run_tgr_batch_inference(
        self,
        windows: List[List[TrackEvent]],
    ) -> np.ndarray:
        """Run TGR for many track windows in one model call.

        Full mode may have tens of mature tracks per frame. Calling the small
        transformer once per track is dominated by Python and dispatcher
        overhead on CPU; batching keeps semantics identical while making the
        online path usable for experiments.
        """
        if not windows:
            return np.zeros((0, 4, 2), dtype=np.float64)
        if self.tgr is None:
            raise RuntimeError("TGR model is not loaded but full mode was requested")
        if self.feature_builder is None:
            raise RuntimeError("Feature builder is not initialized but full mode was requested")

        device = next(self.tgr.parameters()).device
        inputs = self.feature_builder.build_tgr_batch_input(windows)
        track_t = inputs["track_feats"].to(device, non_blocking=True)
        det_t = inputs["det_feats"].to(device, non_blocking=True)
        scalar_t = inputs["scalar_feats"].to(device, non_blocking=True)
        policy_t = inputs["iwg_policy_probs"].to(device, non_blocking=True)
        gate_t = inputs["iwg_gates"].to(device, non_blocking=True)
        has_det_t = inputs["has_detection_mask"].to(device, non_blocking=True)

        with torch.inference_mode():
            g_revised: torch.Tensor = self.tgr(
                track_t, det_t, scalar_t, policy_t, gate_t, has_det_t
            )[0]

        return g_revised.cpu().numpy()

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
        if self.tgr is None:
            raise RuntimeError("TGR model is not loaded but full mode was requested")
        if self.feature_builder is None:
            raise RuntimeError("Feature builder is not initialized but full mode was requested")

        device = next(self.tgr.parameters()).device

        # Use feature builder for TGR input
        inputs = self.feature_builder.build_tgr_input(window_events)
        track_t = inputs["track_feats"].to(device)
        det_t = inputs["det_feats"].to(device)
        scalar_t = inputs["scalar_feats"].to(device)
        policy_t = inputs["iwg_policy_probs"].to(device)
        gate_t = inputs["iwg_gates"].to(device)
        has_det_t = inputs["has_detection_mask"].to(device)

        with torch.inference_mode():
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
            - ``risk`` : (4,) ndarray — semantic risk probabilities.
            - ``cue`` : (3,) ndarray — predicted cues (sigmoid outputs).
            - ``iwg_gate_residual`` : (2,) ndarray — local IWG residual.
        """
        # Fallback when IWG is unavailable
        if self.iwg is None or self.feature_builder is None:
            if self.mode in ("iwg", "full"):
                raise RuntimeError(
                    f"IWG model or feature builder is None in {self.mode} mode"
                )
            self.stats.record_skipped_immature()
            return {
                "policy_probs": np.ones(5, dtype=np.float64) / 5.0,

                "gate": np.ones(2, dtype=np.float64),
                "risk": np.zeros(4, dtype=np.float64),
                "cue": np.ones(3, dtype=np.float64),
                "iwg_gate_residual": np.zeros(2, dtype=np.float64),
            }

        device = next(self.iwg.parameters()).device

        # Use feature builder for IWG input
        inputs = self.feature_builder.build_iwg_input(events_sequence)
        track_t = inputs["track_feats"].to(device)
        det_t = inputs["det_feats"].to(device)
        scalar_t = inputs["scalar_feats"].to(device)
        mask_t = inputs["mask"].to(device)

        with torch.inference_mode():
            outputs = self.iwg(track_t, det_t, scalar_t, mask_t)

        # Squeeze batch dimension
        return {
            "policy_probs": outputs["policy_probs"].squeeze(0).cpu().numpy(),
            "gate": outputs["gate"].squeeze(0).cpu().numpy(),
            "risk": outputs["risk"].squeeze(0).cpu().numpy(),
            "cue": outputs["cue"].squeeze(0).cpu().numpy(),
            "iwg_gate_residual": outputs["iwg_gate_residual"].squeeze(0).cpu().numpy(),
        }

    def run_iwg_batch_inference(
        self,
        event_sequences: List[List[Optional[TrackEvent]]],
    ) -> List[Dict[str, np.ndarray]]:
        """Run IWG for many per-track sequences in one model call."""
        if not event_sequences:
            return []
        if self.iwg is None or self.feature_builder is None:
            if self.mode in ("iwg", "full"):
                raise RuntimeError(
                    f"IWG model or feature builder is None in {self.mode} mode"
                )
            return [
                {
                    "policy_probs": np.ones(5, dtype=np.float64) / 5.0,
                    "gate": np.ones(2, dtype=np.float64),
                    "risk": np.zeros(4, dtype=np.float64),
                    "cue": np.ones(3, dtype=np.float64),
                    "iwg_gate_residual": np.zeros(2, dtype=np.float64),
                }
                for _ in event_sequences
            ]

        device = next(self.iwg.parameters()).device
        inputs = self.feature_builder.build_iwg_batch_input(event_sequences)
        track_t = inputs["track_feats"].to(device, non_blocking=True)
        det_t = inputs["det_feats"].to(device, non_blocking=True)
        scalar_t = inputs["scalar_feats"].to(device, non_blocking=True)
        mask_t = inputs["mask"].to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = self.iwg(track_t, det_t, scalar_t, mask_t)

        policy = outputs["policy_probs"].cpu().numpy()
        gate = outputs["gate"].cpu().numpy()
        cue = outputs["cue"].cpu().numpy()
        risk = outputs["risk"].cpu().numpy()
        iwg_gate_residual = outputs["iwg_gate_residual"].cpu().numpy()
        return [
            {
                "policy_probs": policy[i],
                "gate": gate[i],
                "cue": cue[i],
                "risk": risk[i],
                "iwg_gate_residual": iwg_gate_residual[i],
            }
            for i in range(len(event_sequences))
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_track(self, track_id: int) -> None:
        """Remove all runtime state associated with *track_id*."""
        self.event_buffers.pop(track_id, None)
        self.window_buffers.pop(track_id, None)
        self.checkpoints.cleanup_track(track_id)
