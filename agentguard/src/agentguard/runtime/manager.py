from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.runtime.buffers import EventBuffer, TemporalTokenBuffer, WindowBuffer
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
        joint_model: Optional[torch.nn.Module] = None,
    ):
        self.config = config
        self.mode = config.get("mode", "off")
        self.device = device
        self.iwg = iwg_model
        self.tgr = tgr_model
        self.joint_model = joint_model
        if self.mode == "joint":
            if self.joint_model is None:
                raise RuntimeError("joint mode requires a combined IWGTSRM model")
            if self.tgr is not None:
                raise RuntimeError("joint mode must not instantiate TGR")
            self.iwg = self.joint_model.iwg
        if self.iwg is not None:
            self.iwg.to(device)
            self.iwg.eval()
        if self.tgr is not None:
            self.tgr.to(device)
            self.tgr.eval()
        if self.joint_model is not None:
            self.joint_model.to(device)
            self.joint_model.eval()

        # Per-track state
        self.event_buffers: Dict[int, EventBuffer] = {}
        self.window_buffers: Dict[int, WindowBuffer] = {}
        self.temporal_buffers: Dict[int, TemporalTokenBuffer] = {}
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
        self.joint_output = str(config.get("joint_output", "final"))
        if self.joint_output not in {"base", "final"}:
            raise ValueError("joint_output must be 'base' or 'final'")
        configured_window_size = config.get("joint_window_size")
        if self.mode == "joint" and configured_window_size is None:
            raise ValueError(
                "joint_window_size must be loaded from the combined checkpoint"
            )
        self.joint_window_size = int(configured_window_size or 16)
        if self.joint_window_size < 1:
            raise ValueError("joint_window_size must be positive")
        self.joint_max_frame_gap = int(config.get("joint_max_frame_gap", 30))

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

    def get_or_create_event_buffer(self, track_id: int) -> EventBuffer:
        if track_id not in self.event_buffers:
            self.event_buffers[track_id] = EventBuffer()
        return self.event_buffers[track_id]

    def get_or_create_temporal_buffer(self, track_id: int) -> TemporalTokenBuffer:
        if track_id not in self.temporal_buffers:
            self.temporal_buffers[track_id] = TemporalTokenBuffer(
                max_len=self.joint_window_size,
                max_frame_gap=self.joint_max_frame_gap,
            )
        return self.temporal_buffers[track_id]

    def _reset_joint_history_for_frame(self, track_id: int, frame_id: int) -> bool:
        """Clear IWG and TSRM history before encoding an event across a gap."""
        previous_frames: list[int] = []
        event_buffer = self.event_buffers.get(track_id)
        if event_buffer is not None and event_buffer.events:
            previous_frames.append(int(event_buffer.events[-1].frame_id))
        temporal_buffer = self.temporal_buffers.get(track_id)
        if temporal_buffer is not None and temporal_buffer.tokens:
            previous_frames.append(int(temporal_buffer.tokens[-1]["frame_id"]))
        if not previous_frames:
            return False
        if len(set(previous_frames)) != 1:
            if event_buffer is not None:
                event_buffer.clear()
            if temporal_buffer is not None:
                temporal_buffer.clear()
            return True
        previous_frame = previous_frames[0]
        gap = int(frame_id) - previous_frame
        if 0 < gap <= self.joint_max_frame_gap:
            return False
        if event_buffer is not None:
            event_buffer.clear()
        if temporal_buffer is not None:
            temporal_buffer.clear()
        return True

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

    def run_joint_inference(
        self,
        track_id: int,
        events_sequence: List[Optional[TrackEvent]],
        *,
        frame_id: int,
        has_detection: bool,
    ) -> Dict[str, np.ndarray]:
        return self.run_joint_batch_inference(
            [track_id],
            [events_sequence],
            frame_ids=[frame_id],
            has_detection=[has_detection],
        )[0]

    def run_joint_batch_inference(
        self,
        track_ids: List[int],
        event_sequences: List[List[Optional[TrackEvent]]],
        *,
        frame_ids: List[int],
        has_detection: List[bool],
    ) -> List[Dict[str, np.ndarray]]:
        if self.mode != "joint" or self.joint_model is None:
            raise RuntimeError("run_joint_batch_inference requires joint mode")
        size = len(track_ids)
        if not (
            len(event_sequences) == len(frame_ids) == len(has_detection) == size
        ):
            raise ValueError("joint batch inputs must have equal lengths")
        if size == 0:
            return []
        if len(set(track_ids)) != size:
            raise ValueError("joint batch cannot contain duplicate track ids")
        if self.feature_builder is None:
            raise RuntimeError("joint runtime feature builder is not initialized")

        event_sequences = list(event_sequences)
        for index, (track_id, frame_id) in enumerate(zip(track_ids, frame_ids)):
            if self._reset_joint_history_for_frame(track_id, frame_id):
                sequence = event_sequences[index]
                if not sequence or sequence[-1] is None:
                    raise ValueError("joint event sequence must end with the current event")
                event_sequences[index] = [None] * (len(sequence) - 1) + [sequence[-1]]

        device = next(self.joint_model.parameters()).device
        inputs = self.feature_builder.build_iwg_batch_input(event_sequences)
        track_t = inputs["track_feats"].to(device, non_blocking=True)
        det_t = inputs["det_feats"].to(device, non_blocking=True)
        scalar_t = inputs["scalar_feats"].to(device, non_blocking=True)
        mask_t = inputs["mask"].to(device, non_blocking=True)
        has_current = torch.as_tensor(has_detection, dtype=torch.bool, device=device)
        with torch.inference_mode():
            iwg_outputs = self.joint_model.iwg(track_t, det_t, scalar_t, mask_t)
            iwg_outputs = self.joint_model.apply_unmatched_sentinel(
                iwg_outputs, has_current
            )

        for index, track_id in enumerate(track_ids):
            token = {
                "frame_id": int(frame_ids[index]),
                "event_embedding": iwg_outputs["event_embedding"][index].float().cpu().numpy(),
                "scalar_feats": scalar_t[index, -1].float().cpu().numpy(),
                "base_gate": iwg_outputs["base_gate"][index].float().cpu().numpy(),
                "policy_probs": iwg_outputs["policy_probs"][index].float().cpu().numpy(),
                "cue": iwg_outputs["cue"][index].float().cpu().numpy(),
                "risk": iwg_outputs["risk"][index].float().cpu().numpy(),
                "has_detection": bool(has_detection[index]),
            }
            self.get_or_create_temporal_buffer(track_id).push(token)

        window_size = self.joint_window_size
        event_embedding = np.zeros((size, window_size, 128), dtype=np.float32)
        scalar_feats = np.zeros((size, window_size, 63), dtype=np.float32)
        base_gate = np.zeros((size, window_size, 2), dtype=np.float32)
        policy_probs = np.zeros((size, window_size, 5), dtype=np.float32)
        cue = np.zeros((size, window_size, 3), dtype=np.float32)
        risk = np.zeros((size, window_size, 4), dtype=np.float32)
        padding = np.ones((size, window_size), dtype=np.bool_)
        detection_mask = np.zeros((size, window_size), dtype=np.bool_)
        reset = np.zeros((size, window_size), dtype=np.bool_)
        for batch_index, track_id in enumerate(track_ids):
            tokens = self.temporal_buffers[track_id].get_window()
            offset = window_size - len(tokens)
            reset[batch_index, offset] = True
            for position, token in enumerate(tokens, start=offset):
                event_embedding[batch_index, position] = token["event_embedding"]
                scalar_feats[batch_index, position] = token["scalar_feats"]
                base_gate[batch_index, position] = token["base_gate"]
                policy_probs[batch_index, position] = token["policy_probs"]
                cue[batch_index, position] = token["cue"]
                risk[batch_index, position] = token["risk"]
                padding[batch_index, position] = False
                detection_mask[batch_index, position] = token["has_detection"]

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(device, non_blocking=True)

        with torch.inference_mode():
            temporal = self.joint_model.tsrm(
                event_embedding=tensor(event_embedding),
                scalar_feats=tensor(scalar_feats),
                base_gate=tensor(base_gate),
                policy_probs=tensor(policy_probs),
                cue=tensor(cue),
                risk=tensor(risk),
                padding_mask=tensor(padding),
                has_detection_mask=tensor(detection_mask),
                reset_mask=tensor(reset),
            )
        final = temporal["final_gate"][:, -1].float().cpu().numpy()
        correction = temporal["temporal_gate_correction"][:, -1].float().cpu().numpy()
        strength = temporal["revision_strength"][:, -1, 0].float().cpu().numpy()
        base = iwg_outputs["base_gate"].float().cpu().numpy()
        policy = iwg_outputs["policy_probs"].float().cpu().numpy()
        cue_values = iwg_outputs["cue"].float().cpu().numpy()
        risk_values = iwg_outputs["risk"].float().cpu().numpy()
        applied = base if self.joint_output == "base" else final
        results: List[Dict[str, np.ndarray]] = []
        for index in range(size):
            self.stats.record_joint(
                base[index],
                final[index],
                correction[index],
                float(strength[index]),
                applied[index],
            )
            results.append(
                {
                    "gate": applied[index],
                    "base_gate": base[index],
                    "final_gate": final[index],
                    "temporal_gate_correction": correction[index],
                    "revision_strength": np.asarray(strength[index]),
                    "policy_probs": policy[index],
                    "cue": cue_values[index],
                    "risk": risk_values[index],
                    "event_embedding": iwg_outputs["event_embedding"][index]
                    .float()
                    .cpu()
                    .numpy(),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_track(self, track_id: int) -> None:
        """Remove all runtime state associated with *track_id*."""
        self.event_buffers.pop(track_id, None)
        self.window_buffers.pop(track_id, None)
        self.temporal_buffers.pop(track_id, None)
        self.checkpoints.cleanup_track(track_id)
