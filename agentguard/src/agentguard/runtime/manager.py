from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agentguard.contracts.events import TrackEvent
from agentguard.runtime.buffers import EventBuffer
from agentguard.runtime.statistics import RuntimeStatistics


class AgentGuardRuntime:
    """Coordinate capture and the current IWG + RG-CMA online path.

    ``off`` leaves TrackTrack untouched, ``capture`` only keeps the compact
    event sink active, and ``iwg-rg-cma`` runs the combined model on the most
    the configured number of events for each track.  Track mutation remains in the adapter;
    this class owns model inputs, per-track context, and runtime statistics.
    """

    _MODES = frozenset({"off", "capture", "iwg-rg-cma"})

    def __init__(
        self,
        config: Dict[str, Any],
        rg_cma_model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.mode = str(config.get("mode", "off"))
        if self.mode not in self._MODES:
            raise ValueError(
                f"unsupported AgentGuard mode {self.mode!r}; "
                f"expected one of {sorted(self._MODES)}"
            )
        if self.mode == "iwg-rg-cma" and rg_cma_model is None:
            raise RuntimeError("iwg-rg-cma mode requires a combined RG-CMA model")
        if self.mode != "iwg-rg-cma" and rg_cma_model is not None:
            raise ValueError(f"{self.mode} mode must not load a RG-CMA model")

        self.device = device
        self.rg_cma_model = rg_cma_model
        if self.rg_cma_model is not None:
            self.rg_cma_model.to(device)
            self.rg_cma_model.eval()

        self.event_buffers: Dict[int, EventBuffer] = {}

        # Feature builder is initialized by the TrackTrack integration once
        # the input ReID dimension or checkpoint metadata is known.
        self.feature_builder: Any = None
        self._feature_builder_initialised = False

        self.event_sink: Any = None
        self.rg_cma_output = str(config.get("rg_cma_output", "final"))
        if self.rg_cma_output not in {"base", "final"}:
            raise ValueError("rg_cma_output must be 'base' or 'final'")
        self.rg_cma_alpha = float(config.get("rg_cma_alpha", 1.0))
        if not math.isfinite(self.rg_cma_alpha) or self.rg_cma_alpha < 0.0:
            raise ValueError("rg_cma_alpha must be finite and non-negative")
        self.disable_kf_gate = bool(config.get("disable_kf_gate", False))
        self.disable_ema_gate = bool(config.get("disable_ema_gate", False))
        if self.disable_kf_gate and self.disable_ema_gate:
            raise ValueError(
                "gate-channel ablation must disable at most one of "
                "the KF and EMA gates"
            )
        self.fallback_threshold = float(config.get("fallback_threshold", 0.0))
        if not 0.0 <= self.fallback_threshold <= 1.0:
            raise ValueError("fallback_threshold must be in [0, 1]")

        self.iwg_context_size = int(config.get("iwg_context_size", 6))
        if self.iwg_context_size < 1:
            raise ValueError(
                f"iwg_context_size must be positive, got {self.iwg_context_size}"
            )
        if self.mode == "iwg-rg-cma":
            model_context_size = int(
                getattr(self.rg_cma_model, "context_size", self.iwg_context_size)
            )
            if model_context_size != self.iwg_context_size:
                raise ValueError(
                    "runtime/model context_size mismatch: "
                    f"{self.iwg_context_size} != {model_context_size}"
                )
        self.rg_cma_max_gap = int(config.get("rg_cma_max_gap", 30))
        if self.rg_cma_max_gap < 1:
            raise ValueError(f"rg_cma_max_gap must be positive, got {self.rg_cma_max_gap}")

        self.stats = RuntimeStatistics()

    def init_feature_builder(self, reid_dim: int, norm_stats: Any = None) -> None:
        """Initialize the feature builder once the ReID dimension is known."""
        if self._feature_builder_initialised:
            return
        from agentguard.features.builder import EventFeatureBuilder

        self.feature_builder = EventFeatureBuilder(reid_dim, norm_stats)
        self._feature_builder_initialised = True

    def is_mature_track(self, track) -> bool:
        """Return whether a track has the configured event context."""
        return (
            track.state in (1, 2)
            and len(track.history) >= self.iwg_context_size
        )

    def get_or_create_event_buffer(self, track_id: int) -> EventBuffer:
        if track_id not in self.event_buffers:
            self.event_buffers[track_id] = EventBuffer(
                max_len=self.iwg_context_size
            )
        return self.event_buffers[track_id]

    def _reset_rg_cma_history_for_frame(self, track_id: int, frame_id: int) -> bool:
        event_buffer = self.event_buffers.get(track_id)
        if event_buffer is None or not event_buffer.events:
            return False
        gap = int(frame_id) - int(event_buffer.events[-1].frame_id)
        if 0 < gap <= self.rg_cma_max_gap:
            return False
        event_buffer.clear()
        return True

    def run_iwg_rg_cma_inference(
        self,
        track_id: int,
        events_sequence: List[Optional[TrackEvent]],
        *,
        frame_id: int,
        has_detection: bool,
    ) -> Dict[str, np.ndarray]:
        """Run one RG-CMA request through the batch path."""
        return self.run_iwg_rg_cma_batch_inference(
            [track_id],
            [events_sequence],
            frame_ids=[frame_id],
            has_detection=[has_detection],
        )[0]

    def run_iwg_rg_cma_batch_inference(
        self,
        track_ids: List[int],
        event_sequences: List[List[Optional[TrackEvent]]],
        *,
        frame_ids: List[int],
        has_detection: List[bool],
    ) -> List[Dict[str, np.ndarray]]:
        """Run the combined model for a batch of current track events."""
        if self.mode != "iwg-rg-cma" or self.rg_cma_model is None:
            raise RuntimeError(
                "run_iwg_rg_cma_batch_inference requires iwg-rg-cma mode"
            )

        size = len(track_ids)
        if not (
            len(event_sequences) == len(frame_ids) == len(has_detection) == size
        ):
            raise ValueError("iwg-rg-cma batch inputs must have equal lengths")
        if size == 0:
            return []
        if len(set(track_ids)) != size:
            raise ValueError("iwg-rg-cma batch cannot contain duplicate track ids")
        if self.feature_builder is None:
            raise RuntimeError("iwg-rg-cma feature builder is not initialized")
        if any(len(sequence) != self.iwg_context_size for sequence in event_sequences):
            raise ValueError(
                "iwg-rg-cma online inference requires context sequences of length "
                f"{self.iwg_context_size}"
            )

        sequences = list(event_sequences)
        for index, (track_id, frame_id) in enumerate(zip(track_ids, frame_ids)):
            if self._reset_rg_cma_history_for_frame(track_id, frame_id):
                sequence = sequences[index]
                if not sequence or sequence[-1] is None:
                    raise ValueError(
                        "iwg-rg-cma sequence must end with the current event"
                    )
                sequences[index] = [None] * (len(sequence) - 1) + [sequence[-1]]

        inputs = self.feature_builder.build_iwg_batch_input(sequences)
        device = next(self.rg_cma_model.parameters()).device
        padding = inputs["mask"].to(device, non_blocking=True)
        detection = torch.zeros_like(padding)
        for batch_index, sequence in enumerate(sequences):
            for position, event in enumerate(sequence):
                detection[batch_index, position] = bool(
                    event is not None and event.has_detection
                )

        reset = torch.zeros_like(padding)
        for batch_index in range(size):
            valid_positions = (~padding[batch_index]).nonzero(as_tuple=False)
            if valid_positions.numel() == 0:
                raise ValueError("iwg-rg-cma sequence cannot contain only padding")
            reset[batch_index, int(valid_positions[0, 0])] = True

        with torch.inference_mode():
            outputs = self.rg_cma_model(
                inputs["track_feats"].to(device, non_blocking=True),
                inputs["det_feats"].to(device, non_blocking=True),
                inputs["scalar_feats"].to(device, non_blocking=True),
                padding,
                detection,
                reset,
            )

        base = outputs["base_gate"].float().cpu().numpy()
        model_final = outputs["refined_gate"].float().cpu().numpy()
        model_correction = outputs["gate_correction"].float().cpu().numpy()
        if self.rg_cma_alpha == 1.0:
            final = model_final
            correction = model_correction
        else:
            correction = self.rg_cma_alpha * model_correction
            final = np.clip(base + correction, 0.0, 1.0)
        policy = outputs["policy_probs"].float().cpu().numpy()
        cue = outputs["cue"].float().cpu().numpy()
        applied = (
            base.copy() if self.rg_cma_output == "base" else final.copy()
        )
        # Channel ablations affect only the gate consumed by TrackTrack. Keep
        # model diagnostics (base/final/correction) unchanged for comparison.
        if self.disable_kf_gate:
            applied[:, 0] = 1.0
        if self.disable_ema_gate:
            applied[:, 1] = 1.0

        policy_confidence = np.max(policy, axis=1)
        fallback_mask = (
            (self.fallback_threshold > 0.0)
            & (policy_confidence < self.fallback_threshold)
            & np.asarray(has_detection, dtype=bool)
        )
        applied[fallback_mask] = 1.0

        results: List[Dict[str, np.ndarray]] = []
        correction_bound = getattr(self.rg_cma_model, "correction_bound", None)
        for index in range(size):
            if not has_detection[index]:
                # Keep the unmatched protocol explicit at the Runtime boundary.
                base[index] = 0.0
                final[index] = 0.0
                applied[index] = 0.0
                correction[index] = 0.0
            result = {
                "gate": applied[index],
                "base_gate": base[index],
                "final_gate": final[index],
                "gate_correction": correction[index],
                "policy_probs": policy[index],
                "cue": cue[index],
            }
            self.stats.record_rg_cma(
                result["gate"],
                result["base_gate"],
                result["final_gate"],
                result["gate_correction"],
                matched=bool(has_detection[index]),
                correction_bound=correction_bound,
                policy_confidence=float(policy_confidence[index]),
                fallback_applied=bool(fallback_mask[index]),
            )
            results.append(result)
        return results

    def cleanup_track(self, track_id: int) -> None:
        """Remove all Runtime state associated with a track."""
        self.event_buffers.pop(track_id, None)
