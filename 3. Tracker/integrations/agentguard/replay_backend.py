"""Replay backend that applies TGR replay plans directly to live Track objects.

This module mutates real ``Track`` instances in-place using native TrackTrack
operations (CMC warp, KF predict, ``update_with_gates``, ``mark_lost``)
without depending on ``agentguard.runtime.replay.ReplayEngine``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agentguard.runtime.replay import ReplayPlan
from trackers.cmc import apply_cmc
from integrations.agentguard.converters import (
    detection_to_update_input,
    restore_track_state,
)
from agentguard.contracts.states import DetectionObservation


def _make_track_like(detection: DetectionObservation) -> Any:
    """Build a lightweight detection object compatible with
    ``Track.update_with_gates`` from a ``DetectionObservation``.

    Uses ``detection_to_update_input`` for box/score/feat and adds the
    ``x1y1x2y2`` and ``cxcywh`` properties used by the KF update in
    ``trackers/track.py``.
    """
    update_input = detection_to_update_input(detection)

    class _DetectionProxy:
        def __init__(self, ui):
            self.box = ui["box"]
            self.score = ui["score"]
            self.feat = ui["feat"]

        @property
        def x1y1x2y2(self) -> np.ndarray:
            return self.box.copy()

        @property
        def cxcywh(self) -> np.ndarray:
            x1, y1, x2, y2 = self.box
            return np.array(
                [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                dtype=np.float64,
            )

    return _DetectionProxy(update_input)


class TrackTrackReplayBackend:
    """Applies a TGR ``ReplayPlan`` onto a live ``Track`` object.

    The strategy:
      1. Restore the live Track to the plan's checkpoint snapshot using
         ``restore_track_state``.
      2. For each step in the plan:
         a. Apply the step's warp matrix via ``apply_cmc``.
         b. Call ``live_track.predict()``.
         c. If matched: ``live_track.update_with_gates(frame_id, detection,
            motion_gate, appearance_gate)``.
         d. If unmatched: ``live_track.mark_lost()``.
      3. The live Track ends at the correct post-replay state (no restore
         needed — the mutations are applied in-place).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay_into_live_track(
        self,
        live_track: Any,
        plan: ReplayPlan,
    ) -> None:
        """Replay *plan* onto *live_track*, mutating its state in-place.

        Parameters
        ----------
        live_track : Track
            The live ``Track`` object to update.  Must support
            ``restore_state`` (or ``restore_track_state``), ``predict()``,
            ``update_with_gates()``, and ``mark_lost()``.
        plan : ReplayPlan
            The replay plan produced by the runtime's ``finalize_first_stage``.
        """
        # Step 1: Restore the live track to the checkpoint snapshot.
        if hasattr(live_track, "restore_state"):
            live_track.restore_state(plan.checkpoint)
        else:
            restore_track_state(live_track, plan.checkpoint)

        # Step 2: Replay each step in sequence.
        for step in plan.steps:
            # Apply step-level CMC warp.
            wp = step.warp_matrix
            if wp is not None and not np.allclose(wp, np.eye(2, 3, dtype=np.float64)):
                apply_cmc([live_track], wp)

            # Kalman predict.
            live_track.predict()

            if step.has_detection and step.detection is not None:
                det = _make_track_like(step.detection)
                live_track.update_with_gates(
                    step.frame_id,
                    det,
                    step.motion_gate,
                    step.appearance_gate,
                )
            else:
                old_end_frame = live_track.end_frame_id
                live_track.mark_lost()
                assert live_track.end_frame_id == old_end_frame, (
                    f"end_frame_id changed from {old_end_frame} to "
                    f"{live_track.end_frame_id} during unmatched replay"
                )
