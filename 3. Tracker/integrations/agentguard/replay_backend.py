"""Replay backend that applies TGR replay plans to live Track objects.

This is the only module that knows both TrackTrack internals (Track) and the
AgentGuard replay protocol (ReplayPlan, TrackStateSnapshot).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import TrackStateSnapshot
from agentguard.runtime.replay import ReplayEngine, ReplayPlan


class TrackTrackReplayBackend:
    """Applies a TGR ``ReplayPlan`` onto a live ``Track`` object.

    The strategy is:
      1. Restore the live Track to the plan's checkpoint snapshot.
      2. For each step in the plan, build a minimal ``TrackEvent``
         and feed it to ``ReplayEngine.replay_event`` to compute the
         updated state.
      3. Restore the live Track to the final state snapshot.
    """

    def __init__(self) -> None:
        self._engine = ReplayEngine()

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
            The live ``Track`` object to update.  Must have ``snapshot_state``
            and ``restore_state`` methods.
        plan : ReplayPlan
            The replay plan produced by the runtime's ``finalize_first_stage``.
        """
        current: TrackStateSnapshot = plan.checkpoint

        for step in plan.steps:
            # Build a minimal TrackEvent for the replay engine
            event = TrackEvent(
                event_id=f"replay/{plan.track_id}/{step.frame_id}",
                dataset="",
                sequence="",
                frame_id=step.frame_id,
                track_id=plan.track_id,
                image_width=0,
                image_height=0,
                has_detection=step.has_detection,
                detection=step.detection,
                warp_matrix=step.warp_matrix.copy(),
            )
            gate = np.array(
                [step.motion_gate, step.appearance_gate], dtype=np.float64
            )
            current = self._engine.replay_event(current, event, gate)

        # Restore the computed final state onto the live Track
        live_track.restore_state(current)
