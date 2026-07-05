from __future__ import annotations

from typing import Any, Dict, Optional

from agentguard.contracts.states import TrackStateSnapshot


class CheckpointManager:
    """
    Manages track state checkpoints for TGR window replay.

    Lifecycle
    ---------
    1. On each event, save a snapshot of pre-event track state.
    2. When window is full, the oldest event's pre-recorded state is the
       checkpoint.
    3. After TGR inference: restore checkpoint, replay window events with
       revised gates.
    4. Save new checkpoint (state after replaying oldest event).
    5. Delete oldest event from window.
    6. Continue with current live state unchanged.
    """

    def __init__(self) -> None:
        self.checkpoints: Dict[int, Dict[str, TrackStateSnapshot]] = {}

    def save_checkpoint(
        self,
        track_id: int,
        event_id: str,
        snapshot: TrackStateSnapshot,
    ) -> None:
        """Save a track state snapshot before processing *event_id*."""
        if track_id not in self.checkpoints:
            self.checkpoints[track_id] = {}
        self.checkpoints[track_id][event_id] = snapshot

    def get_checkpoint(
        self,
        track_id: int,
        event_id: str,
    ) -> Optional[TrackStateSnapshot]:
        """Retrieve the checkpoint for *event_id*, or None."""
        track_cps = self.checkpoints.get(track_id)
        if track_cps is None:
            return None
        return track_cps.get(event_id)

    def remove_checkpoint(self, track_id: int, event_id: str) -> None:
        """Remove the checkpoint for *event_id* (no-op if missing)."""
        track_cps = self.checkpoints.get(track_id)
        if track_cps is not None:
            track_cps.pop(event_id, None)

    def cleanup_track(self, track_id: int) -> None:
        """Remove all checkpoints associated with *track_id*."""
        self.checkpoints.pop(track_id, None)
