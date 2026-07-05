from __future__ import annotations

from typing import List, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent


class EventBuffer:
    """
    Per-track event buffer for IWG.
    Maintains up to 6 most recent events (current + 5 historical).
    """

    def __init__(self, max_len: int = 6):
        self.max_len = max_len
        self.events: List[TrackEvent] = []

    def push(self, event: TrackEvent) -> None:
        """Append an event and trim to max_len."""
        self.events.append(event)
        if len(self.events) > self.max_len:
            self.events.pop(0)

    def get_sequence(self) -> List[Optional[TrackEvent]]:
        """Return events as a fixed-length list, left-padded with None.

        The last element is always the most recent event.
        Padding appears on the left when fewer than ``max_len`` events
        have been accumulated.
        """
        events = self.events[-self.max_len :]
        padding = [None] * (self.max_len - len(events))
        return padding + events

    def clear(self) -> None:
        """Remove all events."""
        self.events = []


class WindowBuffer:
    """
    Per-track window buffer for TGR.
    Maintains exactly 4 events for temporal revision.

    When full:
      1. Save checkpoint before oldest event
      2. After TGR inference, replay from checkpoint with revised gates
      3. Delete oldest event
    """

    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.events: List[TrackEvent] = []

    @property
    def is_full(self) -> bool:
        """Check whether the window has reached its capacity."""
        return len(self.events) >= self.window_size

    def push(self, event: TrackEvent) -> None:
        """Append an event to the window (does not trim automatically)."""
        self.events.append(event)

    def get_window(self) -> List[TrackEvent]:
        """Return the most recent events up to ``window_size``."""
        return self.events[-self.window_size :]

    def pop_oldest(self) -> Optional[TrackEvent]:
        """Remove and return the oldest event, or None if empty."""
        if self.events:
            return self.events.pop(0)
        return None

    def clear(self) -> None:
        """Remove all events."""
        self.events = []
