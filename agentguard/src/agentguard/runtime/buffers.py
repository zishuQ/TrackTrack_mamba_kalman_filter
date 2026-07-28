from __future__ import annotations

from typing import List, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent


class EventBuffer:
    """
    Per-track event buffer for IWG.
    Maintains a fixed number of most recent events (current plus history).
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
