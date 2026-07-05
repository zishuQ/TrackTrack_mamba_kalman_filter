from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent


class SimilarEventGrouper:
    """Finds similar events from other sequences for cross-validation.

    For a given event, retrieves the *k* most similar events from the
    pool of all labelled events, subject to:

    - At least 12 valid events must be found.
    - The found events must come from at least 3 different sequences.
    - Events from the current event's sequence are **excluded**.

    Parameters
    ----------
    k : int
        Number of similar events to retrieve (default 32).
    """

    def __init__(self, k: int = 32) -> None:
        self.k = k

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def find_similar(
        self,
        event_descriptor: np.ndarray,
        all_events: List[TrackEvent],
        all_descriptors: List[np.ndarray],
        query_sequence: Optional[str] = None,
    ) -> List[Tuple[TrackEvent, np.ndarray, float]]:
        """Find similar events from other sequences for cross-validation.

        Parameters
        ----------
        event_descriptor : ndarray, shape ``(D,)``
            Descriptor of the query event.
        all_events : list of TrackEvent
            All candidate events (must match *all_descriptors*).
        all_descriptors : list of ndarray
            Descriptors for all candidate events.  Must be the same length
            as *all_events*.
        query_sequence : str or None, optional
            Sequence name of the query event.  If provided, events from
            the same sequence are excluded.

        Returns
        -------
        list of (TrackEvent, ndarray, float)
            Up to *k* similar events, each with its descriptor and cosine
            similarity score.  Events from the same sequence as the query
            are excluded.

        Raises
        ------
        ValueError
            If fewer than 12 valid events are found or if the events come
            from fewer than 3 different sequences.
        """
        if len(all_events) != len(all_descriptors):
            raise ValueError(
                f"Length mismatch: {len(all_events)} events vs "
                f"{len(all_descriptors)} descriptors"
            )

        # Compute similarities and filter
        scored: List[Tuple[TrackEvent, np.ndarray, float]] = []

        for event, desc in zip(all_events, all_descriptors):
            # Exclude query event's own sequence
            if query_sequence is not None and event.sequence == query_sequence:
                continue

            similarity = self._cosine_similarity(event_descriptor, desc)
            scored.append((event, desc, similarity))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[2], reverse=True)

        # Take top-k
        similar = scored[: self.k]

        # Validate constraints
        if len(similar) < 12:
            raise ValueError(
                f"Only {len(similar)} valid similar events found "
                f"(minimum 12 required)."
            )

        sequences = set(e.sequence for e, _, _ in similar)
        if len(sequences) < 3:
            raise ValueError(
                f"Similar events come from only {len(sequences)} different "
                f"sequences (minimum 3 required)."
            )

        return similar

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)
