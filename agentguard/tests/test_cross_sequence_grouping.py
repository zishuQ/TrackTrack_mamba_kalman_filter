"""Test SimilarEventGrouper excludes current sequence and enforces diversity.

The grouper must:
- Exclude events from the query event's own sequence.
- Raise ValueError if fewer than 12 events are found.
- Raise ValueError if events come from fewer than 3 different sequences.
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.verifier.event_grouping import SimilarEventGrouper


def _make_event(seq: str, event_id: str) -> TrackEvent:
    """Helper to create a minimal TrackEvent with given sequence."""
    return TrackEvent(
        event_id=event_id,
        dataset="test",
        sequence=seq,
        frame_id=0,
        track_id=0,
        image_width=640,
        image_height=480,
        has_detection=False,
    )


class TestSimilarEventGrouper:
    """Tests for SimilarEventGrouper."""

    @pytest.fixture
    def grouper(self) -> SimilarEventGrouper:
        return SimilarEventGrouper(k=32)

    def test_excludes_current_sequence(self, grouper):
        """Events from the query's own sequence should be excluded."""
        query_desc = np.ones(16, dtype=np.float64)
        query_seq = "seq_A"

        # Create events from seq_A (will be excluded) and 3+ other sequences
        all_events: List[TrackEvent] = []
        all_descs: List[np.ndarray] = []
        for i in range(5):
            all_events.append(_make_event("seq_A", f"A_{i}"))
            all_descs.append(np.ones(16, dtype=np.float64) * 0.5)
        for seq in ["seq_B", "seq_C", "seq_D", "seq_E"]:
            for i in range(5):
                all_events.append(_make_event(seq, f"{seq}_{i}"))
                all_descs.append(np.ones(16, dtype=np.float64) * 0.9)

        similar = grouper.find_similar(
            query_desc, all_events, all_descs, query_sequence=query_seq
        )

        # No events from seq_A should be present
        for evt, _, _ in similar:
            assert evt.sequence != "seq_A"
        # At least 3 different sequences
        seqs = {e.sequence for e, _, _ in similar}
        assert len(seqs) >= 3

    def test_minimum_12_events(self, grouper):
        """Should raise ValueError if fewer than 12 events found (even across 3+ seqs)."""
        query_desc = np.ones(16, dtype=np.float64)

        all_events = []
        all_descs = []
        for seq in ["seq_B", "seq_C", "seq_D"]:
            for i in range(4):
                all_events.append(_make_event(seq, f"{seq}_{i}"))
                all_descs.append(np.ones(16, dtype=np.float64))

        # Only 12 events total, but grouper requires at least 12 AND >= 3 seqs
        # Actually 12 is exactly the minimum, so this should work
        # Let's use fewer to trigger the error
        all_events = [_make_event("seq_B", f"B_{i}") for i in range(5)]
        all_descs = [np.ones(16, dtype=np.float64) for _ in range(5)]

        with pytest.raises(ValueError, match="Only 5 valid similar events"):
            grouper.find_similar(query_desc, all_events, all_descs, query_sequence="seq_A")

    def test_minimum_3_sequences(self, grouper):
        """Should raise ValueError if events come from < 3 sequences."""
        query_desc = np.ones(16, dtype=np.float64)

        all_events = []
        all_descs = []
        for i in range(8):
            all_events.append(_make_event("seq_B", f"B_{i}"))
            all_descs.append(np.ones(16, dtype=np.float64))
        for i in range(8):
            all_events.append(_make_event("seq_C", f"C_{i}"))
            all_descs.append(np.ones(16, dtype=np.float64))

        with pytest.raises(ValueError, match="only 2 different sequences"):
            grouper.find_similar(query_desc, all_events, all_descs, query_sequence="seq_A")

    def test_diversity_enforced_3_sequences(self, grouper):
        """3 different sequences should be sufficient."""
        query_desc = np.zeros(16, dtype=np.float64)

        all_events = []
        all_descs = []
        for seq in ["seq_B", "seq_C", "seq_D"]:
            for i in range(5):
                all_events.append(_make_event(seq, f"{seq}_{i}"))
                all_descs.append(np.random.RandomState(hash(seq + str(i)) % 2**31).randn(16).astype(np.float64))

        similar = grouper.find_similar(query_desc, all_events, all_descs, query_sequence="seq_A")
        seqs = {e.sequence for e, _, _ in similar}
        assert len(seqs) >= 3

    def test_returns_top_k_similar(self, grouper):
        """Should return up to k most similar events sorted by similarity."""
        k = 12
        grouper = SimilarEventGrouper(k=k)

        query_desc = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Create exactly 12 events across 3 sequences, all equally similar.
        # Top-12 = all 12 events from all 3 sequences.
        all_events = []
        all_descs = []
        for seq, val in [("seq_B", 1.0), ("seq_C", 0.99), ("seq_D", 0.98)]:
            for i in range(4):
                all_events.append(_make_event(seq, f"{seq}_{i}"))
                all_descs.append(np.array([val, 1.0 - val, 0.0], dtype=np.float64))

        similar = grouper.find_similar(query_desc, all_events, all_descs, query_sequence="seq_A")
        assert len(similar) == k

        # All 3 sequences should be represented
        seqs = {e.sequence for e, _, _ in similar}
        assert len(seqs) == 3

        # First 4 should be from seq_B (highest similarity)
        for evt, _, _ in similar[:4]:
            assert evt.sequence == "seq_B"

        # Similarities should be non-increasing
        sims = [s for _, _, s in similar]
        for i in range(len(sims) - 1):
            assert sims[i] >= sims[i + 1] - 1e-10

    def test_length_mismatch_raises(self, grouper):
        """Mismatched events and descriptors lengths should raise ValueError."""
        query_desc = np.ones(16)
        all_events = [_make_event("seq_B", "b0"), _make_event("seq_B", "b1")]
        all_descs = [np.ones(16)]  # only 1 descriptor but 2 events

        with pytest.raises(ValueError, match="Length mismatch"):
            grouper.find_similar(query_desc, all_events, all_descs, query_sequence="seq_A")

    def test_exact_exclusion_of_query_sequence(self, grouper):
        """When query_sequence is None, events from all sequences are valid."""
        # Need >= 3 sequences and >= 12 events total
        query_desc = np.ones(16, dtype=np.float64)
        all_events = []
        all_descs = []
        for seq in ["seq_A", "seq_B", "seq_C"]:
            for i in range(5):
                all_events.append(_make_event(seq, f"{seq}_{i}"))
                all_descs.append(np.ones(16, dtype=np.float64))

        # Without query_sequence, all events from all sequences are valid
        similar = grouper.find_similar(query_desc, all_events, all_descs, query_sequence=None)
        assert len(similar) >= 12

    def test_zero_norm_descriptor(self, grouper):
        """Zero-norm descriptor should result in 0 similarity for all."""
        query_desc = np.zeros(16, dtype=np.float64)
        all_events = []
        all_descs = []
        for seq in ["seq_B", "seq_C", "seq_D"]:
            for i in range(4):
                all_events.append(_make_event(seq, f"{seq}_{i}"))
                all_descs.append(np.ones(16, dtype=np.float64))

        similar = grouper.find_similar(query_desc, all_events, all_descs, query_sequence="seq_A")
        for _, _, sim in similar:
            assert sim == 0.0
