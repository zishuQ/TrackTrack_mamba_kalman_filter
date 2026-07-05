from __future__ import annotations

from typing import List, Set

from agentguard.contracts.events import TrackEvent


def validate_splits(
    train_events: List[TrackEvent],
    val_events: List[TrackEvent],
) -> bool:
    """Verify that no video sequence appears in both train and val splits.

    Raises
    ------
    ValueError
        If cross-contamination (sequence leakage) is detected.

    Returns
    -------
    True
        When the validation passes (returns ``True`` so it can be used
        as an assertion).
    """
    train_sequences = _extract_sequences(train_events)
    val_sequences = _extract_sequences(val_events)
    return check_no_sequence_leakage(train_sequences, val_sequences)


def check_no_sequence_leakage(
    train_sequences: List[str],
    val_sequences: List[str],
) -> bool:
    """Assert that two lists of sequence names have no overlap.

    Parameters
    ----------
    train_sequences : list of str
        Sequence names in the training split.
    val_sequences : list of str
        Sequence names in the validation split.

    Returns
    -------
    bool
        ``True`` if the sets are disjoint.

    Raises
    ------
    ValueError
        If any sequence appears in both sets.
    """
    train_set: Set[str] = set(train_sequences)
    val_set: Set[str] = set(val_sequences)
    overlap = train_set & val_set
    if overlap:
        raise ValueError(
            f"Sequence leakage detected: {sorted(overlap)}. "
            f"No sequence may appear in both train and val splits."
        )
    return True


def _extract_sequences(events: List[TrackEvent]) -> List[str]:
    """Extract unique sequence names from a list of ``TrackEvent`` objects."""
    seen: Set[str] = set()
    sequences: List[str] = []
    for evt in events:
        if evt.sequence not in seen:
            seen.add(evt.sequence)
            sequences.append(evt.sequence)
    return sequences
