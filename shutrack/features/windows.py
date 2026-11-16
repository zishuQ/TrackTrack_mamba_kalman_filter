"""Five-past + current window construction used offline and online.

The window is always causal: it only ever reads events strictly before the
current event.  Events from a different track, sequence/segment or across a
reset boundary are never included; left padding is expressed through an
explicit mask and a learned null-history token in the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shutrack.data.evidence import EventEvidence
from shutrack.schema import (
    CONTEXT_DIM,
    HISTORY_SIZE,
    MOTION_DIM,
)


@dataclass(frozen=True)
class WindowArrays:
    history_motion: np.ndarray        # (HISTORY_SIZE, 12)
    history_track: np.ndarray         # (HISTORY_SIZE, D)
    history_detection: np.ndarray     # (HISTORY_SIZE, D)
    history_context: np.ndarray       # (HISTORY_SIZE, 8)
    history_mask: np.ndarray          # (HISTORY_SIZE,) True = ignored padding
    history_has_detection: np.ndarray  # (HISTORY_SIZE,)
    history_reset: np.ndarray         # (HISTORY_SIZE,) True = invalidated by reset
    current_motion: np.ndarray        # (12,)
    current_track: np.ndarray         # (D,)
    current_detection: np.ndarray     # (D,)
    current_context: np.ndarray       # (8,)
    current_has_detection: bool
    current_frame_id: int


def select_history(
    events: Sequence[EventEvidence],
    current: EventEvidence,
    *,
    max_history: int = HISTORY_SIZE,
) -> list[EventEvidence]:
    """Return up to ``max_history`` causal events from the current segment."""

    selected: list[EventEvidence] = []
    for event in reversed(list(events)):
        if event.track_id != current.track_id:
            break
        if event.segment_id != current.segment_id:
            break
        if event.frame_id >= current.frame_id:
            continue
        selected.append(event)
        if len(selected) >= max_history:
            break
    selected.reverse()
    return selected


def build_window(
    history: Sequence[EventEvidence],
    current: EventEvidence,
    *,
    reset: bool = False,
    reset_mask: np.ndarray | None = None,
) -> WindowArrays:
    """Build one fixed-length causal window.

    ``history`` is chronological.  When ``reset`` is true the history is
    intentionally dropped.  ``reset_mask`` may additionally mark individual
    history slots as invalid (a longer offline sequence may pass its reset
    bookkeeping); it must have shape ``(HISTORY_SIZE,)``.
    """

    selected = [] if reset else select_history(history, current)
    pad = HISTORY_SIZE - len(selected)
    reid_dim = int(current.track_feature.shape[0])

    history_motion = np.zeros((HISTORY_SIZE, MOTION_DIM), dtype=np.float32)
    history_track = np.zeros((HISTORY_SIZE, reid_dim), dtype=np.float32)
    history_detection = np.zeros((HISTORY_SIZE, reid_dim), dtype=np.float32)
    history_context = np.zeros((HISTORY_SIZE, CONTEXT_DIM), dtype=np.float32)
    history_mask = np.ones((HISTORY_SIZE,), dtype=np.bool_)
    history_has_detection = np.zeros((HISTORY_SIZE,), dtype=np.bool_)
    history_reset = np.zeros((HISTORY_SIZE,), dtype=np.bool_)

    for position, event in enumerate(selected, start=pad):
        if event.reid_dim != reid_dim:
            raise ValueError(
                "history and current appearance dimensions differ: "
                f"{event.reid_dim} != {reid_dim}"
            )
        history_motion[position] = event.motion
        history_track[position] = event.track_feature
        history_detection[position] = event.detection_feature
        history_context[position] = event.context
        history_mask[position] = False
        history_has_detection[position] = event.has_detection

    if reset_mask is not None:
        provided = np.asarray(reset_mask, dtype=np.bool_).reshape(-1)
        if provided.shape != (HISTORY_SIZE,):
            raise ValueError(
                f"reset_mask must have shape ({HISTORY_SIZE},), got {provided.shape}"
            )
        history_reset |= provided
        history_mask |= history_reset

    return WindowArrays(
        history_motion=history_motion,
        history_track=history_track,
        history_detection=history_detection,
        history_context=history_context,
        history_mask=history_mask,
        history_has_detection=history_has_detection,
        history_reset=history_reset,
        current_motion=np.asarray(current.motion, dtype=np.float32),
        current_track=np.asarray(current.track_feature, dtype=np.float32),
        current_detection=np.asarray(current.detection_feature, dtype=np.float32),
        current_context=np.asarray(current.context, dtype=np.float32),
        current_has_detection=bool(current.has_detection),
        current_frame_id=int(current.frame_id),
    )


def window_from_sequence(
    events: Sequence[EventEvidence],
    current_index: int,
    *,
    reset_mask: np.ndarray | None = None,
) -> WindowArrays:
    """Offline helper for long sequences: window ending at ``current_index``."""

    if current_index < 0 or current_index >= len(events):
        raise IndexError(
            f"current_index={current_index} outside [0, {len(events)})"
        )
    current = events[current_index]
    history = select_history(events[:current_index], current)
    return build_window(history, current, reset_mask=reset_mask)


def stack_windows(windows: Sequence[WindowArrays]) -> dict[str, np.ndarray]:
    """Stack per-track windows into batched numpy arrays (CPU)."""

    if not windows:
        raise ValueError("cannot stack an empty window list")
    return {
        "history_motion": np.stack([w.history_motion for w in windows], axis=0),
        "history_track": np.stack([w.history_track for w in windows], axis=0),
        "history_detection": np.stack(
            [w.history_detection for w in windows], axis=0
        ),
        "history_context": np.stack([w.history_context for w in windows], axis=0),
        "history_mask": np.stack([w.history_mask for w in windows], axis=0),
        "history_has_detection": np.stack(
            [w.history_has_detection for w in windows], axis=0
        ),
        "history_reset": np.stack([w.history_reset for w in windows], axis=0),
        "current_motion": np.stack([w.current_motion for w in windows], axis=0),
        "current_track": np.stack([w.current_track for w in windows], axis=0),
        "current_detection": np.stack(
            [w.current_detection for w in windows], axis=0
        ),
        "current_context": np.stack([w.current_context for w in windows], axis=0),
        "current_has_detection": np.asarray(
            [w.current_has_detection for w in windows], dtype=np.bool_
        ),
    }


__all__ = [
    "WindowArrays",
    "select_history",
    "build_window",
    "window_from_sequence",
    "stack_windows",
]
