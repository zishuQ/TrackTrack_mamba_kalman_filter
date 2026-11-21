"""Regression tests for the TrackEval-compatible CLEAR matching helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from clear_id_analysis import assignments  # noqa: E402


def test_empty_prediction_frame_keeps_previous_match_continuity() -> None:
    # Three frames, two people.  Frame 1 matches correctly, frame 2 has no
    # tracker boxes at all, frame 3 has crossed positions where both pairings
    # still satisfy IoU >= 0.5.  TrackEval's CLEAR keeps the previous-frame
    # matching through the empty frame; a fresh Hungarian assignment would swap
    # both identities and count two spurious switches.
    gt_frames = {
        1: {
            1: np.array([0.0, 0.0, 10.0, 10.0]),
            2: np.array([2.0, 0.0, 12.0, 10.0]),
        },
        2: {
            1: np.array([0.0, 0.0, 10.0, 10.0]),
            2: np.array([2.0, 0.0, 12.0, 10.0]),
        },
        3: {
            1: np.array([2.0, 0.0, 12.0, 10.0]),
            2: np.array([0.0, 0.0, 10.0, 10.0]),
        },
    }
    track_frames = {
        1: {
            10: np.array([0.0, 0.0, 10.0, 10.0]),
            20: np.array([2.0, 0.0, 12.0, 10.0]),
        },
        # frame 2 intentionally has no predictions
        3: {
            10: np.array([0.0, 0.0, 10.0, 10.0]),
            20: np.array([2.0, 0.0, 12.0, 10.0]),
        },
    }
    result = assignments(gt_frames, track_frames)
    assert result[1] == {1: 10, 2: 20}
    assert result[2] == {}
    assert result[3] == {1: 10, 2: 20}


def test_matching_performed_frame_updates_continuity() -> None:
    # A normal frame with both sides must refresh the continuity map: a stale
    # map would keep the old track instead of the frame-2 match.
    gt_frames = {
        1: {1: np.array([0.0, 0.0, 10.0, 10.0])},
        2: {1: np.array([20.0, 0.0, 30.0, 10.0])},
        3: {1: np.array([2.0, 0.0, 12.0, 10.0])},
    }
    track_frames = {
        1: {5: np.array([0.0, 0.0, 10.0, 10.0]), 6: np.array([20.0, 0.0, 30.0, 10.0])},
        2: {5: np.array([0.0, 0.0, 10.0, 10.0]), 6: np.array([20.0, 0.0, 30.0, 10.0])},
        3: {5: np.array([0.0, 0.0, 10.0, 10.0]), 6: np.array([2.0, 0.0, 12.0, 10.0])},
    }
    result = assignments(gt_frames, track_frames)
    assert result[1] == {1: 5}
    assert result[2] == {1: 6}
    assert result[3] == {1: 6}
