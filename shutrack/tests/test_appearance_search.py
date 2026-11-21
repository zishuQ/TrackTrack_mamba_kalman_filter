"""Unit tests for the appearance positive/competitor window search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from sportsmot_supervision_pilot import find_appearance_pair  # noqa: E402


def entry(index: int, box, gt_id: int, iou: float) -> dict:
    return {
        "index": index,
        "box": np.asarray(box, dtype=np.float64),
        "score": 0.9,
        "gt_id": gt_id,
        "iou": iou,
    }


class FakePilot:
    def __init__(self, matches: dict[int, list[dict]], frame_count: int):
        self.matches = matches
        self.frame_count = frame_count

    def frame_matches(self, frame_index: int):
        return self.matches.get(frame_index + 1, [])


POSITIVE_BOX = [0.0, 0.0, 10.0, 10.0]
COMPETITOR_BOX = [1.0, 0.0, 11.0, 10.0]
FAR_BOX = [50.0, 50.0, 60.0, 60.0]


def test_walks_the_window_until_a_same_frame_pair_exists() -> None:
    pilot = FakePilot(
        {
            2: [entry(10, POSITIVE_BOX, 1, 0.9)],
            3: [entry(11, POSITIVE_BOX, 1, 0.9)],
            4: [
                entry(12, POSITIVE_BOX, 1, 0.9),
                entry(13, COMPETITOR_BOX, 2, 0.8),
            ],
        },
        frame_count=10,
    )
    found = find_appearance_pair(pilot, frame=1, target=1, current_index=99)
    assert found is not None
    positive, positive_frame, competitors = found
    assert positive["index"] == 12
    assert positive_frame == 4
    assert [item["index"] for item in competitors] == [13]


def test_never_uses_the_detection_being_written() -> None:
    pilot = FakePilot(
        {
            2: [
                entry(5, POSITIVE_BOX, 1, 0.9),
                entry(6, COMPETITOR_BOX, 2, 0.8),
            ]
        },
        frame_count=10,
    )
    assert find_appearance_pair(pilot, frame=1, target=1, current_index=5) is None


def test_competitor_must_overlap_the_positive() -> None:
    pilot = FakePilot(
        {2: [entry(10, POSITIVE_BOX, 1, 0.9), entry(11, FAR_BOX, 2, 0.8)]},
        frame_count=10,
    )
    assert find_appearance_pair(pilot, frame=1, target=1, current_index=99) is None


def test_no_pair_in_window_returns_none() -> None:
    pilot = FakePilot(
        {2: [entry(10, POSITIVE_BOX, 1, 0.9)], 3: [entry(11, POSITIVE_BOX, 1, 0.9)]},
        frame_count=10,
    )
    assert find_appearance_pair(pilot, frame=1, target=1, current_index=99) is None


def test_window_respects_frame_count() -> None:
    pilot = FakePilot(
        {2: [entry(10, POSITIVE_BOX, 1, 0.9), entry(11, COMPETITOR_BOX, 2, 0.8)]},
        frame_count=1,
    )
    assert find_appearance_pair(pilot, frame=1, target=1, current_index=99) is None
