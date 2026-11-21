"""Regression tests for GT-continuity in anchor identity recovery."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from paired_identity import recover_identity_labels  # noqa: E402

GT1 = np.array([0.0, 0.0, 10.0, 10.0])
GT2 = np.array([8.0, 0.0, 18.0, 10.0])


class FakePilot:
    def __init__(self, events, gt):
        self.events = events
        self.gt = gt
        self.recomputed = []

    def _compute_motion_endpoints(self, row):
        self.recomputed.append((row["frame_t"], row["track_id"]))


def make_event(frame, track, box, target):
    return {
        "frame_t": frame,
        "track_id": track,
        "detection_box": np.asarray(box, dtype=np.float64).tolist(),
        "target_gt_before": target,
    }


def build_scenario(gap_frame=18, resume_frame=19, last_before_gap=17):
    events = []
    for frame in range(1, last_before_gap + 1):
        if frame <= 3:
            events.append(make_event(frame, 1, GT1, 1))
            events.append(make_event(frame, 2, GT2, 2))
        else:
            # Mutual swap: detections at the other person's box, unresolved vote.
            events.append(make_event(frame, 1, GT2, None))
            events.append(make_event(frame, 2, GT1, None))
    events.append(make_event(resume_frame, 1, GT2, None))
    events.append(make_event(resume_frame, 2, GT1, None))
    gt = {}
    for frame in range(1, last_before_gap + 1):
        gt[frame] = {1: GT1, 2: GT2}
    gt[resume_frame] = {1: GT1, 2: GT2}
    return FakePilot(events, gt)


def test_anchor_not_extended_across_gt_gap() -> None:
    pilot = build_scenario()
    recovered = recover_identity_labels(pilot)
    keys = {(frame, track, gt_id) for frame, track, gt_id in recovered}
    assert (17, 1, 1) in keys and (17, 2, 2) in keys
    assert not any(frame == 19 for frame, _, _ in keys)
    rows_19 = [event for event in pilot.events if event["frame_t"] == 19]
    assert all(event["target_gt_before"] is None for event in rows_19)
    assert not any(event.get("anchor_identity") for event in rows_19)


def test_anchor_still_extends_without_gap() -> None:
    pilot = build_scenario()
    pilot.gt[18] = {1: GT1, 2: GT2}  # continuous GT: the anchor may extend
    recovered = recover_identity_labels(pilot)
    keys = {(frame, track) for frame, track, _ in recovered}
    assert (19, 1) in keys and (19, 2) in keys
    rows_19 = [event for event in pilot.events if event["frame_t"] == 19]
    assert all(event["target_gt_before"] is not None for event in rows_19)


def crossing_scenario(frames):
    events = []
    gt = {}
    for frame, kind in frames.items():
        gt[frame] = {1: GT1, 2: GT2}
        if kind == "correct":
            events.append(make_event(frame, 1, GT1, 1))
            events.append(make_event(frame, 2, GT2, 2))
        elif kind == "cross":
            events.append(make_event(frame, 1, GT2, None))
            events.append(make_event(frame, 2, GT1, None))
    return FakePilot(events, gt)


def test_gap_before_crossing_does_not_reactivate_established_identity() -> None:
    # Frames 1-3 correct, frame 4 has no GT and no write events, frame 5
    # crosses with unknown identity.  The old evidence must not survive.
    pilot = crossing_scenario({1: "correct", 2: "correct", 3: "correct", 5: "cross"})
    recovered = recover_identity_labels(pilot)
    assert [(frame, track) for frame, track, _ in recovered] == []
    rows_5 = [event for event in pilot.events if event["frame_t"] == 5]
    assert all(event["target_gt_before"] is None for event in rows_5)


def test_active_pair_gap_does_not_reform_from_stale_established_evidence() -> None:
    # Frames 1-3 correct, frame 4 crossing starts an anchor, frame 5 has no GT,
    # frames 6-7 keep crossing.  Frame 6 must stay disabled and frame 7 must
    # not relabel from the stale pre-gap establishment.
    pilot = crossing_scenario(
        {1: "correct", 2: "correct", 3: "correct", 4: "cross", 6: "cross", 7: "cross"}
    )
    recovered = recover_identity_labels(pilot)
    frames = sorted({frame for frame, _, _ in recovered})
    assert frames == [4]
    for frame in (6, 7):
        rows = [event for event in pilot.events if event["frame_t"] == frame]
        assert all(event["target_gt_before"] is None for event in rows)


def test_reaccumulated_evidence_can_establish_an_anchor_again() -> None:
    # After the gap, frames 5-7 are consistent again (three matches rebuild the
    # establishment); the crossing at frame 8 may then be anchored.
    pilot = crossing_scenario(
        {
            1: "correct",
            2: "correct",
            3: "correct",
            5: "correct",
            6: "correct",
            7: "correct",
            8: "cross",
        }
    )
    recovered = recover_identity_labels(pilot)
    keys = {(frame, track, gt_id) for frame, track, gt_id in recovered}
    assert (8, 1, 1) in keys and (8, 2, 2) in keys
    assert not any(frame in (5, 6, 7) for frame, _, _ in keys)
