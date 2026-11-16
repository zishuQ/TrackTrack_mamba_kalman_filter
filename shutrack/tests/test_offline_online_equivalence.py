"""Offline window vs per-event online equivalence and causality tests."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from shutrack.features.windows import build_window, stack_windows, window_from_sequence
from shutrack.tests.conftest import make_evidence, make_model


def _sequence() -> list:
    events = []
    for frame_id in range(1, 7):
        events.append(
            make_evidence(
                track_id=1,
                segment_id=0,
                frame_id=frame_id,
                observation_count=frame_id,
                seed=frame_id,
                has_detection=frame_id not in {3},
            )
        )
    for offset, frame_id in enumerate(range(7, 13), start=1):
        events.append(
            make_evidence(
                track_id=1,
                segment_id=1,
                frame_id=frame_id,
                observation_count=offset,
                seed=100 + frame_id,
                has_detection=frame_id not in {8},
            )
        )
    return events


def _tensor_batch(windows) -> dict[str, torch.Tensor]:
    batch = stack_windows(windows)
    return {key: torch.from_numpy(value) for key, value in batch.items()}


def test_offline_windows_match_per_event_online_evaluation():
    events = _sequence()
    model = make_model()
    model.eval()

    matched_indices = [index for index, event in enumerate(events) if event.has_detection]
    offline_windows = [window_from_sequence(events, index) for index in matched_indices]
    with torch.no_grad():
        offline_gates = model(**_tensor_batch(offline_windows))["gates"].numpy()

    buffer: deque = deque(maxlen=5)
    online_gates = []
    previous_segment = None
    previous_frame = None
    for event in events:
        reset = (
            previous_segment is not None
            and (
                event.segment_id != previous_segment
                or event.frame_id - previous_frame > 30
                or event.frame_id <= previous_frame
            )
        )
        if reset:
            buffer.clear()
        if event.has_detection:
            window = build_window(list(buffer), event)
            with torch.no_grad():
                gate = model(**_tensor_batch([window]))["gates"].numpy()[0]
            online_gates.append(gate)
        buffer.append(event)
        previous_segment = event.segment_id
        previous_frame = event.frame_id

    np.testing.assert_allclose(offline_gates, np.asarray(online_gates), atol=1e-5, rtol=1e-5)


def test_future_events_do_not_change_current_window_or_gate():
    events = _sequence()
    current_index = 8
    window_before = window_from_sequence(events, current_index)

    perturbed = list(events)
    for index in range(current_index + 1, len(perturbed)):
        perturbed[index] = make_evidence(
            track_id=1,
            segment_id=events[index].segment_id,
            frame_id=events[index].frame_id,
            observation_count=events[index].observation_count,
            seed=900 + index,
            motion_scale=25.0,
            context_scale=25.0,
        )
    window_after = window_from_sequence(perturbed, current_index)

    for name in (
        "history_motion",
        "history_track",
        "history_detection",
        "history_context",
    ):
        np.testing.assert_allclose(
            getattr(window_before, name), getattr(window_after, name), atol=1e-6
        )

    model = make_model()
    model.eval()
    with torch.no_grad():
        before = model(**_tensor_batch([window_before]))["gates"].numpy()
        after = model(**_tensor_batch([window_after]))["gates"].numpy()
    np.testing.assert_allclose(before, after, atol=1e-6, rtol=1e-6)


def test_windows_never_cross_tracks_or_sequences():
    current = make_evidence(track_id=7, segment_id=2, frame_id=20, seed=20)
    same_track_older_segment = make_evidence(
        track_id=7, segment_id=1, frame_id=19, seed=19
    )
    same_segment_other_track = make_evidence(
        track_id=8, segment_id=2, frame_id=19, seed=18
    )
    same_track_same_segment = make_evidence(
        track_id=7, segment_id=2, frame_id=18, seed=17
    )
    window = build_window(
        [same_track_older_segment, same_segment_other_track, same_track_same_segment],
        current,
    )
    assert window.history_mask.tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(
        window.history_motion[-1], same_track_same_segment.motion
    )
