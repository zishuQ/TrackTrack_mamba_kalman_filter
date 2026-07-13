from __future__ import annotations

import json

import numpy as np
import pytest

from agentguard.datasets.joint_window_dataset import (
    HOLD_BOTH_POLICY,
    JOINT_DATASET_SCHEMA_VERSION,
    MOT17_FRCNN_TRAIN_SEQUENCES,
    MOT17_FRCNN_VAL_SEQUENCES,
    UNMATCHED_CUE,
    UNMATCHED_RISK,
    CompactIWGTSRMWindowDataset,
    build_window_index,
    event_key,
    resolve_joint_sequences,
    segment_track_timelines,
)


def _record(
    sequence: str,
    track_id: int,
    frame_id: int,
    offset: int,
    *,
    matched: bool = True,
    history_count: int | None = None,
) -> dict:
    return {
        "sequence": sequence,
        "track_id": track_id,
        "frame_id": frame_id,
        "event_shard_id": 0,
        "event_offset": offset,
        "event_id": f"{sequence}-{track_id}-{frame_id}",
        "matched": matched,
        "history_count": frame_id if history_count is None else history_count,
    }


def test_full_timeline_segmentation_retains_unmatched_and_never_crosses_tracks():
    records = [
        _record("seq-a", 1, 1, 0),
        _record("seq-a", 1, 2, 1, matched=False),
        _record("seq-a", 1, 3, 2),
        _record("seq-a", 1, 4, 3, history_count=0),
        _record("seq-a", 2, 1, 4),
        _record("seq-b", 1, 1, 5),
        _record("seq-b", 1, 50, 6),
    ]
    segments = segment_track_timelines(records, max_frame_gap=10)

    assert sum(len(segment) for segment in segments) == len(records)
    assert any(not event["matched"] for segment in segments for event in segment)
    for segment in segments:
        assert len({event["sequence"] for event in segment}) == 1
        assert len({event["track_id"] for event in segment}) == 1
    assert [event["frame_id"] for event in segments[0]] == [1, 2, 3]


def test_window_index_is_left_padded_fixed_context_and_segment_local():
    segment = [_record("seq-a", 7, frame, frame) for frame in range(1, 10)]
    windows = build_window_index([segment], window_size=4, window_stride=2)

    assert windows[0]["pad_left"] == 3
    assert len(windows[0]["events"]) == 1
    assert all(len(window["events"]) + window["pad_left"] == 4 for window in windows)
    assert all(
        len(window["iwg_events"]) + window["iwg_pad_left"] == 9
        for window in windows
    )
    assert windows[-1]["events"][-1]["frame_id"] == 9
    assert [event["frame_id"] for event in windows[-1]["iwg_events"]] == list(
        range(1, 10)
    )
    assert all({event["track_id"] for event in window["events"]} == {7} for window in windows)


def test_unmatched_sentinel_contract_is_exact():
    np.testing.assert_array_equal(HOLD_BOTH_POLICY, [0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(UNMATCHED_CUE, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(UNMATCHED_RISK, [0.0, 0.0, 1.0, 0.0])


def test_event_key_is_sequence_scoped_and_stable():
    assert event_key("MOT17-02-FRCNN", 3, 9) == "MOT17-02-FRCNN|3|9"
    assert event_key("a", 3, 9) != event_key("b", 3, 9)


def test_mot17_all_split_ignores_other_detector_directories():
    other_detectors = ["MOT17-02-DPM", "MOT17-11-SDP"]
    selected, train, val = resolve_joint_sequences(
        dataset="MOT17",
        split="all",
        available_sequences=(
            MOT17_FRCNN_TRAIN_SEQUENCES
            + MOT17_FRCNN_VAL_SEQUENCES
            + other_detectors
        ),
        val_sequences=MOT17_FRCNN_VAL_SEQUENCES,
    )
    assert train == sorted(MOT17_FRCNN_TRAIN_SEQUENCES)
    assert val == sorted(MOT17_FRCNN_VAL_SEQUENCES)
    assert selected == sorted(train + val)
    assert set(selected).isdisjoint(other_detectors)


def test_v1_joint_dataset_is_strictly_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"joint_dataset_schema_version": JOINT_DATASET_SCHEMA_VERSION - 1})
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        CompactIWGTSRMWindowDataset(tmp_path, "train")
