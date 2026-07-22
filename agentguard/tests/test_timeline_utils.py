from __future__ import annotations

import hashlib

import numpy as np

from agentguard.datasets import timeline_utils


def _record(sequence: str, track_id: int, frame_id: int, history_count: int, offset: int):
    return {
        "sequence": sequence,
        "track_id": track_id,
        "frame_id": frame_id,
        "history_count": history_count,
        "event_shard_id": 0,
        "event_offset": offset,
        "event_id": f"{sequence}-{track_id}-{frame_id}",
        "matched": frame_id % 2 == 0,
    }


def test_timeline_record_and_event_key_are_stable():
    record = _record("seq", 4, 12, 3, 9)
    assert timeline_utils.event_key("seq", 2, 7) == "seq|2|7"
    assert timeline_utils.compact_timeline_record("seq", record) == {
        "sequence": "seq",
        "event_shard_id": 0,
        "event_offset": 9,
        "event_id": "seq-4-12",
        "frame_id": 12,
        "track_id": 4,
        "matched": True,
        "history_count": 3,
    }


def test_timeline_segments_sort_group_and_reset_on_gap_or_history_drop():
    records = [
        _record("seq", 2, 1, 1, 0),
        _record("seq", 1, 4, 3, 3),
        _record("seq", 1, 2, 2, 2),
        _record("seq", 1, 1, 1, 1),
        _record("seq", 1, 7, 4, 4),
        _record("seq", 1, 8, 1, 5),
    ]

    segments = timeline_utils.segment_track_timelines(records, max_frame_gap=2)

    assert [[item["frame_id"] for item in segment] for segment in segments] == [
        [1, 2, 4],
        [7],
        [8],
        [1],
    ]


def test_file_hashes_include_sorted_names_and_content(tmp_path):
    first = tmp_path / "b.bin"
    second = tmp_path / "a.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    expected = hashlib.sha256()
    expected.update(b"a.bin")
    expected.update(hashlib.sha256(b"second").hexdigest().encode("ascii"))
    expected.update(b"b.bin")
    expected.update(hashlib.sha256(b"first").hexdigest().encode("ascii"))
    assert timeline_utils._sha256_file_set([first, second]) == expected.hexdigest()

    first.write_bytes(b"changed")
    assert timeline_utils._sha256_file_set([first, second]) != expected.hexdigest()


def test_normalization_reads_only_the_requested_training_sequences(monkeypatch, tmp_path):
    opened: list[str] = []

    class Reader:
        def __init__(self, path):
            opened.append(str(path))

        def iter_event_records(self):
            return [
                {"scalar_features": np.zeros(63, dtype=np.float64)},
                {"scalar_features": np.full(63, 2.0, dtype=np.float64)},
            ]

        def close(self):
            pass

    monkeypatch.setattr(timeline_utils, "CompactEventCacheReader", Reader)
    stats = timeline_utils._fit_train_normalization(
        tmp_path,
        "MOT17",
        "train",
        ["train-sequence"],
    )

    assert opened == [str(tmp_path / "MOT17" / "train" / "train-sequence")]
    np.testing.assert_allclose(stats.mean, np.ones(63))
    np.testing.assert_allclose(stats.std, np.ones(63))
