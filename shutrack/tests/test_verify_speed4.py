"""Failure-mode tests for the 4-D speed data validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_speed4 as validator  # noqa: E402


def make_sequence(root: Path, name: str, count: int = 2, skip: tuple[str, ...] = ()) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = count
    keys = [
        {"sequence": name, "frame": index + 1, "track_id": index + 1, "detection_index": index}
        for index in range(rows)
    ]
    arrays = {
        "frame": np.arange(1, rows + 1, dtype=np.int64),
        "track_id": np.arange(1, rows + 1, dtype=np.int64),
        "detection_index": np.arange(rows, dtype=np.int64),
        "gt_id": np.arange(1, rows + 1, dtype=np.int64),
        "window_indices": np.zeros((rows, 5), np.int64),
        "keys_json": np.array(json.dumps(keys)),
        "motion_benefit": np.zeros(rows, np.float32),
        "appearance_benefit": np.zeros(rows, np.float32),
        "valid_motion": np.ones(rows, np.bool_),
        "valid_appearance": np.ones(rows, np.bool_),
        "gt_coverage": np.ones(rows, np.float32),
        "oracle_detection_coverage": np.ones(rows, np.float32),
        "motion_gt_valid": np.ones(rows, np.bool_),
        "pred_box_g0": np.zeros((rows, 4), np.float32),
        "pred_box_g1": np.zeros((rows, 4), np.float32),
        "gt_box_t1": np.zeros((rows, 4), np.float32),
        "appearance_delta": np.zeros(rows, np.float32),
        "appearance_diag_valid": np.ones(rows, np.bool_),
        "appearance_score_template0": np.zeros(rows, np.float32),
        "appearance_score_template1": np.zeros(rows, np.float32),
        "motion_gate": np.full(rows, -1.0, np.float32),
        "appearance_gate": np.full(rows, -1.0, np.float32),
        "kf_velocity_g0": np.zeros((rows, 2), np.float32),
        "kf_velocity_g1": np.zeros((rows, 2), np.float32),
        "gt_velocity": np.zeros((rows, 2), np.float32),
        "kf_velocity4_g0": np.zeros((rows, 4), np.float32),
        "kf_velocity4_g1": np.zeros((rows, 4), np.float32),
        "gt_velocity4": np.zeros((rows, 4), np.float32),
        "gt_size_velocity_central": np.zeros((rows, 2), np.float32),
        "speed_valid": np.ones(rows, np.bool_),
        "size_adj_valid": np.ones(rows, np.bool_),
        "size_central_valid": np.ones(rows, np.bool_),
    }
    np.savez(directory / "samples.npz", **arrays)
    for filename in (
        "event_evidence.npy",
        "event_track.npy",
        "event_detection_index.npy",
        "event_has_detection.npy",
    ):
        if filename in skip:
            continue
        np.save(directory / filename, np.zeros(0))
    (directory / "summary.json").write_text(
        json.dumps({"kf_velocity4_mid_error_max": 0.0, "kf_box_mid_error_max": 0.0})
    )


def prepare(tmp_path, monkeypatch, expected, new_sequences, old_sequences):
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    new_root.mkdir()
    old_root.mkdir()
    for count in new_sequences:
        make_sequence(new_root, count[0], count[1], count[2] if len(count) > 2 else ())
    for count in old_sequences:
        make_sequence(old_root, count[0], count[1])
    monkeypatch.setattr(validator, "NEW", new_root)
    monkeypatch.setattr(validator, "OLD", old_root)
    monkeypatch.setattr(validator, "official_trainval", lambda: set(expected))
    return validator.data_alignment()


def test_empty_new_directory_fails(tmp_path, monkeypatch) -> None:
    report = prepare(tmp_path, monkeypatch, {"a"}, [], [("a", 2)])
    assert report["data_aligned"] is False
    assert report["new_sequences"] == 0
    assert any("no sequences" in reason for reason in report["reasons"])


def test_missing_sequence_fails(tmp_path, monkeypatch) -> None:
    report = prepare(tmp_path, monkeypatch, {"a", "b"}, [("a", 2)], [("a", 2), ("b", 2)])
    assert report["data_aligned"] is False
    assert any("missing 1 sequences" in reason for reason in report["reasons"])


def test_missing_shard_file_fails(tmp_path, monkeypatch) -> None:
    report = prepare(
        tmp_path,
        monkeypatch,
        {"a"},
        [("a", 2, ("event_track.npy",))],
        [("a", 2)],
    )
    assert report["data_aligned"] is False
    assert any("missing shard files" in reason for reason in report["reasons"])


def test_sample_count_mismatch_fails(tmp_path, monkeypatch) -> None:
    report = prepare(tmp_path, monkeypatch, {"a"}, [("a", 3)], [("a", 2)])
    assert report["data_aligned"] is False
    assert any("sample count" in reason for reason in report["reasons"])


def test_matching_history_passes(tmp_path, monkeypatch) -> None:
    report = prepare(tmp_path, monkeypatch, {"a"}, [("a", 2)], [("a", 2)])
    assert report["data_aligned"] is True
    assert report["sequences"] == 1
    assert report["samples"] == 2
    assert report["reasons"] == []


def test_zero_sample_history_fails(tmp_path, monkeypatch) -> None:
    report = prepare(tmp_path, monkeypatch, {"a"}, [("a", 0)], [("a", 0)])
    assert report["data_aligned"] is False
    assert any("zero samples" in reason for reason in report["reasons"])


def test_mismatch_reason_is_attributed_only_to_the_bad_sequence(tmp_path, monkeypatch) -> None:
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    new_root.mkdir()
    old_root.mkdir()
    for name in ("a", "b"):
        make_sequence(new_root, name, 2)
        make_sequence(old_root, name, 2)
    data = np.load(new_root / "a" / "samples.npz")
    arrays = {key: data[key] for key in data.files}
    arrays["motion_benefit"] = arrays["motion_benefit"] + 1.0
    np.savez(new_root / "a" / "samples.npz", **arrays)
    monkeypatch.setattr(validator, "NEW", new_root)
    monkeypatch.setattr(validator, "OLD", old_root)
    monkeypatch.setattr(validator, "official_trainval", lambda: {"a", "b"})
    report = validator.data_alignment()
    assert report["data_aligned"] is False
    assert any(reason.startswith("a:") and "labels" in reason for reason in report["reasons"])
    assert not any(reason.startswith("b:") for reason in report["reasons"])
