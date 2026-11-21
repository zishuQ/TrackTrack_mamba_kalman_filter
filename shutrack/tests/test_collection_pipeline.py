"""Collection-only storage, cleanup and matching equivalence regressions."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from paired_collection_io import atomic_json, atomic_npz, atomic_track_copy, inspect_shard
from generate_trainval_paired import collect_pilot
from sportsmot_supervision_pilot import bind_gt_box, bind_gt_boxes, SequencePilot
from paired_column import alternate_matches, unique_detection_identities


@pytest.mark.parametrize("kind", ["normal", "fortran", "strided", "scalar", "empty"])
def test_streamed_npz_matches_numpy(tmp_path, kind):
    value = np.arange(60, dtype=np.float32).reshape(10, 6)
    if kind == "fortran":
        value = np.asfortranarray(value)
    elif kind == "strided":
        value = value[:, ::2]
    elif kind == "scalar":
        value = np.asarray("hello, collection")
    elif kind == "empty":
        value = np.empty((0, 2048), dtype=np.float16)
    calls = []
    path = tmp_path / "samples.npz"
    atomic_npz(path, {"value": value}, budget=32, guard=lambda: calls.append(1))
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["value"], value)
    if value.size:
        assert calls


def test_interrupted_write_does_not_publish_partial_npz(tmp_path):
    path = tmp_path / "samples.npz"
    atomic_npz(path, {"old": np.arange(3)})
    original = path.read_bytes()
    def fail():
        raise MemoryError("injected")
    with pytest.raises(MemoryError, match="injected"):
        atomic_npz(path, {"new": np.arange(100)}, guard=fail, budget=32)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp.*"))


def test_bounded_track_copy_does_not_close_source(tmp_path):
    source = np.lib.format.open_memmap(tmp_path / "scratch.npy", mode="w+",
                                      dtype=np.float32, shape=(20, 8))
    source[:] = np.arange(160).reshape(20, 8)
    path = tmp_path / "event_track.npy"
    atomic_track_copy(path, source, 13)
    np.testing.assert_array_equal(np.load(path), source[:13])
    assert not source._mmap.closed


def tiny_shard(tmp_path):
    n, events, ranks = 3, 5, 2
    arrays = {key: np.zeros(n, dtype=np.int64) for key in (
        "frame", "track_id", "detection_index", "gt_id", "current_event_index",
        "rank_valid", "motion_gt_valid", "column_det", "column_score",
        "column_offset", "column_rival",
    )}
    arrays.update(window_indices=np.zeros((n, 5), dtype=np.int64),
                  current_competition=np.zeros((n, 3), dtype=np.float32),
                  pred_box_g0=np.zeros((n, 4)), pred_box_g1=np.zeros((n, 4)),
                  gt_box_t1=np.zeros((n, 4)), rank_rows=np.arange(ranks),
                  rank_score=np.ones(ranks))
    for key in ("rank_template0", "rank_detection", "rank_positive", "rank_competitor"):
        arrays[key] = np.zeros((ranks, 8), dtype=np.float16)
    atomic_npz(tmp_path / "samples.npz", arrays)
    np.save(tmp_path / "event_evidence.npy", np.zeros((events, 13), dtype=np.float32))
    np.save(tmp_path / "event_track.npy", np.zeros((events, 8), dtype=np.float32))
    np.save(tmp_path / "event_detection_index.npy", np.zeros(events, dtype=np.int64))
    np.save(tmp_path / "event_has_detection.npy", np.zeros(events, dtype=bool))
    summary = {"complete": True, "samples": events, "events": events,
               "protocol": {"schema": 3, "rank_arrays": True, "column": True}}
    atomic_json(tmp_path / "summary.json", summary)
    return summary


def test_count_repair_uses_headers_and_keeps_numeric_arrays(tmp_path, monkeypatch):
    summary = tiny_shard(tmp_path)
    original = (tmp_path / "samples.npz").read_bytes()
    def forbid_load(*args, **kwargs):
        raise AssertionError("header gate must not load numpy payloads")
    monkeypatch.setattr(np, "load", forbid_load)
    with pytest.raises(ValueError, match="counts"):
        inspect_shard(tmp_path)
    result = inspect_shard(tmp_path, summary["protocol"], repair_counts=True)
    assert result == {"samples": 3, "events": 5, "repaired": True}
    assert json.loads((tmp_path / "summary.json").read_text())["samples"] == 3
    assert (tmp_path / "samples.npz").read_bytes() == original
    assert len(list(tmp_path.glob("summary.before-count-fix.*.json"))) == 1
    assert not inspect_shard(tmp_path)["repaired"]


@pytest.mark.parametrize("failure", ["incomplete", "missing", "truncated"])
def test_header_gate_rejects_invalid_completed_shards(tmp_path, failure):
    summary = tiny_shard(tmp_path)
    if failure == "incomplete":
        summary["complete"] = False
        atomic_json(tmp_path / "summary.json", summary)
    elif failure == "missing":
        (tmp_path / "event_track.npy").unlink()
    else:
        path = tmp_path / "event_track.npy"
        path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_shard(tmp_path, repair_counts=True)


@pytest.mark.parametrize("stage", ["replay", "labels", "appearance", "write"])
def test_collection_always_closes_and_preserves_original_failure(stage):
    closed = []
    def action(current):
        if current == stage:
            raise RuntimeError("original " + stage)
    class Pilot:
        def run_replay(self): action("replay")
        def compute_labels(self): action("labels")
        def appearance_diagnostics(self): action("appearance")
        def write_compact(self, root): action("write")
        def close(self):
            closed.append(True)
            raise OSError("cleanup error")
    with pytest.raises(RuntimeError, match="original"):
        collect_pilot(Pilot(), Path("."))
    assert closed == [True]


def test_close_is_safe_for_partially_initialized_pilot():
    pilot = SequencePilot.__new__(SequencePilot)
    pilot._rank_final_handles = []
    pilot.close()
    pilot.close()


def test_vector_binding_matches_original_thresholds_and_ties():
    rng = np.random.RandomState(42)
    boxes = rng.uniform(-20, 20, (100, 4))
    boxes[:, 2:] = boxes[:, :2] + rng.uniform(1, 20, (100, 2))
    gt = {i: box for i, box in enumerate(boxes[:20])}
    gt[100] = gt[0].copy()
    boxes = np.r_[boxes, [[0, 0, 10, 10], [0, 0, 20, 10]]]
    for truth in (gt, {}, {7: np.array([0, 0, 20, 10])}):
        assert bind_gt_boxes(boxes, truth) == [bind_gt_box(box, truth) for box in boxes]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_changed_row_matching_equals_full_association(dtype):
    from trackers.utils import iterative_assignment
    rng = np.random.RandomState(11)
    def make(index):
        xy = rng.uniform(0, 20, 2)
        box = np.r_[xy, xy + [40, 80]]
        feat = rng.randn(1, 32).astype(dtype)
        feat /= np.linalg.norm(feat)
        return SimpleNamespace(x1y1x2y2=box, feat=feat, score=0.8,
                               velocity=np.zeros((4, 2)),
                               history={1: [box.copy(), 0.85]}, frame_detection_index=index)
    for _ in range(12):
        tracks = [make(i) for i in range(8)]
        high, low, extra = [make(i) for i in range(6)], [make(6)], [make(7)]
        dets = high + low + extra
        _, _, _, meta = iterative_assignment(
            tracks, high, low, extra, 0.8, 0.2, 0.3, 0.1, 5, return_meta=True,
        )
        saved = meta["final_cost"].copy()
        for index in (0, 4, 7):
            clone = copy.copy(tracks[index])
            clone.feat = make(99).feat
            modified = list(tracks)
            modified[index] = clone
            expected = iterative_assignment(
                modified, high, low, extra, 0.8, 0.2, 0.3, 0.1, 5,
            )[0]
            actual = alternate_matches(meta, index, clone, dets, 0.8, 0.2, 0.3, 0.1)
            assert actual == expected
            np.testing.assert_array_equal(meta["final_cost"], saved)


def test_unique_binding_matches_original_column_rules():
    from sportsmot_supervision_pilot import iou_xyxy
    rng = np.random.RandomState(2)
    boxes = rng.uniform(0, 20, (20, 4))
    boxes[:, 2:] = boxes[:, :2] + 20
    dets = [SimpleNamespace(x1y1x2y2=b) for b in boxes]
    for gt in ({i: b for i, b in enumerate(boxes[:4])}, {}, {8: boxes[0]}):
        expected = []
        for det in dets:
            scores = sorted(((iou_xyxy(det.x1y1x2y2, box), gid)
                             for gid, box in gt.items()), reverse=True)
            expected.append(scores[0][1] if scores and scores[0][0] >= 0.5
                            and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.1)
                            else None)
        assert unique_detection_identities(dets, gt) == expected
