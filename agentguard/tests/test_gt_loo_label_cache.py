"""Correctness tests for GT+LOO label-generation caches against frozen code."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.detection_cache import SequenceDetectionCache
from agentguard.data.identity_prototype import (
    IdentityPrototypeBuilder,
    PreparedIdentityCandidates,
)
from agentguard.data.rollout_label_builder import (
    _select_oracle_detection_index,
    best_oracle_detection,
    box_iou,
    build_compact_rollout_labels_for_sequence,
)
from agentguard.data.train_data_v2 import PackedReplayReader, RULES


FROZEN_DIR = Path(__file__).resolve().parent / "frozen_unoptimized"
TOP_KEEP_RATIO = 0.7
FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-10


def load_frozen_builder():
    spec = importlib.util.spec_from_file_location(
        "frozen_unoptimized_rollout_label_builder",
        FROZEN_DIR / "rollout_label_builder.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_frozen_prototype():
    spec = importlib.util.spec_from_file_location(
        "frozen_unoptimized_identity_prototype",
        FROZEN_DIR / "identity_prototype.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.IdentityPrototypeBuilder


FrozenPrototype = load_frozen_prototype()


def reference_kept_indices(features_normed: np.ndarray) -> np.ndarray:
    init_mean = np.mean(features_normed, axis=0)
    norm = np.linalg.norm(init_mean)
    if norm > 0:
        init_mean = init_mean / norm
    similarities = features_normed @ init_mean
    k = max(1, int(np.ceil(features_normed.shape[0] * TOP_KEEP_RATIO)))
    return np.argsort(similarities)[::-1][:k]


def remaining_rows(features_by_detection, excluded):
    feats = [
        np.asarray(feature, dtype=np.float64).reshape(-1)
        for index, feature in features_by_detection.items()
        if index != excluded
    ]
    if not feats:
        return np.zeros((0, 0), dtype=np.float64)
    stacked = np.stack(feats, axis=0)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return stacked / norms


DISCRETE_FIELDS = (
    "event_id",
    "event_shard_id",
    "event_offset",
    "sequence",
    "frame_id",
    "frame_index",
    "track_id",
    "candidate_type",
    "candidate_detection_index",
    "target_gt_id",
    "target_identity_key",
    "detection_gt_id",
    "valid_motion",
    "valid_appearance",
    "valid_horizon_count",
    "sample_type",
    "motion_oracle_hard",
    "appearance_oracle_hard",
    "target_gate",
    "motion_label_mode",
)

FLOAT_FIELDS = (
    "detection_gt_iou",
    "motion_benefit",
    "appearance_benefit",
    "gt_coverage",
    "oracle_detection_coverage",
    "motion_soft_target",
    "appearance_soft_target",
    "motion_safe_target",
    "appearance_safe_target",
    "motion_label_confidence",
    "appearance_label_confidence",
    "sample_weight",
)

VECTOR_FIELDS = (
    "cue_target",
    "risk_targets",
    "policy_soft_target",
    "policy_safe_soft_target",
)


def compare_labels(actual, expected):
    assert len(actual) == len(expected)
    max_err = 0.0
    max_field = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        for field in DISCRETE_FIELDS:
            assert got[field] == want[field], f"{field} mismatch at {index}: {got[field]!r} vs {want[field]!r}"
        for field in FLOAT_FIELDS:
            err = abs(float(got[field]) - float(want[field]))
            if err > max_err:
                max_err = err
                max_field = field
            np.testing.assert_allclose(
                got[field],
                want[field],
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                err_msg=f"{field} mismatch at {index}",
            )
        for field in VECTOR_FIELDS:
            err = float(np.max(np.abs(np.asarray(got[field], dtype=np.float64) - np.asarray(want[field], dtype=np.float64))))
            if err > max_err:
                max_err = err
                max_field = field
            np.testing.assert_allclose(
                np.asarray(got[field], dtype=np.float64),
                np.asarray(want[field], dtype=np.float64),
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                err_msg=f"{field} mismatch at {index}",
            )
    return max_err, max_field


def test_loo_precompute_matches_frozen_exclusion_and_ties():
    rng = np.random.default_rng(3)
    observations = {i: rng.normal(size=16) for i in range(12)}
    # Current detection not among prototype candidates.
    prepared = PreparedIdentityCandidates.from_features_by_detection(observations)
    actual = prepared.leave_one_out(99)
    expected = FrozenPrototype.build_leave_one_out(observations, 99)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        IdentityPrototypeBuilder.build_leave_one_out(observations, 99),
        expected,
    )
    np.testing.assert_array_equal(reference_kept_indices(prepared.features_normed), reference_kept_indices(remaining_rows(observations, 99)))

    # Duplicate detection references stay one candidate.
    unique = {0: observations[0], 1: observations[1], 2: observations[2]}
    for _ in range(7):
        unique.setdefault(1, observations[1] * 4)
    assert IdentityPrototypeBuilder.build_leave_one_out(unique, 0) is None
    assert FrozenPrototype.build_leave_one_out(unique, 0) is None
    assert IdentityPrototypeBuilder.build_leave_one_out(unique, 99) is not None

    # Exclusion leaves fewer than three references.
    tiny = {10: observations[0], 11: observations[1], 12: observations[2]}
    assert IdentityPrototypeBuilder.build_leave_one_out(tiny, 11) is None
    assert FrozenPrototype.build_leave_one_out(tiny, 11) is None

    # Similarity ties: two extra copies of the mean direction plus two equal residuals.
    tied = {
        0: np.array([1.0, 0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0, 0.0]),
        3: np.array([0.0, 1.0, 0.0, 0.0]),
        4: np.array([0.0, 0.0, 1.0, 0.0]),
        5: np.array([1.0, 1.0, 0.0, 0.0]),
    }
    prepared = PreparedIdentityCandidates.from_features_by_detection(tied)
    for excluded in (2, 99, 5):
        actual = prepared.leave_one_out(excluded)
        expected = FrozenPrototype.build_leave_one_out(tied, excluded)
        if expected is None:
            assert actual is None
            continue
        np.testing.assert_array_equal(actual, expected)
        remaining = remaining_rows(tied, excluded)
        np.testing.assert_array_equal(
            reference_kept_indices(remaining),
            reference_kept_indices(
                prepared.features_normed
                if excluded not in prepared._row
                else np.concatenate(
                    (
                        prepared.features_normed[: prepared._row[excluded]],
                        prepared.features_normed[prepared._row[excluded] + 1 :],
                    ),
                    axis=0,
                )
            ),
        )

    # Same identity/candidates/exclusion reuses the identical array object.
    first = prepared.leave_one_out(2)
    second = prepared.leave_one_out(2)
    assert first is second


def test_loo_does_not_subtract_from_final_prototype():
    rng = np.random.default_rng(4)
    observations = {i: rng.normal(size=8) for i in range(6)}
    full = IdentityPrototypeBuilder.build_prototype(list(observations.values()))
    excluded = observations[1]
    approx = full - excluded / 6.0
    actual = IdentityPrototypeBuilder.build_leave_one_out(observations, 1)
    assert actual is not None
    assert not np.allclose(actual, approx / np.linalg.norm(approx))


def _write_detection_cache(directory: Path, boxes, scores, features):
    directory.mkdir(parents=True, exist_ok=True)
    n = len(boxes)
    np.save(directory / "boxes.npy", np.asarray(boxes, dtype=np.float32))
    np.save(directory / "scores.npy", np.asarray(scores, dtype=np.float32))
    np.save(directory / "features.npy", np.asarray(features, dtype=np.float32))
    np.save(directory / "sources.npy", np.zeros(n, dtype=np.int8))
    np.save(directory / "class_ids.npy", np.ones(n, dtype=np.int16))
    np.save(directory / "frame_offsets.npy", np.array([0, 2, n], dtype=np.int64))
    np.save(directory / "target_frame_offsets.npy", np.array([0, 1, 2], dtype=np.int64))
    np.save(directory / "target_detection_indices.npy", np.array([1, 2], dtype=np.int64))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "schema_version": 1,
                "dataset": "MOT17",
                "split": "all",
                "sequence": "seq",
                "num_frames": 2,
                "num_detections": n,
                "reid_dim": int(np.asarray(features).shape[1]),
            }
        )
    )


def test_oracle_reads_boxes_only_and_preserves_ties_and_missing(tmp_path):
    frozen = load_frozen_builder()
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
        ],
        dtype=np.float32,
    )
    features = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    _write_detection_cache(tmp_path, boxes, [0.9, 0.8, 0.7], features)
    cache = SequenceDetectionCache(tmp_path)
    feature_reads = {"count": 0}
    original_features = cache.features

    class Probe:
        def __getitem__(self, item):
            feature_reads["count"] += 1
            return original_features[item]

    cache.features = Probe()

    class Reader:
        detection_cache = cache

        def get_detection(self, detection_index, *, include_feature=True):
            return cache.get_detection(int(detection_index), include_feature=include_feature)

    reader = Reader()
    gt = np.array([0.2, 0.2, 10.2, 10.2], dtype=np.float64)
    before = feature_reads["count"]
    indices, frame_boxes = cache.get_frame_indices_and_boxes(0, view="source")
    assert feature_reads["count"] == before
    assert indices.tolist() == [0, 1]
    np.testing.assert_allclose(frame_boxes, boxes[:2])

    selected = best_oracle_detection(reader, 0, gt)
    assert selected is not None
    assert selected.detection_index == 0
    assert feature_reads["count"] == before + 1
    frozen_selected = frozen.best_oracle_detection(reader, 0, gt)
    assert frozen_selected is not None
    assert frozen_selected.detection_index == selected.detection_index

    # Equal IoU keeps the earlier detection (strict >).
    tied_gt = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    assert box_iou(tied_gt, boxes[0]) == box_iou(np.array([0.0, 0.0, 10.0, 10.0]), boxes[0])
    equal_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    assert _select_oracle_detection_index(np.array([4, 7]), equal_boxes, tied_gt, 0.3) == 4

    assert best_oracle_detection(reader, 0, None) is None
    miss = np.array([200.0, 200.0, 210.0, 210.0])
    assert best_oracle_detection(reader, 0, miss) is None
    assert frozen.best_oracle_detection(reader, 0, miss) is None
    empty_reader = type("Reader", (), {"detection_cache": None})()
    assert best_oracle_detection(empty_reader, 0, gt) is None
    cache.close()


def _synthetic_sequence(tmp_path):
    sequence = "MOT17-09-FRCNN"
    event_dir = tmp_path / "events" / "MOT17/all" / sequence
    detection_dir = tmp_path / "detection" / "MOT17/all" / sequence
    gt_parent = tmp_path / "datasets/MOT17/train"
    gt_dir = gt_parent / sequence / "gt"
    for path in (event_dir, detection_dir, gt_dir):
        path.mkdir(parents=True)
    n, dim = 8, 8
    features = np.array(
        [[1.0, 0.05 * i, 0.1 * (i % 3), 0.0, 0.2, 0.0, 0.01 * i, 0.3] for i in range(n)],
        dtype=np.float32,
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    boxes = np.tile(np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32), (n, 1))
    boxes[6] = np.array([80.0, 80.0, 90.0, 95.0], dtype=np.float32)  # mismatch-like box
    scores = np.ones(n, dtype=np.float32) * 0.9
    scores[5] = 0.4  # below prototype score gate
    for name, value in {
        "boxes": boxes,
        "features": features,
        "scores": scores,
        "sources": np.zeros(n, dtype=np.int8),
        "class_ids": np.ones(n, dtype=np.int16),
        "frame_offsets": np.arange(n + 1, dtype=np.int64),
        "target_frame_offsets": np.arange(n + 1, dtype=np.int64),
        "target_detection_indices": np.arange(n, dtype=np.int64),
    }.items():
        np.save(detection_dir / f"{name}.npy", value)
    (detection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "schema_version": 1,
                "dataset": "MOT17",
                "split": "all",
                "sequence": sequence,
                "num_frames": n,
                "num_detections": n,
                "reid_dim": dim,
            }
        )
    )
    # Frame 8 has no GT row to cover missing-GT / sequence-tail behaviour.
    gt_lines = [f"{i + 1},1,0,0,10,20,1,1,1\n" for i in range(n - 1)]
    gt_lines.append("4,2,80,80,10,15,1,1,1\n")
    (gt_dir / "gt.txt").write_text("".join(gt_lines))
    (gt_dir.parent / "seqinfo.ini").write_text("[Sequence]\nimWidth=100\nimHeight=100\nframeRate=30\nseqLength=8\n")
    states, events, frames, associations = [], [], [], []
    state = {
        "mean": np.array([6.0, 10.0, 12.0, 21.0, 0, 0, 0, 0]),
        "covariance": np.eye(8),
        "box": np.array([0.0, -0.5, 12.0, 20.5]),
        "score": 0.9,
        "state": 1,
        "velocity": np.zeros((4, 2)),
        "end_frame_id": 0,
        "recent_history_frames": np.zeros(0, dtype=np.int32),
        "recent_history_boxes": np.zeros((0, 4)),
        "recent_history_scores": np.zeros(0),
    }
    for i in range(n):
        states.extend([state, state])
        matched = i != 4
        accepted = i if matched else -1
        events.append(
            {
                "event_id": f"MOT17/{sequence}/{i + 1:06d}/000001",
                "frame_id": i + 1,
                "frame_index": i,
                "track_id": 1,
                "matched": matched,
                "accepted_detection_index": accepted,
                "association_shard_id": 0 if matched else -1,
                "association_offset": i if matched else -1,
                "association_track_row": 0 if matched else -1,
                "state_shard_id": 0,
                "frame_start_state_offset": 2 * i,
                "pre_update_state_offset": 2 * i + 1,
                "event_shard_id": 0,
                "event_offset": i,
                "scalar_features": np.ones(63, dtype=np.float32) * i,
                "track_feature": np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "history_count": min(i + 1, 6),
            }
        )
        frames.append(
            {
                "frame_id": i + 1,
                "frame_index": i,
                "effective_warp": np.eye(2, 3),
                "image_width": 100,
                "image_height": 100,
            }
        )
        associations.append(
            {
                "frame_id": i + 1,
                "track_ids": np.array([1]),
                "detection_indices": np.array([i]),
                "final_cost": np.zeros((1, 1)),
                "assignment_threshold": np.ones((1, 1)),
                "detection_source": np.zeros((1, 1), dtype=np.int8),
            }
        )
    for kind, records in {
        "events": events,
        "states": states,
        "frames": frames,
        "associations": associations,
    }.items():
        torch.save(records, event_dir / f"{kind}_00000.pt")
    (event_dir / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "truncated": False,
                "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "dataset": "MOT17",
                "sequence": sequence,
                "num_frames": n,
                "num_events": n,
                "reid_dim": dim,
            }
        )
    )
    return event_dir, detection_dir, gt_parent


def test_synthetic_sequence_matches_frozen_builder(tmp_path):
    frozen = load_frozen_builder()
    event_dir, detection_dir, gt_parent = _synthetic_sequence(tmp_path)
    from agentguard.data.cache_reader import CompactEventCacheReader

    old_reader = CompactEventCacheReader(event_dir, detection_dir, max_cached_shards=4)
    expected, old_summary = frozen.build_compact_rollout_labels_for_sequence(
        event_dir,
        detection_dir,
        gt_parent,
        reader=old_reader,
        persist_stats=False,
        **RULES,
    )
    new_reader = CompactEventCacheReader(event_dir, detection_dir, max_cached_shards=4)
    actual, new_summary = build_compact_rollout_labels_for_sequence(
        event_dir,
        detection_dir,
        gt_parent,
        reader=new_reader,
        persist_stats=False,
        **RULES,
    )
    compare_labels(actual, expected)
    assert new_summary["num_labels"] == old_summary["num_labels"]
    assert new_summary["prototype_count"] == old_summary["prototype_count"]
    assert new_summary["valid_appearance_labels"] == old_summary["valid_appearance_labels"]
    assert new_summary["skipped_unreliable"] == old_summary["skipped_unreliable"]
    assert new_summary["skipped_invalid"] == old_summary["skipped_invalid"]
    assert any(not label["valid_appearance"] or label["detection_gt_id"] == 1 for label in actual)
    # Unmatched event is not labeled as candidate A.
    assert all(int(label["candidate_detection_index"]) >= 0 for label in actual)


def test_oracle_cache_does_not_leak_across_sources(tmp_path):
    frozen = load_frozen_builder()
    event_dir, detection_dir, gt_parent = _synthetic_sequence(tmp_path)
    other = tmp_path / "other_det"
    _write_detection_cache(
        other,
        np.array([[100.0, 100.0, 110.0, 120.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        [0.9, 0.9],
        np.ones((2, 8), dtype=np.float32),
    )
    from agentguard.data.cache_reader import CompactEventCacheReader

    reader_a = CompactEventCacheReader(event_dir, detection_dir, max_cached_shards=2)
    reader_b = CompactEventCacheReader(event_dir, other, max_cached_shards=2)
    gt = np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float64)
    a = best_oracle_detection(reader_a, 0, gt)
    b = best_oracle_detection(reader_b, 0, gt)
    assert a is not None
    assert b is None or a.detection_index != b.detection_index or not np.allclose(a.box, b.box)
    frozen_a = frozen.best_oracle_detection(reader_a, 0, gt)
    assert frozen_a.detection_index == a.detection_index
    reader_a.close()
    reader_b.close()


@pytest.mark.skipif(
    not Path("/home/shang/workspace/TrackTrack/outputs/agentguard/train_data_v2/MOT17/MOT17-09-FRCNN/data.pt").is_file(),
    reason="local MOT17-09-FRCNN v2 data is not present",
)
def test_mot17_09_small_prefix_matches_frozen_and_stored_labels():
    frozen = load_frozen_builder()
    seq = "MOT17-09-FRCNN"
    root = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/train_data_v2/MOT17") / seq
    det = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache/MOT17/all") / seq
    gt = Path("/home/shang/datasets/MOT17/train")
    old_reader = PackedReplayReader(root, det)
    expected, _ = frozen.build_compact_rollout_labels_for_sequence(
        root, det, gt, reader=old_reader, persist_stats=False, max_events=40, **RULES
    )
    new_reader = PackedReplayReader(root, det)
    actual, _ = build_compact_rollout_labels_for_sequence(
        root, det, gt, reader=new_reader, persist_stats=False, max_events=40, **RULES
    )
    compare_labels(actual, expected)
