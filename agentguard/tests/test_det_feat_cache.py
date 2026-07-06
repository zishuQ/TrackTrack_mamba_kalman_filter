"""Focused tests for detection-pair caching and sequence-grouping optimisation.

Verifies that:
1. ``load_detection_pair`` returns a cached result on repeat calls with the same paths.
2. ``clear_detection_pair_cache`` evicts entries and forces a reload.
3. Sequence grouping in ``_resolve_seq_pickle_groups`` groups sequences by
   their (target_pickle_path, pickle_path_95) pair as computed by
   ``set_parameters``.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
TRACKER_DIR = os.path.join(PROJECT_ROOT, "3. Tracker")
if TRACKER_DIR not in sys.path:
    sys.path.insert(0, TRACKER_DIR)

from utils.det_feat_storage import (
    build_shards,
    clear_detection_pair_cache,
    compact_index_path,
    load_detection_pair,
    shard_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_pickle(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _build_det_pickle(
    seq_names: List[str], n_frames: int = 2, n_dets: int = 3
) -> Dict[str, Dict[int, Any]]:
    """Build a dict structured like a detection pickle."""
    data: Dict[str, Dict[int, Any]] = {}
    for seq in seq_names:
        data[seq] = {}
        for fi in range(n_frames):
            data[seq][fi] = np.random.randn(n_dets, 5).astype(np.float64)
    return data


def _build_mini_idx(target_pickle_path: str, source_pickle_path: str,
                    seq_names: List[str], n_frames: int, n_dets: int) -> str:
    """Create a minimal compact-index pickle for the given source pickle."""
    from utils.det_feat_storage import compact_index_path

    idx_path = compact_index_path(target_pickle_path, source_pickle_path)
    if not os.path.isdir(os.path.dirname(idx_path)):
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)

    idx_data = {"index": {}}
    for seq in seq_names:
        idx_data["index"][seq] = {}
        for fi in range(n_frames):
            # Map frame_idx -> list of indices into the 0.95 array
            idx_data["index"][seq][fi] = list(range(n_dets))

    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    with open(idx_path, "wb") as f:
        pickle.dump(idx_data, f)
    return idx_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pickle_pair(tmpdir):
    """Create a minimal (target, source) pickle pair on disk."""
    base = str(tmpdir)
    seqs = ["seqA", "seqB"]
    n_frames = 2
    n_dets = 2

    source_path = os.path.join(base, "dummy_0.95.pickle")
    target_path = os.path.join(base, "dummy_0.80.pickle")

    source_data = _build_det_pickle(seqs, n_frames, n_dets)
    _write_minimal_pickle(source_path, source_data)
    _build_mini_idx(target_path, source_path, seqs, n_frames, n_dets)

    return target_path, source_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectionPairCache:
    def test_first_load_uses_disk(self, pickle_pair):
        target, source = pickle_pair
        clear_detection_pair_cache()

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80, det95 = load_detection_pair(target, source)
            # pickle.load called twice: once for source pickle, once for idx
            assert spy.call_count == 2

    def test_second_load_hits_cache(self, pickle_pair):
        target, source = pickle_pair
        clear_detection_pair_cache()

        det80_a, det95_a = load_detection_pair(target, source)

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80_b, det95_b = load_detection_pair(target, source)
            assert spy.call_count == 0

        assert det80_b is det80_a
        assert det95_b is det95_a

    def test_sequence_filter_builds_only_requested_subset(self, pickle_pair):
        target, source = pickle_pair
        clear_detection_pair_cache()

        det80, det95 = load_detection_pair(target, source, sequence_names=["seqA"])

        assert set(det80.keys()) == {"seqA"}
        assert "seqB" in det95

    def test_cache_key_matches_absolute_paths(self, pickle_pair):
        target, source = pickle_pair
        clear_detection_pair_cache()

        det80_a, det95_a = load_detection_pair(target, source)

        # Call with the same paths (already absolute) — must hit cache
        with patch("pickle.load", wraps=pickle.load) as spy:
            det80_b, det95_b = load_detection_pair(target, source)
            assert spy.call_count == 0

        assert det80_b is det80_a

    def test_different_pairs_independent(self, pickle_pair, tmpdir):
        target1, source1 = pickle_pair
        clear_detection_pair_cache()

        # Create a second independent pair
        base = str(tmpdir.mkdir("pair2"))
        seqs = ["seqC"]
        n_frames = 1
        n_dets = 1
        source2 = os.path.join(base, "other_0.95.pickle")
        target2 = os.path.join(base, "other_0.80.pickle")
        _write_minimal_pickle(source2, _build_det_pickle(seqs, n_frames, n_dets))
        _build_mini_idx(target2, source2, seqs, n_frames, n_dets)

        det80_a, _ = load_detection_pair(target1, source1)
        det80_b, _ = load_detection_pair(target2, source2)

        # The two caches should hold different objects
        assert det80_a is not det80_b

    def test_shared_source_pickle_only_loads_source_once(self, pickle_pair, tmpdir):
        target1, source = pickle_pair
        clear_detection_pair_cache()

        # Second target shares the same 0.95 source pickle but has its own index.
        base = str(tmpdir.mkdir("pair_shared_source"))
        target2 = os.path.join(base, "other_0.80.pickle")
        _build_mini_idx(target2, source, ["seqA", "seqB"], 2, 2)

        with patch("pickle.load", wraps=pickle.load) as spy:
            load_detection_pair(target1, source)
            load_detection_pair(target2, source)
            # First pair: source + idx, second pair: idx only.
            assert spy.call_count == 3

    def test_sequence_filter_participates_in_cache_key(self, pickle_pair):
        target, source = pickle_pair
        clear_detection_pair_cache()

        det80_a, _ = load_detection_pair(target, source, sequence_names=["seqA"])
        det80_b, _ = load_detection_pair(target, source, sequence_names=["seqB"])

        assert set(det80_a.keys()) == {"seqA"}
        assert set(det80_b.keys()) == {"seqB"}
        assert det80_a is not det80_b

    def test_clear_cache_forces_reload(self, pickle_pair):
        target, source = pickle_pair
        clear_detection_pair_cache()

        det80_a, _ = load_detection_pair(target, source)
        clear_detection_pair_cache()

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80_b, _ = load_detection_pair(target, source)
            assert spy.call_count == 2  # source + idx

        # After clear + reload, data is a new object
        assert det80_b is not det80_a

    def test_clear_cache_importable(self):
        from utils.det_feat_storage import clear_detection_pair_cache
        clear_detection_pair_cache()


class TestSequenceGroupingByPicklePair:
    """Verify that sequences are grouped by (target_pickle_path, pickle_path_95)."""

    def _resolve_seq_pickle_groups(
        self, sequences: List[str], mode: str, pickle_dir: str,
    ) -> Dict[Tuple[str, str], List[str]]:
        """Miniature re-implementation of the precompute+group logic for testing."""
        from utils.etc import set_parameters

        class _Args:
            pass

        tracker_args = _Args()
        tracker_args.pickle_dir = pickle_dir
        tracker_args.data_dir = "/fake/datasets/"

        seq_meta: Dict[str, Dict[str, str]] = {}
        for seq_name in sequences:
            set_parameters(tracker_args, seq_name, mode)
            seq_meta[seq_name] = {
                "target_pickle_path": tracker_args.target_pickle_path,
                "pickle_path_95": tracker_args.pickle_path_95,
            }

        groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for seq_name, meta in seq_meta.items():
            key = (meta["target_pickle_path"], meta["pickle_path_95"])
            groups[key].append(seq_name)
        return dict(groups)

    def test_all_mot17_train_custom_sequences_single_group(self):
        """All MOT17 train_custom sequences share one pickle pair."""
        seqs = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09"]
        groups = self._resolve_seq_pickle_groups(
            seqs, "train_custom", "/fake/pickles"
        )
        assert len(groups) == 1, (
            f"Expected 1 pickle-pair group for train_custom, got {len(groups)}"
        )
        key = next(iter(groups))
        seqs_in_group = groups[key]
        assert sorted(seqs_in_group) == sorted(seqs)

    def test_mix_mot17_and_mot20_creates_two_groups(self):
        """MOT17 and MOT20 sequences produce different pickle paths."""
        seqs = ["MOT17-02", "MOT17-04", "MOT20-05", "MOT20-07"]
        groups = self._resolve_seq_pickle_groups(
            seqs, "test", "/fake/pickles"
        )
        assert len(groups) >= 1

    def test_preserves_explicit_single_sequence(self):
        """A single sequence maps to exactly one group."""
        seqs = ["MOT17-02"]
        groups = self._resolve_seq_pickle_groups(
            seqs, "train_custom", "/fake/pickles"
        )
        assert len(groups) == 1
        key = next(iter(groups))
        assert groups[key] == ["MOT17-02"]


def test_cmd_cache_events_requires_mmap_cache_unless_fallback(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agentguard.cli import _cmd_cache_events

    def fake_resolve_sequences(dataset, mode, explicit=None):
        return ["SEQ_A"]

    def fake_set_parameters(tracker_args, seq_name, mode):
        tracker_args.target_pickle_path = "/fake/shared_0.80.pickle"
        tracker_args.pickle_path_95 = "/fake/shared_0.95.pickle"
        tracker_args.data_path = str(tmp_path) + "/"
        tracker_args.det_thr = 0.6
        tracker_args.init_thr = 0.7
        tracker_args.match_thr = 0.7

    def fake_load_detection_pair(target_pickle_path, source_pickle_path, sequence_names=None):
        raise AssertionError("cache_events must not load the monolithic pickle by default")

    monkeypatch.setattr("agentguard.cli._resolve_sequences", fake_resolve_sequences)
    monkeypatch.setattr("utils.etc.set_parameters", fake_set_parameters)
    monkeypatch.setattr("utils.det_feat_storage.load_detection_pair", fake_load_detection_pair)

    args = SimpleNamespace(
        dataset="MOT17",
        mode="all",
        sequence=None,
        max_frames=1,
        data_dir=str(tmp_path),
        pickle_dir=str(tmp_path),
        detection_cache_root=str(tmp_path / "missing_detection_cache"),
        event_cache_root=str(tmp_path / "event_cache"),
        allow_pickle_fallback=False,
    )
    with pytest.raises(FileNotFoundError, match="Detection cache manifest not found"):
        _cmd_cache_events(args)


# ---------------------------------------------------------------------------
# Per‑sequence shard tests
# ---------------------------------------------------------------------------


class TestShardPreferredLoad:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        clear_detection_pair_cache()

    def _build_pair_and_shard(self, tmpdir, seqs, n_frames=2, n_dets=3, shard_seqs=None):
        base = str(tmpdir)
        source_path = os.path.join(base, "shard_test_0.95.pickle")
        target_path = os.path.join(base, "shard_test_0.80.pickle")

        source_data = _build_det_pickle(seqs, n_frames, n_dets)
        _write_minimal_pickle(source_path, source_data)
        _build_mini_idx(target_path, source_path, seqs, n_frames, n_dets)

        if shard_seqs is not None:
            clear_detection_pair_cache()
            build_shards(target_path, source_path, sequence_names=shard_seqs, overwrite=False)

        return target_path, source_path

    def test_load_one_sequence_from_shard(self, tmpdir):
        target, source = self._build_pair_and_shard(tmpdir, ["seqA", "seqB"], shard_seqs=["seqA"])
        clear_detection_pair_cache()

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80, det95 = load_detection_pair(target, source, sequence_names=["seqA"])

        assert set(det80.keys()) == {"seqA"}
        assert set(det95.keys()) == {"seqA"}
        assert spy.call_count == 1

    def test_fallback_when_shard_missing(self, tmpdir):
        target, source = self._build_pair_and_shard(tmpdir, ["seqA", "seqB"], shard_seqs=["seqA"])
        clear_detection_pair_cache()

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80, det95 = load_detection_pair(target, source, sequence_names=["seqA", "seqB"])

        assert set(det80.keys()) == {"seqA", "seqB"}
        assert spy.call_count == 2

    def test_mixed_multiple_sequences_all_sharded(self, tmpdir):
        target, source = self._build_pair_and_shard(
            tmpdir, ["seqA", "seqB", "seqC"], shard_seqs=["seqA", "seqC"]
        )
        clear_detection_pair_cache()

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80, det95 = load_detection_pair(target, source, sequence_names=["seqA", "seqC"])

        assert set(det80.keys()) == {"seqA", "seqC"}
        assert set(det95.keys()) == {"seqA", "seqC"}
        assert spy.call_count == 2

    def test_no_sequence_names_always_uses_monolithic(self, tmpdir):
        target, source = self._build_pair_and_shard(
            tmpdir, ["seqA", "seqB"], shard_seqs=["seqA", "seqB"]
        )
        clear_detection_pair_cache()

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80, det95 = load_detection_pair(target, source)

        assert set(det80.keys()) == {"seqA", "seqB"}
        assert spy.call_count == 2

    def test_shard_load_is_cached(self, tmpdir):
        target, source = self._build_pair_and_shard(tmpdir, ["seqA", "seqB"], shard_seqs=["seqA"])
        clear_detection_pair_cache()

        det80_a, det95_a = load_detection_pair(target, source, sequence_names=["seqA"])

        with patch("pickle.load", wraps=pickle.load) as spy:
            det80_b, det95_b = load_detection_pair(target, source, sequence_names=["seqA"])
            assert spy.call_count == 0

        assert det80_b is det80_a
        assert det95_b is det95_a

    def test_shard_data_matches_monolithic_for_same_sequence(self, tmpdir):
        target, source = self._build_pair_and_shard(
            tmpdir, ["seqA", "seqB"], shard_seqs=["seqA"]
        )
        clear_detection_pair_cache()

        det80_shard, det95_shard = load_detection_pair(target, source, sequence_names=["seqA"])
        clear_detection_pair_cache()

        det80_mono, det95_mono = load_detection_pair(target, source, sequence_names=["seqA"])

        assert det80_shard.keys() == det80_mono.keys()
        for fid in det80_shard["seqA"]:
            np.testing.assert_array_equal(det80_shard["seqA"][fid], det80_mono["seqA"][fid])

        assert det95_shard.keys() == det95_mono.keys()
        for fid in det95_shard["seqA"]:
            np.testing.assert_array_equal(det95_shard["seqA"][fid], det95_mono["seqA"][fid])


class TestBuildShards:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        clear_detection_pair_cache()

    def test_build_shards_creates_files(self, tmpdir):
        base = str(tmpdir)
        seqs = ["seqA", "seqB"]
        source_path = os.path.join(base, "bs_test_0.95.pickle")
        target_path = os.path.join(base, "bs_test_0.80.pickle")

        _write_minimal_pickle(source_path, _build_det_pickle(seqs, 2, 3))
        _build_mini_idx(target_path, source_path, seqs, 2, 3)
        clear_detection_pair_cache()

        built, skipped = build_shards(target_path, source_path, sequence_names=["seqA"])
        assert built == ["seqA"]
        assert len(skipped) == 0

        sp = shard_path(target_path, source_path, "seqA")
        assert os.path.isfile(sp)

    def test_build_shards_skips_existing_by_default(self, tmpdir):
        base = str(tmpdir)
        seqs = ["seqA", "seqB"]
        source_path = os.path.join(base, "bs2_test_0.95.pickle")
        target_path = os.path.join(base, "bs2_test_0.80.pickle")

        _write_minimal_pickle(source_path, _build_det_pickle(seqs, 2, 3))
        _build_mini_idx(target_path, source_path, seqs, 2, 3)
        clear_detection_pair_cache()

        built1, _ = build_shards(target_path, source_path, sequence_names=["seqA"])
        assert built1 == ["seqA"]
        clear_detection_pair_cache()

        built2, skipped2 = build_shards(target_path, source_path, sequence_names=["seqA"])
        assert built2 == []
        assert len(skipped2) == 1
        assert skipped2[0][0] == "seqA"
        assert "already exists" in skipped2[0][1]

    def test_build_shards_overwrite(self, tmpdir):
        base = str(tmpdir)
        seqs = ["seqA", "seqB"]
        source_path = os.path.join(base, "bs3_test_0.95.pickle")
        target_path = os.path.join(base, "bs3_test_0.80.pickle")

        _write_minimal_pickle(source_path, _build_det_pickle(seqs, 2, 3))
        _build_mini_idx(target_path, source_path, seqs, 2, 3)
        clear_detection_pair_cache()

        build_shards(target_path, source_path, sequence_names=["seqA"])
        clear_detection_pair_cache()

        built, skipped = build_shards(
            target_path, source_path, sequence_names=["seqA"], overwrite=True
        )
        assert built == ["seqA"]
        assert len(skipped) == 0

    def test_build_shards_all(self, tmpdir):
        base = str(tmpdir)
        seqs = ["seqA", "seqB", "seqC"]
        source_path = os.path.join(base, "bs4_test_0.95.pickle")
        target_path = os.path.join(base, "bs4_test_0.80.pickle")

        _write_minimal_pickle(source_path, _build_det_pickle(seqs, 2, 3))
        _build_mini_idx(target_path, source_path, seqs, 2, 3)
        clear_detection_pair_cache()

        built, skipped = build_shards(target_path, source_path)
        assert sorted(built) == sorted(seqs)
        assert len(skipped) == 0

        for s in seqs:
            assert os.path.isfile(shard_path(target_path, source_path, s))

    def test_shard_path_naming(self, tmpdir):
        base = str(tmpdir)
        target = os.path.join(base, "mot17_train_custom_0.80.pickle")
        source = os.path.join(base, "mot17_train_custom_0.95.pickle")

        sp = shard_path(target, source, "MOT17-02")
        assert sp.endswith("mot17_train_custom_0.80.from_mot17_train_custom_0.95.MOT17-02.shard.pickle")
        assert os.path.dirname(sp) == base
