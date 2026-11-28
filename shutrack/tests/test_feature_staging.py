"""Feature staging must preserve exact batch values, dtypes and sample order."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from train_supervision_groups import batch_tensors
from shutrack.training import feature_staging
from shutrack.training.feature_staging import (
    SelectedFeatureRows, aligned_stage_chunk_samples, feature_chunks, staged_sequence_rows,
)
from shutrack.training.paired_pipeline import prepared_batches


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("group", ["appearance_rank", "appearance_row_column"])
def test_staged_batch_matches_mmap_with_padding_duplicates_and_column(tmp_path, dtype, group):
    rng = np.random.default_rng(41)
    def mapped(name, shape, kind=dtype):
        path = tmp_path / (name + ".npy")
        np.save(path, rng.normal(size=shape).astype(kind))
        return np.load(path, mmap_mode="r")
    seq = SimpleNamespace(
        event_track=mapped("track", (12, 8)),
        event_evidence=mapped("evidence", (12, 13), np.float32),
        event_det=np.array([-1, 2, 4, 6, -1, 10, 1, 3, 5, 8, 7, 9]),
        event_has=np.array([False, True, True, True, False, True, True, True, True, True, True, True]),
        cache=SimpleNamespace(features=mapped("detection", (15, 8))),
        samples={},
        rank_pos=np.array([-1, 0, 1, -1, 2, 3]),
    )
    seq._rank_arrays = tuple(mapped("rank" + str(i), (4, 8), np.float16) for i in range(4)) + (
        np.array([.8, .7, .9, .6], dtype=np.float32),)
    seq.load_rank_arrays = lambda: None
    seq.small = {
        "window_indices": np.array([[-1,-1,-1,-1,-1],[-1,-1,1,2,3],
                                    [1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]]),
        "current_event_index": np.array([1,4,6,7,9,11]),
        "current_competition": rng.normal(size=(6,3)).astype(np.float32),
        "pred_box_g0": rng.normal(size=(6,4)).astype(np.float32),
        "pred_box_g1": rng.normal(size=(6,4)).astype(np.float32),
        "gt_box_t1": rng.normal(size=(6,4)).astype(np.float32),
        "motion_gt_valid": np.array([True,False,True,True,False,True]),
        "column_det": np.array([14,-1,13,12,-1,14]),
    }
    for key in ("column_offset", "column_rival", "column_score"):
        seq.small[key] = rng.uniform(size=6).astype(np.float32)
    targets = {name: rng.uniform(size=6).astype(np.float32)
               for name in ("motion_safe", "appearance_safe", "appearance_new")}
    targets.update(valid_motion=np.ones(6,dtype=bool),valid_appearance=np.ones(6,dtype=bool))
    rows = np.array([5,0,2,5,1])
    originals = (seq.event_track, seq.cache.features, seq._rank_arrays)
    expected = batch_tensors(seq, rows, targets, group, compact_reid=True)
    with staged_sequence_rows(seq, rows, group) as stats:
        assert stats["bytes"] > 0
        for selected in (rows, rows[:2], rows[2:]):
            actual = batch_tensors(seq, selected, targets, group, compact_reid=True)
            reference = expected if len(selected) == len(rows) else None
            if reference is None:
                positions = slice(0,2) if len(selected) == 2 else slice(2,None)
                reference = {key: value[positions] for key,value in expected.items()}
            assert actual.keys() == reference.keys()
            for key in actual:
                assert actual[key].dtype == reference[key].dtype, key
                np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    assert seq.event_track is originals[0]
    assert seq.cache.features is originals[1]
    assert seq._rank_arrays is originals[2]


def test_selected_rows_restore_order_and_reject_unstaged_rows():
    source = np.arange(80,dtype=np.float32).reshape(10,8)
    staged = SelectedFeatureRows(source, [8,2,8,5], block_bytes=32)
    requested = np.array([[8,5],[2,8]])
    np.testing.assert_array_equal(staged[requested], source[requested])
    with pytest.raises(IndexError, match="not included"):
        staged[np.array([1])]

def test_selected_rows_discard_clean_file_pages_only_when_requested(tmp_path, monkeypatch):
    path = tmp_path / "features.npy"
    values = np.arange(4096, dtype=np.float32).reshape(128, 32)
    np.save(path, values)
    source = np.load(path, mmap_mode="r")
    calls = []
    advice = 4
    monkeypatch.setattr(feature_staging.os, "POSIX_FADV_DONTNEED", advice, raising=False)
    monkeypatch.setattr(
        feature_staging.os, "posix_fadvise",
        lambda fd, offset, size, flag: calls.append((offset, size, flag)), raising=False,
    )
    rows = np.array([15, 1, 15])
    retained = SelectedFeatureRows(source, rows)
    assert calls == []
    discarded = SelectedFeatureRows(source, rows, discard_cache=True)
    assert calls == [(source.offset, source.nbytes, advice)]
    np.testing.assert_array_equal(discarded[rows], retained[rows])


def test_staging_reads_npz_member_offsets_and_sliced_mmap_views(tmp_path):
    from shutrack.training.paired_pipeline import npz_array_view
    source = np.arange(120, dtype=np.float32).reshape(15, 8)
    path = tmp_path / "features.npz"
    np.savez(path, padding=np.zeros(135), features=source)
    with np.load(path, allow_pickle=False) as archive:
        view = npz_array_view(archive, "features")
        for array in (np.asarray(view), view[3:12]):
            selected = np.array([4, 1, 4, 5])
            staged = SelectedFeatureRows(array, selected, block_bytes=256)
            np.testing.assert_array_equal(staged[selected], array[selected])


@pytest.mark.parametrize("chunk_samples", [2, 7, 12])
def test_chunking_preserves_batch_boundaries_and_duplicate_rows(chunk_samples):
    rows = np.array([14, 3, 7, 3, 9, 2, 1, 2, 11, 2, 6, 3, 8, 9, 6])
    batch_size = 4
    aligned = aligned_stage_chunk_samples(chunk_samples, batch_size)
    assert aligned >= batch_size and aligned % batch_size == 0

    def batches_for(enabled):
        actual_rows = []
        for chunk in feature_chunks(rows, aligned, enabled=enabled):
            with prepared_batches(
                chunk, batch_size, lambda selected: {}, (), prefetch=1
            ) as batches:
                actual_rows.extend(row_ids.tolist() for row_ids, _ in batches)
        return actual_rows

    expected = [rows[i : i + batch_size].tolist() for i in range(0, len(rows), batch_size)]
    assert batches_for(True) == batches_for(False) == expected
