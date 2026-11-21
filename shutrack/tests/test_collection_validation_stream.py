"""Bounded validation regressions (tiny synthetic arrays only)."""
import sys
from pathlib import Path
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from paired_collection_io import atomic_npz, stream_nonfinite_rows, collection_lock


@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("masked", [False, True])
def test_stream_finite_matches_full_array(tmp_path, fortran, dtype, masked):
    values = np.ones((13, 9), dtype=dtype)
    values[1, 0] = np.nan
    values[1, 8] = np.inf
    values[9, 5] = -np.inf
    if fortran:
        values = np.asfortranarray(values)
    path = tmp_path / "data.npz"
    atomic_npz(path, {"rank": values}, budget=48)
    mask = np.arange(13) % 2 == 1 if masked else None
    bad = (~np.isfinite(values)).any(axis=1)
    expected = int(np.count_nonzero(bad if mask is None else bad & mask))
    assert stream_nonfinite_rows(path, "rank", mask, budget=48) == expected


def test_stream_finite_empty_and_bad_mask(tmp_path):
    path = tmp_path / "data.npz"
    atomic_npz(path, {"empty": np.zeros((0, 2048)), "values": np.ones((3, 2))})
    assert stream_nonfinite_rows(path, "empty", budget=32) == 0
    with pytest.raises(ValueError, match="mask length"):
        stream_nonfinite_rows(path, "values", np.ones(2, dtype=bool))


def test_collection_lock_rejects_duplicate_then_releases(tmp_path):
    with collection_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already active"):
            with collection_lock(tmp_path):
                pass
    with collection_lock(tmp_path):
        pass
