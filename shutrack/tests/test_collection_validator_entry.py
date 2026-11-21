"""Verify the real validator CLI without loading large rank members."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import verify_final_data as validator
from paired_collection_io import atomic_npz, atomic_json


@pytest.mark.parametrize("bad_rank", [False, True])
def test_validator_streams_rank_and_rejects_nonfinite(tmp_path, monkeypatch, bad_rank):
    directory = tmp_path / "data/MOT20-01"
    directory.mkdir(parents=True)
    n = 3
    arrays = {k: np.zeros(n, dtype=np.float32) for k in (
        "motion_benefit", "appearance_benefit", "appearance_delta", "column_offset",
        "column_rival", "column_score")}
    arrays.update(frame=np.arange(1, 4), track_id=np.ones(n, dtype=np.int64),
                  detection_index=np.arange(n), gt_id=np.ones(n, dtype=np.int64),
                  current_event_index=np.arange(n), window_indices=np.zeros((n, 5), dtype=np.int64),
                  current_competition=np.zeros((n, 3)), motion_gt_valid=np.ones(n, dtype=bool),
                  valid_motion=np.ones(n, dtype=bool), valid_appearance=np.ones(n, dtype=bool),
                  appearance_diag_valid=np.ones(n, dtype=bool),
                  rank_valid=np.array([True, True, False]), rank_rows=np.array([0, 1]),
                  rank_score=np.ones(2), pred_box_g0=np.zeros((n, 4)),
                  pred_box_g1=np.zeros((n, 4)), gt_box_t1=np.zeros((n, 4)),
                  column_det=np.array([1, 2, -1]))
    rank_names = {"rank_template0", "rank_detection", "rank_positive", "rank_competitor"}
    for name in rank_names:
        arrays[name] = np.ones((2, 8), dtype=np.float16)
    if bad_rank:
        arrays["rank_positive"][1, 3] = np.nan
    keys = [{"sequence": "MOT20-01", "frame": i + 1, "track_id": 1, "detection_index": i}
            for i in range(n)]
    arrays["keys_json"] = np.asarray(json.dumps(keys))
    atomic_npz(directory / "samples.npz", arrays)
    for name, value in {
        "event_evidence": np.zeros((n, 13)), "event_track": np.zeros((n, 8)),
        "event_detection_index": np.arange(n), "event_has_detection": np.ones(n, dtype=bool)
    }.items():
        np.save(directory / (name + ".npy"), value)
    atomic_json(directory / "summary.json", {
        "complete": True, "samples": n, "events": n, "column_counts": {"actual_mismatch": 0},
        "protocol": {"schema": 3, "full_write": True, "rank_arrays": True,
                     "column": True, "dataset": "MOT20"}})

    class Cache:
        def __init__(self, path):
            self.num_detections = n
            self.frame_offsets = np.arange(n + 1)
        def close(self):
            pass
    monkeypatch.setattr(validator, "SequenceDetectionCache", Cache)
    # Fails if the validator accidentally reintroduces full rank loading.
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(self, name):
        assert name not in rank_names, "rank feature was materialized in RAM"
        return original(self, name)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["verify_final_data", "--data-root", str(directory.parent),
                                    "--dataset", "MOT20", "--split", "train",
                                    "--sequences", "MOT20-01", "--expect-full-write",
                                    "--require-column", "--out", str(output)])
    if bad_rank:
        with pytest.raises(SystemExit) as exc:
            validator.main()
        assert exc.value.code == 1
    else:
        validator.main()
    report = json.loads(output.read_text())
    assert report["data_ok"] is (not bad_rank)
    assert report["finite_bad"] == int(bad_rank)
    assert report["index_mismatch"] == report["column_mismatch"] == 0
