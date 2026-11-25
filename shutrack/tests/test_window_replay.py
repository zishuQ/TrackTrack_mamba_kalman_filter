"""Window replay correctness: no external data or training jobs required."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from shutrack.training.observation_noise import ObservationNoise, corrupt_observations
from shutrack.training.recurrent_motion import kf_update_gated, warp_predict
from shutrack.training.window_replay import window_index, WINDOW_IMPLEMENTATION
from train_window_groups import validate_resume_config

def observations(n=512, length=8):
    boxes = np.broadcast_to(np.array([640, 360, 100, 100], np.float32),
                            (n, length, 4)).copy()
    return boxes, np.full((n, length), .95, np.float32), np.ones((n, length), bool)

def corrupt(boxes, scores, present, rows, config, epoch=3, valid=None):
    return corrupt_observations(boxes, scores, present,
        np.ones_like(present) if valid is None else valid, rows,
        seed=42, epoch=epoch, width=1280, height=720, config=config)

def test_noise_is_epoch_dependent_and_batch_independent():
    boxes, scores, present = observations()
    rows = np.arange(len(boxes))
    cfg = ObservationNoise(missing=0)
    full = corrupt(boxes, scores, present, rows, cfg)[0]
    chunks = [corrupt(boxes[part], scores[part], present[part], rows[part], cfg)[0]
              for part in np.array_split(rows, 7)]
    np.testing.assert_array_equal(full, np.concatenate(chunks))
    assert not np.array_equal(full, corrupt(boxes, scores, present, rows, cfg, epoch=4)[0])
    np.testing.assert_allclose((full-boxes).std((0, 1)), np.ones(4), atol=.05)

def test_missing_burst_and_original_absence():
    boxes, scores, present = observations()
    present[:, 0] = False
    valid = np.ones_like(present)
    valid[:, :2] = False
    cfg = ObservationNoise(missing=0, burst=1, stay_missing=1)
    noisy, new_scores, keep = corrupt(boxes, scores, present,
                                      np.arange(len(boxes)), cfg, valid=valid)
    assert not keep[:, :2].any()
    assert keep[:, 2].all()
    assert not keep[:, 3:].any()
    assert not new_scores[~keep].any()
    assert not noisy[~keep].any()
    present[:, 2] = False
    _, _, keep = corrupt(boxes, scores, present, np.arange(len(boxes)), cfg, valid=valid)
    assert not keep[:, 2].any()

def test_mottrack_dataset_profiles():
    assert ObservationNoise.for_dataset("SportsMOT").missing == .004
    assert ObservationNoise.for_dataset("MOT17").burst == .02
    mot20 = ObservationNoise.for_dataset("MOT20")
    assert (mot20.position_std, mot20.size_std, mot20.small_area,
            mot20.boundary_margin) == (3, 2, 2500, 20)

def test_window_suffix_respects_missing_state_frame_gap_and_warp():
    warp = np.broadcast_to(np.array([[1, 0, 0], [0, 1, 0]], np.float32), (6, 2, 3)).copy()
    data = SimpleNamespace(
        small={"window_indices": np.array([[0, 1, 2, 3, 4], [-1, -1, 0, 1, 2]]),
               "current_event_index": np.array([5, 3])},
        event_frame=np.arange(6), event_target_row=np.arange(6),
        event_warp_next=warp)
    e, _, valid = window_index(data, 5)
    np.testing.assert_array_equal(e[0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(e[1, valid[1]], [0, 1, 2, 3])
    assert window_index(data, 5) is window_index(data, 5)
    data._window_indices_cache = {}
    data.event_warp_next[3] = 0
    e, _, valid = window_index(data, 5)
    np.testing.assert_array_equal(e[0, valid[0]], [4, 5])
    data._window_indices_cache = {}
    data.event_target_row[4] = -1
    e, _, valid = window_index(data, 5)
    np.testing.assert_array_equal(e[0, valid[0]], [5])

def test_kf_endpoints_and_future_gradients():
    mean = torch.tensor([[100., 80., 35., 70., 2., -1., .2, .1]], dtype=torch.float64)
    cov = torch.diag_embed(torch.arange(1, 9, dtype=torch.float64).unsqueeze(0))
    obs = mean[:, :4] + 2
    score = torch.tensor([.9], dtype=torch.float64)
    m0, p0 = kf_update_gated(mean, cov, obs, score, torch.zeros(1))
    m1, p1 = kf_update_gated(mean, cov, obs, score, torch.ones(1))
    torch.testing.assert_close(m0, mean)
    torch.testing.assert_close(p0, cov)
    assert not torch.equal(m1, m0)
    gate = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    mid, pmid = kf_update_gated(mean, cov, obs, score, gate)
    torch.testing.assert_close(mid, m0 + gate[:, None]*(m1-m0))
    torch.testing.assert_close(pmid, p0 + gate[:, None, None]*(p1-p0))
    warp = torch.tensor([[[.99, -.01, 1], [.01, .99, 2]]], dtype=torch.float64)
    later, _ = warp_predict(mid, pmid, warp)
    later.square().sum().backward()
    assert torch.isfinite(gate.grad).all() and gate.grad.abs().sum() > 0

def test_resume_rejects_changed_objective_and_allows_new_stop():
    original = {"window_implementation": WINDOW_IMPLEMENTATION, "epochs": 100,
                "window_frames": 5, "augment": "none", "stop_after_epoch": 1}
    validate_resume_config(original, dict(original, stop_after_epoch=None))
    with pytest.raises(ValueError, match="window_frames"):
        validate_resume_config(original, dict(original, window_frames=1))
    with pytest.raises(ValueError, match="epochs"):
        validate_resume_config(original, dict(original, epochs=20))
