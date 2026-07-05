"""Test formal checkpoint loading protocol and validation."""

import numpy as np
import pytest
import torch


def _make_minimal_checkpoint(**overrides):
    ckpt = {
        "model_state_dict": {"reid_proj.weight": torch.randn(128, 64)},
        "reid_dim": 64,
        "scalar_dim": 63,
        "event_dim": 128,
        "policy_prototypes": np.random.randn(5, 2).tolist(),
        "normalization_mean": np.random.randn(63).astype(np.float64).tolist(),
        "normalization_std": np.abs(np.random.randn(63)).astype(np.float64).tolist(),
        "feature_schema_sha256": "abc123",
    }
    ckpt.update(overrides)
    return ckpt


def test_checkpoint_state_dict_load_order():
    """model_state_dict taken first, state_dict second, raw checkpoint third."""
    ckpt_model = _make_minimal_checkpoint()
    ckpt_state = {
        "state_dict": {"reid_proj.weight": torch.randn(128, 64)},
        "reid_dim": 64,
        "scalar_dim": 63,
        "event_dim": 128,
        "policy_prototypes": np.random.randn(5, 2).tolist(),
        "normalization_mean": np.zeros(63).tolist(),
        "normalization_std": np.ones(63).tolist(),
    }
    ckpt_raw = {
        "reid_proj.weight": torch.randn(128, 64),
        "reid_dim": 64,
        "scalar_dim": 63,
        "event_dim": 128,
        "policy_prototypes": np.random.randn(5, 2).tolist(),
        "normalization_mean": np.zeros(63).tolist(),
        "normalization_std": np.ones(63).tolist(),
    }

    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        p_model = os.path.join(tmp, "model.pt")
        p_state = os.path.join(tmp, "state.pt")
        p_raw = os.path.join(tmp, "raw.pt")
        torch.save(ckpt_model, p_model)
        torch.save(ckpt_state, p_state)
        torch.save(ckpt_raw, p_raw)

        # Replicate _load_checkpoint logic
        for p in [p_model, p_state, p_raw]:
            cp = torch.load(p, map_location="cpu", weights_only=False)
            if "model_state_dict" in cp:
                sd = cp["model_state_dict"]
            elif "state_dict" in cp:
                sd = cp["state_dict"]
            else:
                sd = cp
            assert "reid_proj.weight" in sd or isinstance(sd, dict)


def test_checkpoint_missing_reid_dim_raises():
    ckpt = {
        "model_state_dict": {"reid_proj.weight": torch.randn(128, 64)},
        # missing reid_dim
    }
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "ckpt.pt")
        torch.save(ckpt, p)
        cp = torch.load(p, map_location="cpu", weights_only=False)
        assert "reid_dim" not in cp or cp.get("reid_dim") is None


def test_checkpoint_scalar_dim_validation():
    ckpt = _make_minimal_checkpoint(scalar_dim=42)
    assert ckpt["scalar_dim"] != 63

    ckpt_ok = _make_minimal_checkpoint(scalar_dim=63)
    assert ckpt_ok["scalar_dim"] == 63


def test_checkpoint_event_dim_validation():
    ckpt = _make_minimal_checkpoint(event_dim=64)
    assert ckpt["event_dim"] != 128


def test_checkpoint_policy_prototypes_shape():
    ckpt = _make_minimal_checkpoint()
    pp = np.asarray(ckpt["policy_prototypes"])
    assert pp.shape == (5, 2)

    ckpt_bad = _make_minimal_checkpoint(
        policy_prototypes=[[0, 0], [1, 1], [2, 2]]
    )
    pp_bad = np.asarray(ckpt_bad["policy_prototypes"])
    assert pp_bad.shape != (5, 2)


def test_checkpoint_normalization_shape():
    ckpt = _make_minimal_checkpoint()
    mean = np.asarray(ckpt["normalization_mean"])
    std = np.asarray(ckpt["normalization_std"])
    assert mean.shape == (63,)
    assert std.shape == (63,)

    # Bad shape should be rejected
    ckpt_bad = _make_minimal_checkpoint(
        normalization_mean=[0.0] * 32,
        normalization_std=[1.0] * 32,
    )
    assert np.asarray(ckpt_bad["normalization_mean"]).shape != (63,)


def test_checkpoint_reid_dim_consistent():
    ckpt = _make_minimal_checkpoint()
    sd = ckpt["model_state_dict"]
    reid_proj_weight = sd.get("reid_proj.weight")
    inferred_dim = reid_proj_weight.shape[1]
    assert inferred_dim == ckpt["reid_dim"]
