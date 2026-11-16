"""Checkpoint schema, roundtrip and legacy rejection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from shutrack.models.shutrack import ShutrackV2Model
from shutrack.training.checkpoint import (
    checkpoint_summary,
    load_shutrack_checkpoint,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from shutrack.tests.conftest import make_model, write_tiny_checkpoint


def _batch(reid_dim: int = 8, batch_size: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        "history_motion": torch.randn(batch_size, 5, 12, generator=generator),
        "history_track": torch.randn(batch_size, 5, reid_dim, generator=generator),
        "history_detection": torch.randn(batch_size, 5, reid_dim, generator=generator),
        "history_context": torch.randn(batch_size, 5, 8, generator=generator),
        "history_mask": torch.zeros(batch_size, 5, dtype=torch.bool),
        "history_has_detection": torch.ones(batch_size, 5, dtype=torch.bool),
        "current_motion": torch.randn(batch_size, 12, generator=generator),
        "current_track": torch.randn(batch_size, reid_dim, generator=generator),
        "current_detection": torch.randn(batch_size, reid_dim, generator=generator),
        "current_context": torch.randn(batch_size, 8, generator=generator),
        "current_has_detection": torch.ones(batch_size, dtype=torch.bool),
        "history_reset": torch.zeros(batch_size, 5, dtype=torch.bool),
    }


def test_checkpoint_roundtrip_is_bitwise_equivalent(tmp_path: Path):
    path = tmp_path / "best.pt"
    write_tiny_checkpoint(path)
    model, checkpoint = load_shutrack_checkpoint(path)
    assert isinstance(model, ShutrackV2Model)
    model.eval()
    original = make_model()
    original.load_state_dict(checkpoint["model_state_dict"], strict=True)
    original.eval()
    batch = _batch()
    with torch.no_grad():
        expected = original(**batch)["gates"]
        actual = model(**batch)["gates"]
    torch.testing.assert_close(expected, actual, atol=0.0, rtol=0.0)
    summary = checkpoint_summary(checkpoint)
    assert summary["architecture"] == "shutrack_hce_rsc"
    assert summary["margin"]["source"] == "unavailable_packed_train_data"


def test_checkpoint_rejects_wrong_architecture(tmp_path: Path):
    path = tmp_path / "best.pt"
    write_tiny_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    payload["architecture"] = "iwg_rg_cma"
    tampered = tmp_path / "tampered.pt"
    save_checkpoint_atomic(tampered, payload)
    with pytest.raises(ValueError, match="not a SHUTrack v2 checkpoint"):
        load_shutrack_checkpoint(tampered)


def test_checkpoint_rejects_legacy_iwg_payload(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_schema": "legacy",
            "correction_bound": 0.05,
            "policy_prototypes": np.zeros((5, 2)),
            "model_state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError) as error:
        load_shutrack_checkpoint(path)
    message = str(error.value)
    assert "legacy IWG/RG-CMA" in message
    assert "strict=False" in message


def test_checkpoint_rejects_feature_schema_mismatch(tmp_path: Path):
    path = tmp_path / "best.pt"
    write_tiny_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    payload["feature_schema_sha256"] = "0" * 64
    tampered = tmp_path / "tampered_schema.pt"
    save_checkpoint_atomic(tampered, payload)
    with pytest.raises(ValueError, match="feature schema hash mismatch"):
        load_shutrack_checkpoint(tampered)


def test_validate_checkpoint_missing_fields(tmp_path: Path):
    payload = {"architecture": "shutrack_hce_rsc"}
    with pytest.raises(ValueError, match="architecture_version"):
        validate_checkpoint(payload)
