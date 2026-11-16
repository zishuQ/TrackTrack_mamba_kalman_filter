"""Packed read-only dataset contract, labels and read-only guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.schema import MOTION_SCALAR_INDICES
from shutrack.tests.conftest import write_tiny_packed_dataset


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_packed_sample_contract(tiny_packed_dir: Path):
    with pytest.warns(RuntimeWarning):
        dataset = PackedShutrackDataset(tiny_packed_dir)
    try:
        assert len(dataset) == 4
        sample = dataset[0]
        assert sample["history_motion"].shape == (5, 12)
        assert sample["history_track"].shape == (5, 8)
        assert sample["history_detection"].shape == (5, 8)
        assert sample["history_context"].shape == (5, 8)
        assert sample["history_mask"].tolist() == [True, True, True, False, False]
        assert sample["current_has_detection"].item() is True
        assert sample["safe_gate_target"].shape == (2,)
        assert sample["valid_channels"].tolist() == [True, True]
        assert sample["sample_weight"].item() == pytest.approx(1.0)
        assert sample["margin_valid"].item() is False
        assert dataset.margin_source == "unavailable_packed_train_data"
        assert dataset.margin_valid_rate == 0.0

        batch = dataset.collate_fn([dataset[index] for index in range(3)])
        assert batch["history_motion"].shape == (3, 5, 12)
        assert batch["safe_gate_target"].shape == (3, 2)
        assert isinstance(batch["sequence"], list)
    finally:
        dataset.close()


def test_packed_scalars_are_selected_not_renormalized(tiny_packed_dir: Path):
    with pytest.warns(RuntimeWarning):
        dataset = PackedShutrackDataset(tiny_packed_dir)
    try:
        packed = torch.load(
            tiny_packed_dir / "seqA" / "data.pt", weights_only=False
        )["arrays"]
        sample = dataset[0]
        current_index = int(packed["event_indices"][0, -1])
        scalar = packed["timeline_scalar_feats"][current_index].numpy()
        expected_motion = [float(scalar[index]) for index in MOTION_SCALAR_INDICES]
        np.testing.assert_allclose(
            sample["current_motion"].numpy(), expected_motion, atol=1e-6
        )
        context = sample["current_context"].numpy()
        assert context[0] == pytest.approx(float(scalar[9]))
        assert context[3] == pytest.approx(float(scalar[58]))
        assert context[4] == pytest.approx(float(scalar[59]))
        assert context[5] == pytest.approx(float(scalar[26]))
        assert context[1] == 0.0 and context[2] == 0.0
        assert context[6] == 0.0 and context[7] == 0.0
    finally:
        dataset.close()


def test_margin_required_mode_reports_missing_fields(tiny_packed_dir: Path):
    with pytest.raises(ValueError, match="raw_cost"):
        PackedShutrackDataset(tiny_packed_dir, margin_mode="required")


def test_margin_disabled_mode_is_explicit_and_quiet(tiny_packed_dir: Path):
    dataset = PackedShutrackDataset(tiny_packed_dir, margin_mode="disabled")
    try:
        assert dataset.margin_source == "disabled"
    finally:
        dataset.close()


def test_dataset_reading_is_read_only(tiny_packed_dir: Path, monkeypatch):
    import agentguard.data.compact_iwg_labels as labels_module
    import agentguard.datasets.iwg_rg_cma_dataset as dataset_module

    def forbidden(*args, **kwargs):
        raise AssertionError("SHUTrack must not call cache/label/index builders")

    monkeypatch.setattr(
        dataset_module, "build_iwg_rg_cma_dataset", forbidden, raising=True
    )
    monkeypatch.setattr(
        labels_module,
        "build_current_rollout_compact_labels_for_sequence",
        forbidden,
        raising=True,
    )

    before = _snapshot(tiny_packed_dir)
    with pytest.warns(RuntimeWarning):
        dataset = PackedShutrackDataset(tiny_packed_dir, max_samples=0)
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            assert torch.isfinite(sample["current_motion"]).all()
    finally:
        dataset.close()
    after = _snapshot(tiny_packed_dir)
    assert before == after


def test_max_samples_truncates_view_only(tiny_packed_dir: Path):
    with pytest.warns(RuntimeWarning):
        dataset = PackedShutrackDataset(tiny_packed_dir, max_samples=2)
    try:
        assert len(dataset) == 2
        assert dataset.total_samples == 4
        assert dataset[1]["sequence"] == "seqA"
    finally:
        dataset.close()


def test_metadata_mismatch_is_rejected(tmp_path: Path):
    root = write_tiny_packed_dataset(tmp_path / "packed")
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["context_size"] = 8
    (root / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="context_size=6"):
        PackedShutrackDataset(root)
