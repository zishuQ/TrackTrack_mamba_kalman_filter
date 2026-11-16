"""Packed read-only dataset contract, labels and read-only guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.schema import COMPETITION_SCALAR_INDICES, EVIDENCE_SCALAR_INDICES
from shutrack.tests.conftest import write_tiny_packed_dataset


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_packed_sample_contract(tiny_packed_dir: Path):
    dataset = PackedShutrackDataset(tiny_packed_dir)
    try:
        assert len(dataset) == 4
        sample = dataset[0]
        assert sample["history_evidence"].shape == (5, 13)
        assert sample["history_track"].shape == (5, 8)
        assert sample["history_detection"].shape == (5, 8)
        assert sample["current_evidence"].shape == (13,)
        assert sample["current_competition"].shape == (3,)
        assert sample["history_mask"].tolist() == [True, True, True, False, False]
        assert sample["current_has_detection"].item() is True
        assert sample["safe_gate_target"].shape == (2,)
        assert sample["valid_channels"].tolist() == [True, True]
        assert sample["sample_weight"].item() == pytest.approx(1.0)
        assert dataset.competition_enabled is True

        batch = dataset.collate_fn([dataset[index] for index in range(3)])
        assert batch["history_evidence"].shape == (3, 5, 13)
        assert batch["current_competition"].shape == (3, 3)
        assert batch["safe_gate_target"].shape == (3, 2)
        assert isinstance(batch["sequence"], list)
    finally:
        dataset.close()


def test_packed_values_are_selected_from_normalized_scalars(tiny_packed_dir: Path):
    dataset = PackedShutrackDataset(tiny_packed_dir)
    try:
        packed = torch.load(
            tiny_packed_dir / "seqA" / "data.pt", weights_only=False
        )["arrays"]
        sample = dataset[0]
        current_index = int(packed["event_indices"][0, -1])
        scalar = packed["timeline_scalar_feats"][current_index].numpy()
        expected_evidence = [float(scalar[index]) for index in EVIDENCE_SCALAR_INDICES] + [1.0]
        np.testing.assert_allclose(
            sample["current_evidence"].numpy(), expected_evidence, atol=1e-6
        )
        expected_competition = [
            float(scalar[index]) for index in COMPETITION_SCALAR_INDICES
        ]
        np.testing.assert_allclose(
            sample["current_competition"].numpy(), expected_competition, atol=1e-6
        )
    finally:
        dataset.close()


def test_competition_disabled_zeros_the_branch(tmp_path: Path):
    root = write_tiny_packed_dataset(tmp_path / "packed")
    dataset = PackedShutrackDataset(root, competition_enabled=False)
    try:
        sample = dataset[0]
        assert sample["current_competition"].abs().sum().item() == 0.0
        assert dataset.competition_enabled is False
        assert dataset.descriptor()["competition_enabled"] is False
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
    dataset = PackedShutrackDataset(tiny_packed_dir, max_samples=0)
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            assert torch.isfinite(sample["current_evidence"]).all()
    finally:
        dataset.close()
    after = _snapshot(tiny_packed_dir)
    assert before == after


def test_max_samples_truncates_view_only(tiny_packed_dir: Path):
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
