"""End-to-end tiny training smoke on a synthetic packed fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shutrack.training.checkpoint import load_shutrack_checkpoint
from shutrack.training.train import evaluate_checkpoint, train_shutrack
from shutrack.tests.conftest import write_tiny_packed_dataset


def test_tiny_training_writes_only_last_and_best(tmp_path: Path):
    dataset_dir = write_tiny_packed_dataset(tmp_path / "packed")
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    config = {
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "device": "cpu",
        "epochs": 2,
        "batch_size": 2,
        "num_workers": 0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        "warmup_steps": 1,
        "grad_clip": 1.0,
        "seed": 42,
        "max_steps": 3,
        "max_samples": 4,
        "log_interval": 1,
        "margin_mode": "auto",
        "d_model": 32,
        "n_heads": 4,
        "hce_layers": 2,
        "ff_dim": 64,
        "appearance_proj_dim": 16,
    }
    with pytest.warns(RuntimeWarning):
        summary = train_shutrack(config)
    assert summary["global_step"] == 3
    assert summary["margin"]["source"] == "unavailable_packed_train_data"
    assert summary["margin"]["valid_rate"] == 0.0
    checkpoints = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    assert checkpoints == ["best.pt", "last.pt"]
    assert (checkpoint_dir / "metrics.jsonl").is_file()
    assert (checkpoint_dir / "training.log").is_file()
    assert (checkpoint_dir / "training_summary.json").is_file()

    metrics = [
        json.loads(line)
        for line in (checkpoint_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert metrics and all(metric["margin_valid_rate"] == 0.0 for metric in metrics)

    model, checkpoint = load_shutrack_checkpoint(checkpoint_dir / "last.pt")
    assert checkpoint["architecture"] == "shutrack_hce_rsc"
    assert checkpoint["optimizer_state_dict"]
    assert model.parameter_report()["total"] > 0

    evaluation = evaluate_checkpoint(
        checkpoint_dir / "last.pt",
        dataset_dir,
        batch_size=2,
        max_batches=1,
        device="cpu",
    )
    assert evaluation["batches"] == 1
    assert evaluation["architecture"] == "shutrack_hce_rsc"

    with pytest.raises(FileExistsError):
        train_shutrack(config)


def test_required_margin_mode_blocks_training(tmp_path: Path):
    dataset_dir = write_tiny_packed_dataset(tmp_path / "packed")
    with pytest.raises(ValueError, match="raw_cost"):
        train_shutrack(
            {
                "dataset_dir": str(dataset_dir),
                "checkpoint_dir": str(tmp_path / "run"),
                "device": "cpu",
                "margin_mode": "required",
                "max_steps": 1,
                "batch_size": 1,
                "max_samples": 1,
            }
        )
