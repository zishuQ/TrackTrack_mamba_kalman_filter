"""End-to-end tiny training smoke on a synthetic packed fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from shutrack.training.checkpoint import load_shutrack_checkpoint
from shutrack.training.train import evaluate_checkpoint, train_shutrack
from shutrack.tests.conftest import write_tiny_packed_dataset


def _base_config(tmp_path: Path, competition_enabled: bool) -> dict:
    return {
        "dataset_dir": str(write_tiny_packed_dataset(tmp_path / "packed")),
        "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
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
        "competition_enabled": competition_enabled,
        "d_model": 32,
        "n_heads": 4,
        "hce_layers": 2,
        "ff_dim": 64,
        "appearance_proj_dim": 16,
    }


@pytest.mark.parametrize("competition_enabled", [True, False])
def test_tiny_training_writes_only_last_and_best(tmp_path: Path, competition_enabled: bool):
    config = _base_config(tmp_path, competition_enabled)
    checkpoint_dir = Path(config["checkpoint_dir"])
    dataset_dir = Path(config["dataset_dir"])
    summary = train_shutrack(config)
    assert summary["global_step"] == 3
    assert summary["competition_enabled"] is competition_enabled
    assert summary["best_metric_name"] == "training_loss"
    assert summary["sequence_load_count"] >= 1
    metrics = [
        json.loads(line)
        for line in (checkpoint_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert metrics and metrics[0]["competition_enabled"] is competition_enabled
    assert metrics[0]["samples_per_second"] > 0
    assert metrics[0]["sequence_load_count"] >= 1
    assert metrics[0]["rss_mb"] > 0
    checkpoints = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    assert checkpoints == ["best.pt", "last.pt"]
    assert (checkpoint_dir / "training.log").is_file()
    assert (checkpoint_dir / "training_summary.json").is_file()

    model, checkpoint = load_shutrack_checkpoint(checkpoint_dir / "last.pt")
    assert checkpoint["architecture"] == "shutrack_hce_rsc"
    assert checkpoint["competition_enabled"] is competition_enabled
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["best_metric_name"] == "training_loss"
    assert model.parameter_report()["total"] > 0

    best_checkpoint = torch.load(checkpoint_dir / "best.pt", weights_only=False)
    last_checkpoint = torch.load(checkpoint_dir / "last.pt", weights_only=False)
    assert last_checkpoint["best_metric"] == pytest.approx(
        best_checkpoint["best_metric"]
    )
    assert last_checkpoint["best_metric"] == pytest.approx(metrics[0]["loss"])

    evaluation = evaluate_checkpoint(
        checkpoint_dir / "last.pt",
        dataset_dir,
        batch_size=2,
        max_batches=1,
        device="cpu",
    )
    assert evaluation["batches"] == 1
    assert evaluation["architecture"] == "shutrack_hce_rsc"
    assert evaluation["competition_enabled"] is competition_enabled

    with pytest.raises(FileExistsError):
        train_shutrack(config)


def test_resume_appends_logs_and_keeps_best_consistent(tmp_path: Path):
    config = _base_config(tmp_path, competition_enabled=True)
    config["max_steps"] = 2
    checkpoint_dir = Path(config["checkpoint_dir"])
    first = train_shutrack(config)
    assert first["global_step"] == 2
    first_metrics = (checkpoint_dir / "metrics.jsonl").read_text().splitlines()
    assert len(first_metrics) == 1
    first_best = torch.load(checkpoint_dir / "last.pt", weights_only=False)[
        "best_metric"
    ]

    resumed_config = dict(config)
    resumed_config["max_steps"] = 3
    resumed_config["resume"] = str(checkpoint_dir / "last.pt")
    second = train_shutrack(resumed_config)
    assert second["global_step"] == 3
    metrics = (checkpoint_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics) == 2
    log = (checkpoint_dir / "training.log").read_text()
    assert log.count("SHUTrack v2 (HCE + RSC) training") == 2
    assert "RESUME from" in log

    last_checkpoint = torch.load(checkpoint_dir / "last.pt", weights_only=False)
    best_checkpoint = torch.load(checkpoint_dir / "best.pt", weights_only=False)
    assert last_checkpoint["best_metric"] == pytest.approx(
        best_checkpoint["best_metric"]
    )
    assert last_checkpoint["best_metric"] <= first_best + 1e-12


def test_resume_contract_mismatch_is_rejected(tmp_path: Path):
    config = _base_config(tmp_path, competition_enabled=True)
    config["max_steps"] = 2
    checkpoint_dir = Path(config["checkpoint_dir"])
    train_shutrack(config)

    mismatched = dict(config)
    mismatched["resume"] = str(checkpoint_dir / "last.pt")
    mismatched["d_model"] = 64
    with pytest.raises(ValueError, match="resume contract mismatch"):
        train_shutrack(mismatched)

    branch_mismatch = dict(config)
    branch_mismatch["resume"] = str(checkpoint_dir / "last.pt")
    branch_mismatch["competition_enabled"] = False
    with pytest.raises(ValueError, match="resume contract mismatch"):
        train_shutrack(branch_mismatch)
