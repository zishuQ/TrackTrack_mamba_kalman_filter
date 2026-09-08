from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agentguard.cli import (
    parse_train_iwg_rg_cma_command,
    validate_train_iwg_rg_cma_command,
)
from agentguard.datasets.iwg_rg_cma_dataset import (
    COMPACT_INDEX_FORMAT,
    PACKED_TRAIN_DATA_FORMAT,
    PACKED_TRAIN_DATA_V2_INDEX_FORMAT,
    StreamingIWGRGCMADataset,
    identify_packed_train_data,
    inspect_packed_train_data,
)
from agentguard.training.train_iwg_rg_cma import (
    numbered_checkpoint_epochs,
    record_checkpoint_plan,
    smoke_train_iwg_rg_cma,
)


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "agentguard"


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_packed_v1(root: Path, *, sequence: str = "MOT17-09-FRCNN") -> Path:
    sequence_dir = root / sequence
    sequence_dir.mkdir(parents=True)
    arrays = {
        "event_indices": torch.tensor(
            [[-1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, 0, 1]],
            dtype=torch.int32,
        ),
        "track_ids": torch.tensor([3, 3], dtype=torch.int64),
        "segment_ids": torch.tensor([4, 4], dtype=torch.int64),
        "safe_gate_target": torch.zeros((2, 2)),
        "policy_safe_soft_target": torch.full((2, 5), 0.2),
        "cue_target": torch.zeros((2, 3)),
        "risk_target": torch.zeros((2, 4)),
        "valid_channels": torch.ones((2, 2), dtype=torch.bool),
        "sample_weight": torch.ones(2),
        "timeline_scalar_feats": torch.zeros((2, 63)),
        "timeline_has_detection": torch.tensor([True, False]),
    }
    torch.save({"metadata": {}, "arrays": arrays}, sequence_dir / "data.pt")
    reid = np.zeros((2, 2, 4), dtype=np.float32)
    reid[0, 0] = 1.0
    np.save(sequence_dir / "reid_features.npy", reid, allow_pickle=False)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": PACKED_TRAIN_DATA_FORMAT,
                "index_format": PACKED_TRAIN_DATA_FORMAT,
                "dataset": "MOT17",
                "split": "all",
                "context_size": 6,
                "reid_dim": 4,
                "scalar_dim": 63,
                "event_dim": 128,
                "max_frame_gap": 30,
                "train_sequences": [sequence],
                "num_train_samples": 2,
                "timeline_counts": {
                    sequence: {
                        "events": 2,
                        "labeled_endpoints": 2,
                    }
                },
                "normalization_mean": [0.0] * 63,
                "normalization_std": [1.0] * 63,
            }
        )
    )
    return root


def _minimal_train_command(
    dataset_dir: Path,
    checkpoint_dir: Path,
    extra: list[str] | None = None,
) -> list[str]:
    command = [
        "python",
        "-m",
        "agentguard.cli",
        "train_iwg_rg_cma",
        "--dataset-dir",
        str(dataset_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--device",
        "cuda",
        "--epochs",
        "100",
        "--batch-size",
        "1024",
        "--num-workers",
        "4",
        "--lr",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--warmup-epochs",
        "1",
        "--grad-clip",
        "1.0",
        "--seed",
        "42",
    ]
    if extra:
        command.extend(extra)
    return command


def test_identify_packed_variants_from_metadata_not_directory_name(tmp_path):
    v1_in_v2_name = tmp_path / "train_data_v2" / "MOT17"
    v1_in_v2_name.mkdir(parents=True)
    (v1_in_v2_name / "metadata.json").write_text(
        json.dumps({"format": "train_data", "index_format": "train_data"})
    )
    assert identify_packed_train_data(
        json.loads((v1_in_v2_name / "metadata.json").read_text())
    ) == PACKED_TRAIN_DATA_FORMAT

    v2_in_v1_name = tmp_path / "train_data" / "MOT17"
    v2_in_v1_name.mkdir(parents=True)
    (v2_in_v1_name / "metadata.json").write_text(
        json.dumps(
            {
                "format": "train_data",
                "schema_version": 2,
                "index_format": "train_data_v2",
            }
        )
    )
    assert identify_packed_train_data(
        json.loads((v2_in_v1_name / "metadata.json").read_text())
    ) == PACKED_TRAIN_DATA_V2_INDEX_FORMAT

    with pytest.raises(ValueError, match="schema_version=2"):
        identify_packed_train_data(
            {"format": "train_data", "schema_version": 2, "index_format": "train_data"}
        )


def test_inspect_packed_train_data_reports_missing_and_incompatible(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        inspect_packed_train_data(missing)

    compact = tmp_path / "compact"
    compact.mkdir()
    (compact / "metadata.json").write_text(
        json.dumps({"format": COMPACT_INDEX_FORMAT, "train_sequences": ["seq"]})
    )
    with pytest.raises(ValueError, match="compact_memmap_v1"):
        inspect_packed_train_data(compact)

    legacy_name = tmp_path / "legacy_name"
    legacy_name.mkdir()
    (legacy_name / "metadata.json").write_text(
        json.dumps({"format": "agentguard_train_data_v1"})
    )
    with pytest.raises(ValueError, match="agentguard_train_data_v1"):
        inspect_packed_train_data(legacy_name)

    incomplete = tmp_path / "incomplete"
    _write_packed_v1(incomplete)
    (incomplete / "MOT17-09-FRCNN" / "reid_features.npy").unlink()
    with pytest.raises(ValueError, match="reid_features.npy"):
        inspect_packed_train_data(incomplete)


def test_inspect_v2_requires_detection_reference(tmp_path):
    root = tmp_path / "v2"
    sequence = "MOT17-09-FRCNN"
    sequence_dir = root / sequence
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "data.pt").write_bytes(b"not-a-real-archive")
    np.save(sequence_dir / "reid_features.npy", np.zeros((2, 4), dtype=np.float32))
    (sequence_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "train_data",
                "schema_version": 2,
                "index_format": "train_data_v2",
                "dataset": "MOT17",
                "sequence": sequence,
                "references": {
                    "detection_cache": str(tmp_path / "missing-det"),
                    "detection_fingerprint": {
                        "manifest.json": {"sha256": "abc", "bytes": 1},
                    },
                },
            }
        )
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "train_data",
                "schema_version": 2,
                "index_format": "train_data_v2",
                "train_sequences": [sequence],
            }
        )
    )
    with pytest.raises(ValueError, match="detection cache"):
        inspect_packed_train_data(root)
    report = inspect_packed_train_data(root, check_detection=False)
    assert report["variant"] == PACKED_TRAIN_DATA_V2_INDEX_FORMAT
    assert report["schema_version"] == 2


def test_inspect_accepts_tiny_packed_v1(tmp_path):
    root = _write_packed_v1(tmp_path / "v1")
    report = inspect_packed_train_data(root)
    assert report["variant"] == PACKED_TRAIN_DATA_FORMAT
    assert report["schema_version"] == 1
    dataset = StreamingIWGRGCMADataset(root)
    try:
        assert dataset.packed_variant == PACKED_TRAIN_DATA_FORMAT
        assert len(dataset) == 2
    finally:
        dataset.close()


def test_checkpoint_every_cli_and_selected_epoch_65(tmp_path):
    dataset_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "ckpts"
    dataset_dir.mkdir()
    checkpoint_dir.mkdir()
    default = parse_train_iwg_rg_cma_command(
        _minimal_train_command(dataset_dir, checkpoint_dir)
    )
    assert default["checkpoint_every"] is None
    assert 65 not in numbered_checkpoint_epochs(default["epochs"])

    every_five = validate_train_iwg_rg_cma_command(
        _minimal_train_command(
            dataset_dir, checkpoint_dir, ["--checkpoint-every", "5"]
        ),
        selected_epoch=65,
    )
    assert every_five["checkpoint_every"] == 5
    assert 65 in every_five["checkpoint_epochs"]
    assert every_five["checkpoint_epochs"][0] == 5
    assert every_five["checkpoint_epochs"][-1] == 100

    with pytest.raises(ValueError, match="positive integer"):
        parse_train_iwg_rg_cma_command(
            _minimal_train_command(
                dataset_dir, checkpoint_dir, ["--checkpoint-every", "0"]
            )
        )
    with pytest.raises(ValueError, match="unrecognized arguments"):
        parse_train_iwg_rg_cma_command(
            _minimal_train_command(
                dataset_dir, checkpoint_dir, ["--freeze-iwg"]
            )
        )
    with pytest.raises(ValueError, match="invalid choice"):
        parse_train_iwg_rg_cma_command(
            _minimal_train_command(
                dataset_dir,
                checkpoint_dir,
                ["--architecture-variant", "without-cma"],
            )
        )


def test_diagnostic_and_log_interval_cli(tmp_path):
    dataset_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "ckpts"
    dataset_dir.mkdir()
    checkpoint_dir.mkdir()
    default = parse_train_iwg_rg_cma_command(
        _minimal_train_command(dataset_dir, checkpoint_dir)
    )
    assert default["diagnostic_interval"] == 0
    assert default["log_interval"] == 0
    restored = parse_train_iwg_rg_cma_command(
        _minimal_train_command(
            dataset_dir,
            checkpoint_dir,
            ["--diagnostic-interval", "1", "--log-interval", "20"],
        )
    )
    assert restored["diagnostic_interval"] == 1
    assert restored["log_interval"] == 20
    with pytest.raises(ValueError, match="non-negative"):
        parse_train_iwg_rg_cma_command(
            _minimal_train_command(
                dataset_dir,
                checkpoint_dir,
                ["--diagnostic-interval", "-1"],
            )
        )


def test_smoke_train_writes_extra_numbered_checkpoints_once(tmp_path, monkeypatch):
    dataset_dir = _write_packed_v1(tmp_path / "data")
    checkpoint_dir = tmp_path / "ckpts"
    saved: list[str] = []
    real_save = torch.save

    def tracking_save(obj, path, *args, **kwargs):
        saved.append(str(Path(path).name))
        return real_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", tracking_save)
    torch.set_num_threads(1)
    summary = smoke_train_iwg_rg_cma(
        {
            "dataset_dir": str(dataset_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "device": "cpu",
            "epochs": 4,
            "batch_size": 2,
            "num_workers": 0,
            "context_size": 6,
            "correction_bound": 0.05,
            "checkpoint_every": 2,
            "seed": 42,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
        }
    )
    assert summary["checkpoint_every"] == 2
    assert summary["checkpoint_epochs"] == [2, 4]
    assert saved.count("iwg_rg_cma_last.pt") == 4
    assert saved.count("iwg_rg_cma_epoch002.pt") == 1
    assert saved.count("iwg_rg_cma_epoch004.pt") == 1
    assert "iwg_rg_cma_epoch001.pt" not in saved
    assert "iwg_rg_cma_epoch003.pt" not in saved
    payload = torch.load(checkpoint_dir / "iwg_rg_cma_last.pt", weights_only=False)
    assert payload["training_config"]["checkpoint_every"] == 2
    assert payload["training_config"]["checkpoint_epochs"] == [2, 4]


def test_default_smoke_cadence_still_saves_final_numbered_epoch(tmp_path):
    dataset_dir = _write_packed_v1(tmp_path / "data")
    checkpoint_dir = tmp_path / "ckpts"
    torch.set_num_threads(1)
    smoke_train_iwg_rg_cma(
        {
            "dataset_dir": str(dataset_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "context_size": 6,
            "correction_bound": 0.05,
        }
    )
    assert (checkpoint_dir / "iwg_rg_cma_last.pt").is_file()
    assert (checkpoint_dir / "iwg_rg_cma_epoch001.pt").is_file()


def test_100e_shell_uses_shared_packed_inspector():
    text = (SCRIPTS / "run_iwg_rg_cma_100e.sh").read_text()
    assert "inspect_packed_train_data" in text
    assert "agentguard_train_data_v1" not in text
    assert "--checkpoint-every" not in text
    assert "outputs/agentguard/train_data/MOT17" in text


def test_finetune_shell_no_longer_requires_compact_memmap():
    text = (SCRIPTS / "run_iwg_rg_cma_mot20_to_mot17_finetune25.sh").read_text()
    assert "inspect_packed_train_data" in text
    assert "compact_memmap_v1" not in text
    assert "--epochs 25" in text
    assert "--checkpoint-every" not in text


def test_cross_dataset_command_keeps_epoch_65_and_checkpoint_every(tmp_path):
    module = _load_script("run_cross_dataset_generalization.py")
    experiment = {
        "dataset_dir": tmp_path / "data",
        "train_split": "all",
    }
    command = module._training_command(experiment, tmp_path / "ckpts")
    config = validate_train_iwg_rg_cma_command(command, selected_epoch=65)
    assert config["checkpoint_every"] == 5
    assert 65 in config["checkpoint_epochs"]
    assert config["architecture_variant"] == "legacy"
    assert config["epochs"] == 100


def test_without_cma_command_is_rejected_without_remapping(tmp_path):
    module = _load_script("run_mot17_without_cma_ablation.py")
    command = module._training_command(tmp_path / "data", tmp_path / "ckpts")
    assert "--architecture-variant" in command
    assert "without-cma" in command
    assert "--checkpoint-every" in command
    with pytest.raises(ValueError, match="without-cma"):
        validate_train_iwg_rg_cma_command(command, selected_epoch=65)
    planned = numbered_checkpoint_epochs(100, checkpoint_every=5)
    assert 65 in planned


def test_ablation_suite_commands_keep_epoch_65_and_report_conflicts(tmp_path):
    module = _load_script("run_mot17_ablation_suite.py")
    context8 = {
        "dataset_dir": tmp_path / "data",
        "architecture_variant": "legacy",
        "context_size": 8,
        "init_checkpoint": "",
        "freeze_iwg": False,
    }
    command = module._training_command(context8, tmp_path / "ckpts")
    config = validate_train_iwg_rg_cma_command(command, selected_epoch=65)
    assert config["context_size"] == 8
    assert 65 in config["checkpoint_epochs"]

    policy = {
        "dataset_dir": tmp_path / "data",
        "architecture_variant": "iwg-policy-only",
        "context_size": 6,
        "init_checkpoint": "",
        "freeze_iwg": False,
    }
    conflicts = module._spec_conflicts(policy)
    assert conflicts
    assert any("iwg-policy-only" in item for item in conflicts)

    cma = {
        "dataset_dir": tmp_path / "data",
        "architecture_variant": "cma-without-reliability-token",
        "context_size": 6,
        "init_checkpoint": tmp_path / "baseline.pt",
        "freeze_iwg": True,
    }
    conflicts = module._spec_conflicts(cma)
    assert any("freeze-iwg" in item for item in conflicts)


def test_record_checkpoint_plan_mutates_training_config():
    config = {"epochs": 100, "init_checkpoint": ""}
    epochs = record_checkpoint_plan(config)
    assert config["checkpoint_every"] is None
    assert list(epochs) == [25, 50, 75, 100]
    config["checkpoint_every"] = 5
    epochs = record_checkpoint_plan(config)
    assert 65 in epochs
    assert config["checkpoint_epochs"][-1] == 100
