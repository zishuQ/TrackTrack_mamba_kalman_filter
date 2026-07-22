from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import ROLLOUT_LABEL_SCHEMA_SHA256
from agentguard.datasets.iwg_attn_dataset import (
    IWG_ATTN_DATASET_SCHEMA_SHA256,
    MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256,
    SPORTSMOT_TRAINVAL_IWG_ATTN_DATASET_SCHEMA_SHA256,
)
from agentguard.models.iwg_rg_cma import (
    IWG_RG_CMA_LEGACY_MODEL_SCHEMA,
    IWG_RG_CMA_LEGACY_MODEL_SCHEMA_SHA256,
    IWG_RG_CMA_MODEL_SCHEMA,
    IWG_RG_CMA_MODEL_SCHEMA_SHA256,
    IWGRGCMA,
    RG_CMA_CORRECTION_BOUND,
    RG_CMA_LEGACY_CORRECTION_BOUND,
)
from agentguard.training.train_iwg_rg_cma import (
    FORMAL_CHECKPOINT_EPOCHS,
    _phase_dataset,
    _run_training_attempt,
    _validate_formal_config,
    SequenceSqrtSampler,
    build_memory_shard_schedule,
    initialize_iwg_rg_cma_model,
    load_iwg_rg_cma_checkpoint,
    validate_iwg_rg_cma_checkpoint_contract,
)


def _checkpoint() -> dict:
    model = IWGRGCMA(16)
    return {
        "model_schema": IWG_RG_CMA_MODEL_SCHEMA,
        "model_schema_sha256": IWG_RG_CMA_MODEL_SCHEMA_SHA256,
        "model_state_dict": model.state_dict(),
        "epoch": 100,
        "reid_dim": 16,
        "scalar_dim": 63,
        "event_dim": 128,
        "context_size": 6,
        "max_frame_gap": 30,
        "correction_bound": RG_CMA_CORRECTION_BOUND,
        "policy_prototypes": np.asarray(POLICY_PROTOTYPE_MATRIX).tolist(),
        "dataset_schema_sha256": IWG_ATTN_DATASET_SCHEMA_SHA256,
        "dataset_sha256": "dataset-hash",
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_mean": np.zeros(63),
        "normalization_std": np.ones(63),
        "training_commit": "commit",
        "training_seed": 42,
        "tracker_seed": 10000,
        "dataset": "MOT20",
    }


def test_iwg_rg_cma_checkpoint_strict_roundtrip(tmp_path):
    checkpoint = _checkpoint()
    validate_iwg_rg_cma_checkpoint_contract(
        checkpoint, expected_dataset_sha256="dataset-hash"
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    model, loaded = load_iwg_rg_cma_checkpoint(path)
    assert isinstance(model, IWGRGCMA)
    assert loaded["epoch"] == 100
    assert model.reliability_mode == "full"

    no_scalar = dict(checkpoint)
    no_scalar["reliability_mode"] = "no-scalar"
    no_scalar_path = tmp_path / "no_scalar_checkpoint.pt"
    torch.save(no_scalar, no_scalar_path)
    no_scalar_loaded, _ = load_iwg_rg_cma_checkpoint(no_scalar_path)
    assert no_scalar_loaded.reliability_mode == "no-scalar"

    invalid = dict(checkpoint)
    invalid["correction_bound"] = 0.2
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_iwg_rg_cma_checkpoint_contract(invalid)

    legacy = dict(checkpoint)
    legacy_model = IWGRGCMA(16, correction_bound=RG_CMA_LEGACY_CORRECTION_BOUND)
    legacy.update(
        {
            "model_schema": IWG_RG_CMA_LEGACY_MODEL_SCHEMA,
            "model_schema_sha256": IWG_RG_CMA_LEGACY_MODEL_SCHEMA_SHA256,
            "model_state_dict": legacy_model.state_dict(),
            "correction_bound": RG_CMA_LEGACY_CORRECTION_BOUND,
        }
    )
    validate_iwg_rg_cma_checkpoint_contract(
        legacy, expected_dataset_sha256="dataset-hash"
    )
    legacy_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(legacy, legacy_path)
    legacy_loaded, _ = load_iwg_rg_cma_checkpoint(legacy_path)
    assert legacy_loaded.correction_bound == RG_CMA_LEGACY_CORRECTION_BOUND

    mot20 = dict(checkpoint)
    mot20["dataset_schema_sha256"] = MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256
    validate_iwg_rg_cma_checkpoint_contract(mot20)

    sportsmot_trainval = dict(checkpoint)
    sportsmot_trainval["dataset_schema_sha256"] = (
        SPORTSMOT_TRAINVAL_IWG_ATTN_DATASET_SCHEMA_SHA256
    )
    validate_iwg_rg_cma_checkpoint_contract(sportsmot_trainval)

    unsupported = dict(checkpoint)
    unsupported["dataset_schema_sha256"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported dataset_schema_sha256"):
        validate_iwg_rg_cma_checkpoint_contract(unsupported)


def test_iwg_rg_cma_formal_batch_size_is_explicit():
    base = {
        "seed": 42,
        "epochs": 100,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "amp": False,
    }
    assert _validate_formal_config({**base, "batch_size": 1024}) == 1024
    assert _validate_formal_config({**base, "batch_size": 2048}) == 2048
    with pytest.raises(ValueError, match="batch_size must be one of"):
        _validate_formal_config({**base, "batch_size": 1536})

    sharded = {
        **base,
        "batch_size": 1024,
        "memory_shards": 5,
        "epochs_per_shard": 10,
        "shard_cycles": 2,
    }
    assert _validate_formal_config(sharded) == 1024
    with pytest.raises(ValueError, match="must equal epochs"):
        _validate_formal_config({**sharded, "epochs_per_shard": 9})

    extended = {
        **base,
        "epochs": 200,
        "batch_size": 1024,
        "memory_shards": 10,
        "epochs_per_shard": 10,
        "shard_cycles": 2,
    }
    assert _validate_formal_config(extended) == 1024
    with pytest.raises(ValueError, match="epochs must be one of"):
        _validate_formal_config({**extended, "epochs": 300, "shard_cycles": 3})


def test_extended_training_has_fixed_checkpoint_cadence():
    assert FORMAL_CHECKPOINT_EPOCHS[100] == {25, 50, 75, 100}
    assert FORMAL_CHECKPOINT_EPOCHS[200] == {50, 100, 150, 200}


def test_iwg_rg_cma_warm_start_has_fixed_finetune_config():
    config = {
        "seed": 42,
        "epochs": 25,
        "num_workers": 4,
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "amp": False,
        "batch_size": 1024,
        "init_checkpoint": "/tmp/parent.pt",
        "memory_shards": 1,
        "epochs_per_shard": 25,
        "shard_cycles": 1,
    }
    assert _validate_formal_config(config) == 1024
    fifty_epoch = {
        **config,
        "epochs": 50,
        "epochs_per_shard": 50,
    }
    assert _validate_formal_config(fifty_epoch) == 1024
    with pytest.raises(ValueError, match="epochs must be one of"):
        _validate_formal_config({**config, "epochs": 100, "epochs_per_shard": 100})
    with pytest.raises(ValueError, match="lr=1e-05"):
        _validate_formal_config({**config, "lr": 1e-4})
    with pytest.raises(ValueError, match="one full-data phase"):
        _validate_formal_config(
            {
                **config,
                "memory_shards": 5,
                "epochs_per_shard": 5,
            }
        )
    with pytest.raises(ValueError, match="unsupported reliability_mode"):
        _validate_formal_config({**config, "reliability_mode": "gate-policy"})


def test_iwg_rg_cma_warm_start_loads_only_strict_model_weights(tmp_path):
    checkpoint = _checkpoint()
    path = tmp_path / "mot20_epoch050.pt"
    torch.save(checkpoint, path)
    model = IWGRGCMA(16)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    provenance = initialize_iwg_rg_cma_model(
        model,
        path,
        dataset_metadata={"reid_dim": 16, "scalar_dim": 63, "event_dim": 128},
    )
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(model.state_dict()[key], value)
    assert provenance["mode"] == "warm_start"
    assert provenance["source_dataset"] == "MOT20"
    assert provenance["source_epoch"] == 100
    assert provenance["optimizer_restored"] is False
    assert provenance["scheduler_restored"] is False
    assert len(provenance["checkpoint_sha256"]) == 64


def test_mot20_memory_shards_cover_each_sequence_once_per_cycle():
    class DatasetStub:
        metadata = {
            "dataset": "MOT20",
            "index_format": "compact_memmap_v1",
            "train_sequences": ["MOT20-A", "MOT20-B"],
            "timeline_counts": {
                "MOT20-A": {"labeled_endpoints": 11},
                "MOT20-B": {"labeled_endpoints": 7},
            },
        }

        def __len__(self):
            return 18

    dataset = DatasetStub()
    config = {
        "epochs": 100,
        "memory_shards": 5,
        "epochs_per_shard": 10,
        "shard_cycles": 2,
    }
    schedule = build_memory_shard_schedule(dataset, config)
    assert schedule["nominal_epochs"] == 100
    assert schedule["effective_full_epochs"] == 20
    assert len(schedule["phases"]) == 10

    for cycle in (1, 2):
        indices = []
        for phase in schedule["phases"]:
            if phase["cycle"] != cycle:
                continue
            indices.extend(list(_phase_dataset(list(range(18)), phase)))
        assert sorted(indices) == list(range(18))
        assert len(indices) == len(set(indices))


def test_randomized_shard_order_is_seeded_and_preserves_each_cycle():
    class DatasetStub:
        metadata = {
            "dataset": "MOT20",
            "index_format": "compact_memmap_v1",
            "train_sequences": ["MOT20-A", "MOT20-B"],
            "timeline_counts": {
                "MOT20-A": {"labeled_endpoints": 11},
                "MOT20-B": {"labeled_endpoints": 7},
            },
        }

        def __len__(self):
            return 18

    config = {
        "epochs": 12,
        "memory_shards": 3,
        "epochs_per_shard": 1,
        "shard_cycles": 4,
        "seed": 42,
        "randomize_shard_order": True,
    }
    schedule = build_memory_shard_schedule(DatasetStub(), config)
    repeated = build_memory_shard_schedule(DatasetStub(), config)
    fixed = build_memory_shard_schedule(
        DatasetStub(), {**config, "randomize_shard_order": False}
    )

    assert schedule["randomize_shard_order"] is True
    assert schedule["shard_order_per_cycle"] == repeated[
        "shard_order_per_cycle"
    ]
    assert schedule["shard_order_per_cycle"] != fixed["shard_order_per_cycle"]
    assert all(
        sorted(order) == [1, 2, 3]
        for order in schedule["shard_order_per_cycle"]
    )

    for cycle in range(1, 5):
        phases = [
            phase for phase in schedule["phases"] if phase["cycle"] == cycle
        ]
        assert [phase["shard"] for phase in phases] == schedule[
            "shard_order_per_cycle"
        ][cycle - 1]
        indices = []
        for phase in phases:
            indices.extend(list(_phase_dataset(list(range(18)), phase)))
        assert sorted(indices) == list(range(18))
        assert len(indices) == len(set(indices))


def test_sqrt_size_sampling_preserves_phase_budget_and_uses_exact_quotas():
    class DatasetStub:
        metadata = {
            "dataset": "MOT20",
            "index_format": "compact_memmap_v1",
            "train_sequences": ["MOT20-A", "MOT20-B", "MOT20-C"],
            "timeline_counts": {
                "MOT20-A": {"labeled_endpoints": 2},
                "MOT20-B": {"labeled_endpoints": 8},
                "MOT20-C": {"labeled_endpoints": 50},
            },
        }

        def __len__(self):
            return 60

    dataset = DatasetStub()
    sqrt_schedule = build_memory_shard_schedule(
        dataset,
        {
            "epochs": 2,
            "memory_shards": 2,
            "epochs_per_shard": 1,
            "shard_cycles": 1,
            "sequence_sampling": "sqrt-size",
            "seed": 42,
        },
    )
    proportional_schedule = build_memory_shard_schedule(
        dataset,
        {
            "epochs": 2,
            "memory_shards": 2,
            "epochs_per_shard": 1,
            "shard_cycles": 1,
            "sequence_sampling": "sample-proportional",
            "seed": 42,
        },
    )

    assert sqrt_schedule["sampling_policy"] == (
        "sequence_sqrt_size_with_replacement"
    )
    assert sqrt_schedule["sequence_sampling_replacement"] is True
    assert sqrt_schedule["sequence_sampling_weights"]["MOT20-A"] < (
        sqrt_schedule["sequence_sampling_weights"]["MOT20-B"]
        < sqrt_schedule["sequence_sampling_weights"]["MOT20-C"]
    )
    assert [phase["samples"] for phase in sqrt_schedule["phases"]] == [
        phase["samples"] for phase in proportional_schedule["phases"]
    ]
    assert sum(
        (int(phase["samples"]) + 3) // 4
        for phase in sqrt_schedule["phases"]
    ) == sum(
        (int(phase["samples"]) + 3) // 4
        for phase in proportional_schedule["phases"]
    )

    phase = sqrt_schedule["phases"][0]
    sampler = SequenceSqrtSampler(
        phase["ranges"],
        phase["sequence_sampling_target_counts"],
        num_samples=int(phase["samples"]),
        seed=42,
    )
    indices = list(sampler)
    assert len(indices) == int(phase["samples"])
    observed = {sequence: 0 for sequence in phase["sequence_sampling_target_counts"]}
    for index in indices:
        offset = 0
        for item in phase["ranges"]:
            end = offset + int(item["samples"])
            if offset <= index < end:
                observed[str(item["sequence"])] += 1
                break
            offset = end
        else:
            raise AssertionError(f"sampler produced an out-of-range index: {index}")
    assert observed == phase["sequence_sampling_target_counts"]
    assert len(set(indices)) < len(indices)


def test_low_memory_training_switches_phases_without_resetting_progress(tmp_path):
    class TinyDataset(torch.utils.data.Dataset):
        metadata = {
            "dataset": "MOT20",
            "split": "all",
            "train_sequences": ["MOT20-A", "MOT20-B"],
            "timeline_counts": {
                "MOT20-A": {"labeled_endpoints": 4},
                "MOT20-B": {"labeled_endpoints": 4},
            },
            "reid_dim": 16,
            "scalar_dim": 63,
            "event_dim": 128,
            "context_size": 6,
            "max_frame_gap": 30,
            "index_format": "compact_memmap_v1",
            "dataset_schema_sha256": MOT20_IWG_ATTN_DATASET_SCHEMA_SHA256,
            "dataset_sha256": "tiny-dataset",
        }

        def __init__(self):
            self.norm_stats = type(
                "Norm", (), {"mean": np.zeros(63), "std": np.ones(63)}
            )()
            self.release_calls = 0

        def __len__(self):
            return 8

        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index)
            padding = torch.zeros(6, dtype=torch.bool)
            return {
                "track_feats": torch.randn(6, 16, generator=generator),
                "det_feats": torch.randn(6, 16, generator=generator),
                "scalar_feats": torch.randn(6, 63, generator=generator),
                "padding_mask": padding,
                "has_detection_mask": ~padding,
                "reset_mask": torch.tensor([True, False, False, False, False, False]),
                "safe_gate_target": torch.tensor([0.25, 0.75]),
                "policy_safe_soft_target": torch.full((5,), 0.2),
                "cue_target": torch.full((3,), 0.5),
                "risk_target": torch.full((4,), 0.5),
                "valid_motion": torch.tensor(True),
                "valid_appearance": torch.tensor(True),
                "sample_weight": torch.tensor(1.0),
            }

        def release_cached_pages(self):
            self.release_calls += 1
            return {"mmap_regions": 0, "files": 0}

    dataset = TinyDataset()
    config = {
        "seed": 42,
        "device": "cpu",
        "epochs": 2,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "grad_clip": 1.0,
        "amp": False,
        "sequence_sampling": "sqrt-size",
        "checkpoint_dir": str(tmp_path),
        "memory_shards": 2,
        "epochs_per_shard": 1,
        "shard_cycles": 1,
    }
    summary = _run_training_attempt(config, dataset=dataset, batch_size=4)
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    checkpoint = torch.load(
        tmp_path / "iwg_rg_cma_last.pt", map_location="cpu", weights_only=False
    )
    assert [item["memory_shard"] for item in metrics] == [1, 2]
    assert [item["optimizer_steps"] for item in metrics] == [1, 2]
    assert summary["effective_full_epochs"] == pytest.approx(1.0)
    assert checkpoint["training_progress"]["global_step"] == 2
    assert checkpoint["training_schedule"]["memory_shards"] == 2
    assert dataset.release_calls == 3


def test_dancetrack_memory_shards_cover_each_sequence_once_per_cycle():
    class DatasetStub:
        metadata = {
            "dataset": "DanceTrack",
            "index_format": "compact_memmap_v1",
            "train_sequences": ["dancetrack0001", "dancetrack0002"],
            "timeline_counts": {
                "dancetrack0001": {"labeled_endpoints": 13},
                "dancetrack0002": {"labeled_endpoints": 8},
            },
        }

        def __len__(self):
            return 21

    schedule = build_memory_shard_schedule(
        DatasetStub(),
        {
            "epochs": 100,
            "memory_shards": 5,
            "epochs_per_shard": 10,
            "shard_cycles": 2,
        },
    )
    assert schedule["effective_full_epochs"] == 20
    for cycle in (1, 2):
        indices = []
        for phase in schedule["phases"]:
            if phase["cycle"] == cycle:
                indices.extend(list(_phase_dataset(list(range(21)), phase)))
        assert sorted(indices) == list(range(21))
        assert len(indices) == len(set(indices))


def test_interleaved_schedule_switches_shards_every_epoch():
    class DatasetStub:
        metadata = {
            "dataset": "SportsMOT",
            "index_format": "compact_memmap_v1",
            "train_sequences": ["sports-a", "sports-b"],
            "timeline_counts": {
                "sports-a": {"labeled_endpoints": 10},
                "sports-b": {"labeled_endpoints": 5},
            },
        }

        def __len__(self):
            return 15

    schedule = build_memory_shard_schedule(
        DatasetStub(),
        {
            "epochs": 100,
            "memory_shards": 5,
            "epochs_per_shard": 1,
            "shard_cycles": 20,
        },
    )
    assert schedule["effective_full_epochs"] == 20
    assert [phase["shard"] for phase in schedule["phases"][:10]] == [
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5
    ]
    assert all(phase["epochs"] == 1 for phase in schedule["phases"])
