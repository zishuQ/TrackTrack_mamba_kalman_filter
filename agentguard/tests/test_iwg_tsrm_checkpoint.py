from __future__ import annotations

import copy

import numpy as np
import pytest

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
)
from agentguard.datasets.joint_window_dataset import JOINT_DATASET_SCHEMA_SHA256
from agentguard.training.train_iwg_tsrm import (
    JOINT_MODEL_SCHEMA,
    validate_checkpoint_contract,
)


def _checkpoint() -> dict:
    return {
        "training_mode": "joint",
        "model_schema_version": JOINT_MODEL_SCHEMA,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "joint_dataset_schema_sha256": JOINT_DATASET_SCHEMA_SHA256,
        "normalization_mean": np.zeros(63),
        "normalization_std": np.ones(63),
        "policy_prototypes": np.asarray(POLICY_PROTOTYPE_MATRIX),
        "training_commit": "test",
        "dataset_metadata_sha256": "test",
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "scaler_state_dict": {},
        "window_size": 16,
        "iwg_warmup_events": 5,
        "iwg_input_size": 21,
        "delta_max": 0.2,
        "temporal_iwg_gradient_scale": 0.0,
        "reid_dim": 16,
        "scalar_dim": 63,
        "event_dim": 128,
    }


def test_v4_combined_checkpoint_contract_accepts_gradient_scale():
    validate_checkpoint_contract(_checkpoint(), expected_training_mode="joint")


def test_v3_combined_checkpoint_is_strictly_rejected():
    checkpoint = copy.deepcopy(_checkpoint())
    checkpoint["model_schema_version"] = "agentguard_iwg_tsrm_v3"
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_checkpoint_contract(checkpoint, expected_training_mode="joint")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("iwg_warmup_events", 0, "must equal 5"),
        ("iwg_input_size", 16, r"window_size \+ 5"),
        ("window_size", 0, "must be positive"),
        ("temporal_iwg_gradient_scale", -0.1, r"must be in \[0, 1\]"),
    ],
)
def test_checkpoint_rejects_incompatible_window_contract(field, value, message):
    checkpoint = copy.deepcopy(_checkpoint())
    checkpoint[field] = value
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_contract(checkpoint, expected_training_mode="joint")
