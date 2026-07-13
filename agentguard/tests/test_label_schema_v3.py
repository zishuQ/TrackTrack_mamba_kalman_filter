from __future__ import annotations

import copy

import pytest

from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    make_cue_target,
    make_risk_targets,
    validate_rollout_label,
)


def _label() -> dict:
    motion = 0.2
    appearance = 0.8
    motion_confidence = 0.4
    appearance_confidence = 0.7
    return {
        "motion_soft_target": motion,
        "appearance_soft_target": appearance,
        "motion_safe_target": 0.6,
        "appearance_safe_target": 0.9,
        "motion_label_confidence": motion_confidence,
        "appearance_label_confidence": appearance_confidence,
        "cue_target": make_cue_target(motion_confidence, appearance_confidence),
        "policy_soft_target": [0.2] * 5,
        "policy_safe_soft_target": [0.2] * 5,
        "motion_benefit": -0.1,
        "appearance_benefit": 0.1,
        "valid_motion": True,
        "valid_appearance": True,
        "risk_targets": make_risk_targets(
            motion, appearance, motion_confidence, appearance_confidence
        ),
        "sample_weight": 1.0,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
    }


def test_rollout_label_v3_contract_accepts_complete_record():
    validate_rollout_label(_label())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label_schema_version", 2, "label_schema_version mismatch"),
        ("label_schema_sha256", "legacy", "label_schema_sha256 mismatch"),
        ("feature_schema_sha256", "legacy", "feature_schema_sha256 mismatch"),
        ("motion_soft_target", float("nan"), "non-finite"),
        ("motion_safe_target", 1.1, "must be in"),
        ("policy_soft_target", [0.1] * 5, "must sum to 1"),
        ("policy_safe_soft_target", [1.0, 0.0], "must have shape"),
        ("risk_targets", [0.0, 0.0, 0.0, float("inf")], "non-finite"),
    ],
)
def test_rollout_label_v3_contract_rejects_incompatible_values(field, value, message):
    label = copy.deepcopy(_label())
    label[field] = value
    with pytest.raises(ValueError, match=message):
        validate_rollout_label(label)


def test_rollout_label_v3_contract_rejects_missing_cue_or_risk():
    for field in ("cue_target", "risk_targets"):
        label = copy.deepcopy(_label())
        del label[field]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_rollout_label(label)
