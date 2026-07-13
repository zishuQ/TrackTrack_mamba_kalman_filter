from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)


ROLLOUT_LABEL_SCHEMA_VERSION = 3
ROLLOUT_LABEL_SCHEMA_DESCRIPTOR = {
    "name": "agentguard_rollout_labels",
    "version": ROLLOUT_LABEL_SCHEMA_VERSION,
    "benefit_semantics": "L_skip_minus_L_write",
    "base_gate_fields": ["motion_soft_target", "appearance_soft_target"],
    "final_gate_fields": ["motion_safe_target", "appearance_safe_target"],
    "cue_fields": [
        "motion_label_confidence",
        "appearance_label_confidence",
        "joint_label_confidence",
    ],
    "policy_dim": 5,
    "risk_fields": [
        "motion_harm",
        "appearance_harm",
        "insufficient_evidence",
        "cross_modal_conflict",
    ],
    "risk_dim": 4,
    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
}
ROLLOUT_LABEL_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        ROLLOUT_LABEL_SCHEMA_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


REQUIRED_ROLLOUT_LABEL_FIELDS = (
    "motion_soft_target",
    "appearance_soft_target",
    "motion_safe_target",
    "appearance_safe_target",
    "motion_label_confidence",
    "appearance_label_confidence",
    "cue_target",
    "policy_soft_target",
    "policy_safe_soft_target",
    "motion_benefit",
    "appearance_benefit",
    "valid_motion",
    "valid_appearance",
    "risk_targets",
    "sample_weight",
    "label_schema_version",
    "label_schema_sha256",
    "feature_schema_sha256",
    "cache_schema_version",
)


def make_cue_target(motion_confidence: float, appearance_confidence: float) -> list[float]:
    motion = float(motion_confidence)
    appearance = float(appearance_confidence)
    return [motion, appearance, max(motion, appearance)]


def make_risk_targets(
    motion_soft_target: float,
    appearance_soft_target: float,
    motion_confidence: float,
    appearance_confidence: float,
) -> list[float]:
    motion = float(motion_soft_target)
    appearance = float(appearance_soft_target)
    return [
        1.0 - motion,
        1.0 - appearance,
        1.0 - min(float(motion_confidence), float(appearance_confidence)),
        abs(motion - appearance),
    ]


def _finite_unit_interval(name: str, value: Any, shape: tuple[int, ...] = ()) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"{name} must be in [0, 1]")
    return array


def validate_rollout_label(label: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ROLLOUT_LABEL_FIELDS if field not in label]
    if missing:
        raise ValueError(f"rollout label v3 missing required fields: {missing}")

    if int(label["label_schema_version"]) != ROLLOUT_LABEL_SCHEMA_VERSION:
        raise ValueError(
            "label_schema_version mismatch: "
            f"{label['label_schema_version']} != {ROLLOUT_LABEL_SCHEMA_VERSION}"
        )
    if str(label["label_schema_sha256"]) != ROLLOUT_LABEL_SCHEMA_SHA256:
        raise ValueError(
            "label_schema_sha256 mismatch: "
            f"{label['label_schema_sha256']!r} != {ROLLOUT_LABEL_SCHEMA_SHA256!r}"
        )
    if int(label["cache_schema_version"]) != COMPACT_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "cache_schema_version mismatch: "
            f"{label['cache_schema_version']} != {COMPACT_CACHE_SCHEMA_VERSION}"
        )
    if str(label["feature_schema_sha256"]) != FEATURE_SCHEMA_SHA256:
        raise ValueError(
            "feature_schema_sha256 mismatch: "
            f"{label['feature_schema_sha256']!r} != {FEATURE_SCHEMA_SHA256!r}"
        )

    for field in (
        "motion_soft_target",
        "appearance_soft_target",
        "motion_safe_target",
        "appearance_safe_target",
        "motion_label_confidence",
        "appearance_label_confidence",
    ):
        _finite_unit_interval(field, label[field])
    _finite_unit_interval("cue_target", label["cue_target"], (3,))
    _finite_unit_interval("risk_targets", label["risk_targets"], (4,))

    for field in ("policy_soft_target", "policy_safe_soft_target"):
        policy = _finite_unit_interval(field, label[field], (5,))
        if not np.isclose(float(policy.sum()), 1.0, atol=1e-6):
            raise ValueError(f"{field} must sum to 1, got {float(policy.sum())}")

    for field in ("motion_benefit", "appearance_benefit", "sample_weight"):
        value = np.asarray(label[field], dtype=np.float64)
        if value.shape != () or not np.isfinite(value).all():
            raise ValueError(f"{field} must be a finite scalar")
    if float(label["sample_weight"]) < 0.0:
        raise ValueError("sample_weight must be non-negative")
