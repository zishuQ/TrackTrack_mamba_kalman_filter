from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.contracts.events import TrackEvent
from agentguard.rollout.losses import sigmoid

_BENEFIT_EPS: float = 1e-3
_BENEFIT_CLIP_MAX: float = 0.1


def compute_dataset_stats(all_benefits: Dict[str, List[float]]) -> Dict[str, float]:
    tau_m = _compute_tau(all_benefits.get("motion_benefits", []))
    tau_a = _compute_tau(all_benefits.get("appearance_benefits", []))
    stats = {"tau_motion": tau_m, "tau_appearance": tau_a}

    dataset_name = os.environ.get("AG_GUARD_DATASET", "default")
    save_dir = Path("outputs") / "agentguard" / "labels" / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "dataset_stats.json"
    if not save_path.exists():
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def compute_soft_target(benefit: float, tau: float) -> float:
    return sigmoid(float(benefit) / max(float(tau), 1e-12))


compute_soft_targets = compute_soft_target


def compute_label_confidence(
    benefit: float,
    tau: float,
    coverage: float,
) -> float:
    magnitude = min(
        abs(float(benefit)) / max(2.0 * float(tau), 1e-12),
        1.0,
    )
    return float(np.clip(coverage, 0.0, 1.0) * magnitude)


def compute_safe_soft_target(
    benefit: float,
    tau: float,
    confidence: float,
    fallback: float = 1.0,
) -> float:
    oracle_soft = compute_soft_target(benefit, tau)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return float(confidence * oracle_soft + (1.0 - confidence) * float(fallback))


def compute_policy_soft_target(
    gate: np.ndarray,
    prototypes: Optional[np.ndarray] = None,
) -> np.ndarray:
    if prototypes is None:
        prototypes = POLICY_PROTOTYPE_MATRIX
    gate = np.asarray(gate, dtype=np.float64).reshape(1, 2)
    diffs = np.asarray(prototypes, dtype=np.float64) - gate
    logits = -np.sum(diffs ** 2, axis=1) / 0.1
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def hard_gate_from_benefit(benefit: float) -> int:
    return 1 if float(benefit) >= -1e-6 else 0


def build_rollout_labels(
    events: List[TrackEvent],
    motion_benefits: List[float],
    appearance_benefits: List[float],
    dataset_stats: Dict[str, float],
    identity_prototypes: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    tau_m = dataset_stats.get("tau_motion", 0.01)
    tau_a = dataset_stats.get("tau_appearance", 0.01)

    labels: List[Dict[str, Any]] = []
    for idx, event in enumerate(events):
        b_m = motion_benefits[idx] if idx < len(motion_benefits) else 0.0
        b_a = appearance_benefits[idx] if idx < len(appearance_benefits) else 0.0
        motion_target = compute_soft_target(b_m, tau_m)
        appearance_target = compute_soft_target(b_a, tau_a)
        motion_confidence = compute_label_confidence(b_m, tau_m, 1.0)
        appearance_confidence = compute_label_confidence(b_a, tau_a, 1.0)
        motion_safe = compute_safe_soft_target(b_m, tau_m, motion_confidence)
        appearance_safe = compute_safe_soft_target(
            b_a,
            tau_a,
            appearance_confidence,
        )
        gate = np.array([motion_target, appearance_target], dtype=np.float64)
        safe_gate = np.array([motion_safe, appearance_safe], dtype=np.float64)

        sample_type = "matched" if event.has_detection else "unmatched"
        labels.append(
            {
                "event_id": event.event_id,
                "track_id": event.track_id,
                "frame_id": event.frame_id,
                "sequence": event.sequence,
                "candidate_type": "A",
                "motion_benefit": float(b_m),
                "appearance_benefit": float(b_a),
                "motion_target": float(motion_target),
                "appearance_target": float(appearance_target),
                "motion_soft_target": float(motion_target),
                "appearance_soft_target": float(appearance_target),
                "motion_safe_target": float(motion_safe),
                "appearance_safe_target": float(appearance_safe),
                "motion_label_confidence": float(motion_confidence),
                "appearance_label_confidence": float(appearance_confidence),
                "motion_oracle_hard": hard_gate_from_benefit(b_m),
                "appearance_oracle_hard": hard_gate_from_benefit(b_a),
                "target_gate": [
                    hard_gate_from_benefit(b_m),
                    hard_gate_from_benefit(b_a),
                ],
                "policy_soft_target": compute_policy_soft_target(gate),
                "policy_safe_soft_target": compute_policy_soft_target(safe_gate),
                "label_schema_version": 2,
                "sample_type": sample_type,
                "valid_motion": True,
                "valid_appearance": True,
                "sample_weight": 1.0,
            }
        )
    return labels


def _compute_tau(benefits: List[float]) -> float:
    arr = np.asarray(benefits, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    non_zero = np.abs(finite[np.abs(finite) > 1e-12])
    if len(non_zero) == 0:
        return _BENEFIT_CLIP_MAX
    return float(np.clip(float(np.median(non_zero)), _BENEFIT_EPS, _BENEFIT_CLIP_MAX))
