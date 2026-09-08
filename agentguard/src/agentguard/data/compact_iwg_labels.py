from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    validate_rollout_label,
)


COMPACT_IWG_LABEL_SCHEMA_V1_DESCRIPTOR = {
    "name": "agentguard_compact_iwg_safe_labels",
    "version": 1,
    "source": "current_timeline_candidate_a_rollout_v3",
    "join_policy": "native_event_shard_and_offset",
    "target_label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
}
COMPACT_IWG_LABEL_SCHEMA_V1_SHA256 = hashlib.sha256(
    json.dumps(
        COMPACT_IWG_LABEL_SCHEMA_V1_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

COMPACT_IWG_LABEL_SCHEMA_VERSION = 2
COMPACT_IWG_LABEL_SCHEMA_DESCRIPTOR = {
    **COMPACT_IWG_LABEL_SCHEMA_V1_DESCRIPTOR,
    "version": COMPACT_IWG_LABEL_SCHEMA_VERSION,
    "target_fields": [
        "safe_gate_target",
        "oracle_gate_target",
        "gate_confidence",
    ],
}
COMPACT_IWG_LABEL_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        COMPACT_IWG_LABEL_SCHEMA_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
SUPPORTED_COMPACT_IWG_LABEL_SCHEMA_SHA256 = frozenset(
    {COMPACT_IWG_LABEL_SCHEMA_V1_SHA256, COMPACT_IWG_LABEL_SCHEMA_SHA256}
)

COMPACT_LABEL_ARRAY_FILES_V1 = {
    "event_keys": "event_keys.npy",
    "safe_gate_target": "safe_gate_target.npy",
    "policy_safe_soft_target": "policy_safe_soft_target.npy",
    "cue_target": "cue_target.npy",
    "risk_target": "risk_target.npy",
    "valid_channels": "valid_channels.npy",
    "sample_weight": "sample_weight.npy",
}
COMPACT_LABEL_ARRAY_FILES = {
    **COMPACT_LABEL_ARRAY_FILES_V1,
    "oracle_gate_target": "oracle_gate_target.npy",
    "gate_confidence": "gate_confidence.npy",
}
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_arrays(
    output_dir: Path,
    *,
    event_keys: np.ndarray,
    safe_gate: np.ndarray,
    oracle_gate: np.ndarray,
    gate_confidence: np.ndarray,
    policy: np.ndarray,
    cue: np.ndarray,
    risk: np.ndarray,
    valid: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, str]:
    values = {
        "event_keys": event_keys,
        "safe_gate_target": safe_gate,
        "oracle_gate_target": oracle_gate,
        "gate_confidence": gate_confidence,
        "policy_safe_soft_target": policy,
        "cue_target": cue,
        "risk_target": risk,
        "valid_channels": valid,
        "sample_weight": sample_weight,
    }
    hashes: dict[str, str] = {}
    for name, value in values.items():
        path = output_dir / COMPACT_LABEL_ARRAY_FILES[name]
        np.save(path, value, allow_pickle=False)
        hashes[path.name] = _sha256_file(path)
    return hashes


def build_current_rollout_compact_labels_for_sequence(
    *,
    sequence: str,
    event_cache_dir: str | Path,
    detection_cache_dir: str | Path,
    gt_root: str | Path,
    output_root: str | Path,
    future_frames: int = 5,
    max_events: int = 0,
    motion_normalization: str = "pred",
    prototype_mode: str = "standard",
) -> dict[str, Any]:
    from agentguard.data.rollout_label_builder import (
        build_compact_rollout_labels_for_sequence,
    )

    if motion_normalization not in {"pred", "gt"}:
        raise ValueError(f"unknown motion normalization: {motion_normalization}")
    if prototype_mode not in {"standard", "leave_one_out"}:
        raise ValueError(f"unknown prototype mode: {prototype_mode}")
    generation_rules = {
        "motion_normalization": motion_normalization,
        "prototype_mode": prototype_mode,
    }
    event_cache_dir = Path(event_cache_dir).resolve()
    detection_cache_dir = Path(detection_cache_dir).resolve()
    gt_root = Path(gt_root).resolve()
    output_root = Path(output_root).resolve()
    final_dir = output_root / sequence
    cache_manifest_path = event_cache_dir / "manifest.json"
    detection_manifest_path = detection_cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    detection_manifest = json.loads(detection_manifest_path.read_text())
    dataset = str(detection_manifest.get("dataset", ""))
    if not dataset or str(cache_manifest.get("dataset", "")) != dataset:
        raise ValueError(
            "event and detection cache manifests must name the same dataset"
        )
    cache_manifest_sha256 = _sha256_file(cache_manifest_path)
    detection_manifest_sha256 = _sha256_file(detection_manifest_path)
    compact_schema_version = COMPACT_IWG_LABEL_SCHEMA_VERSION
    compact_schema_sha256 = COMPACT_IWG_LABEL_SCHEMA_SHA256
    motion_target_mode = "nsa"
    motion_label_mode = "nsa_rollout"
    manifest_path = final_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("complete")
            and existing.get("compact_label_schema_sha256")
            == compact_schema_sha256
            and existing.get("event_cache_manifest_sha256")
            == cache_manifest_sha256
            and existing.get("detection_cache_manifest_sha256")
            == detection_manifest_sha256
            and int(existing.get("future_frames", -1)) == int(future_frames)
            and existing.get("motion_label_mode", "nsa_rollout")
            == motion_label_mode
            and int(existing.get("max_events", 0)) == int(max_events)
            and existing.get("generation_rules", {
                "motion_normalization": "pred", "prototype_mode": "standard",
            }) == generation_rules
        ):
            return existing
        raise FileExistsError(f"refusing to overwrite stale compact labels: {final_dir}")
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite compact labels: {final_dir}")

    labels, rollout_summary = build_compact_rollout_labels_for_sequence(
        event_cache_dir,
        detection_cache_dir,
        gt_root,
        future_frames=int(future_frames),
        max_events=int(max_events),
        motion_normalization=motion_normalization,
        prototype_mode=prototype_mode,
    )
    for index, label in enumerate(labels):
        try:
            validate_rollout_label(label)
        except ValueError as exc:
            raise ValueError(f"invalid current rollout label {sequence}:{index}: {exc}") from exc
        if str(label.get("candidate_type", "A")) != "A":
            raise ValueError(f"non-A label produced in current rollout for {sequence}")
    if not labels:
        raise RuntimeError(f"no current rollout labels produced for {sequence}")

    event_keys = np.asarray(
        [
            (int(label["event_shard_id"]) << 32) | int(label["event_offset"])
            for label in labels
        ],
        dtype=np.int64,
    )
    order = np.argsort(event_keys, kind="stable")
    event_keys = event_keys[order]
    if np.any(event_keys[1:] == event_keys[:-1]):
        raise ValueError(f"duplicate current rollout event key in {sequence}")
    safe_gate = np.asarray(
        [
            [label["motion_safe_target"], label["appearance_safe_target"]]
            for label in labels
        ],
        dtype=np.float32,
    )[order]
    oracle_gate = np.asarray(
        [
            [label["motion_soft_target"], label["appearance_soft_target"]]
            for label in labels
        ],
        dtype=np.float32,
    )[order]
    gate_confidence = np.asarray(
        [
            [label["motion_label_confidence"], label["appearance_label_confidence"]]
            for label in labels
        ],
        dtype=np.float32,
    )[order]
    policy = np.asarray(
        [label["policy_safe_soft_target"] for label in labels], dtype=np.float32
    )[order]
    cue = np.asarray([label["cue_target"] for label in labels], dtype=np.float32)[
        order
    ]
    risk = np.asarray([label["risk_targets"] for label in labels], dtype=np.float32)[
        order
    ]
    valid = np.asarray(
        [
            [bool(label["valid_motion"]), bool(label["valid_appearance"])]
            for label in labels
        ],
        dtype=np.bool_,
    )[order]
    sample_weight = np.asarray(
        [float(label["sample_weight"]) for label in labels], dtype=np.float32
    )[order]

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f"{sequence}.incomplete.{os.getpid()}"
    temporary.mkdir()
    file_hashes = _write_arrays(
        temporary,
        event_keys=event_keys,
        safe_gate=safe_gate,
        oracle_gate=oracle_gate,
        gate_confidence=gate_confidence,
        policy=policy,
        cue=cue,
        risk=risk,
        valid=valid,
        sample_weight=sample_weight,
    )
    manifest = {
        "complete": True,
        "generation_rules": generation_rules,
        "dataset": dataset,
        "sequence": sequence,
        "event_cache_dir": str(event_cache_dir),
        "event_cache_manifest_sha256": cache_manifest_sha256,
        "detection_cache_dir": str(detection_cache_dir),
        "detection_cache_manifest_sha256": detection_manifest_sha256,
        "gt_root": str(gt_root),
        "future_frames": int(future_frames),
        "max_events": int(max_events),
        "event_source": str(cache_manifest.get("event_source", "nsa")),
        "motion_target_mode": motion_target_mode,
        "motion_label_mode": motion_label_mode,
        "source_labels": len(labels),
        "retained_labels": len(labels),
        "retention_rate": 1.0,
        "rollout_summary": rollout_summary,
        "compact_label_schema_version": compact_schema_version,
        "compact_label_schema_sha256": compact_schema_sha256,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "array_files": COMPACT_LABEL_ARRAY_FILES,
        "array_sha256": file_hashes,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(final_dir)
    return manifest


def load_compact_label_arrays(
    sequence_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sequence_dir = Path(sequence_dir)
    manifest = json.loads((sequence_dir / "manifest.json").read_text())
    required = {
        "complete": True,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"compact label manifest mismatch: {mismatches}")
    if manifest.get("compact_label_schema_sha256") not in SUPPORTED_COMPACT_IWG_LABEL_SCHEMA_SHA256:
        raise ValueError(
            "unsupported compact label schema: "
            f"{manifest.get('compact_label_schema_sha256')!r}"
        )
    schema_hash = manifest.get("compact_label_schema_sha256")
    if schema_hash == COMPACT_IWG_LABEL_SCHEMA_V1_SHA256:
        array_files = COMPACT_LABEL_ARRAY_FILES_V1
    elif schema_hash == COMPACT_IWG_LABEL_SCHEMA_SHA256:
        array_files = COMPACT_LABEL_ARRAY_FILES
    else:
        raise ValueError(
            "unsupported compact label schema: "
            f"{schema_hash!r}"
        )
    declared_files = manifest.get("array_files", array_files)
    if declared_files != array_files:
        raise ValueError(
            f"compact label array file contract mismatch in {sequence_dir}"
        )
    arrays = {
        name: np.load(sequence_dir / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in array_files.items()
    }
    count = int(manifest["retained_labels"])
    if any(value.shape[0] != count for value in arrays.values()):
        raise ValueError(f"compact label array length mismatch in {sequence_dir}")
    return manifest, arrays


def convert_json_label_directory_to_compact(
    *,
    dataset: str,
    source_root: str | Path,
    output_root: str | Path,
    sequences: list[str],
    event_cache_root: str | Path | None = None,
    detection_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Migrate an existing v3 JSON label directory to compact arrays.

    This is an explicit one-time migration path. Runtime dataset loading only
    accepts compact labels and never falls back to the JSON representation.
    """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    event_cache_root = Path(event_cache_root).resolve() if event_cache_root else None
    detection_cache_root = (
        Path(detection_cache_root).resolve() if detection_cache_root else None
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"source label directory not found: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite compact labels: {output_root}")

    output_root.mkdir(parents=True)
    sequence_summaries: dict[str, dict[str, Any]] = {}
    for sequence in sequences:
        source_path = source_root / f"{sequence}_labels.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"source labels not found: {source_path}")
        records = json.loads(source_path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"source labels must be a JSON list: {source_path}")

        labels: list[dict[str, Any]] = []
        for index, label in enumerate(records):
            if not isinstance(label, dict):
                raise ValueError(f"source label is not an object: {source_path}:{index}")
            validate_rollout_label(label)
            if str(label.get("candidate_type", "A")) == "A":
                labels.append(label)
        if not labels:
            raise RuntimeError(f"no candidate-A labels found in {source_path}")

        event_keys = np.asarray(
            [
                (int(label["event_shard_id"]) << 32)
                | int(label["event_offset"])
                for label in labels
            ],
            dtype=np.int64,
        )
        order = np.argsort(event_keys, kind="stable")
        event_keys = event_keys[order]
        if np.any(event_keys[1:] == event_keys[:-1]):
            raise ValueError(f"duplicate label event key in {source_path}")

        safe_gate = np.asarray(
            [
                [label["motion_safe_target"], label["appearance_safe_target"]]
                for label in labels
            ],
            dtype=np.float32,
        )[order]
        oracle_gate = np.asarray(
            [
                [label["motion_soft_target"], label["appearance_soft_target"]]
                for label in labels
            ],
            dtype=np.float32,
        )[order]
        gate_confidence = np.asarray(
            [
                [
                    label["motion_label_confidence"],
                    label["appearance_label_confidence"],
                ]
                for label in labels
            ],
            dtype=np.float32,
        )[order]
        policy = np.asarray(
            [label["policy_safe_soft_target"] for label in labels],
            dtype=np.float32,
        )[order]
        cue = np.asarray(
            [label["cue_target"] for label in labels], dtype=np.float32
        )[order]
        risk = np.asarray(
            [label["risk_targets"] for label in labels], dtype=np.float32
        )[order]
        valid = np.asarray(
            [
                [bool(label["valid_motion"]), bool(label["valid_appearance"])]
                for label in labels
            ],
            dtype=np.bool_,
        )[order]
        sample_weight = np.asarray(
            [float(label["sample_weight"]) for label in labels],
            dtype=np.float32,
        )[order]

        sequence_dir = output_root / sequence
        temporary = output_root / f"{sequence}.incomplete.{os.getpid()}"
        temporary.mkdir()
        array_sha256 = _write_arrays(
            temporary,
            event_keys=event_keys,
            safe_gate=safe_gate,
            oracle_gate=oracle_gate,
            gate_confidence=gate_confidence,
            policy=policy,
            cue=cue,
            risk=risk,
            valid=valid,
            sample_weight=sample_weight,
        )
        manifest: dict[str, Any] = {
            "complete": True,
            "dataset": dataset,
            "sequence": sequence,
            "source_format": "rollout_labels_json_v3",
            "source_label_file_sha256": _sha256_file(source_path),
            "source_labels": len(records),
            "retained_labels": len(labels),
            "retention_rate": len(labels) / max(len(records), 1),
            "compact_label_schema_version": COMPACT_IWG_LABEL_SCHEMA_VERSION,
            "compact_label_schema_sha256": COMPACT_IWG_LABEL_SCHEMA_SHA256,
            "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
            "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "array_files": COMPACT_LABEL_ARRAY_FILES,
            "array_sha256": array_sha256,
        }
        if event_cache_root is not None:
            event_manifest = (
                event_cache_root / dataset / "all" / sequence / "manifest.json"
            )
            if event_manifest.is_file():
                manifest["event_cache_manifest_sha256"] = _sha256_file(event_manifest)
        if detection_cache_root is not None:
            detection_manifest = (
                detection_cache_root / dataset / "all" / sequence / "manifest.json"
            )
            if detection_manifest.is_file():
                manifest["detection_cache_manifest_sha256"] = _sha256_file(
                    detection_manifest
                )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(sequence_dir)
        sequence_summaries[sequence] = {
            "source_labels": len(records),
            "retained_labels": len(labels),
            "source_label_file_sha256": manifest["source_label_file_sha256"],
        }

    source_summary = source_root / "summary.json"
    summary: dict[str, Any] = {
        "complete": True,
        "dataset": dataset,
        "mode": "all",
        "split": "all",
        "label_mode": "all_a_only",
        "motion_label_mode": "nsa_rollout",
        "candidate_types": ["A"],
        "num_labels": sum(item["retained_labels"] for item in sequence_summaries.values()),
        "num_sequences": len(sequence_summaries),
        "sequences": sequence_summaries,
        "source_format": "rollout_labels_json_v3",
        "source_summary_sha256": _sha256_file(source_summary)
        if source_summary.is_file()
        else "",
        "compact_label_schema_version": COMPACT_IWG_LABEL_SCHEMA_VERSION,
        "compact_label_schema_sha256": COMPACT_IWG_LABEL_SCHEMA_SHA256,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
    }
    summary["retention_rate"] = summary["num_labels"] / max(
        sum(item["source_labels"] for item in sequence_summaries.values()), 1
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
