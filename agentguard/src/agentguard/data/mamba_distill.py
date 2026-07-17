from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.compact_iwg_labels import (
    COMPACT_LABEL_ARRAY_FILES,
    MAMBA_DISTILL_COMPACT_LABEL_SCHEMA_SHA256,
    MAMBA_DISTILL_COMPACT_LABEL_SCHEMA_VERSION,
    MAMBA_DISTILL_LABEL_ARRAY_FILES,
    _sha256_file,
    _write_arrays,
    load_compact_label_arrays,
)
from agentguard.data.gt_reader import GTReader
from agentguard.data.identity_vote import TrackIdentityVoteState
from agentguard.motion.mamba_shadow import MambaShadowReader, sha256_file
from agentguard.rollout.losses import motion_frame_loss
from agentguard.rollout_labels import compute_policy_soft_target
from agentguard.v0_pipeline import gt_box_for_id, match_detection_to_gt


TEACHER_WEIGHT_CAP = 0.5
ADVANTAGE_HORIZON = 5


@dataclass
class _PendingAdvantage:
    label_position: int
    start_frame: int
    target_gt_id: int
    nsa_losses: list[float]
    mamba_losses: list[float]


def _packed_event_key(record: dict[str, Any]) -> int:
    return (int(record["event_shard_id"]) << 32) | int(record["event_offset"])


def _joined_records(
    reader: CompactEventCacheReader,
    teacher: MambaShadowReader,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    event_iter = reader.iter_event_records()
    teacher_iter = teacher.iter_records()
    count = 0
    while True:
        try:
            event_record = next(event_iter)
        except StopIteration:
            event_record = None
        try:
            teacher_record = next(teacher_iter)
        except StopIteration:
            teacher_record = None
        if event_record is None or teacher_record is None:
            if event_record is not None or teacher_record is not None:
                raise ValueError("teacher and NSA event streams have unequal lengths")
            break
        count += 1
        event_id = str(event_record["event_id"])
        if str(teacher_record["event_id"]) != event_id:
            raise ValueError(
                f"teacher/NSA event_id mismatch at record {count}: "
                f"{teacher_record['event_id']!r} != {event_id!r}"
            )
        expected = {
            "frame_id": int(event_record["frame_id"]),
            "track_id": int(event_record["track_id"]),
            "matched": bool(event_record["matched"]),
            "accepted_detection_index": int(
                event_record.get("accepted_detection_index", -1)
            ),
        }
        actual = {
            "frame_id": int(teacher_record["frame_id"]),
            "track_id": int(teacher_record["track_id"]),
            "matched": bool(teacher_record["matched"]),
            "accepted_detection_index": int(
                teacher_record["accepted_detection_index"]
            ),
        }
        if actual != expected:
            raise ValueError(
                f"teacher/NSA event metadata mismatch for {event_id}: "
                f"{actual} != {expected}"
            )
        yield event_record, teacher_record


def _projection_gate(
    nsa_prior_box: np.ndarray,
    detection_box: np.ndarray,
    mamba_posterior_box: np.ndarray,
    image_width: int,
    image_height: int,
) -> float:
    scale = np.asarray(
        [image_width, image_height, image_width, image_height], dtype=np.float64
    )
    prior = np.asarray(nsa_prior_box, dtype=np.float64) / scale
    detection = np.asarray(detection_box, dtype=np.float64) / scale
    posterior = np.asarray(mamba_posterior_box, dtype=np.float64) / scale
    update = detection - prior
    gate = float(np.dot(update, posterior - prior) / (np.dot(update, update) + 1e-8))
    return float(np.clip(gate, 0.0, 1.0))


def compute_sequence_distill_statistics(
    *,
    sequence: str,
    event_cache_dir: str | Path,
    detection_cache_dir: str | Path,
    teacher_dir: str | Path,
    nsa_label_dir: str | Path,
    gt_root: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    label_manifest, labels = load_compact_label_arrays(nsa_label_dir)
    label_keys = np.asarray(labels["event_keys"], dtype=np.int64)
    label_count = len(label_keys)
    advantage = np.full(label_count, np.nan, dtype=np.float32)
    coverage = np.zeros(label_count, dtype=np.float32)
    projection_gate = np.full(label_count, np.nan, dtype=np.float32)
    joined_labels = np.zeros(label_count, dtype=np.bool_)

    reader = CompactEventCacheReader(
        event_cache_dir,
        detection_cache_dir,
        max_cached_shards=8,
    )
    teacher = MambaShadowReader(teacher_dir)
    gt_reader = GTReader(str(gt_root), sequence)
    votes = TrackIdentityVoteState()
    pending: dict[int, list[_PendingAdvantage]] = defaultdict(list)
    matched_count = 0
    unmatched_count = 0
    joined_count = 0
    reliable_count = 0
    gap_nsa_losses: list[float] = []
    gap_mamba_losses: list[float] = []
    projection_capture: list[float] = []

    def finalize(item: _PendingAdvantage) -> None:
        valid = len(item.nsa_losses)
        if valid == 0:
            raise ValueError(
                f"distill endpoint has no valid GT frames: {sequence}:"
                f"{item.label_position}"
            )
        position = item.label_position
        advantage[position] = float(
            np.mean(np.asarray(item.nsa_losses) - np.asarray(item.mamba_losses))
        )
        coverage[position] = float(valid / (ADVANTAGE_HORIZON + 1))
        joined_labels[position] = True

    try:
        for event_record, teacher_record in _joined_records(reader, teacher):
            joined_count += 1
            matched = bool(event_record["matched"])
            matched_count += int(matched)
            unmatched_count += int(not matched)
            track_id = int(event_record["track_id"])
            frame_id = int(event_record["frame_id"])

            still_pending: list[_PendingAdvantage] = []
            for item in pending.get(track_id, []):
                if frame_id > item.start_frame + ADVANTAGE_HORIZON:
                    finalize(item)
                else:
                    still_pending.append(item)
            pending[track_id] = still_pending

            target_gt_id = votes.resolve_before_current(sequence, track_id)
            detection_gt_id = -1
            detection_box = None
            if matched:
                detection_index = int(event_record["accepted_detection_index"])
                detection = reader.get_detection(detection_index)
                detection_box = np.asarray(detection["box"], dtype=np.float64)
                detection_gt_id, _detection_iou, _gt_box = match_detection_to_gt(
                    detection_box,
                    gt_reader.get_gt_for_frame(frame_id),
                    min_iou=0.5,
                )

            key = _packed_event_key(event_record)
            label_position = int(np.searchsorted(label_keys, key))
            is_label = (
                label_position < label_count
                and int(label_keys[label_position]) == key
            )
            if is_label:
                if not matched or detection_box is None:
                    raise ValueError(f"compact label points to unmatched event: {sequence}:{key}")
                if target_gt_id is None:
                    raise ValueError(
                        f"compact label lost causal GT identity: {sequence}:{key}"
                    )
                projection_gate[label_position] = _projection_gate(
                    teacher_record["nsa_prior_box"],
                    detection_box,
                    teacher_record["mamba_posterior_box"],
                    int(reader.manifest["image_width"]),
                    int(reader.manifest["image_height"]),
                )
                pending[track_id].append(
                    _PendingAdvantage(
                        label_position=label_position,
                        start_frame=frame_id,
                        target_gt_id=int(target_gt_id),
                        nsa_losses=[],
                        mamba_losses=[],
                    )
                )

            for item in pending.get(track_id, []):
                gt_box = gt_box_for_id(
                    gt_reader, frame_id, int(item.target_gt_id)
                )
                if gt_box is None:
                    continue
                item.nsa_losses.append(
                    motion_frame_loss(gt_box, teacher_record["nsa_prior_box"])
                )
                item.mamba_losses.append(
                    motion_frame_loss(gt_box, teacher_record["mamba_prior_box"])
                )

            if target_gt_id is not None:
                current_gt = gt_box_for_id(gt_reader, frame_id, int(target_gt_id))
                if current_gt is not None:
                    reliable_count += 1
                    nsa_loss = motion_frame_loss(
                        current_gt, teacher_record["nsa_prior_box"]
                    )
                    mamba_loss = motion_frame_loss(
                        current_gt, teacher_record["mamba_prior_box"]
                    )
                    if int(teacher_record["gap"]) >= 2:
                        gap_nsa_losses.append(nsa_loss)
                        gap_mamba_losses.append(mamba_loss)
                    if matched and detection_box is not None:
                        nsa_full_loss = motion_frame_loss(
                            current_gt, teacher_record["nsa_full_update_box"]
                        )
                        mamba_posterior_loss = motion_frame_loss(
                            current_gt, teacher_record["mamba_posterior_box"]
                        )
                    else:
                        nsa_full_loss = float("nan")
                        mamba_posterior_loss = float("nan")
                    if (
                        matched
                        and detection_box is not None
                        and mamba_posterior_loss < nsa_full_loss
                    ):
                        gate = _projection_gate(
                            teacher_record["nsa_prior_box"],
                            detection_box,
                            teacher_record["mamba_posterior_box"],
                            int(reader.manifest["image_width"]),
                            int(reader.manifest["image_height"]),
                        )
                        projected = (
                            (1.0 - gate)
                            * np.asarray(teacher_record["nsa_prior_box"])
                            + gate * detection_box
                        )
                        projected_loss = motion_frame_loss(current_gt, projected)
                        projection_capture.append(
                            float(
                                (nsa_full_loss - projected_loss)
                                / max(
                                    nsa_full_loss - mamba_posterior_loss,
                                    1e-12,
                                )
                            )
                        )
            votes.add_current_observation(sequence, track_id, detection_gt_id)

        for items in pending.values():
            for item in items:
                finalize(item)
    finally:
        reader.close()

    if not joined_labels.all():
        missing = np.flatnonzero(~joined_labels)
        raise ValueError(
            f"teacher did not cover all compact labels in {sequence}: "
            f"{len(missing)} missing, first={missing[:8].tolist()}"
        )
    if not np.isfinite(advantage).all() or not np.isfinite(projection_gate).all():
        raise ValueError(f"non-finite distill statistics in {sequence}")
    event_manifest = json.loads((Path(event_cache_dir) / "manifest.json").read_text())
    expected_counts = {
        "records": int(event_manifest["num_events"]),
        "matched": int(event_manifest["num_matched_events"]),
        "unmatched": int(event_manifest["num_unmatched_events"]),
    }
    actual_counts = {
        "records": joined_count,
        "matched": matched_count,
        "unmatched": unmatched_count,
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"teacher/NSA count mismatch for {sequence}: "
            f"{actual_counts} != {expected_counts}"
        )

    summary = {
        "sequence": sequence,
        "join_rate": 1.0,
        "records": joined_count,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "labels": label_count,
        "reliable_gt_events": reliable_count,
        "gap_nsa_losses": np.asarray(gap_nsa_losses, dtype=np.float64),
        "gap_mamba_losses": np.asarray(gap_mamba_losses, dtype=np.float64),
        "projection_capture": np.asarray(projection_capture, dtype=np.float64),
        "nsa_label_manifest_sha256": _sha256_file(
            Path(nsa_label_dir) / "manifest.json"
        ),
        "teacher_manifest_sha256": sha256_file(Path(teacher_dir) / "manifest.json"),
        "teacher_config_sha256": teacher.manifest["teacher_config_sha256"],
        "teacher_checkpoint_path": teacher.manifest["teacher_config"][
            "checkpoint_path"
        ],
        "teacher_checkpoint_sha256": teacher.manifest["teacher_config"][
            "checkpoint_sha256"
        ],
        "teacher_config": teacher.manifest["teacher_config"],
        "event_cache_manifest_sha256": _sha256_file(
            Path(event_cache_dir) / "manifest.json"
        ),
        "detection_cache_manifest_sha256": _sha256_file(
            Path(detection_cache_dir) / "manifest.json"
        ),
        "source_labels": int(label_manifest["retained_labels"]),
    }
    return {
        "event_keys": label_keys,
        "advantage": advantage,
        "coverage": coverage,
        "projection_gate": projection_gate,
    }, summary


def _bootstrap_improvement_ci(
    nsa_losses: np.ndarray,
    mamba_losses: np.ndarray,
    *,
    seed: int = 42,
    samples: int = 2000,
) -> tuple[float, float]:
    if len(nsa_losses) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(nsa_losses), size=len(nsa_losses))
        nsa_mean = float(nsa_losses[selected].mean())
        values[index] = (
            nsa_mean - float(mamba_losses[selected].mean())
        ) / max(nsa_mean, 1e-12)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def audit_mamba_shadow(
    sequence_results: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]],
) -> dict[str, Any]:
    nsa = np.concatenate(
        [summary["gap_nsa_losses"] for _, summary in sequence_results.values()]
    )
    mamba = np.concatenate(
        [summary["gap_mamba_losses"] for _, summary in sequence_results.values()]
    )
    capture = np.concatenate(
        [summary["projection_capture"] for _, summary in sequence_results.values()]
    )
    improvement = (
        float(nsa.mean() - mamba.mean()) / max(float(nsa.mean()), 1e-12)
        if len(nsa)
        else float("nan")
    )
    ci_low, ci_high = _bootstrap_improvement_ci(nsa, mamba)
    capture_median = float(np.median(capture)) if len(capture) else float("nan")
    structural_pass = all(
        float(summary["join_rate"]) == 1.0
        for _, summary in sequence_results.values()
    )
    admission_checks = {
        "gap_improvement_at_least_2pct": bool(improvement >= 0.02),
        "bootstrap_ci_lower_positive": bool(ci_low > 0.0),
        "projection_capture_median_at_least_50pct": bool(
            capture_median >= 0.5
        ),
    }
    per_sequence = {}
    for sequence, (_arrays, summary) in sequence_results.items():
        per_sequence[sequence] = {
            key: value
            for key, value in summary.items()
            if not isinstance(value, np.ndarray)
        }
        seq_nsa = summary["gap_nsa_losses"]
        seq_mamba = summary["gap_mamba_losses"]
        per_sequence[sequence]["gap_ge_2_events"] = len(seq_nsa)
        per_sequence[sequence]["gap_improvement"] = (
            float(seq_nsa.mean() - seq_mamba.mean())
            / max(float(seq_nsa.mean()), 1e-12)
            if len(seq_nsa)
            else None
        )
    return {
        "structural_pass": structural_pass,
        "full_mot20": set(sequence_results) == set(("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")),
        "gap_ge_2_events": len(nsa),
        "gap_motion_loss_improvement": improvement,
        "bootstrap_95_ci": [ci_low, ci_high],
        "projection_capture_events": len(capture),
        "projection_capture_median": capture_median,
        "admission_checks": admission_checks,
        "admission_pass": structural_pass and all(admission_checks.values()),
        "per_sequence": per_sequence,
    }


def build_mamba_distill_labels(
    *,
    sequence_results: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]],
    nsa_label_root: str | Path,
    output_root: str | Path,
    tau_adv: float,
) -> dict[str, Any]:
    nsa_label_root = Path(nsa_label_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Mamba distill labels: {output_root}")
    temporary_root = output_root.with_name(
        f"{output_root.name}.incomplete.{os.getpid()}"
    )
    if temporary_root.exists():
        raise FileExistsError(f"stale Mamba distill label directory: {temporary_root}")
    temporary_root.mkdir(parents=True)
    sequence_manifests: dict[str, Any] = {}
    try:
        for sequence, (distill, source_summary) in sequence_results.items():
            source_dir = nsa_label_root / sequence
            source_manifest, source = load_compact_label_arrays(source_dir)
            final_dir = temporary_root / sequence
            final_dir.mkdir()
            advantage = np.asarray(distill["advantage"], dtype=np.float32)
            coverage = np.asarray(distill["coverage"], dtype=np.float32)
            projection = np.asarray(distill["projection_gate"], dtype=np.float32)
            nsa_target = np.asarray(source["safe_gate_target"][:, 0], dtype=np.float32)
            positive = advantage > 0.0
            teacher_weight = np.zeros_like(advantage, dtype=np.float32)
            teacher_weight[positive] = (
                TEACHER_WEIGHT_CAP
                * coverage[positive]
                * np.clip(advantage[positive] / float(tau_adv), 0.0, 1.0)
            )
            teacher_weight = np.clip(
                teacher_weight, 0.0, TEACHER_WEIGHT_CAP
            ).astype(np.float32)
            hybrid_target = (
                (1.0 - teacher_weight) * nsa_target
                + teacher_weight * projection
            ).astype(np.float32)
            safe_gate = np.asarray(source["safe_gate_target"], dtype=np.float32).copy()
            safe_gate[:, 0] = hybrid_target
            policy = np.stack(
                [
                    compute_policy_soft_target(
                        np.asarray([hybrid_target[index], safe_gate[index, 1]])
                    )
                    for index in range(len(hybrid_target))
                ],
                axis=0,
            ).astype(np.float32)
            if not (
                np.isfinite(hybrid_target).all()
                and np.isfinite(teacher_weight).all()
                and ((hybrid_target >= 0.0) & (hybrid_target <= 1.0)).all()
            ):
                raise ValueError(f"invalid Mamba hybrid target in {sequence}")
            fallback = ~positive
            if not np.array_equal(hybrid_target[fallback], nsa_target[fallback]):
                raise AssertionError("non-positive Mamba advantage must exactly fall back to NSA")

            hashes = _write_arrays(
                final_dir,
                event_keys=np.asarray(source["event_keys"]),
                safe_gate=safe_gate,
                policy=policy,
                cue=np.asarray(source["cue_target"]),
                risk=np.asarray(source["risk_target"]),
                valid=np.asarray(source["valid_channels"]),
                sample_weight=np.asarray(source["sample_weight"]),
            )
            extras = {
                "teacher_weight": teacher_weight,
                "nsa_motion_target": nsa_target,
                "hybrid_motion_target": hybrid_target,
                "mamba_projection_gate": projection,
                "mamba_advantage": advantage,
                "mamba_coverage": coverage,
            }
            for name, values in extras.items():
                path = final_dir / MAMBA_DISTILL_LABEL_ARRAY_FILES[name]
                np.save(path, values, allow_pickle=False)
                hashes[path.name] = _sha256_file(path)
            manifest = {
                **source_manifest,
                "motion_target_mode": "mamba_hybrid",
                "source_nsa_label_dir": str(source_dir),
                "source_nsa_label_manifest_sha256": _sha256_file(
                    source_dir / "manifest.json"
                ),
                "compact_label_schema_version": (
                    MAMBA_DISTILL_COMPACT_LABEL_SCHEMA_VERSION
                ),
                "compact_label_schema_sha256": (
                    MAMBA_DISTILL_COMPACT_LABEL_SCHEMA_SHA256
                ),
                "teacher_weight_cap": TEACHER_WEIGHT_CAP,
                "advantage_horizon": ADVANTAGE_HORIZON,
                "tau_adv": float(tau_adv),
                "teacher_config_sha256": source_summary[
                    "teacher_config_sha256"
                ],
                "teacher_checkpoint_path": source_summary[
                    "teacher_checkpoint_path"
                ],
                "teacher_checkpoint_sha256": source_summary[
                    "teacher_checkpoint_sha256"
                ],
                "teacher_config": source_summary["teacher_config"],
                "teacher_manifest_sha256": source_summary[
                    "teacher_manifest_sha256"
                ],
                "event_cache_manifest_sha256": source_summary[
                    "event_cache_manifest_sha256"
                ],
                "detection_cache_manifest_sha256": source_summary[
                    "detection_cache_manifest_sha256"
                ],
                "distill_array_files": MAMBA_DISTILL_LABEL_ARRAY_FILES,
                "array_sha256": hashes,
                "teacher_weight_mean": float(teacher_weight.mean()),
                "teacher_active_rate": float((teacher_weight > 0.0).mean()),
            }
            (final_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            sequence_manifests[sequence] = manifest

        summary = {
            "complete": set(sequence_manifests) == set(("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")),
            "dataset": "MOT20",
            "split": "all_a_only",
            "motion_target_mode": "mamba_hybrid",
            "sequences": sorted(sequence_manifests),
            "teacher_weight_cap": TEACHER_WEIGHT_CAP,
            "advantage_horizon": ADVANTAGE_HORIZON,
            "tau_adv": float(tau_adv),
            "per_sequence": sequence_manifests,
        }
        (temporary_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        temporary_root.rename(output_root)
        return summary
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
