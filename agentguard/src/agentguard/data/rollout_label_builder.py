"""Build current compact rollout labels from cached tracking events."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.states import DetectionObservation
from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.gt_reader import GTReader
from agentguard.data.identity_prototype import IdentityPrototypeBuilder
from agentguard.data.identity_vote import TrackIdentityVoteState
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    make_cue_target,
    make_risk_targets,
    validate_rollout_label,
)
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.appearance import compute_appearance_benefit
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.motion import compute_motion_benefit
from agentguard.rollout_labels import (
    compute_dataset_stats,
    compute_label_confidence,
    compute_policy_soft_target,
    compute_safe_soft_target,
    compute_soft_target,
    hard_gate_from_benefit,
)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def detection_observation(
    reader: CompactEventCacheReader,
    detection_index: int,
) -> DetectionObservation:
    det = reader.get_detection(int(detection_index))
    feature = np.asarray(det["feature"], dtype=np.float64).reshape(1, -1)
    return DetectionObservation(
        detection_index=int(detection_index),
        box=np.asarray(det["box"], dtype=np.float64),
        score=float(det["score"]),
        feature=feature,
        source=int(det["source"]),
        class_id=int(det["class_id"]),
    )


def match_detection_to_gt(
    det_box: np.ndarray,
    gt_entries: Any,
    min_iou: float = 0.5,
) -> Tuple[int, float, Optional[np.ndarray]]:
    best_id = -1
    best_iou = float(min_iou)
    best_box: Optional[np.ndarray] = None
    for gt_box, gt_id in gt_entries:
        iou = box_iou(det_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_id = int(gt_id)
            best_box = np.asarray(gt_box, dtype=np.float64)
    return best_id, best_iou, best_box


def gt_box_for_id(
    gt_reader: GTReader,
    frame_id: int,
    gt_id: int,
) -> Optional[np.ndarray]:
    indexed = gt_reader.get_gt_box(int(frame_id), int(gt_id))
    if indexed is not None:
        return indexed
    for gt_box, tid in gt_reader.get_gt_for_frame(int(frame_id)):
        if int(tid) == int(gt_id):
            return np.asarray(gt_box, dtype=np.float64)
    return None


def best_oracle_detection(
    reader: CompactEventCacheReader,
    frame_index: int,
    gt_box: Optional[np.ndarray],
    min_iou: float = 0.3,
) -> Optional[DetectionObservation]:
    if gt_box is None or reader.detection_cache is None:
        return None
    frame = reader.detection_cache.get_frame(int(frame_index), view="source")
    best_idx = -1
    best_iou = float(min_iou)
    for det_idx, det_box in zip(frame["detection_indices"], frame["boxes"]):
        iou = box_iou(gt_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = int(det_idx)
    if best_idx < 0:
        return None
    return detection_observation(reader, best_idx)


def _warp_for_frame(reader: CompactEventCacheReader, frame_index: int) -> np.ndarray:
    frame = reader.get_frame_record(int(frame_index))
    if frame is None:
        return np.eye(2, 3, dtype=np.float64)
    return np.asarray(frame.get("warp_matrix", frame.get("effective_warp")), dtype=np.float64)


def _build_future_context(
    reader: CompactEventCacheReader,
    gt_reader: GTReader,
    frame_id: int,
    target_gt_id: int,
    future_frames: int,
) -> Tuple[
    Optional[np.ndarray],
    List[Optional[np.ndarray]],
    List[Optional[DetectionObservation]],
    List[np.ndarray],
    int,
    int,
]:
    current_gt = gt_box_for_id(gt_reader, frame_id, target_gt_id)
    future_gt: List[Optional[np.ndarray]] = []
    future_oracle: List[Optional[DetectionObservation]] = []
    future_warps: List[np.ndarray] = []
    gt_count = 1 if current_gt is not None else 0
    oracle_count = 0
    total_frames = int(
        reader.manifest.get("total_sequence_frames")
        or reader.manifest.get("num_frames")
        or 0
    )
    for step in range(1, future_frames + 1):
        fid = int(frame_id) + step
        if total_frames and fid > total_frames:
            break
        gt_box = gt_box_for_id(gt_reader, fid, target_gt_id)
        future_gt.append(gt_box)
        if gt_box is not None:
            gt_count += 1
        oracle_det = best_oracle_detection(reader, fid - 1, gt_box)
        if oracle_det is not None:
            oracle_count += 1
        future_oracle.append(oracle_det)
        future_warps.append(_warp_for_frame(reader, fid - 1))
    return current_gt, future_gt, future_oracle, future_warps, gt_count, oracle_count


def build_compact_rollout_labels_for_sequence(
    event_cache_dir: str | os.PathLike[str],
    detection_cache_dir: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    *,
    max_events: int = 0,
    future_frames: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reader = CompactEventCacheReader(
        event_cache_dir,
        detection_cache_dir,
        max_cached_shards=16,
    )
    gt_reader = GTReader(str(gt_root), str(reader.manifest["sequence"]))
    motion_model = NSAKalmanFilter()
    vote_state = TrackIdentityVoteState()
    proto_features: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    target_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

    try:
        for record in reader.iter_event_records():
            event = reader.materialize_training_event(record)
            target_gt_id = vote_state.resolve_before_current(event.sequence, event.track_id)
            identity_key = vote_state.get_target_identity_key(event.track_id)
            detection_gt_id = -1
            detection_iou = 0.0
            if event.has_detection and event.detection is not None:
                detection_gt_id, detection_iou, _ = match_detection_to_gt(
                    event.detection.box,
                    gt_reader.get_gt_for_frame(event.frame_id),
                    min_iou=0.5,
                )
                if (
                    detection_gt_id >= 0
                    and detection_iou >= 0.7
                    and event.detection.score >= 0.6
                ):
                    proto_features[(event.sequence, detection_gt_id)].append(
                        event.detection.feature.reshape(-1).astype(np.float32)
                    )
            target_info[(int(record["event_shard_id"]), int(record["event_offset"]))] = {
                "target_gt_id": target_gt_id,
                "target_identity_key": identity_key,
                "detection_gt_id": detection_gt_id,
                "detection_gt_iou": detection_iou,
            }
            vote_state.add_current_observation(event.sequence, event.track_id, detection_gt_id)

        prototypes: Dict[Tuple[str, int], np.ndarray] = {}
        for key, feats in proto_features.items():
            if len(feats) >= 3:
                prototypes[key] = IdentityPrototypeBuilder.build_prototype(feats)
        proto_features.clear()

        raw_labels: List[Dict[str, Any]] = []
        motion_benefits: List[float] = []
        appearance_benefits: List[float] = []
        summary = {
            "reliable_identity_events": 0,
            "candidate_a_count": 0,
            "valid_motion_labels": 0,
            "valid_appearance_labels": 0,
            "future_gt_coverage_count": 0,
            "future_oracle_coverage_count": 0,
            "prototype_count": len(prototypes),
            "skipped_unreliable": 0,
            "skipped_invalid": 0,
        }

        for record in reader.iter_event_records():
            if max_events > 0 and len(raw_labels) >= max_events:
                break
            info = target_info[(int(record["event_shard_id"]), int(record["event_offset"]))]
            target_gt_id = info["target_gt_id"]
            if target_gt_id is None:
                summary["skipped_unreliable"] += 1
                continue

            event = reader.materialize_training_event(record)
            summary["reliable_identity_events"] += 1
            current_gt, future_gt, future_oracle, future_warps, gt_count, oracle_count = (
                _build_future_context(
                    reader,
                    gt_reader,
                    event.frame_id,
                    int(target_gt_id),
                    future_frames,
                )
            )
            if current_gt is None:
                summary["skipped_invalid"] += 1
                continue

            proto = prototypes.get((event.sequence, int(target_gt_id)))
            accepted_detection_index = int(record.get("accepted_detection_index", -1))
            candidate_specs = (
                [
                    {
                        "candidate_type": "A",
                        "detection_index": accepted_detection_index,
                        "detection_gt_id": int(info["detection_gt_id"]),
                        "detection_gt_iou": float(info["detection_gt_iou"]),
                    }
                ]
                if accepted_detection_index >= 0
                else []
            )
            wrote_any = False
            for spec in candidate_specs:
                if max_events > 0 and len(raw_labels) >= max_events:
                    break
                candidate_event = event
                ctx = RolloutContext(
                    frame_id=candidate_event.frame_id,
                    target_gt_id=int(target_gt_id),
                    pre_update_state=candidate_event.pre_update_state
                    or candidate_event.frame_start_state,
                    current_candidate=candidate_event.detection,
                    current_gt_box=current_gt,
                    future_gt_boxes=future_gt,
                    future_oracle_detections=future_oracle,
                    future_warp_matrices=future_warps,
                    identity_prototype=proto,
                )

                valid_motion = bool(
                    candidate_event.has_detection and candidate_event.detection is not None
                )
                valid_appearance = bool(valid_motion and proto is not None)
                b_m = 0.0
                b_a = 0.0
                motion_valid_count = 0
                appearance_valid_count = 0
                if valid_motion:
                    try:
                        b_m, _, _, m_mask = compute_motion_benefit(
                            ctx,
                            motion_model,
                            future_frames=future_frames,
                            include_current=True,
                        )
                        motion_valid_count = int(np.asarray(m_mask, dtype=bool).sum())
                        valid_motion = motion_valid_count > 0
                    except Exception:
                        valid_motion = False
                        b_m = 0.0
                if valid_appearance:
                    try:
                        b_a, _, _, _, _, a_mask = compute_appearance_benefit(
                            ctx,
                            future_frames=future_frames,
                            include_current=True,
                        )
                        appearance_valid_count = int(np.asarray(a_mask, dtype=bool).sum())
                        valid_appearance = appearance_valid_count > 0
                    except Exception:
                        valid_appearance = False
                        b_a = 0.0

                if not valid_motion and not valid_appearance:
                    continue

                ctype = str(spec["candidate_type"])
                summary[f"candidate_{ctype.lower()}_count"] += 1
                summary["valid_motion_labels"] += int(valid_motion)
                summary["valid_appearance_labels"] += int(valid_appearance)
                summary["future_gt_coverage_count"] += gt_count
                summary["future_oracle_coverage_count"] += oracle_count
                motion_benefits.append(float(b_m))
                appearance_benefits.append(float(b_a))
                raw_labels.append(
                    {
                        "event_id": event.event_id,
                        "event_shard_id": int(record["event_shard_id"]),
                        "event_offset": int(record["event_offset"]),
                        "state_shard_id": int(record["state_shard_id"]),
                        "frame_start_state_offset": int(record["frame_start_state_offset"]),
                        "pre_update_state_offset": int(record["pre_update_state_offset"]),
                        "association_shard_id": int(record.get("association_shard_id", -1)),
                        "association_offset": int(record.get("association_offset", -1)),
                        "sequence": event.sequence,
                        "frame_id": int(event.frame_id),
                        "frame_index": int(record.get("frame_index", event.frame_id - 1)),
                        "track_id": int(event.track_id),
                        "candidate_type": ctype,
                        "candidate_detection_index": int(spec["detection_index"]),
                        "target_gt_id": int(target_gt_id),
                        "target_identity_key": list(
                            info["target_identity_key"]
                            or (event.sequence, int(target_gt_id))
                        ),
                        "detection_gt_id": int(spec["detection_gt_id"]),
                        "detection_gt_iou": float(spec["detection_gt_iou"]),
                        "motion_benefit": float(b_m),
                        "motion_label_mode": "nsa_rollout",
                        "appearance_benefit": float(b_a),
                        "valid_motion": bool(valid_motion),
                        "valid_appearance": bool(valid_appearance),
                        "valid_horizon_count": int(
                            max(motion_valid_count, appearance_valid_count)
                        ),
                        "gt_coverage": float(gt_count / max(future_frames + 1, 1)),
                        "oracle_detection_coverage": float(
                            oracle_count / max(future_frames, 1)
                        ),
                        "sample_type": "matched" if candidate_event.has_detection else "unmatched",
                        "sample_weight": 1.0,
                    }
                )
                wrote_any = True

            if not wrote_any:
                summary["skipped_invalid"] += 1

        stats = compute_dataset_stats(
            {
                "motion_benefits": [
                    b for b, lbl in zip(motion_benefits, raw_labels) if lbl["valid_motion"]
                ],
                "appearance_benefits": [
                    b
                    for b, lbl in zip(appearance_benefits, raw_labels)
                    if lbl["valid_appearance"]
                ],
            }
        )
        for lbl in raw_labels:
            motion_soft = (
                compute_soft_target(lbl["motion_benefit"], stats["tau_motion"])
                if lbl["valid_motion"]
                else 0.5
            )
            appearance_soft = (
                compute_soft_target(lbl["appearance_benefit"], stats["tau_appearance"])
                if lbl["valid_appearance"]
                else 0.5
            )
            motion_confidence = (
                compute_label_confidence(
                    lbl["motion_benefit"],
                    stats["tau_motion"],
                    lbl["gt_coverage"],
                )
                if lbl["valid_motion"]
                else 0.0
            )
            appearance_confidence = (
                compute_label_confidence(
                    lbl["appearance_benefit"],
                    stats["tau_appearance"],
                    lbl["oracle_detection_coverage"],
                )
                if lbl["valid_appearance"]
                else 0.0
            )
            motion_safe = compute_safe_soft_target(
                lbl["motion_benefit"],
                stats["tau_motion"],
                motion_confidence,
            )
            appearance_safe = compute_safe_soft_target(
                lbl["appearance_benefit"],
                stats["tau_appearance"],
                appearance_confidence,
            )
            lbl["motion_soft_target"] = float(motion_soft)
            lbl["appearance_soft_target"] = float(appearance_soft)
            lbl["motion_target"] = float(motion_soft)
            lbl["appearance_target"] = float(appearance_soft)
            lbl["motion_safe_target"] = float(motion_safe)
            lbl["appearance_safe_target"] = float(appearance_safe)
            lbl["motion_label_confidence"] = float(motion_confidence)
            lbl["appearance_label_confidence"] = float(appearance_confidence)
            lbl["cue_target"] = make_cue_target(
                motion_confidence, appearance_confidence
            )
            lbl["risk_targets"] = make_risk_targets(
                motion_soft,
                appearance_soft,
                motion_confidence,
                appearance_confidence,
            )
            lbl["label_schema_version"] = ROLLOUT_LABEL_SCHEMA_VERSION
            lbl["label_schema_sha256"] = ROLLOUT_LABEL_SCHEMA_SHA256
            lbl["feature_schema_sha256"] = FEATURE_SCHEMA_SHA256
            lbl["cache_schema_version"] = COMPACT_CACHE_SCHEMA_VERSION
            lbl["motion_oracle_hard"] = hard_gate_from_benefit(lbl["motion_benefit"])
            lbl["appearance_oracle_hard"] = hard_gate_from_benefit(
                lbl["appearance_benefit"]
            )
            lbl["target_gate"] = [
                lbl["motion_oracle_hard"],
                lbl["appearance_oracle_hard"],
            ]
            lbl["policy_soft_target"] = compute_policy_soft_target(
                np.array([motion_soft, appearance_soft], dtype=np.float64)
            ).tolist()
            lbl["policy_safe_soft_target"] = compute_policy_soft_target(
                np.array([motion_safe, appearance_safe], dtype=np.float64)
            ).tolist()

        summary.update(
            {
                "num_labels": len(raw_labels),
                "motion_benefit_positive": int(
                    sum(1 for b in motion_benefits if b > 1e-9)
                ),
                "motion_benefit_negative": int(
                    sum(1 for b in motion_benefits if b < -1e-9)
                ),
                "appearance_benefit_positive": int(
                    sum(1 for b in appearance_benefits if b > 1e-9)
                ),
                "appearance_benefit_negative": int(
                    sum(1 for b in appearance_benefits if b < -1e-9)
                ),
                "dataset_stats": stats,
                "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
                "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            }
        )
        return raw_labels, summary
    finally:
        reader.close()


__all__ = ["build_compact_rollout_labels_for_sequence"]
