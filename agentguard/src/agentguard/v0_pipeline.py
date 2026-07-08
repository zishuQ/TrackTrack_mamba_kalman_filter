from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from agentguard.contracts.states import DetectionObservation
from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.gt_reader import GTReader
from agentguard.data.identity_prototype import IdentityPrototypeBuilder
from agentguard.data.identity_vote import TrackIdentityVoteState
from agentguard.features.builder import EventFeatureBuilder
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.appearance import compute_appearance_benefit
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.motion import compute_motion_benefit
from agentguard.rollout_labels import (
    compute_dataset_stats,
    compute_policy_soft_target,
    compute_soft_target,
    hard_gate_from_benefit,
)


SAMPLE_TYPE_TO_ID = {
    "A": 0,
    "B": 1,
    "C": 2,
    "matched": 0,
    "unmatched": 3,
}


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


def detection_observation(reader: CompactEventCacheReader, detection_index: int) -> DetectionObservation:
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
    gt_entries: Iterable[Tuple[np.ndarray, int]],
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


def gt_box_for_id(gt_reader: GTReader, frame_id: int, gt_id: int) -> Optional[np.ndarray]:
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
    total_frames = int(reader.manifest.get("total_sequence_frames") or reader.manifest.get("num_frames") or 0)
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


def _detection_gt(
    reader: CompactEventCacheReader,
    gt_reader: GTReader,
    frame_id: int,
    detection_index: int,
    min_iou: float = 0.5,
    cache: Optional[Dict[Tuple[int, int], Tuple[int, float, Optional[np.ndarray]]]] = None,
) -> Tuple[int, float, Optional[np.ndarray]]:
    key = (int(frame_id), int(detection_index))
    if cache is not None and key in cache:
        return cache[key]
    det = reader.get_detection(int(detection_index))
    result = match_detection_to_gt(
        np.asarray(det["box"], dtype=np.float64),
        gt_reader.get_gt_for_frame(int(frame_id)),
        min_iou=min_iou,
    )
    if cache is not None:
        cache[key] = result
    return result


def _frame_detection_gt_assignments(
    reader: CompactEventCacheReader,
    gt_reader: GTReader,
    frame_id: int,
    frame_index: int,
    cache: Dict[int, Dict[int, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    frame_index = int(frame_index)
    if frame_index in cache:
        return cache[frame_index]
    if reader.detection_cache is None:
        cache[frame_index] = {}
        return cache[frame_index]
    frame = reader.detection_cache.get_frame(frame_index, view="source")
    gt_entries = list(gt_reader.get_gt_for_frame(int(frame_id)))
    assignments: Dict[int, Dict[str, Any]] = {}
    for det_idx, det_box, source in zip(frame["detection_indices"], frame["boxes"], frame["sources"]):
        gt_id, gt_iou, gt_box = match_detection_to_gt(
            np.asarray(det_box, dtype=np.float64),
            gt_entries,
            min_iou=0.5,
        )
        assignments[int(det_idx)] = {
            "gt_id": int(gt_id),
            "gt_iou": float(gt_iou),
            "gt_box": gt_box,
            "box": np.asarray(det_box, dtype=np.float64),
            "source": int(source),
        }
    cache[frame_index] = assignments
    return assignments


def _association_candidate_indices(
    reader: CompactEventCacheReader,
    record: Dict[str, Any],
) -> Tuple[Optional[Dict[str, np.ndarray]], int, np.ndarray]:
    assoc = reader.get_association(
        int(record.get("association_shard_id", -1)),
        int(record.get("association_offset", -1)),
    )
    row_idx = int(record.get("association_track_row", -1))
    if assoc is None or row_idx < 0:
        return None, -1, np.zeros((0,), dtype=np.int64)
    final_cost = np.asarray(assoc.get("final_cost", []), dtype=np.float64)
    det_indices = np.asarray(assoc.get("detection_indices", []), dtype=np.int64)
    if final_cost.ndim != 2 or row_idx >= final_cost.shape[0] or det_indices.size != final_cost.shape[1]:
        return None, -1, np.zeros((0,), dtype=np.int64)
    return assoc, row_idx, det_indices


def _build_candidate_specs(
    reader: CompactEventCacheReader,
    gt_reader: GTReader,
    record: Dict[str, Any],
    frame_id: int,
    frame_index: int,
    target_gt_id: int,
    current_gt_box: np.ndarray,
    frame_assignment_cache: Dict[int, Dict[int, Dict[str, Any]]],
    candidate_types: set[str],
) -> List[Dict[str, Any]]:
    accepted = int(record.get("accepted_detection_index", -1))
    frame_assignments = _frame_detection_gt_assignments(
        reader,
        gt_reader,
        frame_id,
        frame_index,
        frame_assignment_cache,
    )
    specs: List[Dict[str, Any]] = []
    if accepted >= 0 and "A" in candidate_types:
        assigned = frame_assignments.get(accepted)
        gt_id = int(assigned["gt_id"]) if assigned is not None else -1
        gt_iou = float(assigned["gt_iou"]) if assigned is not None else 0.0
        specs.append(
            {
                "candidate_type": "A",
                "detection_index": accepted,
                "detection_gt_id": int(gt_id),
                "detection_gt_iou": float(gt_iou),
            }
        )

    assoc, row_idx, det_indices = _association_candidate_indices(reader, record)
    if assoc is not None and "B" in candidate_types:
        final_cost = np.asarray(assoc["final_cost"], dtype=np.float64)
        best_b: Optional[Dict[str, Any]] = None
        for col_idx, det_idx in enumerate(det_indices.tolist()):
            det_idx = int(det_idx)
            if det_idx == accepted:
                continue
            assigned = frame_assignments.get(det_idx)
            if assigned is None:
                continue
            gt_id = int(assigned["gt_id"])
            gt_iou = float(assigned["gt_iou"])
            if gt_id < 0 or int(gt_id) == int(target_gt_id):
                continue
            cost = float(final_cost[row_idx, col_idx])
            if best_b is None or cost < best_b["final_cost"]:
                best_b = {
                    "candidate_type": "B",
                    "detection_index": det_idx,
                    "detection_gt_id": int(gt_id),
                    "detection_gt_iou": float(gt_iou),
                    "final_cost": cost,
                }
        if best_b is not None:
            specs.append(best_b)

    if frame_assignments and "C" in candidate_types:
        source_priority = {2: 0, 1: 1, 0: 2}
        best_c: Optional[Dict[str, Any]] = None
        for det_idx, assigned in frame_assignments.items():
            det_idx = int(det_idx)
            if det_idx == accepted:
                continue
            gt_id = int(assigned["gt_id"])
            gt_iou = float(assigned["gt_iou"])
            if int(gt_id) != int(target_gt_id):
                continue
            priority = int(source_priority.get(int(assigned["source"]), 99))
            target_iou = box_iou(np.asarray(assigned["box"], dtype=np.float64), current_gt_box)
            spec = {
                "candidate_type": "C",
                "detection_index": det_idx,
                "detection_gt_id": int(gt_id),
                "detection_gt_iou": float(gt_iou),
                "source_priority": priority,
                "target_iou": float(target_iou),
            }
            if (
                best_c is None
                or priority < best_c["source_priority"]
                or (priority == best_c["source_priority"] and target_iou > best_c["target_iou"])
            ):
                best_c = spec
        if best_c is not None:
            specs.append(best_c)

    return specs


def build_compact_rollout_labels_for_sequence(
    event_cache_dir: str | os.PathLike[str],
    detection_cache_dir: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    *,
    max_events: int = 0,
    future_frames: int = 5,
    candidate_types: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_types = {str(t).upper() for t in (candidate_types or {"A"})}
    invalid_types = candidate_types.difference({"A", "B", "C"})
    if invalid_types:
        raise ValueError(f"Unknown candidate type(s): {sorted(invalid_types)}")
    if not candidate_types:
        candidate_types = {"A"}

    reader = CompactEventCacheReader(event_cache_dir, detection_cache_dir)
    gt_reader = GTReader(str(gt_root), str(reader.manifest["sequence"]))
    motion_model = NSAKalmanFilter()
    vote_state = TrackIdentityVoteState()
    proto_features: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    target_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    frame_assignment_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}

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
                        event.detection.feature.reshape(-1).astype(np.float64)
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

        raw_labels: List[Dict[str, Any]] = []
        motion_benefits: List[float] = []
        appearance_benefits: List[float] = []
        summary = {
            "reliable_identity_events": 0,
            "candidate_a_count": 0,
            "candidate_b_count": 0,
            "candidate_c_count": 0,
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
            candidate_specs = _build_candidate_specs(
                reader,
                gt_reader,
                record,
                event.frame_id,
                int(record.get("frame_index", event.frame_id - 1)),
                int(target_gt_id),
                current_gt,
                frame_assignment_cache,
                candidate_types,
            )
            wrote_any = False
            for spec in candidate_specs:
                if max_events > 0 and len(raw_labels) >= max_events:
                    break
                candidate_event = (
                    event
                    if spec["candidate_type"] == "A"
                    else reader.materialize_training_event(
                        record,
                        candidate_detection_index=int(spec["detection_index"]),
                    )
                )
                ctx = RolloutContext(
                    frame_id=candidate_event.frame_id,
                    target_gt_id=int(target_gt_id),
                    pre_update_state=candidate_event.pre_update_state or candidate_event.frame_start_state,
                    current_candidate=candidate_event.detection,
                    current_gt_box=current_gt,
                    future_gt_boxes=future_gt,
                    future_oracle_detections=future_oracle,
                    future_warp_matrices=future_warps,
                    identity_prototype=proto,
                )

                valid_motion = bool(candidate_event.has_detection and candidate_event.detection is not None)
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
                        "target_identity_key": list(info["target_identity_key"] or (event.sequence, int(target_gt_id))),
                        "detection_gt_id": int(spec["detection_gt_id"]),
                        "detection_gt_iou": float(spec["detection_gt_iou"]),
                        "motion_benefit": float(b_m),
                        "appearance_benefit": float(b_a),
                        "valid_motion": bool(valid_motion),
                        "valid_appearance": bool(valid_appearance),
                        "valid_horizon_count": int(max(motion_valid_count, appearance_valid_count)),
                        "gt_coverage": float(gt_count / max(future_frames + 1, 1)),
                        "oracle_detection_coverage": float(oracle_count / max(future_frames, 1)),
                        "sample_type": "matched" if candidate_event.has_detection else "unmatched",
                        "sample_weight": 1.0,
                    }
                )
                wrote_any = True

            if not wrote_any:
                summary["skipped_invalid"] += 1

        stats = compute_dataset_stats(
            {
                "motion_benefits": [b for b, lbl in zip(motion_benefits, raw_labels) if lbl["valid_motion"]],
                "appearance_benefits": [
                    b for b, lbl in zip(appearance_benefits, raw_labels) if lbl["valid_appearance"]
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
            lbl["motion_soft_target"] = float(motion_soft)
            lbl["appearance_soft_target"] = float(appearance_soft)
            lbl["motion_target"] = float(motion_soft)
            lbl["appearance_target"] = float(appearance_soft)
            lbl["motion_oracle_hard"] = hard_gate_from_benefit(lbl["motion_benefit"])
            lbl["appearance_oracle_hard"] = hard_gate_from_benefit(lbl["appearance_benefit"])
            lbl["target_gate"] = [lbl["motion_oracle_hard"], lbl["appearance_oracle_hard"]]
            lbl["policy_soft_target"] = compute_policy_soft_target(
                np.array([motion_soft, appearance_soft], dtype=np.float64)
            ).tolist()

        summary.update(
            {
                "num_labels": len(raw_labels),
                "motion_benefit_positive": int(sum(1 for b in motion_benefits if b > 1e-9)),
                "motion_benefit_negative": int(sum(1 for b in motion_benefits if b < -1e-9)),
                "appearance_benefit_positive": int(sum(1 for b in appearance_benefits if b > 1e-9)),
                "appearance_benefit_negative": int(sum(1 for b in appearance_benefits if b < -1e-9)),
                "dataset_stats": stats,
            }
        )
        return raw_labels, summary
    finally:
        reader.close()


def load_label_records(label_dir: str | os.PathLike[str], max_samples: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(Path(label_dir).glob("*_labels.json")):
        with path.open("r") as f:
            seq_records = json.load(f)
        for record in seq_records:
            records.append(record)
            if max_samples > 0 and len(records) >= max_samples:
                return records
    return records


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_norm_stats_from_records(
    records: List[Dict[str, Any]],
    event_cache_root: str | os.PathLike[str],
    detection_cache_root: str | os.PathLike[str],
    dataset: str,
    split: str,
):
    from agentguard.features.normalization import NormalizationStats

    readers: Dict[str, CompactEventCacheReader] = {}
    scalars: List[np.ndarray] = []
    try:
        for record in records:
            seq = record["sequence"]
            if seq not in readers:
                readers[seq] = CompactEventCacheReader(
                    Path(event_cache_root) / dataset / split / seq,
                    Path(detection_cache_root) / dataset / split / seq,
                )
            event_record = readers[seq].get_event_record(
                int(record["event_shard_id"]),
                int(record["event_offset"]),
            )
            event = readers[seq].materialize_training_event(event_record)
            if event.scalar_features is not None:
                scalar = np.asarray(event.scalar_features, dtype=np.float64).reshape(-1)
                if scalar.size == 63:
                    scalars.append(scalar)
    finally:
        for reader in readers.values():
            reader.close()
    stats = NormalizationStats()
    stats.fit(scalars)
    return stats


class _CompactDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        event_cache_root: str | os.PathLike[str],
        detection_cache_root: str | os.PathLike[str],
        dataset: str,
        split: str,
        feature_builder: EventFeatureBuilder,
    ) -> None:
        self.records = records
        self.event_cache_root = Path(event_cache_root)
        self.detection_cache_root = Path(detection_cache_root)
        self.dataset = dataset
        self.split = split
        self.feature_builder = feature_builder
        self._readers: Dict[str, CompactEventCacheReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def _reader(self, sequence: str) -> CompactEventCacheReader:
        if sequence not in self._readers:
            self._readers[sequence] = CompactEventCacheReader(
                self.event_cache_root / self.dataset / self.split / sequence,
                self.detection_cache_root / self.dataset / self.split / sequence,
            )
        return self._readers[sequence]

    def _event(self, record: Dict[str, Any]):
        reader = self._reader(record["sequence"])
        event_record = reader.get_event_record(
            int(record["event_shard_id"]),
            int(record["event_offset"]),
        )
        candidate_detection_index = int(record.get("candidate_detection_index", -1))
        candidate_type = str(record.get("candidate_type", "A"))
        if candidate_detection_index >= 0 and (
            candidate_type != "A"
            or candidate_detection_index != int(event_record.get("accepted_detection_index", -1))
        ):
            event = reader.materialize_training_event(
                event_record,
                candidate_detection_index=candidate_detection_index,
            )
        else:
            event = reader.materialize_training_event(event_record)
        event.iwg_gate = np.array(
            [record.get("motion_soft_target", 1.0), record.get("appearance_soft_target", 1.0)],
            dtype=np.float64,
        )
        event.iwg_policy_probs = np.asarray(
            record.get("policy_soft_target", [0.2] * 5),
            dtype=np.float64,
        )
        return event

    @staticmethod
    def _targets(label: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        sample_type = label.get("candidate_type") or label.get("sample_type", "matched")
        return {
            "motion_target": torch.tensor(float(label.get("motion_soft_target", label.get("motion_target", 0.5))), dtype=torch.float),
            "appearance_target": torch.tensor(float(label.get("appearance_soft_target", label.get("appearance_target", 0.5))), dtype=torch.float),
            "policy_soft_target": torch.tensor(label.get("policy_soft_target", [0.2] * 5), dtype=torch.float),
            "sample_type": torch.tensor(SAMPLE_TYPE_TO_ID.get(str(sample_type), 0), dtype=torch.long),
            "valid_motion": torch.tensor(bool(label.get("valid_motion", True)), dtype=torch.bool),
            "valid_appearance": torch.tensor(bool(label.get("valid_appearance", True)), dtype=torch.bool),
            "sample_weight": torch.tensor(float(label.get("sample_weight", 1.0)), dtype=torch.float),
        }


class CompactV0IWGDataset(_CompactDatasetBase):
    def __init__(self, *args, max_history: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_history = int(max_history)
        self._track_index: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
        for idx, record in enumerate(self.records):
            self._track_index[(record["sequence"], int(record["track_id"]))].append(
                (int(record["frame_id"]), idx)
            )
        for entries in self._track_index.values():
            entries.sort(key=lambda item: item[0])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        entries = self._track_index[(record["sequence"], int(record["track_id"]))]
        pos = next((i for i, (_, rec_idx) in enumerate(entries) if rec_idx == idx), -1)
        prev_indices: List[int] = []
        if pos >= 0:
            prev_indices = [rec_idx for _, rec_idx in entries[max(0, pos - self.max_history):pos]]
        seq_records = [self.records[i] for i in prev_indices] + [record]
        events = [self._event(r) for r in seq_records]
        seq_len = self.max_history + 1
        if len(events) < seq_len:
            events = [None] * (seq_len - len(events)) + events
        inputs = self.feature_builder.build_iwg_input(events)
        return {
            "track_feats": inputs["track_feats"].squeeze(0),
            "det_feats": inputs["det_feats"].squeeze(0),
            "scalar_feats": inputs["scalar_feats"].squeeze(0),
            "mask": inputs["mask"].squeeze(0),
            "targets": self._targets(record),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from agentguard.datasets.iwg_dataset import IWGDataset

        return IWGDataset.collate_fn(batch)


class CompactV0TGRDataset(_CompactDatasetBase):
    def __init__(self, *args, window_size: int = 4, window_stride: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_size = int(window_size)
        self.window_stride = max(int(window_stride), 1)
        self._windows = self._build_windows()

    def _build_windows(self) -> List[List[int]]:
        groups: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
        for idx, record in enumerate(self.records):
            groups[(record["sequence"], int(record["track_id"]))].append((int(record["frame_id"]), idx))
        windows: List[List[int]] = []
        for entries in groups.values():
            entries.sort(key=lambda item: item[0])
            for i in range(0, max(0, len(entries) - self.window_size + 1), self.window_stride):
                chunk = entries[i:i + self.window_size]
                if len(chunk) != self.window_size:
                    continue
                if all(chunk[j + 1][0] - chunk[j][0] <= 1 for j in range(self.window_size - 1)):
                    windows.append([idx for _, idx in chunk])
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self._windows[idx]
        records = [self.records[i] for i in indices]
        events = [self._event(r) for r in records]
        inputs = self.feature_builder.build_tgr_input(events)
        target_gate = np.asarray([r.get("target_gate", [1.0, 1.0]) for r in records], dtype=np.float64)
        sample_weight = np.asarray([r.get("sample_weight", 1.0) for r in records], dtype=np.float64)
        valid_motion = np.asarray([r.get("valid_motion", True) for r in records], dtype=bool)
        valid_appearance = np.asarray([r.get("valid_appearance", True) for r in records], dtype=bool)
        return {
            "track_feats": inputs["track_feats"].squeeze(0),
            "det_feats": inputs["det_feats"].squeeze(0),
            "scalar_feats": inputs["scalar_feats"].squeeze(0),
            "iwg_policy_probs": inputs["iwg_policy_probs"].squeeze(0),
            "iwg_gates": inputs["iwg_gates"].squeeze(0),
            "has_detection_mask": inputs["has_detection_mask"].squeeze(0),
            "target_gate": torch.from_numpy(target_gate).float(),
            "sample_weight": torch.from_numpy(sample_weight).float(),
            "valid_motion": torch.from_numpy(valid_motion),
            "valid_appearance": torch.from_numpy(valid_appearance),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from agentguard.datasets.tgr_dataset import TGRDataset

        stacked = TGRDataset.collate_fn(batch)
        stacked["valid_motion"] = torch.stack([b["valid_motion"] for b in batch])
        stacked["valid_appearance"] = torch.stack([b["valid_appearance"] for b in batch])
        return stacked
