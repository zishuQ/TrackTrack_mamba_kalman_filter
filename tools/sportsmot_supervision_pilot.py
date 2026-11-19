#!/usr/bin/env python3
"""SportsMOT supervision pilot on three sequences.

One A-e200 online replay per sequence produces, for every model-supervised
match event: the exact A window inputs, the legacy rollout labels rebuilt with
the repository's own rule (identity vote, identity prototypes, NSA/appearance
rollouts), the g=0/g=1 next-frame motion endpoints, and appearance
discrimination diagnostics against future GT positives and true same-frame
competitors.

Outputs per sequence:
  events.jsonl   per-event keys, labels, masks, motion endpoints, appearance scores
  samples.npz    window tensors (evidence/competition/track features), detection
                 cache indices, labels and motion boxes ready for batch loading
  summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE),
    str(WORKTREE / "3. Tracker"),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "tools"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentguard.contracts.states import DetectionObservation, TrackStateSnapshot
from agentguard.data.identity_prototype import IdentityPrototypeBuilder
from agentguard.data.identity_vote import TrackIdentityVoteState
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.rollout.appearance import compute_appearance_benefit, ema_update
from agentguard.rollout.motion import compute_motion_benefit
from agentguard.rollout.context import RolloutContext
from agentguard.rollout_labels import (
    compute_label_confidence,
    compute_policy_soft_target,
    compute_safe_soft_target,
    compute_soft_target,
)
from agentguard.data.label_schema import make_cue_target, make_risk_targets

DEFAULT_CACHE = MAIN / "outputs/agentguard/detection_cache"
DEFAULT_CKPT = (
    MAIN
    / "outputs/agentguard/experiments/shutrack_a_submit_20261117/SportsMOT/checkpoints/last.pt"
)
DEFAULT_OUT = (
    MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_pilot_20261118"
)
GT_ROOT = Path("/home/shang/datasets/SportsMOT/dataset")
SEQUENCES = ("v_9MHDmAMxO5I_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")
FUTURE_FRAMES = 5


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def diou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    iou = iou_xyxy(a, b)
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    return iou - ((acx - bcx) ** 2 + (acy - bcy) ** 2) / max(c2, 1e-9)


def load_gt(sequence: str, split: str) -> dict[int, dict[int, np.ndarray]]:
    frames: dict[int, dict[int, np.ndarray]] = {}
    path = GT_ROOT / split / sequence / "gt" / "gt.txt"
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        gt_id = int(float(fields[1]))
        if len(fields) >= 8 and (float(fields[6]) <= 0 or float(fields[7]) != 1):
            continue
        x, y, w, h = map(float, fields[2:6])
        frames.setdefault(frame, {})[gt_id] = np.array([x, y, x + w, y + h])
    return frames


def load_warps(sequence: str) -> dict[int, np.ndarray]:
    path = WORKTREE / "3. Tracker" / "trackers" / "cmc" / f"GMC-sportsmot-{sequence}.txt"
    warps: dict[int, np.ndarray] = {}
    with path.open() as handle:
        for index, line in enumerate(handle):
            tokens = line.strip().split("\t")
            if len(tokens) < 7:
                continue
            warp = np.array(
                [
                    [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                    [float(tokens[4]), float(tokens[5]), float(tokens[6])],
                ],
                dtype=np.float64,
            )
            warps[index + 1] = warp
    return warps


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))


def interpolate_box(box0: np.ndarray, box1: np.ndarray, gate: float) -> np.ndarray:
    return box0 + float(gate) * (box1 - box0)


def kf_update_predict(mean, covariance, measurement, score, warp, kf):
    mean_full, cov_full = kf.update(
        np.asarray(mean, dtype=np.float64).copy(),
        np.asarray(covariance, dtype=np.float64).copy(),
        np.asarray(measurement, dtype=np.float64),
        float(score),
    )
    return mean_full, cov_full


def warp_predict(mean, covariance, warp, kf):
    rot = np.asarray(warp, dtype=np.float64)[:, :2]
    rot8 = np.kron(np.eye(4, dtype=np.float64), rot)
    trans = np.asarray(warp, dtype=np.float64)[:, 2]
    moved = rot8 @ np.asarray(mean, dtype=np.float64)
    moved = moved.copy()
    moved[:2] += trans
    moved_cov = rot8 @ np.asarray(covariance, dtype=np.float64) @ rot8.T
    pred_mean, pred_cov = kf.predict(moved, moved_cov)
    return pred_mean, pred_cov


def mean_to_box(mean: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    return np.array(
        [
            mean[0] - mean[2] / 2,
            mean[1] - mean[3] / 2,
            mean[0] + mean[2] / 2,
            mean[1] + mean[3] / 2,
        ]
    )


class SequencePilot:
    def __init__(self, sequence: str, split: str, checkpoint: Path, out_dir: Path, max_frames: int = 0):
        self.sequence = sequence
        self.split = split
        self.checkpoint = checkpoint
        self.out_dir = out_dir / sequence
        self.max_frames = max_frames
        self.gt = load_gt(sequence, split)
        self.warps = load_warps(sequence)
        self.events: list[dict] = []
        self.adapter_events: dict[tuple[int, int], dict] = {}
        self.windows: dict[tuple[int, int], dict] = {}
        self.window_frames: dict[tuple[int, int], list[int]] = {}
        self.frame_det_index: dict[tuple[int, int], int] = {}
        self.track_events: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        self.vote = TrackIdentityVoteState()
        self.frame_matches_cache: dict[int, list[dict]] = {}
        self.det_gt: dict[tuple[int, int], tuple[int, float]] = {}
        self.prototypes: dict[int, np.ndarray] = {}
        self.kf = None
        self.frame_matches_precomputed = False

    # ------------------------------------------------------------------
    def frame_matches(self, frame_index: int) -> list[dict]:
        cached = self.frame_matches_cache.get(frame_index)
        if cached is not None:
            return cached
        frame_id = frame_index + 1
        entries: list[dict] = []
        indices = self.det_cache.get_frame(frame_index, view="target")["detection_indices"]
        array = self.det_cache.get_frame_array(frame_index, view="target")
        gt_boxes = self.gt.get(frame_id, {})
        if array is not None:
            for row, global_index in zip(array, indices):
                box = np.asarray(row[:4], dtype=np.float64)
                score = float(row[4])
                best_id, best_iou = -1, 0.5
                for gt_id, gt_box in gt_boxes.items():
                    value = iou_xyxy(box, gt_box)
                    if value > best_iou:
                        best_iou = value
                        best_id = int(gt_id)
                entries.append(
                    {
                        "index": int(global_index),
                        "box": box,
                        "score": score,
                        "gt_id": best_id,
                        "iou": float(best_iou),
                    }
                )
                self.det_gt[(frame_id, int(global_index))] = (best_id, float(best_iou))
        self.frame_matches_cache[frame_index] = entries
        return entries

    def detection_observation(self, index: int) -> DetectionObservation:
        det = self.det_cache.get_detection(int(index))
        return DetectionObservation(
            detection_index=int(index),
            box=np.asarray(det["box"], dtype=np.float64),
            score=float(det["score"]),
            feature=np.asarray(det["feature"], dtype=np.float64).reshape(1, -1),
            source=int(det["source"]),
            class_id=int(det["class_id"]),
        )

    def build_prototypes(self) -> None:
        features: dict[int, list[np.ndarray]] = defaultdict(list)
        for frame_index in range(self.det_cache.num_frames):
            for entry in self.frame_matches(frame_index):
                if entry["gt_id"] >= 0 and entry["iou"] >= 0.7 and entry["score"] >= 0.6:
                    det = self.det_cache.get_detection(entry["index"])
                    features[entry["gt_id"]].append(
                        np.asarray(det["feature"], dtype=np.float32).reshape(-1)
                    )
        self.prototypes = {
            gt_id: IdentityPrototypeBuilder.build_prototype(values)
            for gt_id, values in features.items()
            if len(values) >= 3
        }

    def future_context(self, frame_id: int, target_gt_id: int):
        current_gt = self.gt.get(frame_id, {}).get(target_gt_id)
        future_gt: list[np.ndarray | None] = []
        future_oracle: list[DetectionObservation | None] = []
        future_warps: list[np.ndarray] = []
        gt_count = 1 if current_gt is not None else 0
        oracle_count = 0
        for step in range(1, FUTURE_FRAMES + 1):
            fid = frame_id + step
            if fid > self.det_cache.num_frames:
                break
            gt_box = self.gt.get(fid, {}).get(target_gt_id)
            future_gt.append(gt_box)
            if gt_box is not None:
                gt_count += 1
            oracle = None
            if gt_box is not None:
                frame_index = fid - 1
                array = self.det_cache.get_frame_array(frame_index, view="source")
                indices = self.det_cache.get_frame(frame_index, view="source")[
                    "detection_indices"
                ]
                best_iou = 0.3
                best_index = -1
                if array is not None and len(array) > 0:
                    for row, global_index in zip(array, indices):
                        value = iou_xyxy(
                            gt_box, np.asarray(row[:4], dtype=np.float64)
                        )
                        if value > best_iou:
                            best_iou = value
                            best_index = int(global_index)
                if best_index >= 0:
                    oracle = self.detection_observation(best_index)
                    oracle_count += 1
            future_oracle.append(oracle)
            future_warps.append(self.warps.get(fid, np.eye(2, 3)))
        return current_gt, future_gt, future_oracle, future_warps, gt_count, oracle_count

    # ------------------------------------------------------------------
    def install_hooks(self):
        from trackers import tracker as tracker_module
        from trackers.track import Track
        import shutrack.runtime.adapter as runtime_adapter
        from shutrack.integration.tracktrack import ShutrackTrackerAdapter

        self._restore = []
        pilot = self

        original_update = tracker_module.Tracker.update
        original_update_no_det = tracker_module.Tracker.update_without_detections
        original_init_tracks = tracker_module.Tracker.init_tracks
        original_track_update = Track.update
        original_track_update_gates = Track.update_with_gates
        original_build_window = runtime_adapter.build_window
        original_build_event = ShutrackTrackerAdapter.build_matched_event

        def tracker_update(self, dets, dets_95):
            result = original_update(self, dets, dets_95)
            if self._current_warp is not None:
                pilot.warps_online[int(self.frame_id)] = np.asarray(self._current_warp).copy()
            return result

        def tracker_update_no_det(self):
            result = original_update_no_det(self)
            if self._current_warp is not None:
                pilot.warps_online[int(self.frame_id)] = np.asarray(self._current_warp).copy()
            return result

        def init_tracks(self, dets):
            before = {int(t.track_id) for t in self.tracks}
            original_init_tracks(self, dets)
            for track in self.tracks:
                if int(track.track_id) not in before:
                    pilot.replay_track_ids.add(int(track.track_id))

        def record_event(track, frame_id, detection, motion_gate, appearance_gate, source):
            snapshot = track.snapshot_state(compact_history=True)
            det_index = int(getattr(detection, "frame_detection_index", -1))
            det_box = np.asarray(detection.x1y1x2y2, dtype=np.float64)
            det_score = float(detection.score)
            det_feature = np.asarray(detection.feat, dtype=np.float32).reshape(-1)
            gt_id_match, gt_iou = pilot.det_gt.get((frame_id, det_index), (-1, 0.0))
            target_before = pilot.vote.resolve_before_current(pilot.sequence, track.track_id)
            pilot.vote.add_current_observation(pilot.sequence, track.track_id, gt_id_match)
            pilot.frame_det_index[(int(track.track_id), int(frame_id))] = det_index
            pilot.track_events[int(track.track_id)].append(
                (int(frame_id), int(gt_id_match), snapshot.feature.reshape(-1).copy())
            )
            record = {
                "sequence": pilot.sequence,
                "frame_t": int(frame_id),
                "track_id": int(track.track_id),
                "detection_index": det_index,
                "source": source,
                "state_before": int(track.state),
                "observation_count": int(track.observation_count),
                "gap_before": int(frame_id - track.end_frame_id)
                if int(track.end_frame_id) > 0
                else -1,
                "motion_gate": None if motion_gate is None else float(motion_gate),
                "appearance_gate": None if appearance_gate is None else float(appearance_gate),
                "detection_gt_id": int(gt_id_match),
                "detection_gt_iou": float(gt_iou),
                "target_gt_before": None if target_before is None else int(target_before),
                "detection_box": det_box.tolist(),
                "detection_score": det_score,
                "detection_feature": None,
                "pre_feature": snapshot.feature.reshape(-1).astype(np.float32).tolist(),
                "snapshot": snapshot,
                "detection_measurement": np.asarray(detection.cxcywh, dtype=np.float64).tolist(),
                "mean_prior": np.asarray(snapshot.mean, dtype=np.float64).copy(),
                "cov_prior": np.asarray(snapshot.covariance, dtype=np.float64).copy(),
                "det_feature": det_feature,
            }
            pilot.events.append(record)
            pilot._compute_motion_endpoints(record)
            return record

        def track_update(self, frame_id, detection):
            source = (
                "new_path_unmature" if int(self.state) == 0 else "tracked_lost_unmature"
            )
            record_event(self, frame_id, detection, None, None, source)
            return original_track_update(self, frame_id, detection)

        def track_update_gates(self, frame_id, detection, motion_gate, appearance_gate):
            record_event(self, frame_id, detection, motion_gate, appearance_gate, "gated")
            return original_track_update_gates(
                self, frame_id, detection, motion_gate, appearance_gate
            )

        def build_window(history, current, *args, **kwargs):
            window = original_build_window(history, current, *args, **kwargs)
            pilot.windows[(int(current.track_id), int(current.frame_id))] = window
            pilot.window_frames[(int(current.track_id), int(current.frame_id))] = [
                int(event.frame_id) for event in history
            ]
            return window

        def build_matched_event(adapter_self, track, detection, track_idx, det_idx, *args, **kwargs):
            event = original_build_event(
                adapter_self, track, detection, track_idx, det_idx, *args, **kwargs
            )
            pilot.adapter_events[(int(adapter_self.frame_id), int(track.track_id))] = {
                "evidence": event.evidence,
                "detection_index": int(getattr(detection, "frame_detection_index", -1)),
            }
            return event

        tracker_module.Tracker.update = tracker_update
        tracker_module.Tracker.update_without_detections = tracker_update_no_det
        tracker_module.Tracker.init_tracks = init_tracks
        Track.update = track_update
        Track.update_with_gates = track_update_gates
        runtime_adapter.build_window = build_window
        ShutrackTrackerAdapter.build_matched_event = build_matched_event
        self._restore = [
            (tracker_module.Tracker, "update", original_update),
            (tracker_module.Tracker, "update_without_detections", original_update_no_det),
            (tracker_module.Tracker, "init_tracks", original_init_tracks),
            (Track, "update", original_track_update),
            (Track, "update_with_gates", original_track_update_gates),
            (runtime_adapter, "build_window", original_build_window),
            (ShutrackTrackerAdapter, "build_matched_event", original_build_event),
        ]

    def remove_hooks(self):
        for owner, name, original in getattr(self, "_restore", []):
            setattr(owner, name, original)

    def _compute_motion_endpoints(self, record: dict) -> None:
        frame_id = record["frame_t"]
        target = record["target_gt_before"]
        warp_next = self.warps.get(frame_id + 1)
        mean_prior = record["mean_prior"]
        cov_prior = record["cov_prior"]
        valid = False
        box0 = box1 = None
        if warp_next is not None and mean_prior is not None and cov_prior is not None:
            measurement = np.asarray(record["detection_measurement"], dtype=np.float64)
            score = float(record["detection_score"])
            mean_full, cov_full = kf_update_predict(
                mean_prior, cov_prior, measurement, score, warp_next, self.kf
            )
            mean0, cov0 = warp_predict(mean_prior, cov_prior, warp_next, self.kf)
            mean1, cov1 = warp_predict(mean_full, cov_full, warp_next, self.kf)
            box0 = mean_to_box(mean0).tolist()
            box1 = mean_to_box(mean1).tolist()
            valid = True
            velocity0 = np.asarray(mean0[4:6], dtype=np.float64) / np.array(
                [float(self.img_w), float(self.img_h)]
            )
            velocity1 = np.asarray(mean1[4:6], dtype=np.float64) / np.array(
                [float(self.img_w), float(self.img_h)]
            )
            mean_mid = 0.5 * (
                np.asarray(mean_prior, dtype=np.float64)
                + np.asarray(mean_full, dtype=np.float64)
            )
            mean_mid_pred, _ = warp_predict(mean_mid, cov_prior, warp_next, self.kf)
            velocity_mid_actual = np.asarray(mean_mid_pred[4:6], dtype=np.float64) / np.array(
                [float(self.img_w), float(self.img_h)]
            )
            velocity_mid_linear = 0.5 * (velocity0 + velocity1)
            record["kf_velocity_g0"] = velocity0.tolist()
            record["kf_velocity_g1"] = velocity1.tolist()
            record["kf_velocity_mid_error"] = float(
                np.max(np.abs(velocity_mid_actual - velocity_mid_linear))
            )
        record["pred_box_g0"] = box0
        record["pred_box_g1"] = box1
        record["motion_forward_valid"] = valid
        gt_t1 = None
        if target is not None:
            gt_t1 = self.gt.get(frame_id + 1, {}).get(int(target))
        record["gt_box_t1"] = None if gt_t1 is None else np.asarray(gt_t1).tolist()
        record["motion_gt_valid"] = bool(valid and gt_t1 is not None)

        gt_now = None
        if target is not None:
            gt_now = self.gt.get(frame_id, {}).get(int(target))
        speed_valid = bool(
            valid and gt_now is not None and gt_t1 is not None and warp_next is not None
        )
        record["speed_valid"] = speed_valid
        if speed_valid:
            center_t = np.array(
                [
                    (gt_now[0] + gt_now[2]) / 2.0,
                    (gt_now[1] + gt_now[3]) / 2.0,
                ],
                dtype=np.float64,
            )
            center_t1 = np.array(
                [
                    (gt_t1[0] + gt_t1[2]) / 2.0,
                    (gt_t1[1] + gt_t1[3]) / 2.0,
                ],
                dtype=np.float64,
            )
            rot = np.asarray(warp_next, dtype=np.float64)[:, :2]
            trans = np.asarray(warp_next, dtype=np.float64)[:, 2]
            warped_center = rot @ center_t + trans
            displacement = center_t1 - warped_center
            record["gt_velocity"] = [
                float(displacement[0] / float(self.img_w)),
                float(displacement[1] / float(self.img_h)),
            ]
        else:
            record["gt_velocity"] = None

    # ------------------------------------------------------------------
    def run_replay(self) -> None:
        import run as run_module
        from agentguard.data.detection_cache import SequenceDetectionCache
        from trackers.track import Track
        from agentguard.contracts.states import TrackStateSnapshot
        from trackers.kalman_filter import create_kalman_filter

        self.kf = create_kalman_filter("nsa")
        self.warps_online = {}
        self.replay_track_ids = set()
        args = run_module.make_parser().parse_args([])
        args.dataset = "SportsMOT"
        args.mode = self.split
        args.sequences = [self.sequence]
        args.seed = 10000
        args.kf_type = "nsa"
        args.agentguard_mode = "shutrack-v2"
        args.shutrack_checkpoint = str(self.checkpoint)
        args.agentguard_device = "cpu"
        args.detection_cache_root = str(DEFAULT_CACHE)
        args.skip_eval = True
        args.torch_num_threads = 4
        args.torch_num_interop_threads = 1
        run_module.set_parameters(args, self.sequence, self.split)
        seqinfo = GT_ROOT / self.split / self.sequence / "seqinfo.ini"
        for line in seqinfo.read_text().splitlines():
            if line.startswith("frameRate"):
                args.max_time_lost = int(line.split("=")[-1]) * 2
            if line.startswith("imWidth"):
                args.img_w = int(line.split("=")[-1])
            if line.startswith("imHeight"):
                args.img_h = int(line.split("=")[-1])
        self.img_w = int(args.img_w)
        self.img_h = int(args.img_h)
        from trackers import tracker as tracker_module
        from agentguard.data.detection_cache import SequenceDetectionCache as _SDC

        self.det_cache = _SDC(
            run_module._sequence_detection_cache_dir(args, self.sequence),
            frame_array_cache_size=4,
        )
        try:
            self.build_prototypes()
            self.install_hooks()
            try:
                tracker = tracker_module.Tracker(args, self.sequence)
                frame_count = self.det_cache.num_frames
                if self.max_frames > 0:
                    frame_count = min(frame_count, self.max_frames)
                self.frame_count = frame_count
                for frame_index in range(frame_count):
                    target = self.det_cache.get_frame(frame_index, view="target")
                    source = self.det_cache.get_frame(frame_index, view="source")
                    args.agentguard_target_detection_indices = target["detection_indices"]
                    args.agentguard_source_detection_indices = source["detection_indices"]
                    det_frame = self.det_cache.get_frame_array(frame_index, view="target")
                    det_frame_95 = self.det_cache.get_frame_array(frame_index, view="source")
                    if det_frame is not None and len(det_frame) > 0:
                        tracker.update(det_frame, det_frame_95)
                    else:
                        tracker.update_without_detections()
            finally:
                self.remove_hooks()
        except Exception:
            self.det_cache.close()
            raise
        mismatches = [
            float(np.max(np.abs(self.warps_online[frame] - self.warps[frame])))
            for frame in self.warps_online
            if frame in self.warps
        ]
        self.warp_error = max(mismatches) if mismatches else float("nan")

    def write_compact(self, data_root: Path) -> dict:
        """Save a compact, mmap-friendly training shard for this sequence."""
        sequence_dir = Path(data_root) / self.sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        mature = [
            record
            for record in self.events
            if (record["frame_t"], record["track_id"]) in self.adapter_events
            and (record["track_id"], record["frame_t"]) in self.windows
        ]
        mature.sort(key=lambda record: (record["frame_t"], record["track_id"]))
        event_index: dict[tuple[int, int], int] = {}
        event_evidence: list[np.ndarray] = []
        event_track: list[np.ndarray] = []
        event_det: list[int] = []
        event_has: list[bool] = []

        def register(key, evidence, track, det_index, has_detection):
            if key in event_index:
                return event_index[key]
            index = len(event_evidence)
            event_index[key] = index
            event_evidence.append(np.asarray(evidence, dtype=np.float32))
            event_track.append(np.asarray(track, dtype=np.float32))
            event_det.append(int(det_index))
            event_has.append(bool(has_detection))
            return index

        count = len(mature)
        window_indices = np.full((count, 5), -1, dtype=np.int64)
        current_event_index = np.zeros(count, dtype=np.int64)
        current_competition = np.zeros((count, 3), dtype=np.float32)
        for row, record in enumerate(mature):
            key = (record["track_id"], record["frame_t"])
            window = self.windows[key]
            frames = self.window_frames[key]
            pad = 5 - len(frames)
            for slot in range(5):
                if bool(window.history_mask[slot]):
                    continue
                history_frame = frames[slot - pad]
                window_indices[row, slot] = register(
                    (record["track_id"], history_frame),
                    window.history_evidence[slot],
                    window.history_track[slot],
                    self.frame_det_index.get((record["track_id"], history_frame), -1),
                    window.history_has_detection[slot],
                )
            current_event_index[row] = register(
                key,
                window.current_evidence,
                window.current_track,
                record["detection_index"],
                window.current_has_detection,
            )
            current_competition[row] = window.current_competition

        source_ids = {"tracked_lost_unmature": 0, "new_path_unmature": 1, "gated": 2}
        rank_valid = np.asarray(
            [bool(record.get("rank_valid", False)) for record in mature], dtype=np.bool_
        )
        rank_rows = np.nonzero(rank_valid)[0]

        def stack_rank(key: str) -> np.ndarray:
            if rank_rows.size == 0:
                return np.zeros((0, 2048), dtype=np.float16)
            return np.stack(
                [
                    np.asarray(mature[index][key], dtype=np.float32)
                    for index in rank_rows
                ]
            ).astype(np.float16)

        rank_template0 = stack_rank("pre_feature")
        rank_detection = stack_rank("det_feature")
        rank_positive = stack_rank("rank_positive_feature")
        rank_competitor = stack_rank("rank_competitor_feature")
        rank_score = np.asarray(
            [float(mature[index].get("detection_score", 0.0)) for index in rank_rows],
            dtype=np.float32,
        )
        speed_valid = np.asarray(
            [bool(record.get("speed_valid", False)) for record in mature], dtype=np.bool_
        )
        kf_velocity_g0 = np.asarray(
            [record.get("kf_velocity_g0") or [0.0, 0.0] for record in mature],
            dtype=np.float32,
        )
        kf_velocity_g1 = np.asarray(
            [record.get("kf_velocity_g1") or [0.0, 0.0] for record in mature],
            dtype=np.float32,
        )
        gt_velocity = np.asarray(
            [record.get("gt_velocity") or [0.0, 0.0] for record in mature],
            dtype=np.float32,
        )
        velocity_mid_errors = [
            float(record["kf_velocity_mid_error"])
            for record in mature
            if record.get("kf_velocity_mid_error") is not None
        ]
        arrays = {
            "frame": np.asarray([record["frame_t"] for record in mature], dtype=np.int64),
            "track_id": np.asarray([record["track_id"] for record in mature], dtype=np.int64),
            "detection_index": np.asarray(
                [record["detection_index"] for record in mature], dtype=np.int64
            ),
            "source": np.asarray(
                [source_ids.get(record["source"], -1) for record in mature], dtype=np.int64
            ),
            "gt_id": np.asarray(
                [
                    -1 if record.get("target_gt_before") is None else record["target_gt_before"]
                    for record in mature
                ],
                dtype=np.int64,
            ),
            "motion_benefit": np.asarray(
                [record.get("motion_benefit", 0.0) for record in mature], np.float32
            ),
            "appearance_benefit": np.asarray(
                [record.get("appearance_benefit", 0.0) for record in mature], np.float32
            ),
            "valid_motion": np.asarray(
                [bool(record.get("valid_motion", False)) for record in mature], np.bool_
            ),
            "valid_appearance": np.asarray(
                [bool(record.get("valid_appearance", False)) for record in mature], np.bool_
            ),
            "gt_coverage": np.asarray(
                [record.get("gt_coverage", 0.0) for record in mature], np.float32
            ),
            "oracle_detection_coverage": np.asarray(
                [record.get("oracle_detection_coverage", 0.0) for record in mature],
                np.float32,
            ),
            "motion_gt_valid": np.asarray(
                [bool(record.get("motion_gt_valid", False)) for record in mature], np.bool_
            ),
            "pred_box_g0": np.asarray(
                [record.get("pred_box_g0") or [0.0] * 4 for record in mature], np.float32
            ),
            "pred_box_g1": np.asarray(
                [record.get("pred_box_g1") or [0.0] * 4 for record in mature], np.float32
            ),
            "gt_box_t1": np.asarray(
                [record.get("gt_box_t1") or [0.0] * 4 for record in mature], np.float32
            ),
            "appearance_delta": np.asarray(
                [
                    (
                        record.get("appearance_score_template1", 0.0)
                        - record.get("appearance_score_template0", 0.0)
                        if record.get("appearance_diag_valid")
                        else 0.0
                    )
                    for record in mature
                ],
                np.float32,
            ),
            "appearance_diag_valid": np.asarray(
                [bool(record.get("appearance_diag_valid", False)) for record in mature],
                np.bool_,
            ),
            "appearance_score_template0": np.asarray(
                [record.get("appearance_score_template0", 0.0) for record in mature],
                np.float32,
            ),
            "appearance_score_template1": np.asarray(
                [record.get("appearance_score_template1", 0.0) for record in mature],
                np.float32,
            ),
            "motion_gate": np.asarray(
                [
                    -1.0 if record.get("motion_gate") is None else record["motion_gate"]
                    for record in mature
                ],
                np.float32,
            ),
            "appearance_gate": np.asarray(
                [
                    -1.0 if record.get("appearance_gate") is None else record["appearance_gate"]
                    for record in mature
                ],
                np.float32,
            ),
            "window_indices": window_indices,
            "current_event_index": current_event_index,
            "current_competition": current_competition,
            "rank_valid": rank_valid,
            "rank_rows": rank_rows.astype(np.int64),
            "rank_template0": rank_template0,
            "rank_detection": rank_detection,
            "rank_positive": rank_positive,
            "rank_competitor": rank_competitor,
            "rank_score": rank_score,
            "speed_valid": speed_valid,
            "kf_velocity_g0": kf_velocity_g0,
            "kf_velocity_g1": kf_velocity_g1,
            "gt_velocity": gt_velocity,
            "keys_json": json.dumps(
                [
                    {
                        "sequence": record["sequence"],
                        "frame": record["frame_t"],
                        "track_id": record["track_id"],
                        "detection_index": record["detection_index"],
                    }
                    for record in mature
                ]
            ),
        }
        np.savez(sequence_dir / "samples.npz", **arrays)
        np.save(
            sequence_dir / "event_evidence.npy",
            np.stack(event_evidence) if event_evidence else np.zeros((0, 13), np.float32),
        )
        np.save(
            sequence_dir / "event_track.npy",
            np.stack(event_track) if event_track else np.zeros((0, 1), np.float32),
        )
        np.save(
            sequence_dir / "event_detection_index.npy",
            np.asarray(event_det, dtype=np.int64),
        )
        np.save(
            sequence_dir / "event_has_detection.npy",
            np.asarray(event_has, dtype=np.bool_),
        )
        summary = {
            "sequence": self.sequence,
            "split": self.split,
            "samples": count,
            "events": len(event_evidence),
            "frames": int(self.frame_count),
            "valid_motion": int(arrays["valid_motion"].sum()),
            "valid_appearance": int(arrays["valid_appearance"].sum()),
            "motion_gt_valid": int(arrays["motion_gt_valid"].sum()),
            "appearance_diag_valid": int(arrays["appearance_diag_valid"].sum()),
            "rank_valid": int(rank_valid.sum()),
            "speed_valid": int(speed_valid.sum()),
            "kf_velocity_mid_error_max": (
                float(max(velocity_mid_errors)) if velocity_mid_errors else None
            ),
            "img_width": int(self.img_w),
            "img_height": int(self.img_h),
            "unique_keys": len(
                {
                    (record["sequence"], record["frame_t"], record["track_id"], record["detection_index"])
                    for record in mature
                }
            ),
            "warp_max_abs_error": float(self.warp_error),
            "prototypes": len(self.prototypes),
            "complete": True,
        }
        (sequence_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary

    def close(self) -> None:
        cache = getattr(self, "det_cache", None)
        if cache is not None:
            cache.close()
            self.det_cache = None

    # ------------------------------------------------------------------
    def compute_labels(self) -> None:
        label_rows = []
        for record in self.events:
            target = record["target_gt_before"]
            mature = (record["frame_t"], record["track_id"]) in self.adapter_events
            if not mature or record["detection_index"] < 0:
                continue
            if target is None:
                record["labels_valid"] = False
                continue
            current_gt, future_gt, future_oracle, future_warps, gt_count, oracle_count = (
                self.future_context(record["frame_t"], int(target))
            )
            if current_gt is None:
                record["labels_valid"] = False
                continue
            snapshot: TrackStateSnapshot = record["snapshot"]
            candidate = DetectionObservation(
                detection_index=record["detection_index"],
                box=np.asarray(record["detection_box"], dtype=np.float64),
                score=float(record["detection_score"]),
                feature=np.asarray(record["det_feature"], dtype=np.float64).reshape(1, -1),
                source=0,
                class_id=1,
            )
            ctx = RolloutContext(
                frame_id=record["frame_t"],
                target_gt_id=int(target),
                pre_update_state=snapshot,
                current_candidate=candidate,
                current_gt_box=current_gt,
                future_gt_boxes=future_gt,
                future_oracle_detections=future_oracle,
                future_warp_matrices=future_warps,
                identity_prototype=self.prototypes.get(int(target)),
            )
            valid_motion = True
            valid_appearance = self.prototypes.get(int(target)) is not None
            b_m = b_a = 0.0
            m_valid = a_valid = 0
            try:
                b_m, _, _, m_mask = compute_motion_benefit(
                    ctx, NSAKalmanFilter(), future_frames=FUTURE_FRAMES, include_current=True
                )
                m_valid = int(np.asarray(m_mask, dtype=bool).sum())
                valid_motion = m_valid > 0
            except Exception:
                valid_motion = False
                b_m = 0.0
            try:
                b_a, _, _, _, _, a_mask = compute_appearance_benefit(
                    ctx, future_frames=FUTURE_FRAMES, include_current=True
                )
                a_valid = int(np.asarray(a_mask, dtype=bool).sum())
                valid_appearance = valid_appearance and a_valid > 0
            except Exception:
                valid_appearance = False
                b_a = 0.0
            record.update(
                {
                    "labels_valid": bool(valid_motion or valid_appearance),
                    "motion_benefit": float(b_m),
                    "appearance_benefit": float(b_a),
                    "valid_motion": bool(valid_motion),
                    "valid_appearance": bool(valid_appearance),
                    "motion_valid_count": int(m_valid),
                    "appearance_valid_count": int(a_valid),
                    "gt_coverage": float(gt_count / (FUTURE_FRAMES + 1)),
                    "oracle_detection_coverage": float(oracle_count / FUTURE_FRAMES),
                }
            )
            label_rows.append(record)
        motion_benefits = [
            row["motion_benefit"] for row in label_rows if row.get("valid_motion")
        ]
        appearance_benefits = [
            row["appearance_benefit"] for row in label_rows if row.get("valid_appearance")
        ]
        tau_motion = self._tau(motion_benefits)
        tau_appearance = self._tau(appearance_benefits)
        self.tau_motion = tau_motion
        self.tau_appearance = tau_appearance
        for row in label_rows:
            motion_soft = (
                compute_soft_target(row["motion_benefit"], tau_motion)
                if row.get("valid_motion")
                else 0.5
            )
            appearance_soft = (
                compute_soft_target(row["appearance_benefit"], tau_appearance)
                if row.get("valid_appearance")
                else 0.5
            )
            motion_conf = (
                compute_label_confidence(
                    row["motion_benefit"], tau_motion, row["gt_coverage"]
                )
                if row.get("valid_motion")
                else 0.0
            )
            appearance_conf = (
                compute_label_confidence(
                    row["appearance_benefit"],
                    tau_appearance,
                    row["oracle_detection_coverage"],
                )
                if row.get("valid_appearance")
                else 0.0
            )
            motion_safe = compute_safe_soft_target(
                row["motion_benefit"], tau_motion, motion_conf
            )
            appearance_safe = compute_safe_soft_target(
                row["appearance_benefit"], tau_appearance, appearance_conf
            )
            row.update(
                {
                    "motion_soft_target": float(motion_soft),
                    "appearance_soft_target": float(appearance_soft),
                    "motion_safe_target": float(motion_safe),
                    "appearance_safe_target": float(appearance_safe),
                    "motion_label_confidence": float(motion_conf),
                    "appearance_label_confidence": float(appearance_conf),
                    "cue_target": list(make_cue_target(motion_conf, appearance_conf)),
                    "risk_target": list(
                        make_risk_targets(
                            motion_soft, appearance_soft, motion_conf, appearance_conf
                        )
                    ),
                    "policy_safe_soft_target": compute_policy_soft_target(
                        np.array([motion_safe, appearance_safe])
                    ).tolist(),
                    "sample_weight": 1.0,
                }
            )

    @staticmethod
    def _tau(benefits: list[float]) -> float:
        array = np.asarray(benefits, dtype=np.float64)
        finite = array[np.isfinite(array)]
        nonzero = np.abs(finite[np.abs(finite) > 1e-12])
        if nonzero.size == 0:
            return 0.1
        return float(np.clip(float(np.median(nonzero)), 1e-3, 0.1))

    # ------------------------------------------------------------------
    def appearance_diagnostics(self) -> None:
        # Latest track template per GT id before each frame is reconstructed
        # from the event stream.
        gt_templates: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for record in self.events:
            target = record.get("target_gt_before")
            if target is not None:
                gt_templates[int(target)].append(
                    (record["frame_t"], np.asarray(record["pre_feature"], dtype=np.float32))
                )
        for values in gt_templates.values():
            values.sort(key=lambda item: item[0])

        def template_at(gt_id: int, frame: int):
            best = None
            for event_frame, feature in gt_templates.get(gt_id, []):
                if event_frame <= frame:
                    best = feature
                else:
                    break
            return best

        for record in self.events:
            mature = (record["frame_t"], record["track_id"]) in self.adapter_events
            if not mature:
                continue
            target = record.get("target_gt_before")
            record["appearance_diag_valid"] = False
            if target is None:
                continue
            frame = record["frame_t"]
            template0 = np.asarray(record["pre_feature"], dtype=np.float64)
            det_feature = np.asarray(record["det_feature"], dtype=np.float64)
            template1 = ema_update(
                template0, det_feature, float(record["detection_score"]), 0.95
            )
            positive = None
            positive_frame = None
            for step in range(1, FUTURE_FRAMES + 1):
                fid = frame + step
                if fid > self.frame_count:
                    break
                candidates = [
                    entry
                    for entry in self.frame_matches(fid - 1)
                    if entry["gt_id"] == int(target) and entry["iou"] >= 0.5
                ]
                if candidates:
                    positive = max(candidates, key=lambda item: item["iou"])
                    positive_frame = fid
                    break
            if positive is None:
                continue
            competitors = [
                entry
                for entry in self.frame_matches(positive_frame - 1)
                if entry["gt_id"] not in (-1, int(target))
                and entry["iou"] >= 0.5
                and iou_xyxy(entry["box"], positive["box"]) >= 0.3
            ]
            if not competitors:
                record["appearance_positive_frame"] = int(positive_frame)
                continue
            positive_feature = np.asarray(
                self.det_cache.get_detection(positive["index"])["feature"], dtype=np.float64
            )
            competitor = min(
                competitors,
                key=lambda entry: cosine_distance(
                    template0,
                    np.asarray(
                        self.det_cache.get_detection(entry["index"])["feature"],
                        dtype=np.float64,
                    ),
                ),
            )
            competitor_feature = np.asarray(
                self.det_cache.get_detection(competitor["index"])["feature"], dtype=np.float64
            )
            d0_pos = cosine_distance(template0, positive_feature)
            d0_comp = cosine_distance(template0, competitor_feature)
            d1_pos = cosine_distance(template1, positive_feature)
            d1_comp = cosine_distance(template1, competitor_feature)
            record["rank_valid"] = True
            record["rank_positive_feature"] = positive_feature.astype(np.float32).copy()
            record["rank_competitor_feature"] = competitor_feature.astype(np.float32).copy()
            contested = []
            for other_id, gt_box in self.gt.get(positive_frame, {}).items():
                if int(other_id) == int(target):
                    continue
                if iou_xyxy(positive["box"], gt_box) >= 0.5:
                    other_template = template_at(int(other_id), positive_frame)
                    if other_template is not None:
                        contested.append(
                            {
                                "other_gt_id": int(other_id),
                                "gt_overlap_iou": float(iou_xyxy(positive["box"], gt_box)),
                                "d_positive_to_target_template": cosine_distance(
                                    positive_feature, template_at(int(target), positive_frame)
                                )
                                if template_at(int(target), positive_frame) is not None
                                else None,
                                "d_positive_to_other_template": cosine_distance(
                                    positive_feature, other_template
                                ),
                            }
                        )
            mismatch_frame = None
            recovery_frame = None
            for event_frame, detection_gt, _feature in self.track_events[record["track_id"]]:
                if event_frame <= frame:
                    continue
                if detection_gt != int(target) and mismatch_frame is None:
                    mismatch_frame = event_frame
                if mismatch_frame is not None and detection_gt == int(target):
                    recovery_frame = event_frame
                    break
            record.update(
                {
                    "appearance_diag_valid": True,
                    "appearance_positive_frame": int(positive_frame),
                    "appearance_positive_index": int(positive["index"]),
                    "appearance_positive_iou": float(positive["iou"]),
                    "appearance_competitor_index": int(competitor["index"]),
                    "appearance_competitor_gt_id": int(competitor["gt_id"]),
                    "appearance_competitor_iou": float(competitor["iou"]),
                    "appearance_d0_positive": d0_pos,
                    "appearance_d0_competitor": d0_comp,
                    "appearance_score_template0": float(d0_comp - d0_pos),
                    "appearance_d1_positive": d1_pos,
                    "appearance_d1_competitor": d1_comp,
                    "appearance_score_template1": float(d1_comp - d1_pos),
                    "appearance_contested": contested,
                    "identity_mismatch_frame": mismatch_frame,
                    "identity_recovery_frame": recovery_frame,
                }
            )

    # ------------------------------------------------------------------
    def write_outputs(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        event_path = self.out_dir / "events.jsonl"
        with event_path.open("w") as handle:
            for record in self.events:
                payload = {
                    key: value
                    for key, value in record.items()
                    if key
                    in {
                        "sequence",
                        "frame_t",
                        "track_id",
                        "detection_index",
                        "source",
                        "state_before",
                        "observation_count",
                        "gap_before",
                        "motion_gate",
                        "appearance_gate",
                        "detection_gt_id",
                        "detection_gt_iou",
                        "target_gt_before",
                        "detection_box",
                        "detection_score",
                        "motion_forward_valid",
                        "gt_box_t1",
                        "motion_gt_valid",
                        "labels_valid",
                        "motion_benefit",
                        "appearance_benefit",
                        "valid_motion",
                        "valid_appearance",
                        "motion_valid_count",
                        "appearance_valid_count",
                        "gt_coverage",
                        "oracle_detection_coverage",
                        "motion_soft_target",
                        "appearance_soft_target",
                        "motion_safe_target",
                        "appearance_safe_target",
                        "motion_label_confidence",
                        "appearance_label_confidence",
                        "sample_weight",
                        "appearance_diag_valid",
                        "appearance_positive_frame",
                        "appearance_positive_index",
                        "appearance_positive_iou",
                        "appearance_competitor_index",
                        "appearance_competitor_gt_id",
                        "appearance_competitor_iou",
                        "appearance_d0_positive",
                        "appearance_d0_competitor",
                        "appearance_score_template0",
                        "appearance_d1_positive",
                        "appearance_d1_competitor",
                        "appearance_score_template1",
                        "appearance_contested",
                        "identity_mismatch_frame",
                        "identity_recovery_frame",
                    }
                }
                if "pred_box_g0" in record:
                    payload["pred_box_g0"] = record["pred_box_g0"]
                    payload["pred_box_g1"] = record["pred_box_g1"]
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

        mature = [
            record
            for record in self.events
            if (record["frame_t"], record["track_id"]) in self.adapter_events
            and (record["track_id"], record["frame_t"]) in self.windows
        ]
        arrays: dict[str, np.ndarray] = {}
        keys = []
        for index, record in enumerate(mature):
            window = self.windows[(record["track_id"], record["frame_t"])]
            history_frames = self.window_frames[(record["track_id"], record["frame_t"])]
            arrays.setdefault("history_evidence", []).append(window.history_evidence)
            arrays.setdefault("history_competition", []).append(
                np.zeros((5, 3), np.float32)
            )
            arrays.setdefault("history_has_detection", []).append(
                window.history_has_detection
            )
            arrays.setdefault("history_mask", []).append(window.history_mask)
            arrays.setdefault("history_reset", []).append(window.history_reset)
            arrays.setdefault("history_track", []).append(window.history_track)
            arrays.setdefault("current_evidence", []).append(window.current_evidence)
            arrays.setdefault("current_competition", []).append(
                window.current_competition
            )
            arrays.setdefault("current_track", []).append(window.current_track)
            arrays.setdefault("current_has_detection", []).append(
                window.current_has_detection
            )
            history_indices = np.array(
                [
                    self.frame_det_index.get((record["track_id"], frame), -1)
                    for frame in history_frames
                ]
                + [-1] * (5 - len(history_frames)),
                dtype=np.int64,
            )
            arrays.setdefault("history_detection_indices", []).append(history_indices)
            arrays.setdefault("current_detection_index", []).append(
                record["detection_index"]
            )
            arrays.setdefault("frame", []).append(record["frame_t"])
            arrays.setdefault("track_id", []).append(record["track_id"])
            arrays.setdefault("gt_id", []).append(
                -1 if record.get("target_gt_before") is None else record["target_gt_before"]
            )
            arrays.setdefault("valid_motion", []).append(
                bool(record.get("valid_motion", False))
            )
            arrays.setdefault("valid_appearance", []).append(
                bool(record.get("valid_appearance", False))
            )
            arrays.setdefault("motion_safe_target", []).append(
                record.get("motion_safe_target", 0.0)
            )
            arrays.setdefault("appearance_safe_target", []).append(
                record.get("appearance_safe_target", 0.0)
            )
            arrays.setdefault("motion_soft_target", []).append(
                record.get("motion_soft_target", 0.0)
            )
            arrays.setdefault("appearance_soft_target", []).append(
                record.get("appearance_soft_target", 0.0)
            )
            arrays.setdefault("sample_weight", []).append(
                record.get("sample_weight", 1.0)
            )
            arrays.setdefault("pred_box_g0", []).append(
                record.get("pred_box_g0") or [0.0] * 4
            )
            arrays.setdefault("pred_box_g1", []).append(
                record.get("pred_box_g1") or [0.0] * 4
            )
            arrays.setdefault("gt_box_t1", []).append(
                record.get("gt_box_t1") or [0.0] * 4
            )
            arrays.setdefault("motion_gt_valid", []).append(
                bool(record.get("motion_gt_valid", False))
            )
            keys.append(
                {
                    "sequence": record["sequence"],
                    "frame": record["frame_t"],
                    "track_id": record["track_id"],
                    "detection_index": record["detection_index"],
                    "source": record["source"],
                }
            )
        for name, values in arrays.items():
            dtype = np.bool_ if name in {
                "history_has_detection",
                "history_mask",
                "history_reset",
                "current_has_detection",
                "valid_motion",
                "valid_appearance",
                "motion_gt_valid",
            } else np.float32 if name not in {
                "history_detection_indices",
                "current_detection_index",
                "frame",
                "track_id",
                "gt_id",
            } else np.int64
            arrays[name] = np.asarray(values, dtype=dtype)
        np.savez_compressed(
            self.out_dir / "samples.npz",
            **arrays,
            keys_json=json.dumps(keys),
            tau_motion=self.tau_motion,
            tau_appearance=self.tau_appearance,
        )

        unique = len({(k["sequence"], k["frame"], k["track_id"], k["detection_index"]) for k in keys})
        summary = {
            "sequence": self.sequence,
            "frames": int(self.frame_count),
            "matched_write_events": len(self.events),
            "gated_events": len(
                [1 for record in self.events if record["motion_gate"] is not None]
            ),
            "mature_samples": len(mature),
            "unique_sample_keys": unique,
            "labels_with_identity": len(
                [1 for record in self.events if record.get("labels_valid")]
            ),
            "valid_motion_labels": len(
                [1 for record in self.events if record.get("valid_motion")]
            ),
            "valid_appearance_labels": len(
                [1 for record in self.events if record.get("valid_appearance")]
            ),
            "motion_gt_valid": int(arrays["motion_gt_valid"].sum())
            if len(mature)
            else 0,
            "appearance_diag_valid": len(
                [1 for record in self.events if record.get("appearance_diag_valid")]
            ),
            "tau_motion": self.tau_motion,
            "tau_appearance": self.tau_appearance,
            "warp_max_abs_error": float(self.warp_error),
            "prototypes": len(self.prototypes),
            "replay_track_ids": len(self.replay_track_ids),
            "outputs": {
                "events": str(event_path),
                "samples": str(self.out_dir / "samples.npz"),
            },
        }
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    started = time.time()
    results = []
    for sequence in args.sequences:
        seq_started = time.time()
        pilot = SequencePilot(
            sequence,
            args.split,
            Path(args.checkpoint),
            Path(args.out_dir),
            max_frames=args.max_frames,
        )
        pilot.run_replay()
        pilot.compute_labels()
        pilot.appearance_diagnostics()
        summary = pilot.write_outputs()
        pilot.close()
        summary["seconds"] = time.time() - seq_started
        results.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    (Path(args.out_dir) / "summary_all.json").write_text(
        json.dumps(
            {"sequences": results, "total_seconds": time.time() - started},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
