#!/usr/bin/env python3
"""SportsMOT trainval motion supervision side table for SHUTrack A.

One faithful online replay of the TrackTrack tracker in SHUTrack-v2 mode
(SportsMOT e200 checkpoint, same thresholds/caches as the formal test run)
records every matched write event at frame t, then advances the pre-write
Kalman state through the *actual* motion write semantics for gate values
g in {0, 0.5, 1}, the online CMC warp used at frame t+1, and the KF predict
step.  The result is joined with the GT identity bound to the track before
the write, producing the side table used for velocity/position supervision
diagnostics.

Conventions
-----------
* frames are 1-based; frame t is the write event, t+1 is the next frame.
* boxes are pixel xyxy (x1, y1, x2, y2).
* KF velocity is (vx, vy) at the box centre, pixels/frame, from state[4:6].
* GT velocity target is c_{t+1} - W_{t+1}(c_t), where W_{t+1} is the CMC
  warp read by the online tracker at frame t+1 (rotation + translation,
  the same matrix apply_cmc uses on KF states).

The script never touches existing caches or training data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = Path("/home/shang/workspace/TrackTrack")
TRACKER_DIR = WORKTREE / "3. Tracker"
AGENTGUARD_SRC = WORKTREE / "agentguard" / "src"
for path in (str(TRACKER_DIR), str(AGENTGUARD_SRC), str(WORKTREE)):
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_CACHE = MAIN / "outputs/agentguard/detection_cache"
DATA_ROOT = Path("/home/shang/datasets")
DEFAULT_OUT = MAIN / "outputs/agentguard/experiments/shutrack_a_motion_supervision_20261118"
DEFAULT_PACKED = MAIN / "outputs/agentguard/train_data/SportsMOT"


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
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
    c_diag2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    center_dist2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    return iou - center_dist2 / max(c_diag2, 1e-9)


def load_gt(gt_path: Path):
    """Return {frame_id: [(gt_id, xyxy), ...]} for active GT rows."""
    frames: dict[int, list[tuple[int, np.ndarray]]] = {}
    with gt_path.open() as handle:
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            gt_id = int(float(fields[1]))
            if len(fields) >= 8:
                active = float(fields[6])
                cls = float(fields[7])
                if active <= 0 or cls != 1:
                    continue
            x, y, w, h = map(float, fields[2:6])
            box = np.array([x, y, x + w, y + h], dtype=np.float64)
            frames.setdefault(frame, []).append((gt_id, box))
    return frames


def build_args(sequence: str, split: str, checkpoint: Path, cache_root: Path):
    import run as run_module

    args = run_module.make_parser().parse_args([])
    args.dataset = "SportsMOT"
    args.mode = split
    args.sequences = [sequence]
    args.seed = 10000
    args.kf_type = "nsa"
    args.agentguard_mode = "shutrack-v2"
    args.shutrack_checkpoint = str(checkpoint)
    args.agentguard_device = "cpu"
    args.detection_cache_root = str(cache_root)
    args.use_post = False
    args.skip_eval = True
    args.max_frames = 0
    args.noise_pos_std = 0.0
    args.noise_size_std = 0.0
    args.drop_rate = 0.0
    args.disable_gmc = False
    args.no_reid = False
    args.agentguard_disable_kf_gate = False
    args.agentguard_disable_ema_gate = False
    args.shutrack_min_observations = 6
    args.shutrack_early_enable = False
    args.torch_num_threads = 4
    args.torch_num_interop_threads = 1
    return args


def forward_state(mean, covariance, kalman_filter, warp):
    """apply_cmc + predict, exactly as the online tracker does."""
    if warp is not None:
        rot = np.asarray(warp, dtype=np.float64)[:, :2]
        rot8 = np.kron(np.eye(4, dtype=np.float64), rot)
        trans = np.asarray(warp, dtype=np.float64)[:, 2]
        mean = rot8 @ mean
        mean = mean.copy()
        mean[:2] += trans
        covariance = rot8 @ covariance @ rot8.T
    return kalman_filter.predict(mean, covariance)


def gate_state(mean_prior, cov_prior, kalman_filter, measurement, score, gate):
    """Motion write semantics from Track.update_with_gates (steps 2-3)."""
    mean_full, cov_full = kalman_filter.update(
        mean_prior.copy(), cov_prior.copy(), np.asarray(measurement, dtype=np.float64), float(score)
    )
    gate = float(min(max(gate, 0.0), 1.0))
    if gate == 1.0:
        return mean_full, cov_full
    if gate == 0.0:
        return mean_prior.copy(), cov_prior.copy()
    mean_final = mean_prior + gate * (mean_full - mean_prior)
    cov_final = (1.0 - gate) * cov_prior + gate * cov_full
    cov_final = 0.5 * (cov_final + cov_final.T)
    return mean_final, cov_final


def state_box_velocity(mean):
    mean = np.asarray(mean, dtype=np.float64)
    box = np.array(
        [
            mean[0] - mean[2] / 2.0,
            mean[1] - mean[3] / 2.0,
            mean[0] + mean[2] / 2.0,
            mean[1] + mean[3] / 2.0,
        ]
    )
    return box, np.array([mean[4], mean[5]], dtype=np.float64)


class Recorder:
    def __init__(self, gt_frames, seq, width, height, clones_per_group=80):
        self.gt_frames = gt_frames
        self.sequence = seq
        self.width = float(width)
        self.height = float(height)
        self.events: list[dict] = []
        self.warps: dict[int, np.ndarray] = {}
        self.pred_snapshots: dict[tuple[int, int], np.ndarray] = {}
        self.current_frame = [0]
        self.bindings: dict[int, int] = {}
        self.clones_per_group = int(clones_per_group)
        self.clone_counts = {"consecutive": 0, "rematch": 0, "new": 0}
        self.clones: dict[int, tuple] = {}
        self.replay_track_ids: set[int] = set()
        self.initiated = 0

    def match_gt(self, box, frame_id):
        best = None
        best_iou = 0.5
        for gt_id, gt_box in self.gt_frames.get(int(frame_id), []):
            value = iou_xyxy(box, gt_box)
            if value >= best_iou:
                best_iou = value
                best = gt_id
        return best

    def bind_track(self, track, frame_id):
        if track.track_id in self.bindings:
            return
        hit = self.match_gt(track.x1y1x2y2, frame_id)
        if hit is not None:
            self.bindings[track.track_id] = hit

    def record_event(self, track, frame_id, detection, motion_gate, appearance_gate, source):
        self.replay_track_ids.add(int(track.track_id))
        gt_id_before = self.bindings.get(track.track_id)
        detection_box = np.asarray(detection.x1y1x2y2, dtype=np.float64).copy()
        if gt_id_before is None:
            hit = self.match_gt(detection_box, frame_id)
            if hit is not None:
                # Bind now, but this event has no identity established before the
                # write; diagnostics exclude it from GT targets.
                self.bindings[track.track_id] = hit
        record = {
            "sequence": self.sequence,
            "frame_t": int(frame_id),
            "track_id": int(track.track_id),
            "detection_index": int(getattr(detection, "frame_detection_index", -1)),
            "source": source,
            "state_before": int(track.state),
            "observation_count_before": int(track.observation_count),
            "gap_before": int(frame_id - track.end_frame_id)
            if int(track.end_frame_id) > 0
            else -1,
            "gt_id": int(gt_id_before) if gt_id_before is not None else None,
            "gt_bound_before": gt_id_before is not None,
            "detection_box": detection_box.tolist(),
            "detection_score": float(detection.score),
            "detection_measurement": np.asarray(detection.cxcywh, dtype=np.float64).tolist(),
            "mean_prior": np.asarray(track.mean, dtype=np.float64).copy().tolist(),
            "cov_prior": np.asarray(track.covariance, dtype=np.float64).copy().tolist(),
            "motion_gate_online": None if motion_gate is None else float(motion_gate),
            "appearance_gate_online": None if appearance_gate is None else float(appearance_gate),
        }
        self.events.append(record)
        group = classify_group(record)
        if (
            motion_gate is not None
            and self.clone_counts.get(group, 0) < self.clones_per_group
        ):
            self.clone_counts[group] = self.clone_counts.get(group, 0) + 1
            clone = copy.deepcopy(
                track,
                memo={
                    id(track.args): track.args,
                    id(track.kalman_filter): track.kalman_filter,
                },
            )
            self.clones[len(self.events) - 1] = (
                clone,
                detection,
                np.asarray(detection.cxcywh, dtype=np.float64).copy(),
                float(detection.score),
            )

    # ------------------------------------------------------------------
    def attach_targets(self, event: dict, gate_predictions: dict):
        frame_t = int(event["frame_t"])
        frame_t1 = frame_t + 1
        warp_next = self.warps.get(frame_t1)
        event["frame_t1"] = frame_t1
        event["warp_t1"] = None if warp_next is None else np.asarray(warp_next).tolist()
        event["valid_warp_t1"] = warp_next is not None
        event["valid_pred_t1"] = warp_next is not None

        gt_id = event["gt_id"]
        gt_box_t = None
        gt_box_t1 = None
        if gt_id is not None and event["gt_bound_before"]:
            for candidate_id, candidate_box in self.gt_frames.get(frame_t, []):
                if candidate_id == gt_id:
                    gt_box_t = candidate_box
                    break
            for candidate_id, candidate_box in self.gt_frames.get(frame_t1, []):
                if candidate_id == gt_id:
                    gt_box_t1 = candidate_box
                    break
        event["gt_box_t"] = None if gt_box_t is None else gt_box_t.tolist()
        event["gt_box_t1"] = None if gt_box_t1 is None else gt_box_t1.tolist()
        event["valid_gt_t"] = gt_box_t is not None
        event["valid_gt_t1"] = gt_box_t1 is not None

        if (
            gt_box_t is not None
            and gt_box_t1 is not None
            and warp_next is not None
            and event["gt_bound_before"]
        ):
            rot = np.asarray(warp_next, dtype=np.float64)[:, :2]
            trans = np.asarray(warp_next, dtype=np.float64)[:, 2]
            center_t = np.array(
                [
                    (gt_box_t[0] + gt_box_t[2]) / 2.0,
                    (gt_box_t[1] + gt_box_t[3]) / 2.0,
                ]
            )
            center_t1 = np.array(
                [
                    (gt_box_t1[0] + gt_box_t1[2]) / 2.0,
                    (gt_box_t1[1] + gt_box_t1[3]) / 2.0,
                ]
            )
            warped_center_t = rot @ center_t + trans
            velocity = center_t1 - warped_center_t
            event["gt_velocity"] = velocity.tolist()
            event["valid_velocity"] = True
        else:
            event["gt_velocity"] = None
            event["valid_velocity"] = False

        for name, payload in gate_predictions.items():
            event[f"pred_box_{name}"] = payload[0].tolist()
            event[f"pred_velocity_{name}"] = payload[1].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="v_00HRwkvvjtQ_c001")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--packed-root", default=str(DEFAULT_PACKED))
    parser.add_argument("--max-frames", type=int, default=0)
    args_ns = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from agentguard.data.detection_cache import SequenceDetectionCache
    import run as run_module
    from trackers import tracker as tracker_module
    from trackers.track import Track, TrackState
    from trackers.cmc import apply_cmc

    sequence = args_ns.sequence
    split = args_ns.split
    checkpoint = Path(args_ns.checkpoint)
    cache_root = Path(args_ns.cache_root)
    out_dir = Path(args_ns.out_dir)
    side_dir = out_dir / "side_tables"
    diag_dir = out_dir / "diagnostics"
    side_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    data_dir = DATA_ROOT
    seqinfo = data_dir / "SportsMOT" / "dataset" / split / sequence / "seqinfo.ini"
    gt_path = seqinfo.parent / "gt" / "gt.txt"
    gt_frames = load_gt(gt_path)
    width = height = None
    for line in seqinfo.read_text().splitlines():
        if line.startswith("imWidth"):
            width = int(line.split("=")[-1])
        if line.startswith("imHeight"):
            height = int(line.split("=")[-1])
    frame_rate = None
    for line in seqinfo.read_text().splitlines():
        if line.startswith("frameRate"):
            frame_rate = int(line.split("=")[-1])

    args = build_args(sequence, split, checkpoint, cache_root)
    run_module.set_parameters(args, sequence, split)
    args.max_time_lost = (frame_rate or 25) * 2
    args.img_w = width
    args.img_h = height

    recorder = Recorder(gt_frames, sequence, width, height)
    started = time.time()

    original_update = tracker_module.Tracker.update
    original_update_no_det = tracker_module.Tracker.update_without_detections
    original_init_tracks = tracker_module.Tracker.init_tracks
    original_track_update = Track.update
    original_track_update_gates = Track.update_with_gates
    original_predict = Track.predict

    def tracker_update(self, dets, dets_95):
        recorder.current_frame[0] = int(self.frame_id) + 1
        result = original_update(self, dets, dets_95)
        if self._current_warp is not None:
            recorder.warps[int(self.frame_id)] = np.asarray(self._current_warp).copy()
        return result

    def tracker_update_no_det(self):
        recorder.current_frame[0] = int(self.frame_id) + 1
        result = original_update_no_det(self)
        if self._current_warp is not None:
            recorder.warps[int(self.frame_id)] = np.asarray(self._current_warp).copy()
        return result

    def init_tracks(self, dets):
        before = {int(t.track_id) for t in self.tracks}
        original_init_tracks(self, dets)
        recorder.initiated += sum(
            1 for t in self.tracks if int(t.track_id) not in before
        )
        for track in self.tracks:
            if int(track.track_id) not in before:
                recorder.replay_track_ids.add(int(track.track_id))
                recorder.bind_track(track, int(self.frame_id))

    def track_update(self, frame_id, detection):
        source = (
            "new_path_unmature" if int(self.state) == 0 else "tracked_lost_unmature"
        )
        recorder.record_event(self, frame_id, detection, None, None, source)
        return original_track_update(self, frame_id, detection)

    def track_update_gates(self, frame_id, detection, motion_gate, appearance_gate):
        recorder.record_event(
            self,
            frame_id,
            detection,
            motion_gate,
            appearance_gate,
            "gated",
        )
        return original_track_update_gates(
            self, frame_id, detection, motion_gate, appearance_gate
        )

    def track_predict(self):
        original_predict(self)
        frame = recorder.current_frame[0]
        recorder.pred_snapshots[(int(frame), int(self.track_id))] = np.asarray(
            self.mean, dtype=np.float64
        ).copy()

    tracker_module.Tracker.update = tracker_update
    tracker_module.Tracker.update_without_detections = tracker_update_no_det
    tracker_module.Tracker.init_tracks = init_tracks
    Track.update = track_update
    Track.update_with_gates = track_update_gates
    Track.predict = track_predict

    det_cache = SequenceDetectionCache(
        run_module._sequence_detection_cache_dir(args, sequence),
        frame_array_cache_size=4,
    )
    tracker = tracker_module.Tracker(args, sequence)
    frame_count = det_cache.num_frames
    if args_ns.max_frames > 0:
        frame_count = min(frame_count, args_ns.max_frames)
    replay_started = time.time()
    try:
        for frame_index in range(frame_count):
            frame_id = frame_index + 1
            target = det_cache.get_frame(frame_index, view="target")
            source = det_cache.get_frame(frame_index, view="source")
            args.agentguard_target_detection_indices = target["detection_indices"]
            args.agentguard_source_detection_indices = source["detection_indices"]
            det_frame = det_cache.get_frame_array(frame_index, view="target")
            det_frame_95 = det_cache.get_frame_array(frame_index, view="source")
            if det_frame is not None and len(det_frame) > 0:
                tracker.update(det_frame, det_frame_95)
            else:
                tracker.update_without_detections()
            if frame_index % 200 == 0:
                print(
                    f"[replay] frame {frame_id}/{frame_count} events={len(recorder.events)}",
                    flush=True,
                )
    finally:
        det_cache.close()
        args.agentguard_target_detection_indices = None
        args.agentguard_source_detection_indices = None
        tracker_module.Tracker.update = original_update
        tracker_module.Tracker.update_without_detections = original_update_no_det
        tracker_module.Tracker.init_tracks = original_init_tracks
        Track.update = original_track_update
        Track.update_with_gates = original_track_update_gates
        Track.predict = original_predict
    replay_seconds = time.time() - replay_started
    print(
        f"[replay] done {frame_count} frames, {len(recorder.events)} write events, "
        f"{replay_seconds:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Counterfactual branches and GT join
    # ------------------------------------------------------------------
    branch_started = time.time()
    for event in recorder.events:
        mean_prior = np.asarray(event["mean_prior"], dtype=np.float64)
        cov_prior = np.asarray(event["cov_prior"], dtype=np.float64)
        measurement = np.asarray(event["detection_measurement"], dtype=np.float64)
        score = float(event["detection_score"])
        warp_next = recorder.warps.get(int(event["frame_t"]) + 1)
        kf = None
        # All tracks share the same NSA filter class; use a fresh instance for
        # the branch math (predict/update are stateless).
        from trackers.kalman_filter import create_kalman_filter

        kf = create_kalman_filter("nsa")
        predictions = {}
        if warp_next is not None:
            for name, gate in (("g0", 0.0), ("g1", 1.0), ("g05", 0.5)):
                mean_g, cov_g = gate_state(
                    mean_prior, cov_prior, kf, measurement, score, gate
                )
                mean_n, cov_n = forward_state(mean_g, cov_g, kf, warp_next)
                predictions[name] = state_box_velocity(mean_n)
        recorder.attach_targets(event, predictions)

        gate_online = event.get("motion_gate_online")
        if warp_next is not None and gate_online is not None:
            mean_g, cov_g = gate_state(
                mean_prior, cov_prior, kf, measurement, score, gate_online
            )
            mean_n, cov_n = forward_state(mean_g, cov_g, kf, warp_next)
            box_o, vel_o = state_box_velocity(mean_n)
            event["pred_box_online_gate"] = box_o.tolist()
            event["pred_velocity_online_gate"] = vel_o.tolist()
        else:
            event["pred_box_online_gate"] = None
            event["pred_velocity_online_gate"] = None
    branch_seconds = time.time() - branch_started

    # ------------------------------------------------------------------
    # Interpolation and online-parity checks
    # ------------------------------------------------------------------
    interp_started = time.time()
    interp_box_max = 0.0
    interp_vel_max = 0.0
    interp_box_sum = 0.0
    interp_vel_sum = 0.0
    interp_count = 0
    by_group_errors: dict[str, list[float]] = {"consecutive": [], "rematch": [], "new": []}
    parity_box_max = 0.0
    parity_vel_max = 0.0
    parity_count = 0
    for event in recorder.events:
        if not event.get("valid_warp_t1") or "pred_box_g05" not in event:
            continue
        b0 = np.asarray(event["pred_box_g0"])
        b1 = np.asarray(event["pred_box_g1"])
        b5 = np.asarray(event["pred_box_g05"])
        v0 = np.asarray(event["pred_velocity_g0"])
        v1 = np.asarray(event["pred_velocity_g1"])
        v5 = np.asarray(event["pred_velocity_g05"])
        err_box = float(np.max(np.abs(b5 - 0.5 * (b0 + b1))))
        err_vel = float(np.max(np.abs(v5 - 0.5 * (v0 + v1))))
        interp_box_max = max(interp_box_max, err_box)
        interp_vel_max = max(interp_vel_max, err_vel)
        interp_box_sum += err_box
        interp_vel_sum += err_vel
        interp_count += 1
        group = classify_group(event)
        by_group_errors[group].append(err_vel)

        snapshot = recorder.pred_snapshots.get(
            (int(event["frame_t"]) + 1, int(event["track_id"]))
        )
        if snapshot is not None and event.get("pred_box_online_gate") is not None:
            box_online, vel_online = state_box_velocity(snapshot)
            parity_box_max = max(
                parity_box_max,
                float(np.max(np.abs(box_online - np.asarray(event["pred_box_online_gate"])))),
            )
            parity_vel_max = max(
                parity_vel_max,
                float(np.max(np.abs(vel_online - np.asarray(event["pred_velocity_online_gate"])))),
            )
            parity_count += 1

    # Genuine-method check on cloned tracks (the actual Track.update_with_gates
    # path, for g in {0, 0.5, 1}).
    genuine_box_max = 0.0
    genuine_vel_max = 0.0
    genuine_count = 0
    for index, (clone, detection, measurement, score) in recorder.clones.items():
        event = recorder.events[index]
        warp_next = recorder.warps.get(int(event["frame_t"]) + 1)
        if warp_next is None:
            continue
        for name, gate in (("g0", 0.0), ("g1", 1.0), ("g05", 0.5)):
            if f"pred_box_{name}" not in event:
                continue
            track = copy.deepcopy(
                clone,
                memo={
                    id(clone.args): clone.args,
                    id(clone.kalman_filter): clone.kalman_filter,
                },
            )
            track.update_with_gates(int(event["frame_t"]), detection, gate, 1.0)
            apply_cmc([track], warp_next)
            track.predict()
            box, vel = track.x1y1x2y2, np.array([track.mean[4], track.mean[5]])
            genuine_box_max = max(
                genuine_box_max,
                float(np.max(np.abs(box - np.asarray(event[f"pred_box_{name}"])))),
            )
            genuine_vel_max = max(
                genuine_vel_max,
                float(np.max(np.abs(vel - np.asarray(event[f"pred_velocity_{name}"])))),
            )
            genuine_count += 1

    # ------------------------------------------------------------------
    # Diagnostics with the online model gate
    # ------------------------------------------------------------------
    diagnostics_started = time.time()
    interpolation_seconds = diagnostics_started - interp_started
    diag: dict[str, dict] = {}

    def gate_optimal_velocity(v0, v1, target):
        candidates = [0.0, 1.0]
        for index, dim in ((0, width), (1, height)):
            delta = float(v1[index] - v0[index])
            if abs(delta) > 1e-12:
                candidates.append(
                    float(min(1.0, max(0.0, (target[index] - v0[index]) / delta)))
                )
        best_gate = 0.0
        best_error = None
        for gate in candidates:
            error = velocity_error(v0, v1, target, gate, width, height)
            if best_error is None or error < best_error:
                best_error = error
                best_gate = gate
        return best_gate, best_error

    for key in ("consecutive", "rematch", "new"):
        diag[key] = {
            "events": 0,
            "valid_velocity": 0,
            "with_online_gate": 0,
            "velocity_error": {"current": [], "g0": [], "g1": [], "optimal": []},
            "box_error": {"current": [], "g0": [], "g1": [], "optimal": []},
            "box_worse_at_velocity_opt_vs_current": 0,
            "box_worse_at_velocity_opt_vs_g1": 0,
            "gate_current": [],
            "gate_optimal": [],
        }

    for event in recorder.events:
        group = classify_group(event)
        stats = diag[group]
        stats["events"] += 1
        if not event.get("valid_velocity"):
            continue
        stats["valid_velocity"] += 1
        gate_current = event.get("motion_gate_online")
        if gate_current is None or event.get("pred_box_online_gate") is None:
            continue
        stats["with_online_gate"] += 1
        v0 = np.asarray(event["pred_velocity_g0"])
        v1 = np.asarray(event["pred_velocity_g1"])
        box0 = np.asarray(event["pred_box_g0"])
        box1 = np.asarray(event["pred_box_g1"])
        target = np.asarray(event["gt_velocity"])
        gt_box = np.asarray(event["gt_box_t1"])
        gate_opt, error_opt = gate_optimal_velocity(v0, v1, target)
        current_error = velocity_error(v0, v1, target, gate_current, width, height)
        stats["velocity_error"]["current"].append(current_error)
        stats["velocity_error"]["g0"].append(
            velocity_error(v0, v1, target, 0.0, width, height)
        )
        stats["velocity_error"]["g1"].append(
            velocity_error(v0, v1, target, 1.0, width, height)
        )
        stats["velocity_error"]["optimal"].append(error_opt)
        box_current = 1.0 - diou_xyxy(
            interpolate_box(box0, box1, gate_current), gt_box
        )
        box_g0 = 1.0 - diou_xyxy(box0, gt_box)
        box_g1 = 1.0 - diou_xyxy(box1, gt_box)
        box_opt = 1.0 - diou_xyxy(
            interpolate_box(box0, box1, gate_opt), gt_box
        )
        stats["box_error"]["current"].append(box_current)
        stats["box_error"]["g0"].append(box_g0)
        stats["box_error"]["g1"].append(box_g1)
        stats["box_error"]["optimal"].append(box_opt)
        if box_opt > box_current + 1e-9:
            stats["box_worse_at_velocity_opt_vs_current"] += 1
        if box_opt > box_g1 + 1e-9:
            stats["box_worse_at_velocity_opt_vs_g1"] += 1
        stats["gate_current"].append(float(gate_current))
        stats["gate_optimal"].append(float(gate_opt))

    def summarize(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.9)),
        }

    diag_out = {}
    for key, stats in diag.items():
        entry = {"events": stats["events"], "valid_velocity": stats["valid_velocity"],
                 "with_online_gate": stats["with_online_gate"]}
        for family in ("velocity_error", "box_error"):
            entry[family] = {
                name: summarize(values) for name, values in stats[family].items()
            }
        entry["box_worse_at_velocity_opt_vs_current"] = stats[
            "box_worse_at_velocity_opt_vs_current"
        ]
        entry["box_worse_at_velocity_opt_vs_g1"] = stats[
            "box_worse_at_velocity_opt_vs_g1"
        ]
        entry["gate_current"] = summarize(stats["gate_current"])
        entry["gate_optimal"] = summarize(stats["gate_optimal"])
        if stats["gate_current"] and stats["gate_optimal"]:
            entry["gate_mean_abs_delta"] = float(
                np.mean(
                    np.abs(
                        np.asarray(stats["gate_current"])
                        - np.asarray(stats["gate_optimal"])
                    )
                )
            )
        vel_current = entry["velocity_error"].get("current")
        vel_optimal = entry["velocity_error"].get("optimal")
        if vel_current and vel_optimal and vel_current["mean"] > 0:
            entry["velocity_error_reduction_fraction"] = float(
                1.0 - vel_optimal["mean"] / vel_current["mean"]
            )
        diag_out[key] = entry
    diagnostics_seconds = time.time() - diagnostics_started

    # ------------------------------------------------------------------
    # Old packed-data alignment audit
    # ------------------------------------------------------------------
    packed_dir = Path(args_ns.packed_root) / sequence
    alignment = {"packed_dir": str(packed_dir), "exists": packed_dir.is_dir()}
    if packed_dir.is_dir():
        import torch as _torch

        packed = _torch.load(
            packed_dir / "data.pt", map_location="cpu", weights_only=False
        )
        arrays = packed["arrays"]
        metadata = packed["metadata"]
        timeline_counts = metadata.get("timeline_counts", {})
        old_track_ids = {int(x) for x in np.asarray(arrays["track_ids"]).tolist()}
        old_segment_ids = {int(x) for x in np.asarray(arrays["segment_ids"]).tolist()}
        replay_track_ids = recorder.replay_track_ids
        overlap = len(old_track_ids & replay_track_ids)
        alignment.update(
            {
                "packed_arrays": sorted(arrays.keys()),
                "packed_samples": int(np.asarray(arrays["safe_gate_target"]).shape[0]),
                "timeline_events": int(np.asarray(arrays["timeline_scalar_feats"]).shape[0]),
                "timeline_counts": timeline_counts,
                "old_unique_track_ids": len(old_track_ids),
                "old_segment_ids": len(old_segment_ids),
                "replay_track_ids": len(replay_track_ids),
                "replay_segments_initiated": int(recorder.initiated),
                "track_id_overlap": int(overlap),
                "track_id_overlap_fraction": float(
                    overlap / max(len(old_track_ids), 1)
                ),
                "event_key_columns_present": {
                    "frame": False,
                    "detection_index": False,
                    "match_stage": False,
                    "event_key": False,
                    "timeline_index": True,
                    "track_id": True,
                    "segment_id": True,
                },
                "verdict": (
                    "packed data.pt does not store frame ids, detection indices, "
                    "match stage or event keys; event_indices index a timeline whose "
                    "source event cache has been deleted, so replay events cannot be "
                    "linked sample-by-sample. Track ids are replay-dependent "
                    "(auxiliary only). Falling back to same-replay generation."
                ),
            }
        )
        del packed
    else:
        alignment["verdict"] = "packed sequence directory missing"

    # ------------------------------------------------------------------
    # Write side table, diagnostics, report
    # ------------------------------------------------------------------
    side_path = side_dir / f"sportsmot_{split}_{sequence}.jsonl"
    with side_path.open("w") as handle:
        for event in recorder.events:
            # cov_prior is only needed for the in-memory branch computation.
            payload = {k: v for k, v in event.items() if k != "cov_prior"}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    summary = {
        "sequence": sequence,
        "split": split,
        "checkpoint": str(checkpoint),
        "frames": int(frame_count),
        "width": int(width),
        "height": int(height),
        "events": len(recorder.events),
        "events_with_online_gate": sum(
            1 for e in recorder.events if e.get("motion_gate_online") is not None
        ),
        "valid_velocity_events": sum(
            1 for e in recorder.events if e.get("valid_velocity")
        ),
        "groups": {
            "consecutive": sum(
                1 for e in recorder.events if classify_group(e) == "consecutive"
            ),
            "rematch": sum(
                1 for e in recorder.events if classify_group(e) == "rematch"
            ),
            "new": sum(1 for e in recorder.events if classify_group(e) == "new"),
        },
        "interpolation": {
            "events": interp_count,
            "box_max_abs_error": interp_box_max,
            "box_mean_abs_error": interp_box_sum / max(interp_count, 1),
            "velocity_max_abs_error": interp_vel_max,
            "velocity_mean_abs_error": interp_vel_sum / max(interp_count, 1),
        },
        "online_parity": {
            "events": parity_count,
            "box_max_abs_error": parity_box_max,
            "velocity_max_abs_error": parity_vel_max,
        },
        "genuine_method_check": {
            "branches": genuine_count,
            "box_max_abs_error": genuine_box_max,
            "velocity_max_abs_error": genuine_vel_max,
        },
        "diagnostics": diag_out,
        "alignment": alignment,
        "timing_seconds": {
            "replay": replay_seconds,
            "branch_and_join": branch_seconds,
            "interpolation_and_parity": interpolation_seconds,
            "diagnostics": diagnostics_seconds,
            "total": time.time() - started,
        },
        "conventions": {
            "frame_indexing": "1-based",
            "box_format": "pixel xyxy",
            "velocity": "box-centre (vx, vy), pixels/frame",
            "velocity_target": "c_{t+1} - W_{t+1}(c_t) with the online CMC warp at t+1",
            "gate_semantics": "motion write strength in [0, 1]; 0=skeep KF prior, 1=full update",
        },
        "side_table": str(side_path),
    }
    summary_path = diag_dir / f"summary_{sequence}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("events", "valid_velocity_events", "groups", "interpolation", "online_parity", "genuine_method_check")}, indent=2))
    print(f"side table: {side_path}")
    print(f"summary:    {summary_path}")


def classify_group(event: dict) -> str:
    state = int(event.get("state_before", -1))
    gap = int(event.get("gap_before", -1))
    if state == 1:  # Tracked
        return "consecutive"
    if state == 2:  # Lost
        return "rematch" if gap > 1 else "consecutive"
    return "new"


def velocity_error(v0, v1, target, gate, width, height):
    pred = v0 + float(gate) * (v1 - v0)
    return float(
        abs(pred[0] - target[0]) / width + abs(pred[1] - target[1]) / height
    )


def interpolate_box(box0, box1, gate):
    return box0 + float(gate) * (box1 - box0)


if __name__ == "__main__":
    main()
