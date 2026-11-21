#!/usr/bin/env python3
"""Paired missed-detection diagnostic from the same last matched event.

For every (sequence, frame, GT id) where all compared checkpoints tracked a
detection, and the detector produces no detection for that GT over the next
k frames (k = 1, 3, 5), roll each checkpoint's captured KF state forward with
the online GMC warp + KF predict and compare the predicted box with the GT
box at frame + k: normalised centre error, size error and IoU.

The no-detection interval is defined by the shared detection stream (no
cached detection with IoU >= 0.5 to that GT), so all groups see exactly the
same interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE / "tools"),
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from sportsmot_supervision_pilot import SequencePilot, mean_to_box, warp_predict  # noqa: E402

VAL_MAP = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
GAPS = (1, 3, 5)


class StateCapturePilot(SequencePilot):
    def __init__(self, sequence, split, checkpoint, out_dir):
        super().__init__(sequence, split, checkpoint, out_dir)
        self.states: dict[tuple[int, int], np.ndarray] = {}

    def build_prototypes(self) -> None:
        for frame_index in range(self.det_cache.num_frames):
            self.frame_matches(frame_index)
        self.prototypes = {}

    def install_hooks(self) -> None:
        from trackers.track import Track
        from shutrack.integration.tracktrack import ShutrackTrackerAdapter
        from paired_missing import capture_detection_pool

        self._restore = []
        capture_detection_pool(self, ShutrackTrackerAdapter)
        pilot = self

        def wrap(original):
            def update(track, frame, detection, *args, **kwargs):
                index = int(getattr(detection, "frame_detection_index", -1))
                tid = int(track.track_id)
                before = pilot.vote.resolve_before_current(pilot.sequence, tid)
                identity = pilot.det_gt.get((frame, index), (-1, 0.0))[0]
                result = original(track, frame, detection, *args, **kwargs)
                pilot.vote.add_current_observation(pilot.sequence, tid, identity)
                pilot.events.append({"frame_t": frame, "track_id": tid, "detection_index": index,
                                     "target_gt_before": before})
                pilot.states[(frame, tid)] = track.mean.copy()
                return result
            return update

        for name in ("update", "update_with_gates"):
            original = getattr(Track, name)
            self._restore.append((Track, name, original))
            setattr(Track, name, wrap(original))


def box_metrics(pred: np.ndarray, gt: np.ndarray, img_w: float, img_h: float) -> dict:
    pred_center = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
    gt_center = np.array([(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2])
    center_error = float(
        np.linalg.norm(
            [(pred_center[0] - gt_center[0]) / img_w, (pred_center[1] - gt_center[1]) / img_h]
        )
    )
    pred_size = np.array([pred[2] - pred[0], pred[3] - pred[1]])
    gt_size = np.array([gt[2] - gt[0], gt[3] - gt[1]])
    size_error = float(
        0.5
        * (
            abs(pred_size[0] - gt_size[0]) / img_w
            + abs(pred_size[1] - gt_size[1]) / img_h
        )
    )
    inter_w = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
    inter_h = max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    inter = inter_w * inter_h
    area_pred = max(0.0, pred_size[0]) * max(0.0, pred_size[1])
    area_gt = max(0.0, gt_size[0]) * max(0.0, gt_size[1])
    union = area_pred + area_gt - inter
    iou = float(inter / union) if union > 0 else 0.0
    return {"center": center_error, "size": size_error, "iou": iou}


def replay_states(sequence, split, checkpoint, out_dir):
    pilot = StateCapturePilot(sequence, split, checkpoint, out_dir)
    pilot.run_replay()
    events = {}
    assigned = {}
    for record in pilot.events:
        frame = int(record["frame_t"])
        index = int(record["detection_index"])
        gt_id = pilot.unique_gt.get((frame, index))
        if gt_id is None:
            continue
        track_id = int(record["track_id"])
        assigned.setdefault((frame, gt_id), set()).add(track_id)
        if record.get("target_gt_before") != gt_id:
            continue
        state = pilot.states.get((frame, track_id))
        if state is not None:
            events[(frame, gt_id, index)] = (track_id, state)
    result = {
        "events": events,
        "assigned": assigned,
        "detections": pilot.visible,
        "clear_visible": pilot.clear_visible,
        "gt": pilot.gt,
        "warps": pilot.warps,
        "kf": pilot.kf,
        "img_w": float(pilot.img_w),
        "img_h": float(pilot.img_h),
        "frame_count": int(pilot.frame_count),
    }
    pilot.close()
    return result


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def summarize_paired(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "stderr": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "stderr": float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
    }


def empty_metrics(checkpoints):
    collected = {
        name: {gap: {"center": [], "size": [], "iou": []} for gap in GAPS}
        for name in checkpoints
    }
    recovery = {name: {gap: {"observable": 0, "same_track": 0} for gap in GAPS}
                for name in checkpoints}
    return collected, recovery


def measure_sequence(job):
    import torch
    torch.set_num_threads(1)
    sequence, checkpoints, log_root = job
    collected, recovery = empty_metrics(checkpoints)
    group_data = {}
    reference = None
    for name, checkpoint in checkpoints.items():
        data = replay_states(sequence, "val", checkpoint, log_root)
        group_data[name] = data
        if reference is None:
            reference = data
    shared = set(group_data[next(iter(checkpoints))]["events"])
    for name in checkpoints:
        shared &= set(group_data[name]["events"])
    gt = reference["gt"]
    warps = reference["warps"]
    detections = reference["detections"]
    kf = reference["kf"]
    img_w = reference["img_w"]
    img_h = reference["img_h"]
    frame_count = reference["frame_count"]
    sequence_rows = {gap: 0 for gap in GAPS}
    used = set()
    for frame, gt_id, det_index in sorted(shared):
        if (frame, gt_id) in used:
            continue
        if any(len(data["assigned"].get((frame, gt_id), ())) != 1 for data in group_data.values()):
            continue
        used.add((frame, gt_id))
        first_return = frame + 1
        while (first_return <= frame_count and gt_id in gt.get(first_return, {})
               and gt_id not in detections.get(first_return, set())):
            first_return += 1
        observable = first_return <= frame_count and gt_id in reference["clear_visible"].get(first_return, set())
        for gap in GAPS:
            target_frame = frame + gap
            if target_frame > frame_count:
                continue
            gt_box = gt.get(target_frame, {}).get(gt_id)
            if gt_box is None or first_return <= target_frame:
                continue
            if any(
                gt_id in detections.get(frame + step, set())
                for step in range(1, gap + 1)
            ):
                continue
            if any((frame + step) not in warps for step in range(1, gap + 1)):
                continue
            sequence_rows[gap] += 1
            for name in checkpoints:
                start_id, state = group_data[name]["events"][(frame, gt_id, det_index)]
                if observable:
                    recovery[name][gap]["observable"] += 1
                    return_ids = group_data[name]["assigned"].get((first_return, gt_id), set())
                    recovery[name][gap]["same_track"] += int(return_ids == {start_id})
                mean = np.asarray(state, dtype=np.float64)
                covariance = np.zeros((8, 8), dtype=np.float64)
                for step in range(1, gap + 1):
                    mean, covariance = warp_predict(
                        mean, covariance, warps[frame + step], kf
                    )
                metrics = box_metrics(
                    mean_to_box(mean), np.asarray(gt_box), img_w, img_h
                )
                for key in ("center", "size", "iou"):
                    collected[name][gap][key].append(metrics[key])
    print(f"{sequence}: shared={len(shared)} rows={sequence_rows}", flush=True)
    return sequence, sequence_rows, collected, recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path")
    parser.add_argument("--sequences-file", default=str(VAL_MAP))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--log-root",
        default=str(
            MAIN
            / "outputs/agentguard/experiments/shutrack_a_supervision_speed4_20261119/logs/paired_gap"
        ),
    )
    args = parser.parse_args()

    sequences = [
        line.strip().split()[0]
        for line in Path(args.sequences_file).read_text().splitlines()
        if line.strip() and line.strip() != "name"
    ]
    if args.only:
        wanted = set(args.only)
        sequences = [sequence for sequence in sequences if sequence in wanted]
    checkpoints = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        checkpoints[name] = Path(path)
    log_root = Path(args.log_root)

    collected, recovery = empty_metrics(checkpoints)
    per_sequence = {}
    jobs = [(sequence, checkpoints, log_root) for sequence in sequences]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for sequence, counts, values, recoveries in pool.map(measure_sequence, jobs):
            per_sequence[sequence] = counts
            for name in checkpoints:
                for gap in GAPS:
                    for metric, items in values[name][gap].items():
                        collected[name][gap][metric].extend(items)
                    for key, count in recoveries[name][gap].items():
                        recovery[name][gap][key] += count

    output = {"protocol": "verified_identity_same_detection_v2", "recovery": recovery,
              "groups": {}, "paired_vs": {}, "per_sequence_rows": per_sequence}
    names = list(checkpoints)
    for name in names:
        output["groups"][name] = {
            str(gap): {
                key: summarize(values)
                for key, values in collected[name][gap].items()
            }
            for gap in GAPS
        }
    baseline = names[0]
    for name in names[1:]:
        output["paired_vs"][name] = {}
        for gap in GAPS:
            output["paired_vs"][name][str(gap)] = {
                key: summarize_paired(
                    [
                        value - base
                        for value, base in zip(
                            collected[name][gap][key], collected[baseline][gap][key]
                        )
                    ]
                )
                for key in ("center", "size", "iou")
            }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["groups"], indent=2, sort_keys=True))
    print("saved", args.output)


if __name__ == "__main__":
    main()
