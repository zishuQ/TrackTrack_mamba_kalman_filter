#!/usr/bin/env python3
"""Center-prediction error after consecutive missed detections.

Replays the SportsMOT tracker with a given checkpoint (same entry, caches,
seed and NSA settings as the val evaluation), captures every track's output
box per frame, and compares it with the GT box of the track's resolved
identity at the 1st, 3rd and 5th consecutive missed-detection frame.
Errors are normalised by image width/height.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (str(WORKTREE / "tools"), str(WORKTREE), str(WORKTREE / "agentguard/src"), str(WORKTREE / "3. Tracker")):
    if path not in sys.path:
        sys.path.insert(0, path)

from sportsmot_supervision_pilot import SequencePilot  # noqa: E402

VAL_MAP = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
GAP_INDICES = (1, 3, 5)


class GapPilot(SequencePilot):
    def __init__(self, sequence, split, checkpoint, out_dir):
        super().__init__(sequence, split, checkpoint, out_dir)
        self.frame_tracks: dict[int, list[tuple[int, np.ndarray]]] = {}

    def build_prototypes(self) -> None:
        # Identity votes need the per-frame detection->GT matches; prototypes
        # themselves are unused for this diagnostic.
        for frame_index in range(self.det_cache.num_frames):
            self.frame_matches(frame_index)
        self.prototypes = {}

    def install_hooks(self) -> None:
        from trackers import tracker as tracker_module

        super().install_hooks()
        current_update = tracker_module.Tracker.update
        current_no_det = tracker_module.Tracker.update_without_detections
        pilot = self

        def update(tracker_self, dets, dets_95):
            result = current_update(tracker_self, dets, dets_95)
            pilot._capture_frame(tracker_self)
            return result

        def update_no_det(tracker_self):
            result = current_no_det(tracker_self)
            pilot._capture_frame(tracker_self)
            return result

        tracker_module.Tracker.update = update
        tracker_module.Tracker.update_without_detections = update_no_det
        # Restore our wrappers before the base hooks so the originals win last.
        self._restore = [
            (tracker_module.Tracker, "update", current_update),
            (tracker_module.Tracker, "update_without_detections", current_no_det),
        ] + self._restore

    def _capture_frame(self, tracker) -> None:
        frame = int(tracker.frame_id)
        entries = []
        for track in tracker.tracks:
            if track.mean is None:
                continue
            entries.append(
                (int(track.track_id), np.asarray(track.x1y1x2y2, dtype=np.float64).copy())
            )
        self.frame_tracks[frame] = entries


def replay(sequence: str, split: str, checkpoint: Path, out_dir: Path) -> dict:
    pilot = GapPilot(sequence, split, checkpoint, out_dir)
    pilot.run_replay()
    # Resolved identity per track over its detection events (online vote order).
    detections: dict[int, set[int]] = {}
    identity_at: dict[int, list[tuple[int, int]]] = {}
    for record in sorted(pilot.events, key=lambda item: (item["frame_t"], item["track_id"])):
        frame = int(record["frame_t"])
        track_id = int(record["track_id"])
        detections.setdefault(track_id, set()).add(frame)
        resolved = record.get("target_gt_before")
        if resolved is not None:
            identity_at.setdefault(track_id, []).append((frame, int(resolved)))
    gt = pilot.gt
    img_w = float(pilot.img_w)
    img_h = float(pilot.img_h)

    def identity_for(track_id: int, frame: int):
        best = None
        for event_frame, gt_id in identity_at.get(track_id, []):
            if event_frame <= frame:
                best = gt_id
            else:
                break
        return best

    errors = {index: [] for index in GAP_INDICES}
    per_sequence_gap_frames = 0
    boxes: dict[int, dict[int, np.ndarray]] = {}
    for frame, entries in pilot.frame_tracks.items():
        for track_id, box in entries:
            boxes.setdefault(track_id, {})[frame] = box
    for track_id, box_by_frame in boxes.items():
        captured = sorted(box_by_frame)
        if not captured:
            continue
        detection_frames = detections.get(track_id, set())
        detection_frames = detection_frames | {captured[0]}
        miss_run = 0
        for frame in captured:
            if frame in detection_frames:
                miss_run = 0
                continue
            miss_run += 1
            per_sequence_gap_frames += 1
            if miss_run not in GAP_INDICES:
                continue
            gt_id = identity_for(track_id, frame)
            if gt_id is None:
                continue
            gt_box = gt.get(frame, {}).get(gt_id)
            if gt_box is None:
                continue
            box = box_by_frame[frame]
            pred_center = np.array(
                [(box[0] + box[2]) / 2.0 / img_w, (box[1] + box[3]) / 2.0 / img_h]
            )
            gt_center = np.array(
                [
                    (gt_box[0] + gt_box[2]) / 2.0 / img_w,
                    (gt_box[1] + gt_box[3]) / 2.0 / img_h,
                ]
            )
            error = float(np.linalg.norm(pred_center - gt_center))
            errors[miss_run].append(error)
    pilot.close()
    return {
        "gap_frames": per_sequence_gap_frames,
        "errors": {str(index): values for index, values in errors.items()},
    }


def summarize(errors: list[float]) -> dict:
    if not errors:
        return {"count": 0, "mean": None, "median": None}
    array = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path")
    parser.add_argument("--sequences-file", default=str(VAL_MAP))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-root", default=str(MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119/logs"))
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
    output_path = Path(args.output)
    if output_path.is_file():
        output = json.loads(output_path.read_text())
    else:
        output = {}
    for name in checkpoints:
        output.setdefault(name, {"sequences": {}, "combined": {}})
    for name, checkpoint in checkpoints.items():
        combined = {index: [] for index in GAP_INDICES}
        for sequence in sequences:
            entry = replay(sequence, "val", checkpoint, log_root)
            output[name]["sequences"][sequence] = entry
            for index in GAP_INDICES:
                combined[index].extend(entry["errors"][str(index)])
            print(
                f"{name} {sequence}: gap_frames={entry['gap_frames']} "
                f"e1={len(entry['errors']['1'])} e3={len(entry['errors']['3'])} "
                f"e5={len(entry['errors']['5'])}",
                flush=True,
            )
        output[name]["combined"] = {
            str(index): summarize(values) for index, values in combined.items()
        }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {name: output[name]["combined"] for name in output}, indent=2, sort_keys=True
    ))


if __name__ == "__main__":
    main()
