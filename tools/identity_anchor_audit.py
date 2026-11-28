#!/usr/bin/env python3
"""Audit anchor identity recovery for one sequence.

``--old-binding`` reproduces the pre-fix GT binding (target view only, best
IoU initialised at 0.5 with a strict comparison); the default uses the fixed
binding.  The anchor function itself is the fixed one.
"""

from __future__ import annotations

import argparse
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

from sportsmot_supervision_pilot import SequencePilot, iou_xyxy  # noqa: E402
from paired_identity import recover_identity_labels  # noqa: E402


class OldBindingPilot(SequencePilot):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--old-binding", action="store_true")
    parser.add_argument("--log-root", default="/tmp/identity_anchor_audit")
    args = parser.parse_args()
    pilot_class = OldBindingPilot if args.old_binding else SequencePilot
    pilot = pilot_class(
        args.sequence, args.split, args.checkpoint, Path(args.log_root) / args.sequence
    )
    pilot.run_replay()
    recovered = recover_identity_labels(pilot)
    rows = [
        {"frame": int(frame), "track_id": int(track), "gt_id": int(gt)}
        for frame, track, gt in recovered
    ]
    payload = {
        "sequence": args.sequence,
        "split": args.split,
        "old_binding": bool(args.old_binding),
        "recovered": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    pilot.close()
    frames = sorted({row["frame"] for row in rows})
    print(
        json.dumps(
            {
                "sequence": args.sequence,
                "old_binding": bool(args.old_binding),
                "recovered": len(rows),
                "frames": [frames[0], frames[-1]] if frames else [],
            }
        )
    )


if __name__ == "__main__":
    main()
