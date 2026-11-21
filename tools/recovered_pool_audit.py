#!/usr/bin/env python3
"""Compare label-side recovered-detection sets before/after the online-IoU fix."""

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

from agentguard.data.detection_cache import SequenceDetectionCache  # noqa: E402
from sportsmot_supervision_pilot import (  # noqa: E402
    DEFAULT_CACHE,
    max_box_iou,
    recovered_detection_rows,
)

DET_THR = 0.55


def old_recovered(target, source, source_indices):
    if target is None or len(target) == 0:
        return set()
    if source is None or len(source) == 0:
        return set()
    keep = max_box_iou(source[:, :4], target[:, :4]) < 0.97
    return {
        int(index)
        for row, index in zip(source[keep], source_indices[keep])
        if float(row[4]) > DET_THR
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cache = SequenceDetectionCache(DEFAULT_CACHE / "SportsMOT" / args.split / args.sequence)
    frames = 0
    differences = []
    old_total = new_total = 0
    try:
        for frame_index in range(cache.num_frames):
            target = cache.get_frame_array(frame_index, view="target")
            source = cache.get_frame_array(frame_index, view="source")
            source_indices = cache.get_frame(frame_index, view="source")[
                "detection_indices"
            ]
            old = old_recovered(target, source, source_indices)
            new = {
                int(index)
                for index, _ in recovered_detection_rows(
                    target, source, source_indices, DET_THR
                )
            }
            frames += 1
            old_total += len(old)
            new_total += len(new)
            if old != new:
                differences.append(
                    {
                        "frame": frame_index + 1,
                        "old_only": sorted(old - new),
                        "new_only": sorted(new - old),
                    }
                )
    finally:
        cache.close()
    payload = {
        "sequence": args.sequence,
        "split": args.split,
        "frames": frames,
        "old_total": old_total,
        "new_total": new_total,
        "frames_with_difference": len(differences),
        "differences": differences,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "sequence": args.sequence,
                "frames": frames,
                "old_total": old_total,
                "new_total": new_total,
                "frames_with_difference": len(differences),
            }
        )
    )


if __name__ == "__main__":
    main()
