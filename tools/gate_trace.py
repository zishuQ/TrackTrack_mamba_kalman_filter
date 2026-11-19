#!/usr/bin/env python3
"""Trace per-event gate decisions during an online replay of one sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE),
    str(WORKTREE / "3. Tracker"),
    str(WORKTREE / "agentguard/src"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from agentguard.data.detection_cache import SequenceDetectionCache
import run as run_module
from trackers import tracker as tracker_module
from trackers.track import Track


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args_ns = parser.parse_args()

    args = run_module.make_parser().parse_args([])
    args.dataset = "SportsMOT"
    args.mode = args_ns.split
    args.sequences = [args_ns.sequence]
    args.seed = 10000
    args.kf_type = "nsa"
    args.agentguard_mode = "shutrack-v2"
    args.shutrack_checkpoint = str(args_ns.checkpoint)
    args.agentguard_device = "cpu"
    args.detection_cache_root = str(MAIN / "outputs/agentguard/detection_cache")
    args.skip_eval = True
    args.torch_num_threads = 4
    args.torch_num_interop_threads = 1
    run_module.set_parameters(args, args_ns.sequence, args_ns.split)
    args.max_time_lost = 50
    args.img_w = 1280
    args.img_h = 720

    records = []
    original = Track.update_with_gates

    def traced(self, frame_id, detection, motion_gate, appearance_gate):
        records.append(
            {
                "frame": int(frame_id),
                "track_id": int(self.track_id),
                "observation_count": int(self.observation_count),
                "motion_gate": float(motion_gate),
                "appearance_gate": float(appearance_gate),
            }
        )
        return original(self, frame_id, detection, motion_gate, appearance_gate)

    Track.update_with_gates = traced
    det_cache = SequenceDetectionCache(
        run_module._sequence_detection_cache_dir(args, args_ns.sequence),
        frame_array_cache_size=4,
    )
    tracker = tracker_module.Tracker(args, args_ns.sequence)
    try:
        for frame_index in range(det_cache.num_frames):
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
    finally:
        det_cache.close()
        Track.update_with_gates = original
    Path(args_ns.out).write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    print(f"records={len(records)} -> {args_ns.out}")


if __name__ == "__main__":
    main()
