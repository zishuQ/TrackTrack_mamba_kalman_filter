#!/usr/bin/env python3
"""Record the two predicted gate distributions on a test sequence.

Runs the real tracker entry (no GT needed) with the final checkpoint and hooks
Track.update_with_gates, reporting motion/appearance gate statistics.  Used to
prove that inference uses the learned gates rather than the fixed g=1 write
used only for data collection.
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
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentguard.data.detection_cache import SequenceDetectionCache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("SportsMOT", "MOT17", "MOT20"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--detection-cache-root", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import run as run_module
    from trackers import tracker as tracker_module
    from trackers.track import Track

    run_parser = run_module.make_parser().parse_args([])
    run_parser.dataset = args.dataset
    run_parser.mode = "test"
    run_parser.sequences = [args.sequence]
    run_parser.seed = 10000
    run_parser.kf_type = "nsa"
    run_parser.agentguard_mode = "shutrack-v2"
    run_parser.shutrack_checkpoint = args.checkpoint
    run_parser.agentguard_device = "cpu"
    run_parser.detection_cache_root = args.detection_cache_root
    run_parser.skip_eval = True
    run_parser.torch_num_threads = 4
    run_parser.torch_num_interop_threads = 1
    run_module.set_parameters(run_parser, args.sequence, "test")
    cache = SequenceDetectionCache(
        run_module._sequence_detection_cache_dir(run_parser, args.sequence),
        frame_array_cache_size=4,
    )
    gates: list[tuple[float, float]] = []
    plain = 0
    original_gated = Track.update_with_gates
    original_update = Track.update

    def gated(track_self, frame_id, detection, motion_gate, appearance_gate):
        gates.append((float(motion_gate), float(appearance_gate)))
        return original_gated(track_self, frame_id, detection, motion_gate, appearance_gate)

    def plain_update(track_self, frame_id, detection):
        nonlocal plain
        plain += 1
        return original_update(track_self, frame_id, detection)

    Track.update_with_gates = gated
    Track.update = plain_update
    try:
        tracker = tracker_module.Tracker(run_parser, args.sequence)
        limit = min(cache.num_frames, max(1, int(args.frames)))
        for frame_index in range(limit):
            target = cache.get_frame(frame_index, view="target")
            source = cache.get_frame(frame_index, view="source")
            run_parser.agentguard_target_detection_indices = target["detection_indices"]
            run_parser.agentguard_source_detection_indices = source["detection_indices"]
            det = cache.get_frame_array(frame_index, view="target")
            det_95 = cache.get_frame_array(frame_index, view="source")
            if det is not None and len(det) > 0:
                tracker.update(det, det_95)
            else:
                tracker.update_without_detections()
    finally:
        Track.update_with_gates = original_gated
        Track.update = original_update
        cache.close()

    array = np.asarray(gates, dtype=np.float64)
    payload = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "frames": limit,
        "gated_writes": int(array.shape[0]),
        "plain_updates": int(plain),
        "motion_gate": {
            "mean": float(array[:, 0].mean()) if array.size else None,
            "std": float(array[:, 0].std()) if array.size else None,
            "min": float(array[:, 0].min()) if array.size else None,
            "max": float(array[:, 0].max()) if array.size else None,
            "fraction_at_one": float((array[:, 0] >= 0.999).mean()) if array.size else None,
        },
        "appearance_gate": {
            "mean": float(array[:, 1].mean()) if array.size else None,
            "std": float(array[:, 1].std()) if array.size else None,
            "min": float(array[:, 1].min()) if array.size else None,
            "max": float(array[:, 1].max()) if array.size else None,
            "fraction_at_one": float((array[:, 1] >= 0.999).mean()) if array.size else None,
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
