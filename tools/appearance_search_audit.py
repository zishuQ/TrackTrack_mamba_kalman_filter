#!/usr/bin/env python3
"""Audit the appearance positive/competitor search for one sequence.

Runs the same e200 replay and records, for every mature event with a valid
appearance pair, the chosen positive/competitor detection indices.  Used to
reproduce and compare the early-exit fix and the GT-binding fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from sportsmot_supervision_pilot import SequencePilot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-root", default="/tmp/appearance_search_audit")
    args = parser.parse_args()

    pilot = SequencePilot(
        args.sequence, args.split, args.checkpoint, Path(args.log_root) / args.sequence
    )
    pilot.run_replay()
    pilot.appearance_diagnostics()
    valid = [record for record in pilot.events if record.get("appearance_diag_valid")]
    rows = [
        {
            "frame": int(record["frame_t"]),
            "track_id": int(record["track_id"]),
            "gt_id": int(record["target_gt_before"]),
            "positive_frame": int(record.get("appearance_positive_frame", -1)),
            "positive_index": int(record.get("appearance_positive_index", -1)),
            "competitor_index": int(record.get("appearance_competitor_index", -1)),
        }
        for record in valid
    ]
    payload = {
        "sequence": args.sequence,
        "split": args.split,
        "events": len(pilot.events),
        "valid": len(rows),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    pilot.close()
    print(json.dumps({"sequence": args.sequence, "valid": len(rows), "events": len(pilot.events)}))


if __name__ == "__main__":
    main()
