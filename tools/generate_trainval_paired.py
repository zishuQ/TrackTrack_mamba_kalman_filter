#!/usr/bin/env python3
"""Generate compact paired supervision data for all SportsMOT trainval sequences."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (str(WORKTREE / "tools"), str(WORKTREE), str(WORKTREE / "agentguard/src"), str(WORKTREE / "3. Tracker")):
    if path not in sys.path:
        sys.path.insert(0, path)

from sportsmot_supervision_pilot import SequencePilot, DEFAULT_CKPT

ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
DATA = ROOT / "data"
LOGS = ROOT / "logs"


def val_sequences() -> set[str]:
    path = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and line.strip() != "name"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    data = root / "data"
    logs = root / "logs"
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (MAIN / "outputs/agentguard/train_data/SportsMOT/metadata.json").read_text()
    )
    sequences = list(metadata["train_sequences"])
    if args.only:
        sequences = [seq for seq in sequences if seq in set(args.only)]
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    val_set = val_sequences()

    state = {
        "total": len(sequences),
        "done": [],
        "failed": [],
        "skipped": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state_path = root / "generation_state.json"
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    for index, sequence in enumerate(sequences, start=1):
        target = data / sequence / "summary.json"
        if target.is_file():
            try:
                summary = json.loads(target.read_text())
                if summary.get("complete"):
                    state["skipped"].append(sequence)
                    print(f"[{index}/{len(sequences)}] skip {sequence}", flush=True)
                    continue
            except Exception:
                pass
        split = "val" if sequence in val_set else "train"
        started = time.time()
        try:
            pilot = SequencePilot(
                sequence, split, Path(args.checkpoint), logs / sequence
            )
            pilot.run_replay()
            pilot.compute_labels()
            pilot.appearance_diagnostics()
            summary = pilot.write_compact(data)
            pilot.close()
            summary["seconds"] = time.time() - started
            summary["split"] = split
            state["done"].append(sequence)
            print(
                f"[{index}/{len(sequences)}] {sequence} split={split} "
                f"samples={summary['samples']} valid_v={summary['valid_motion']} "
                f"gt={summary['motion_gt_valid']} app={summary['appearance_diag_valid']} "
                f"{summary['seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            error = traceback.format_exc()
            (logs / f"{sequence}.error.log").write_text(error)
            state["failed"].append({"sequence": sequence, "error": str(exc)})
            print(f"[{index}/{len(sequences)}] FAILED {sequence}: {exc}", flush=True)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"done": len(state["done"]), "failed": len(state["failed"]), "skipped": len(state["skipped"])}))


if __name__ == "__main__":
    main()
