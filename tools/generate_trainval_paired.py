#!/usr/bin/env python3
"""Generate compact paired supervision data for all SportsMOT trainval sequences."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (str(WORKTREE / "tools"), str(WORKTREE), str(WORKTREE / "agentguard/src"), str(WORKTREE / "3. Tracker")):
    if path not in sys.path:
        sys.path.insert(0, path)

from sportsmot_supervision_pilot import SequencePilot, DEFAULT_CACHE
from paired_training_input import full_write_checkpoint, official_sequences, replay_split
from agentguard.data.detection_cache import SequenceDetectionCache
from paired_collection_io import atomic_json, inspect_shard, collection_lock

ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
DATA = ROOT / "data"
LOGS = ROOT / "logs"
# Label semantics: 3 = appearance window search + online IoU recovered pool +
# identity-evidence invalidation after a GT gap.  Schema-2 shards are stale.
LABEL_SEMANTICS_VERSION = 3


def existing_summary_state(summary: dict, protocol: dict) -> str:
    """Return 'incompatible', 'complete', or 'incomplete' for an existing shard."""
    if summary.get("protocol") != protocol:
        return "incompatible"
    return "complete" if summary.get("complete") else "incomplete"


def collect_pilot(pilot, data_root, *, anchor=False):
    """Close resources on every Python exception without hiding its cause."""
    try:
        pilot.run_replay()
        if anchor:
            from paired_identity import recover_identity_labels
            recover_identity_labels(pilot)
        pilot.compute_labels()
        pilot.appearance_diagnostics()
        return pilot.write_compact(data_root)
    finally:
        failing = sys.exc_info()[0] is not None
        try:
            pilot.close()
        except Exception:
            if not failing:
                raise
            print("Collection cleanup also failed:\n" + traceback.format_exc(), file=sys.stderr)


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
    parser.add_argument("--dataset", choices=("SportsMOT", "MOT17", "MOT20", "DanceTrack"), default="SportsMOT")
    parser.add_argument("--checkpoint", type=Path, help="Teacher checkpoint for gated replay.")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--skip-rank-arrays",
        action="store_true",
        help="omit the 2048-d rank features (smaller shards for motion-only groups)",
    )
    parser.add_argument("--full-write", action="store_true")
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--anchor-identities", action="store_true")
    parser.add_argument("--column-probe", action="store_true")
    parser.add_argument("--gap3", action="store_true")
    parser.add_argument(
        "--lean",
        action="store_true",
        help="low-memory replay: finalize labels/rank features per event and "
        "release heavy state instead of holding the whole sequence",
    )
    args = parser.parse_args()
    if not args.full_write and args.checkpoint is None:
        parser.error("--checkpoint is required unless --full-write is set")
    root = Path(args.root)
    data = root / "data"
    logs = root / "logs"
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    sequences = sorted(official_sequences("train", args.dataset) | (official_sequences("val", args.dataset) if args.split == "trainval" else set()))
    if args.only:
        sequences = [seq for seq in sequences if seq in set(args.only)]
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise ValueError("No sequences selected")
    val_set = val_sequences()
    temp = tempfile.TemporaryDirectory(dir=root)
    checkpoint = args.checkpoint
    if args.full_write:
        first = sequences[0]
        cache = SequenceDetectionCache(DEFAULT_CACHE / args.dataset / replay_split(first, args.dataset) / first)
        try:
            checkpoint = Path(temp.name) / "full_write.pt"
            full_write_checkpoint(checkpoint, cache.reid_dim)
        finally:
            cache.close()
    protocol = {"schema": LABEL_SEMANTICS_VERSION, "full_write": args.full_write, "anchor": args.anchor_identities,
                "column": args.column_probe, "gap3": args.gap3, "rank_arrays": not args.skip_rank_arrays,
                "teacher_sha256": None if args.full_write else hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    if args.dataset != "SportsMOT":
        protocol["dataset"] = args.dataset

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

    pilot_class = SequencePilot
    if args.column_probe:
        from paired_column import ColumnPilot
        pilot_class = ColumnPilot

    for index, sequence in enumerate(sequences, start=1):
        with collection_lock(data / sequence):
            target = data / sequence / "summary.json"
            if target.is_file():
                summary = json.loads(target.read_text())
                state_kind = existing_summary_state(summary, protocol)
                if state_kind == "incompatible":
                    raise ValueError(
                        f"{target}: incompatible replay or label semantics "
                        f"(schema {LABEL_SEMANTICS_VERSION}); use a new output directory"
                    )
                if state_kind == "complete":
                    try:
                        check = inspect_shard(target.parent, protocol, repair_counts=True)
                    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                        print(f"[{index}/{len(sequences)}] rebuild incomplete {sequence}: {exc}", flush=True)
                    else:
                        state["skipped"].append(sequence)
                        print(f"[{index}/{len(sequences)}] skip {sequence} counts={check}", flush=True)
                        atomic_json(state_path, state)
                        continue
            split = replay_split(sequence, args.dataset)
            started = time.time()
            try:
                pilot = pilot_class(
                    sequence, split, checkpoint, logs / sequence, dataset=args.dataset
                )
                pilot.protocol = protocol
                pilot.lean = bool(args.lean)
                if args.skip_rank_arrays:
                    pilot.store_rank_features = False
                atomic_json(data / sequence / "summary.json", {
                    "sequence": sequence, "protocol": protocol, "complete": False,
                    "stage": "collecting",
                })
                summary = collect_pilot(pilot, data, anchor=args.anchor_identities)
                inspect_shard(data / sequence, protocol)
                summary["seconds"] = time.time() - started
                summary["split"] = split
                atomic_json(data / sequence / "summary.json", summary)
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
            atomic_json(state_path, state)

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_json(state_path, state)
    temp.cleanup()
    print(json.dumps({"done": len(state["done"]), "failed": len(state["failed"]), "skipped": len(state["skipped"])}))
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
