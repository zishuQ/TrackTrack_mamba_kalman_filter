"""Real replay parity for collection changes; never starts model training."""

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sportsmot_supervision_pilot as current
from paired_column import ColumnPilot
from paired_training_input import full_write_checkpoint
from paired_collection_io import atomic_json, inspect_shard


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("reference", "new", "compare"), required=True)
    parser.add_argument("--dataset", default="SportsMOT")
    parser.add_argument("--sequence", default="v_9MHDmAMxO5I_c003")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    base = ROOT / "outputs/agentguard/experiments/shutrack_collection_fix_20261121"
    sequence_root = base / "parity" / args.sequence
    if args.case == "compare":
        old, new = sequence_root / "reference/data" / args.sequence, sequence_root / "new/data" / args.sequence
        compared = 0
        with np.load(old / "samples.npz", allow_pickle=False) as a, np.load(new / "samples.npz", allow_pickle=False) as b:
            assert a.files == b.files
            for name in a.files:
                x, y = a[name], b[name]
                assert x.shape == y.shape and x.dtype == y.dtype, name
                assert np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y), name
                compared += 1
            count = len(a["frame"])
        for name in ("event_track", "event_evidence", "event_detection_index", "event_has_detection"):
            np.testing.assert_array_equal(np.load(old / (name + ".npy")), np.load(new / (name + ".npy")))
        old_summary, new_summary = (json.loads((d / "summary.json").read_text()) for d in (old, new))
        assert old_summary["column_counts"] == new_summary["column_counts"]
        assert new_summary["samples"] == count
        report = {"sequence": args.sequence, "samples": count, "npz_fields_exact": compared,
                  "event_arrays_exact": 4, "column_counts": new_summary["column_counts"],
                  "header_gate": inspect_shard(new)}
        atomic_json(sequence_root / "comparison.json", report)
        print(json.dumps(report, indent=2))
        return
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sequence_root.mkdir(parents=True, exist_ok=True)
    checkpoint = sequence_root / "full_write.pt"
    full_write_checkpoint(checkpoint, 2048)
    if args.case == "reference":
        reference = load_module("collection_reference_pilot", base / "reference_pilot.py")
        reference.WORKTREE = ROOT
        reference.MAIN = ROOT
        reference.DEFAULT_CACHE = current.DEFAULT_CACHE
        reference.GT_ROOT = current.GT_ROOT
        cls = load_module("collection_reference_column", base / "reference_column.py").ColumnPilot
    else:
        cls = ColumnPilot
    split = "all" if args.dataset != "SportsMOT" else "val"
    out = sequence_root / args.case
    pilot = cls(args.sequence, split, checkpoint, out / "logs",
                max_frames=args.frames, dataset=args.dataset)
    pilot.lean = True
    pilot.protocol = {"schema": 3, "full_write": True, "anchor": False,
                      "column": True, "gap3": False, "rank_arrays": True,
                      "teacher_sha256": None}
    if args.dataset != "SportsMOT":
        pilot.protocol["dataset"] = args.dataset
    started = time.perf_counter()
    try:
        pilot.run_replay()
        prototype_path = out / "prototypes.npz"
        np.savez(prototype_path, **{str(k): v for k, v in pilot.prototypes.items()})
        pilot.compute_labels()
        pilot.appearance_diagnostics()
        summary = pilot.write_compact(out / "data")
        stats = {
            "case": args.case, "seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "summary": summary,
        }
        atomic_json(out / "stats.json", stats)
        print(json.dumps(stats, indent=2))
    finally:
        pilot.close()


if __name__ == "__main__":
    main()
