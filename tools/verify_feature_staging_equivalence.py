#!/usr/bin/env python3
"""Verify staged feature rows reproduce the mmap inputs tensor-for-tensor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT, ROOT / "3. Tracker", ROOT / "agentguard/src"):
    sys.path.insert(0, str(path))
import train_supervision_groups as base  # noqa: E402
from shutrack.training.defaults import MAINLINE_STAGE_CHUNK_SAMPLES  # noqa: E402
from shutrack.training.feature_staging import (  # noqa: E402
    aligned_stage_chunk_samples, feature_chunks, staged_sequence_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequence", default="MOT20-01")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-samples", type=int, default=MAINLINE_STAGE_CHUNK_SAMPLES)
    parser.add_argument("--group", default="appearance_row_column")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    data = base.SequenceData(args.sequence, "all", data_root, "MOT20")
    data.open_cache()
    taus = base.load_global_taus([args.sequence], data_root)
    targets = base.derive_targets(data.samples, taus)
    n = min(args.rows, data.n)
    rng = np.random.RandomState(42)
    rows = rng.choice(data.n, n, replace=False).astype(np.int64)
    batch_size = args.batch_size
    aligned = aligned_stage_chunk_samples(args.chunk_samples, batch_size)
    expected_rows = [rows[start:start + batch_size] for start in range(0, n, batch_size)]
    mismatch = []
    staged_chunks = []
    checked_batches = 0
    for chunk in feature_chunks(rows, aligned, enabled=True):
        chunk_size = (len(chunk) + batch_size - 1) // batch_size
        reference = [
            base.batch_tensors(data, selected, targets, args.group, compact_reid=True)
            for selected in expected_rows[checked_batches:checked_batches + chunk_size]
        ]
        with staged_sequence_rows(data, chunk, args.group) as staged:
            staged_chunks.append(staged)
            for start, ref_batch in zip(range(0, len(chunk), batch_size), reference, strict=True):
                selected = chunk[start:start + batch_size]
                if not np.array_equal(selected, expected_rows[checked_batches]):
                    mismatch.append({"key": "row_indices", "batch": checked_batches})
                staged_batch = base.batch_tensors(data, selected, targets, args.group,
                                                  compact_reid=True)
                for key in ref_batch:
                    a = np.asarray(ref_batch[key])
                    b = np.asarray(staged_batch[key])
                    if a.dtype != b.dtype or not np.array_equal(a, b):
                        mismatch.append({"key": key, "batch": checked_batches,
                                         "shape_a": list(a.shape), "shape_b": list(b.shape)})
                checked_batches += 1
    result = {"sequence": args.sequence, "rows": n, "batches": checked_batches,
              "aligned_chunk_samples": aligned, "staged_chunks": staged_chunks,
              "mismatches": mismatch,
              "identical": not mismatch and checked_batches == len(expected_rows)}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    data.close_cache()
    if not result["identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
