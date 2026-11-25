#!/usr/bin/env python3
"""Build small, row-bound target sidecars from existing endpoint/GT arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shutrack.training.motion_targets import (
    GRID, SOURCE_KEYS, TIE_TOLERANCE, generate_targets, numpy_diou, source_fingerprint,
)


def generate_sequence(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(source / "samples.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in SOURCE_KEYS}
    fingerprint = source_fingerprint(arrays)
    metadata_path = destination / "metadata.json"
    target_path = destination / "motion_gate_target.npy"
    if metadata_path.is_file() and target_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_sha256"] != fingerprint:
            raise ValueError(f"{source.name}: existing labels belong to different source rows")
        target = np.load(target_path, allow_pickle=False)
        if target.shape != (len(arrays["frame"]),) or target.dtype != np.float32:
            raise ValueError(f"{source.name}: invalid saved target shape/dtype")
        if (metadata["grid"] != GRID.tolist() or metadata["tie_tolerance"] != TIE_TOLERANCE
                or metadata["target_sha256"] != hashlib.sha256(target.tobytes()).hexdigest()):
            raise ValueError(f"{source.name}: saved gate labels differ from their policy/checksum")
        return metadata
    valid = arrays["motion_gt_valid"].astype(bool)
    target = generate_targets(arrays["pred_box_g0"], arrays["pred_box_g1"],
                              arrays["gt_box_t1"], valid)
    rows = np.flatnonzero(valid)
    a = arrays["pred_box_g0"][rows].astype(np.float64)
    b = arrays["pred_box_g1"][rows].astype(np.float64)
    gt = arrays["gt_box_t1"][rows].astype(np.float64)
    full_loss = 1 - numpy_diou(b, gt)
    chosen_loss = 1 - numpy_diou(a + target[rows, None] * (b - a), gt)
    # Independent sparse audit of the saved label's loss and tie policy.
    check = rows[np.linspace(0, max(0, len(rows)-1), min(128,len(rows)), dtype=int)]
    if len(check):
        aa = arrays["pred_box_g0"][check].astype(np.float64)
        dd = arrays["pred_box_g1"][check].astype(np.float64) - aa
        losses = 1 - numpy_diou(aa[None] + GRID[:,None,None]*dd[None],
                                arrays["gt_box_t1"][check].astype(np.float64)[None])
        eligible = losses <= losses.min(0)[None] + TIE_TOLERANCE
        expected = GRID[eligible.argmax(0)].astype(np.float32)
        np.testing.assert_array_equal(target[check], expected)
    metadata = {
        "sequence": source.name, "source_sha256": fingerprint, "rows": len(target),
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "valid_rows": int(valid.sum()), "dtype": "float32", "grid": GRID.tolist(),
        "tie_tolerance": TIE_TOLERANCE, "invalid_target": 1.0,
        "target_mean": float(target[valid].mean()) if valid.any() else None,
        "target_below_one_fraction": float((target[valid]<1).mean()) if valid.any() else None,
        "target_at_most_half_fraction": float((target[valid]<=.5).mean()) if valid.any() else None,
        "full_write_loss": float(full_loss.mean()) if len(rows) else None,
        "target_loss": float(chosen_loss.mean()) if len(rows) else None,
        "audited_rows": len(check), "bytes": target.nbytes,
    }
    temporary = destination / "motion_gate_target.tmp.npy"
    np.save(temporary, target, allow_pickle=False)
    temporary.replace(target_path)
    temporary_json = destination / "metadata.tmp.json"
    temporary_json.write_text(json.dumps(metadata, indent=2)+"\n")
    temporary_json.replace(metadata_path)
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--sequences", nargs="*")
    args=parser.parse_args()
    data=Path(args.data_root)
    names=args.sequences or sorted(p.name for p in data.iterdir() if (p/"samples.npz").is_file())
    records=[]
    for name in names:
        record=generate_sequence(data/name, Path(args.target_root)/name)
        records.append(record)
        print(json.dumps(record), flush=True)
    print(json.dumps({"sequences":len(records),"rows":sum(r["rows"] for r in records),
                      "label_bytes":sum(r["bytes"] for r in records)}),flush=True)


if __name__=="__main__":
    main()
