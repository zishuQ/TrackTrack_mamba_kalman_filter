#!/usr/bin/env python3
"""Precompute pairwise motion-gate ranking labels from existing caches.

For every sample row (motion_gt_valid):
  keep      : g_star == 1 and L(g=0.5) > L(g=1) + 1e-6
  suppress  : g_star <  1 and L(g=1) > L_best + 1e-6
Only three scalar arrays are added per sequence:
  motion_rank_label.npy (uint8: 0 none, 1 keep, 2 suppress)
  motion_rank_gain.npy  (L(1) - L_best for suppress rows, else 0)
  motion_rank_damage.npy(L(0.5) - L(1) for keep rows, else 0)
Source rows and the gate-target checksum are bound in metadata.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from shutrack.training.motion_targets import SOURCE_KEYS, source_fingerprint, numpy_diou

TOL = 1e-6


def generate(sequence, data_dir, target_dir, out_dir):
    with np.load(data_dir / "samples.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in SOURCE_KEYS}
    metadata = json.loads((target_dir / "metadata.json").read_text())
    fingerprint = source_fingerprint(arrays)
    if metadata["source_sha256"] != fingerprint:
        raise ValueError(f"{sequence}: target sidecar does not match source rows")
    target = np.load(target_dir / "motion_gate_target.npy", allow_pickle=False)
    if metadata["target_sha256"] != hashlib.sha256(target.tobytes()).hexdigest():
        raise ValueError(f"{sequence}: target checksum differs")

    valid = arrays["motion_gt_valid"].astype(bool)
    g0 = arrays["pred_box_g0"].astype(np.float64)
    g1 = arrays["pred_box_g1"].astype(np.float64)
    gt = arrays["gt_box_t1"].astype(np.float64)
    star = target.astype(np.float64)
    box_half = g0 + 0.5 * (g1 - g0)
    box_best = g0 + star[:, None] * (g1 - g0)
    loss1 = 1.0 - numpy_diou(g1, gt)
    loss_half = 1.0 - numpy_diou(box_half, gt)
    loss_best = 1.0 - numpy_diou(box_best, gt)

    keep = valid & (star >= 1.0 - TOL) & (loss_half > loss1 + TOL)
    suppress = valid & (star < 1.0 - TOL) & (loss1 > loss_best + TOL)
    label = np.zeros(len(target), dtype=np.uint8)
    label[keep] = 1
    label[suppress] = 2
    gain = np.where(suppress, loss1 - loss_best, 0.0).astype(np.float32)
    damage = np.where(keep, loss_half - loss1, 0.0).astype(np.float32)

    destination = Path(out_dir) / sequence
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "motion_rank_label.tmp.npy", label, allow_pickle=False)
    (destination / "motion_rank_label.tmp.npy").replace(destination / "motion_rank_label.npy")
    np.save(destination / "motion_rank_gain.tmp.npy", gain, allow_pickle=False)
    (destination / "motion_rank_gain.tmp.npy").replace(destination / "motion_rank_gain.npy")
    np.save(destination / "motion_rank_damage.tmp.npy", damage, allow_pickle=False)
    (destination / "motion_rank_damage.tmp.npy").replace(destination / "motion_rank_damage.npy")
    record = {
        "sequence": sequence, "rows": int(len(target)), "valid_rows": int(valid.sum()),
        "keep_rows": int(keep.sum()), "suppress_rows": int(suppress.sum()),
        "gain_mean": float(gain[suppress].mean()) if suppress.any() else None,
        "damage_mean": float(damage[keep].mean()) if keep.any() else None,
        "source_sha256": fingerprint, "target_sha256": metadata["target_sha256"],
        "label_sha256": hashlib.sha256(label.tobytes()).hexdigest(),
        "gain_sha256": hashlib.sha256(gain.tobytes()).hexdigest(),
        "damage_sha256": hashlib.sha256(damage.tobytes()).hexdigest(),
        "policy": {"keep": "g_star==1 and L(0.5)>L(1)+1e-6",
                   "suppress": "g_star<1 and L(1)>L_best+1e-6",
                   "tolerance": TOL, "dtype": "uint8/float32"},
    }
    temporary = destination / "metadata.tmp.json"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination / "metadata.json")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    target_root = Path(args.target_root)
    names = args.sequences or sorted(
        p.name for p in target_root.iterdir() if (p / "metadata.json").is_file()
    )
    records = []
    for name in names:
        if not (data_root / name / "samples.npz").is_file():
            continue
        records.append(generate(name, data_root / name, target_root / name, args.out))
    print(json.dumps({
        "sequences": len(records),
        "valid": sum(r["valid_rows"] for r in records),
        "keep": sum(r["keep_rows"] for r in records),
        "suppress": sum(r["suppress_rows"] for r in records),
    }), flush=True)


if __name__ == "__main__":
    main()
