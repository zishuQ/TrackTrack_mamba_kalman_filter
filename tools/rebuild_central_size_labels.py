"""Rebuild only the optional central labels of an existing paired replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from sportsmot_supervision_pilot import load_gt, load_warps


def rebuild(shard, output):
    with np.load(shard / "samples.npz", allow_pickle=False) as samples:
        keys = {k: samples[k] for k in ("frame", "track_id", "gt_id", "detection_index")}
        valid = samples["size_adj_valid"].copy()
        previous_mask = samples["size_central_valid"]
        previous = samples["gt_size_velocity_central"]
    info = json.loads((shard / "summary.json").read_text())
    gt = load_gt(shard.name, info["split"])
    warps = load_warps(shard.name)
    scale = np.array([info["img_width"], info["img_height"]])
    labels = np.zeros((len(valid), 2), np.float32)
    old_error = 0.0
    for t in np.unique(keys["frame"][valid]):
        rows = np.flatnonzero((keys["frame"] == t) & valid)
        warp1, warp2 = warps.get(t + 1), warps.get(t + 2)
        if warp1 is None or warp2 is None:
            valid[rows] = False
            continue
        a, b = warp1[:, :2], warp2[:, :2]
        if not np.isfinite(b).all() or np.linalg.cond(b) > 1e8:
            valid[rows] = False
            continue
        now, future = gt.get(t, {}), gt.get(t + 2, {})
        keep = [r for r in rows if keys["gt_id"][r] in now and keys["gt_id"][r] in future]
        valid[rows] = False
        if not keep:
            continue
        valid[keep] = True
        ids = keys["gt_id"][keep]
        box0 = np.stack([now[g] for g in ids])
        box2 = np.stack([future[g] for g in ids])
        size0 = (box0[:, 2:] - box0[:, :2]) @ a.T
        size2 = box2[:, 2:] - box2[:, :2]
        labels[keep] = (np.linalg.solve(b, size2.T).T - size0) / (2 * scale)
        expected_old = (size2 @ b - size0) / (2 * scale)
        old_error = max(old_error, float(np.max(np.abs(expected_old - previous[keep]))))
    if old_error > 1e-6 or np.any(valid & ~previous_mask):
        raise ValueError(f"{shard.name}: source shard does not reproduce its legacy central labels")
    np.savez_compressed(output / (shard.name + ".npz"), **keys,
                        gt_size_velocity_central=labels, size_central_valid=valid)
    return {"valid": int(valid.sum()), "rows": len(valid), "source_max_error": old_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {s.parent.name: rebuild(s.parent, args.output)
               for s in sorted(args.data_root.glob("*/samples.npz"))}
    if not results:
        raise ValueError("No source shards")
    (args.output / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    print("sequences", len(results), "rows", sum(r["rows"] for r in results.values()),
          "valid", sum(r["valid"] for r in results.values()))


if __name__ == "__main__":
    main()
