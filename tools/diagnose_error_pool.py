#!/usr/bin/env python3
"""Compare real detections with error-pool simulated observations.

For a paired sample of GT-anchored events: decode a sampled residual onto the
GT box, then run the same NSA-KF g0/g1 endpoint computation used by the replay
data.  Reports error distributions, harmful-event ratios and endpoint losses
side by side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentguard/src"))
from agentguard.motion.nsa_numpy import NSAKalmanFilter  # noqa: E402
from shutrack.training.motion_targets import numpy_diou  # noqa: E402


def warp_predict(mean, covariance, warp, kf):
    rot = np.asarray(warp, np.float64)[:, :2]
    rot8 = np.kron(np.eye(4, dtype=np.float64), rot)
    trans = np.asarray(warp, np.float64)[:, 2]
    moved = rot8 @ np.asarray(mean, np.float64)
    moved = moved.copy(); moved[:2] += trans
    moved_cov = rot8 @ np.asarray(covariance, np.float64) @ rot8.T
    return kf.predict(moved, moved_cov)


def box(mean):
    return np.array([mean[0]-mean[2]/2, mean[1]-mean[3]/2,
                     mean[0]+mean[2]/2, mean[1]+mean[3]/2])


def diou(a, b):
    """Use the shared definition, already tested against the training objective."""
    return float(numpy_diou(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    pool = np.load(Path(args.pool) / "error_pool.npz")["relative"]
    rng = np.random.RandomState(42)
    data_root = Path(args.data_root)
    kf = NSAKalmanFilter()
    sequences = sorted(p.name for p in data_root.iterdir()
                       if (p / "samples.npz").is_file())
    rows = []
    for sequence in sequences:
        d = data_root / sequence
        with np.load(d / "samples.npz", allow_pickle=False) as s:
            event = np.asarray(s["current_event_index"], np.int64)
            has = np.load(d / "event_has_detection.npy")[event]
            valid = (np.asarray(s["motion_gt_valid"], bool)
                     & np.load(d / "gt_valid_t0.npy") & has)
        measurement_all = np.load(d / "event_det_measurement.npy", mmap_mode="r")
        gt0_all = np.load(d / "gt_box_t0.npy", mmap_mode="r")
        det = np.asarray(measurement_all[event], np.float64)
        gt0 = np.asarray(gt0_all, np.float64)
        # Same reliability criterion as the pool: real detection vs GT IoU >= 0.5.
        d_x1 = det[:,0]-det[:,2]/2; d_y1 = det[:,1]-det[:,3]/2
        d_x2 = det[:,0]+det[:,2]/2; d_y2 = det[:,1]+det[:,3]/2
        g_x1 = gt0[:,0]-gt0[:,2]/2; g_y1 = gt0[:,1]-gt0[:,3]/2
        g_x2 = gt0[:,0]+gt0[:,2]/2; g_y2 = gt0[:,1]+gt0[:,3]/2
        iw = np.maximum(0.0, np.minimum(d_x2,g_x2)-np.maximum(d_x1,g_x1))
        ih = np.maximum(0.0, np.minimum(d_y2,g_y2)-np.maximum(d_y1,g_y1))
        inter = iw*ih
        area_d = np.maximum(0.0,d_x2-d_x1)*np.maximum(0.0,d_y2-d_y1)
        area_g = np.maximum(0.0,g_x2-g_x1)*np.maximum(0.0,g_y2-g_y1)
        union = area_d+area_g-inter
        iou = np.where(union>1e-9, inter/np.maximum(union,1e-9), 0.0)
        valid &= iou >= 0.5
        idx = np.nonzero(valid)[0]
        take = min(len(idx), max(1, args.samples // len(sequences)))
        rows.append((sequence, idx[rng.choice(len(idx), take, replace=False)]))
    real_rel, sim_rel = [], []
    real_raw, sim_raw = [], []
    real_l0, real_l1, sim_l0, sim_l1 = [], [], [], []
    real_harm = sim_harm = 0
    count = 0
    endpoint_err = 0.0
    for sequence, idx in rows:
        d = data_root / sequence
        with np.load(d / "samples.npz", allow_pickle=False) as s:
            event = np.asarray(s["current_event_index"], np.int64)[idx]
            gt1 = np.asarray(s["gt_box_t1"], np.float64)[idx]
            valid1 = np.asarray(s["motion_gt_valid"], bool)[idx]
            stored0 = np.asarray(s["pred_box_g0"], np.float64)[idx]
            stored1 = np.asarray(s["pred_box_g1"], np.float64)[idx]
        mean = np.asarray(np.load(d / "event_state_mean.npy", mmap_mode="r")[event],
                          np.float64)
        cov = np.asarray(np.load(d / "event_state_cov.npy", mmap_mode="r")[event],
                         np.float64).reshape(-1, 8, 8)
        measurement = np.asarray(np.load(d / "event_det_measurement.npy",
                                         mmap_mode="r")[event], np.float64)
        score = np.asarray(np.load(d / "event_det_score.npy", mmap_mode="r")[event],
                           np.float64)
        warp = np.asarray(np.load(d / "event_warp_next.npy", mmap_mode="r")[event],
                          np.float64).reshape(-1, 2, 3)
        gt0 = np.load(d / "gt_box_t0.npy", mmap_mode="r")[idx]
        picks = rng.randint(0, len(pool), size=len(idx))
        residual = pool[picks].astype(np.float64)
        sim_measurement = np.asarray(gt0, np.float64).copy()
        sim_measurement[:, 0] += residual[:, 0] * gt0[:, 2]
        sim_measurement[:, 1] += residual[:, 1] * gt0[:, 3]
        sim_measurement[:, 2] *= np.exp(residual[:, 2])
        sim_measurement[:, 3] *= np.exp(residual[:, 3])
        sim_measurement[:, 2:] = np.maximum(sim_measurement[:, 2:], 1.0)
        for i in range(len(idx)):
            full_m, full_c = kf.update(mean[i].copy(), cov[i].copy(),
                                       measurement[i].copy(), float(score[i]))
            m0, _ = warp_predict(mean[i], cov[i], warp[i], kf)
            m1, _ = warp_predict(full_m, full_c, warp[i], kf)
            b0, b1 = box(m0), box(m1)
            sim_full_m, sim_full_c = kf.update(mean[i].copy(), cov[i].copy(),
                                               sim_measurement[i].copy(),
                                               float(score[i]))
            s0, _ = warp_predict(mean[i], cov[i], warp[i], kf)
            s1, _ = warp_predict(sim_full_m, sim_full_c, warp[i], kf)
            sb0, sb1 = box(s0), box(s1)
            if i < 5:
                endpoint_err = max(endpoint_err,
                                   float(np.max(np.abs(b0-stored0[i]))),
                                   float(np.max(np.abs(b1-stored1[i]))))
            g = gt1[i]
            rl0, rl1 = 1-diou(b0, g), 1-diou(b1, g)
            sl0, sl1 = 1-diou(sb0, g), 1-diou(sb1, g)
            real_l0.append(rl0); real_l1.append(rl1)
            sim_l0.append(sl0); sim_l1.append(sl1)
            real_harm += int(valid1[i] and rl1 > rl0 + 1e-6)
            sim_harm += int(valid1[i] and sl1 > sl0 + 1e-6)
            # observation-vs-GT residuals in relative and pixel space
            for tag, meas in (("real", measurement[i]), ("sim", sim_measurement[i])):
                gcx, gcy, gw, gh = gt0[i]
                rel = [(meas[0]-gcx)/max(gw, 1e-9), (meas[1]-gcy)/max(gh, 1e-9),
                       np.log(max(meas[2], 1e-3)/max(gw, 1e-9)),
                       np.log(max(meas[3], 1e-3)/max(gh, 1e-9))]
                raw = [meas[0]-gcx, meas[1]-gcy, meas[2]-gw, meas[3]-gh]
                (real_rel if tag == "real" else sim_rel).append(rel)
                (real_raw if tag == "real" else sim_raw).append(raw)
            count += 1
    def stats(values):
        arr = np.asarray(values, np.float64)
        return {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                "p01": np.percentile(arr, 1, axis=0).tolist(),
                "p50": np.percentile(arr, 50, axis=0).tolist(),
                "p99": np.percentile(arr, 99, axis=0).tolist()}
    result = {
        "metric": "shared_training_diou",
        "prior_policy": "fixed_real_replay_state",
        "samples": count, "real_vs_stored_endpoint_max_abs": endpoint_err,
        "observation_error": {"real_relative": stats(real_rel),
                              "sim_relative": stats(sim_rel),
                              "real_pixels": stats(real_raw),
                              "sim_pixels": stats(sim_raw)},
        "harmful_ratio": {"real": real_harm/count, "sim": sim_harm/count},
        "endpoint_loss": {
            "real_g0": float(np.mean(real_l0)), "real_g1": float(np.mean(real_l1)),
            "sim_g0": float(np.mean(sim_l0)), "sim_g1": float(np.mean(sim_l1)),
            "real_delta_g1_minus_g0": float(np.mean(np.asarray(real_l1)-np.asarray(real_l0))),
            "sim_delta_g1_minus_g0": float(np.mean(np.asarray(sim_l1)-np.asarray(sim_l0))),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
