#!/usr/bin/env python3
"""Mechanism study: why does the motion gate saturate at 1 on SportsMOT?

Uses the stored schema-3 samples and event evidence only (no model runs).
For each dataset it reports:
  1. endpoint errors of the two t+1 motion hypotheses (g=0 prior vs g=1 updated),
  2. harmful-update rate and its dependence on detection score, observation
     interval, sample density and motion speed,
  3. whether the model's actual motion-gate inputs (13-d evidence + 3-d
     competition) can predict which endpoint is better, using sequence-held-out
     cross-validation (logistic regression and gradient boosting),
  4. the per-event oracle gain over the constant g=1 policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEQUENCES = {
    "SportsMOT": [
        "v_9MHDmAMxO5I_c003",
        "v_1yHWGw8DH4A_c077",
        "v_2j7kLB-vEEk_c001",
        "v_HdiyOtliFiw_c004",
        "v_gQNyhv8y0QY_c012",
    ],
    "MOT20": ["MOT20-02", "MOT20-03"],
    "MOT17": ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-11-FRCNN"],
}
ROOTS = {
    "MOT20": ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT20/data",
    "MOT17": ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT17/data",
    "SportsMOT": ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data",
}


def diou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None) * np.clip(
        np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None
    )
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    d2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    cw = np.maximum(ax2, bx2) - np.minimum(ax1, bx1)
    ch = np.maximum(ay2, by2) - np.minimum(ay1, by1)
    c2 = cw * cw + ch * ch
    return iou - np.where(c2 > 0, d2 / np.maximum(c2, 1e-12), 0.0)


GRID = np.arange(0, 1.0001, 0.1)


def load_dataset(dataset: str):
    rows = []
    for seq in SEQUENCES[dataset]:
        root = ROOTS[dataset] / seq
        z = np.load(root / "samples.npz", allow_pickle=False)
        ev = np.load(root / "event_evidence.npy", mmap_mode="r")
        valid = z["motion_gt_valid"].astype(bool)
        idx = z["current_event_index"]
        b0 = z["pred_box_g0"].astype(np.float64)
        b1 = z["pred_box_g1"].astype(np.float64)
        gt = z["gt_box_t1"].astype(np.float64)
        keep = valid
        seq_rows = {
            "seq": seq,
            "frame": z["frame"][keep].astype(np.int64),
            "evidence": np.asarray(ev[idx[keep]]),
            "competition": np.asarray(z["current_competition"])[keep].astype(np.float64),
            "b0": b0[keep],
            "b1": b1[keep],
            "gt": gt[keep],
            "velocity": z["gt_velocity4"][keep].astype(np.float64) if "gt_velocity4" in z else None,
            "speed_valid": z["speed_valid"][keep].astype(bool) if "speed_valid" in z else None,
        }
        rows.append(seq_rows)
    return rows


def frame_density(seq_rows) -> np.ndarray:
    frames = seq_rows["frame"]
    uniq, counts = np.unique(frames, return_counts=True)
    lookup = dict(zip(uniq.tolist(), counts.tolist()))
    return np.array([lookup[f] for f in frames.tolist()], dtype=np.float64)


def quantile_bins(values: np.ndarray, n: int = 5):
    edges = np.quantile(values, np.linspace(0, 1, n + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return [(None, None, len(values))]
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for k in range(len(edges) - 1):
        mask = idx == k
        out.append((float(edges[k]), float(edges[k + 1]), mask))
    return out


def analyse(dataset: str) -> dict:
    seqs = load_dataset(dataset)
    feats, comps, harmful, e0s, e1s, speeds, dens, scores, intervals, groups = [], [], [], [], [], [], [], [], [], []
    for s in seqs:
        d0, d1 = diou(s["b0"], s["gt"]), diou(s["b1"], s["gt"])
        e0, e1 = 1 - d0, 1 - d1
        harmful.append(e1 > e0 + 1e-9)
        e0s.append(e0)
        e1s.append(e1)
        feats.append(s["evidence"])
        comps.append(s["competition"])
        speed = np.full(len(e0), np.nan)
        if s["velocity"] is not None and s["speed_valid"] is not None:
            v = s["velocity"]
            speed[s["speed_valid"]] = np.sqrt(v[s["speed_valid"], 0] ** 2 + v[s["speed_valid"], 1] ** 2)
        speeds.append(speed)
        dens.append(frame_density(s))
        scores.append(s["evidence"][:, 10].astype(np.float64))
        intervals.append(s["evidence"][:, 11].astype(np.float64))
        groups.append(np.full(len(e0), s["seq"], dtype=object))
    harm = np.concatenate(harmful)
    e0 = np.concatenate(e0s)
    e1 = np.concatenate(e1s)
    X = np.concatenate(feats).astype(np.float64)
    C = np.concatenate(comps)
    speed = np.concatenate(speeds)
    dens = np.concatenate(dens)
    score = np.concatenate(scores)
    interval = np.concatenate(intervals)
    group = np.concatenate(groups)

    result = {"dataset": dataset, "n": int(len(harm)), "harmful_rate": float(harm.mean())}
    result["diou_g0"] = float((1 - e0).mean())
    result["diou_g1"] = float((1 - e1).mean())
    result["speed_median"] = float(np.nanmedian(speed))
    result["density_median"] = float(np.median(dens))
    result["score_median"] = float(np.median(score))
    result["interval_median"] = float(np.median(interval))

    def rates(values, valid=None):
        sel = np.isfinite(values)
        if valid is not None:
            sel &= valid
        if sel.sum() < 10:
            return []
        edges = np.unique(np.quantile(values[sel], np.linspace(0, 1, 6)))
        if edges.size < 3:
            return [{"lo": None, "hi": None, "n": int(sel.sum()), "harmful": round(float(harm[sel].mean()), 4)}]
        idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
        out = []
        for k in range(len(edges) - 1):
            m = (idx == k) & sel
            if m.sum() == 0:
                continue
            out.append({"lo": round(float(edges[k]), 4), "hi": round(float(edges[k + 1]), 4),
                        "n": int(m.sum()), "harmful": round(float(harm[m].mean()), 4)})
        return out

    result["harmful_by_score"] = rates(score)
    result["harmful_by_interval"] = rates(interval)
    result["harmful_by_density"] = rates(dens)
    result["harmful_by_speed"] = rates(speed)

    # oracle per-event gate vs constant g=1
    loss1 = float(e1.mean())
    best = np.full(len(e1), np.inf)
    for g in GRID:
        dg = diou(np.concatenate([s["b0"] + g * (s["b1"] - s["b0"]) for s in seqs]), np.concatenate([s["gt"] for s in seqs]))
        loss = 1 - dg
        best = np.minimum(best, loss)
    result["loss_const_g1"] = loss1
    result["loss_oracle_per_event"] = float(best.mean())
    result["oracle_gain_pct"] = float((loss1 - best.mean()) / loss1 * 100)

    # ---- probe: can the model's own motion-gate inputs predict harmful? ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.concatenate([X, C], axis=1)
    rng = np.random.default_rng(0)
    cap = 40000
    if len(features) > cap:
        take = rng.choice(len(features), cap, replace=False)
        features, harm_s, group_s = features[take], harm[take], group[take]
    else:
        harm_s, group_s = harm, group
    probe = {"majority_baseline": float(max(harm_s.mean(), 1 - harm_s.mean()))}
    for name, clf in [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("gbm", HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)),
    ]:
        aucs = []
        for held in sorted(set(group_s.tolist())):
            test = group_s == held
            if test.all() or (~test).sum() < 100:
                continue
            clf.fit(features[~test], harm_s[~test])
            score = clf.predict_proba(features[test])[:, 1]
            if len(np.unique(harm_s[test])) > 1:
                aucs.append(float(roc_auc_score(harm_s[test], score)))
        probe[name + "_auc_by_heldout_seq"] = [round(a, 3) for a in aucs]
        # in-sample AUC for reference
        clf.fit(features, harm_s)
        probe[name + "_auc_insample"] = round(float(roc_auc_score(harm_s, clf.predict_proba(features)[:, 1])), 3)
    result["probe"] = probe

    # single-feature AUCs
    single = {}
    names = [f"ev{i}" for i in range(13)] + [f"comp{i}" for i in range(3)]
    for k, name in enumerate(names):
        try:
            single[name] = round(float(roc_auc_score(harm_s, features[:, k])), 3)
        except ValueError:
            single[name] = None
    result["single_feature_auc"] = single
    return result


def main() -> None:
    out = {}
    for dataset in ("SportsMOT", "MOT17", "MOT20"):
        out[dataset] = analyse(dataset)
        print(json.dumps(out[dataset], indent=1, ensure_ascii=False), flush=True)
    dest = ROOT / "outputs/agentguard/experiments/chatgpt_mot_review_20261121/motion_gate_mechanism.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("saved", dest)


if __name__ == "__main__":
    sys.exit(main())
