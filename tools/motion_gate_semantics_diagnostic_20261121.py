#!/usr/bin/env python3
"""Motion-gate semantics diagnostic: prediction vs detection error, and the
position/velocity decomposition of the two next-frame endpoints.

Read-only diagnostics. Replays a handful of clips with the existing tracker,
detection cache and GMC warps; writes nothing into the data directories.

For every matched event it records:
  - pre-update KF prior box and the matched detection box (current frame),
  - the stored g=0/g=1 next-frame endpoints and their GT,
  - four local counterfactuals from the same prior state: A no update,
    B position/size-only, C velocity-only, D full update,
  - GT motion speed / acceleration / turn, and whether the matched detection
    belongs to the track's resolved GT identity.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
for path in (str(WORKTREE / "tools"), str(WORKTREE), str(WORKTREE / "agentguard/src"), str(WORKTREE / "3. Tracker")):
    sys.path.insert(0, path)

from sportsmot_supervision_pilot import (  # noqa: E402
    DEFAULT_CACHE,
    SequencePilot,
    kf_update_predict,
    mean_to_box,
    warp_predict,
)
from paired_training_input import full_write_checkpoint, replay_split  # noqa: E402
from agentguard.data.detection_cache import SequenceDetectionCache  # noqa: E402

OUT = WORKTREE / "outputs/agentguard/experiments/chatgpt_mot_review_20261121"

CLIPS = [
    # (dataset, sequence, max_frames, sport)
    ("SportsMOT", "v_-6Os86HzwCs_c003", 0, "basketball"),
    ("SportsMOT", "v_-6Os86HzwCs_c007", 0, "basketball"),
    ("SportsMOT", "v_1yHWGw8DH4A_c047", 0, "football"),
    ("SportsMOT", "v_1yHWGw8DH4A_c601", 0, "football"),
    ("SportsMOT", "v_0kUtTtmLaJA_c005", 0, "volleyball"),
    ("SportsMOT", "v_0kUtTtmLaJA_c006", 0, "volleyball"),
    ("MOT20", "MOT20-02", 350, "control"),
    ("MOT20", "MOT20-03", 350, "control"),
]


def diou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
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


def box_diou(box: np.ndarray, gt: np.ndarray) -> float:
    return float(diou(np.asarray(box)[None], np.asarray(gt)[None])[0])


def predict_box(mean: np.ndarray, cov: np.ndarray, warp: np.ndarray, kf) -> np.ndarray:
    pred_mean, _ = warp_predict(mean, cov, warp, kf)
    return mean_to_box(pred_mean)


def gt_motion(gt, warps, frame_id: int, target: int, img_w: int, img_h: int):
    gt_now = gt.get(frame_id, {}).get(target)
    gt_t1 = gt.get(frame_id + 1, {}).get(target)
    gt_prev = gt.get(frame_id - 1, {}).get(target)
    warp_t = warps.get(frame_id)
    warp_t1 = warps.get(frame_id + 1)
    if gt_now is None or gt_t1 is None or warp_t1 is None:
        return None
    c_now = np.array([(gt_now[0] + gt_now[2]) / 2, (gt_now[1] + gt_now[3]) / 2])
    c_t1 = np.array([(gt_t1[0] + gt_t1[2]) / 2, (gt_t1[1] + gt_t1[3]) / 2])
    rot = np.asarray(warp_t1)[:, :2]
    trans = np.asarray(warp_t1)[:, 2]
    v = (c_t1 - (rot @ c_now + trans)) / np.array([img_w, img_h], dtype=np.float64)
    out = {"speed": float(np.linalg.norm(v)), "accel": None, "turn_deg": None}
    if gt_prev is not None and warp_t is not None:
        c_prev = np.array([(gt_prev[0] + gt_prev[2]) / 2, (gt_prev[1] + gt_prev[3]) / 2])
        rot_t = np.asarray(warp_t)[:, :2]
        trans_t = np.asarray(warp_t)[:, 2]
        v_prev = (c_now - (rot_t @ c_prev + trans_t)) / np.array([img_w, img_h], dtype=np.float64)
        out["accel"] = float(np.linalg.norm(v - v_prev))
        denom = float(np.linalg.norm(v) * np.linalg.norm(v_prev))
        if denom > 1e-12:
            cos = float(np.clip(np.dot(v, v_prev) / denom, -1.0, 1.0))
            out["turn_deg"] = float(np.degrees(np.arccos(cos)))
    return out


def replay_clip(dataset: str, sequence: str, max_frames: int, checkpoint: Path, workdir: Path):
    split = replay_split(sequence, dataset)
    pilot = SequencePilot(sequence, split, checkpoint, workdir, max_frames=max_frames, dataset=dataset)
    pilot.protocol = {
        "schema": 3, "full_write": True, "anchor": False, "column": False,
        "gap3": False, "rank_arrays": False, "teacher_sha256": None,
    }
    pilot.lean = False
    pilot.store_rank_features = False
    started = time.time()
    try:
        pilot.run_replay()
        rows = []
        excluded = {"no_target": 0, "no_prior": 0, "no_warp_next": 0, "no_gt_t1": 0, "nonfinite": 0}
        reproduce = {"g0": 0.0, "g1": 0.0}
        for rec in pilot.events:
            frame_id = int(rec["frame_t"])
            target = rec.get("target_gt_before")
            if target is None:
                excluded["no_target"] += 1
                continue
            mean_prior = rec.get("mean_prior")
            cov_prior = rec.get("cov_prior")
            if mean_prior is None or cov_prior is None:
                excluded["no_prior"] += 1
                continue
            warp_next = pilot.warps.get(frame_id + 1)
            if warp_next is None:
                excluded["no_warp_next"] += 1
                continue
            gt_t1 = pilot.gt.get(frame_id + 1, {}).get(int(target))
            if gt_t1 is None:
                excluded["no_gt_t1"] += 1
                continue
            mean_prior = np.asarray(mean_prior, dtype=np.float64)
            cov_prior = np.asarray(cov_prior, dtype=np.float64)
            measurement = np.asarray(rec["detection_measurement"], dtype=np.float64)
            score = float(rec["detection_score"])
            mean_full, cov_full = kf_update_predict(mean_prior, cov_prior, measurement, score, warp_next, pilot.kf)

            mean_b = mean_full.copy(); mean_b[4:8] = mean_prior[4:8]
            mean_c = mean_prior.copy(); mean_c[4:8] = mean_full[4:8]
            states = {"A": (mean_prior, cov_full), "B": (mean_b, cov_full),
                      "C": (mean_c, cov_full), "D": (mean_full, cov_full)}
            boxes_t1 = {k: predict_box(m, c, warp_next, pilot.kf) for k, (m, c) in states.items()}
            boxes_now = {k: mean_to_box(m) for k, (m, _) in states.items()}

            if not all(np.isfinite(boxes_t1[k]).all() for k in boxes_t1):
                excluded["nonfinite"] += 1
                continue

            stored0 = rec.get("pred_box_g0")
            stored1 = rec.get("pred_box_g1")
            if stored0 is not None:
                reproduce["g0"] = max(reproduce["g0"], float(np.max(np.abs(np.asarray(stored0) - boxes_t1["A"]))))
            if stored1 is not None:
                reproduce["g1"] = max(reproduce["g1"], float(np.max(np.abs(np.asarray(stored1) - boxes_t1["D"]))))

            gt_now = pilot.gt.get(frame_id, {}).get(int(target))
            det_box = np.asarray(rec["detection_box"], dtype=np.float64)
            det_gt_id = int(rec.get("detection_gt_id", -1))
            motion = gt_motion(pilot.gt, pilot.warps, frame_id, int(target), pilot.img_w, pilot.img_h)

            row = {
                "clip": sequence, "frame": frame_id, "track_id": int(rec["track_id"]),
                "det_gt_id": det_gt_id, "target_gt_id": int(target),
                "mismatch": det_gt_id != int(target),
                "unlabeled_detection": det_gt_id < 0,
                "diou_t1": {k: box_diou(boxes_t1[k], gt_t1) for k in boxes_t1},
                "diou_now": ({k: box_diou(boxes_now[k], gt_now) for k in boxes_now} if gt_now is not None else None),
                "det_diou_now": (box_diou(det_box, gt_now) if gt_now is not None else None),
                "speed": None if motion is None else motion["speed"],
                "accel": None if motion is None else motion["accel"],
                "turn_deg": None if motion is None else motion["turn_deg"],
                "det_score": score,
            }
            rows.append(row)
        duration = time.time() - started
    finally:
        try:
            pilot.close()
        except Exception:
            pass
    return rows, excluded, reproduce, duration, pilot.frame_count


def representatives(rows, k: int = 3):
    def entry(r):
        d = r["diou_t1"]
        return {
            "clip": r["clip"], "frame": r["frame"], "track_id": r["track_id"],
            "det_gt_id": r["det_gt_id"], "target_gt_id": r["target_gt_id"],
            "mismatch": r["mismatch"],
            "diou_t1": {key: round(d[key], 4) for key in "ABCD"},
            "pos_gain": round(d["B"] - d["A"], 4),
            "vel_gain": round(d["C"] - d["A"], 4),
            "full_gain": round(d["D"] - d["A"], 4),
            "speed": None if r["speed"] is None else round(r["speed"], 5),
            "turn_deg": None if r["turn_deg"] is None else round(r["turn_deg"], 1),
            "det_score": round(r["det_score"], 3),
        }

    pos_help_vel_harm = [r for r in rows if r["diou_t1"]["B"] > r["diou_t1"]["A"] and r["diou_t1"]["C"] < r["diou_t1"]["A"]]
    pos_help_vel_harm.sort(key=lambda r: (r["diou_t1"]["B"] - r["diou_t1"]["A"]) - (r["diou_t1"]["C"] - r["diou_t1"]["A"]))
    vel_help_pos_harm = [r for r in rows if r["diou_t1"]["C"] > r["diou_t1"]["A"] and r["diou_t1"]["B"] < r["diou_t1"]["A"]]
    vel_help_pos_harm.sort(key=lambda r: (r["diou_t1"]["C"] - r["diou_t1"]["A"]) - (r["diou_t1"]["B"] - r["diou_t1"]["A"]))
    harmful = [r for r in rows if r["diou_t1"]["D"] < r["diou_t1"]["A"]]
    harmful.sort(key=lambda r: r["diou_t1"]["D"] - r["diou_t1"]["A"])
    mismatch_harmful = [r for r in harmful if r["mismatch"]]

    def take(items):
        seen = set()
        out = []
        for r in items:
            key = (r["frame"], r["track_id"])
            if key in seen:
                continue
            seen.add(key)
            out.append(entry(r))
            if len(out) >= k:
                break
        return out

    return {
        "pos_help_vel_harm": take(pos_help_vel_harm),
        "vel_help_pos_harm": take(vel_help_pos_harm),
        "most_harmful": take(harmful),
        "mismatch_harmful": take(mismatch_harmful),
    }


def summarize(rows, clip, sport, dataset):
    n = len(rows)
    if n == 0:
        return {"clip": clip, "sport": sport, "n": 0}
    d = {k: np.array([r["diou_t1"][k] for r in rows]) for k in "ABCD"}
    a = d["A"]
    full_gain = d["D"] - a
    pos_gain = d["B"] - a
    vel_gain = d["C"] - a
    harmful = d["D"] < d["A"]
    mismatch = np.array([r["mismatch"] for r in rows])
    unlabeled = np.array([r["unlabeled_detection"] for r in rows])
    has_now = np.array([r["diou_now"] is not None for r in rows])
    prior_err = np.array([1 - r["diou_now"]["A"] if r["diou_now"] is not None else np.nan for r in rows])
    det_err = np.array([1 - r["det_diou_now"] if r["det_diou_now"] is not None else np.nan for r in rows])
    speed = np.array([r["speed"] if r["speed"] is not None else np.nan for r in rows])
    accel = np.array([r["accel"] if r["accel"] is not None else np.nan for r in rows])
    turn = np.array([r["turn_deg"] if r["turn_deg"] is not None else np.nan for r in rows])

    def rate(mask):
        return round(float(mask.mean()), 4) if len(mask) else None

    out = {
        "clip": clip, "sport": sport, "dataset": dataset, "n": n,
        "harmful_rate": rate(harmful),
        "full_gain_mean": round(float(full_gain.mean()), 5),
        "pos_gain_mean": round(float(pos_gain.mean()), 5),
        "vel_gain_mean": round(float(vel_gain.mean()), 5),
        "pos_help_rate": rate(pos_gain > 0),
        "vel_help_rate": rate(vel_gain > 0),
        "pos_help_vel_harm_rate": rate((pos_gain > 0) & (vel_gain < 0)),
        "vel_help_pos_harm_rate": rate((vel_gain > 0) & (pos_gain < 0)),
        "pos_help_vel_harm_gain_sum": round(float(pos_gain[(pos_gain > 0) & (vel_gain < 0)].sum()), 4),
        "pos_help_vel_harm_loss_sum": round(float(vel_gain[(pos_gain > 0) & (vel_gain < 0)].sum()), 4),
        "vel_help_pos_harm_gain_sum": round(float(vel_gain[(vel_gain > 0) & (pos_gain < 0)].sum()), 4),
        "vel_help_pos_harm_loss_sum": round(float(pos_gain[(vel_gain > 0) & (pos_gain < 0)].sum()), 4),
        "oracle4_vs_D_gain": round(float((np.maximum.reduce([d["A"], d["B"], d["C"], d["D"]]) - d["D"]).mean()), 5),
        "prior_err_mean": round(float(np.nanmean(prior_err)), 5) if has_now.any() else None,
        "det_err_mean": round(float(np.nanmean(det_err)), 5) if has_now.any() else None,
        "prior_minus_det_err_mean": round(float(np.nanmean(det_err - prior_err)), 5) if has_now.any() else None,
        "mismatch_rate": rate(mismatch),
        "unlabeled_det_rate": rate(unlabeled),
        "mismatch_harmful_rate": rate(harmful[mismatch]) if mismatch.any() else None,
        "clean_harmful_rate": rate(harmful[~mismatch]) if (~mismatch).any() else None,
        "mismatch_full_gain_mean": round(float(full_gain[mismatch].mean()), 5) if mismatch.any() else None,
        "clean_full_gain_mean": round(float(full_gain[~mismatch].mean()), 5) if (~mismatch).any() else None,
        "speed_median": round(float(np.nanmedian(speed)), 6) if np.isfinite(speed).any() else None,
        "accel_median": round(float(np.nanmedian(accel)), 6) if np.isfinite(accel).any() else None,
        "turn_median": round(float(np.nanmedian(turn)), 3) if np.isfinite(turn).any() else None,
    }
    # conditional: full gain by prior/detection error quantiles
    if has_now.any():
        m = has_now & np.isfinite(prior_err) & np.isfinite(det_err)
        if m.sum() >= 50:
            pe = prior_err[m]; de = det_err[m]; fg = full_gain[m]
            pe_edges = np.unique(np.quantile(pe, [0, .2, .4, .6, .8, 1]))
            de_edges = np.unique(np.quantile(de, [0, .2, .4, .6, .8, 1]))
            table = []
            for i in range(len(pe_edges) - 1):
                for j in range(len(de_edges) - 1):
                    mm = (pe >= pe_edges[i]) & (pe <= pe_edges[i + 1]) & (de >= de_edges[j]) & (de <= de_edges[j + 1])
                    if mm.sum() >= 10:
                        table.append({
                            "prior_err": [round(float(pe_edges[i]), 4), round(float(pe_edges[i + 1]), 4)],
                            "det_err": [round(float(de_edges[j]), 4), round(float(de_edges[j + 1]), 4)],
                            "n": int(mm.sum()),
                            "full_gain": round(float(fg[mm].mean()), 5),
                            "harmful": round(float((fg[mm] < 0).mean()), 4),
                        })
            out["gain_by_prior_det_err"] = table
    # conditional by speed/accel/turn quartiles
    for name, arr in (("speed", speed), ("accel", accel), ("turn", turn)):
        sel = np.isfinite(arr)
        if sel.sum() < 50:
            continue
        edges = np.unique(np.quantile(arr[sel], [0, .25, .5, .75, 1]))
        bins = []
        for i in range(len(edges) - 1):
            mm = sel & (arr >= edges[i]) & (arr <= edges[i + 1])
            if mm.sum() >= 10:
                bins.append({
                    "range": [round(float(edges[i]), 5), round(float(edges[i + 1]), 5)],
                    "n": int(mm.sum()),
                    "full_gain": round(float(full_gain[mm].mean()), 5),
                    "harmful": round(float(harmful[mm].mean()), 4),
                    "pos_help": round(float((pos_gain[mm] > 0).mean()), 4),
                    "vel_help": round(float((vel_gain[mm] > 0).mean()), 4),
                })
        out["by_" + name] = bins
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT / "motion_gate_semantics.json"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        tempdir = Path(temp)
        checkpoint = tempdir / "full_write.pt"
        first = CLIPS[0]
        split = replay_split(first[1], first[0])
        cache = SequenceDetectionCache(DEFAULT_CACHE / first[0] / split / first[1])
        try:
            full_write_checkpoint(checkpoint, cache.reid_dim)
        finally:
            cache.close()
        for dataset, sequence, max_frames, sport in CLIPS:
            print(f"[diagnostic] {dataset}/{sequence} max_frames={max_frames or 'all'}", flush=True)
            rows, excluded, reproduce, duration, frames = replay_clip(
                dataset, sequence, max_frames, checkpoint, tempdir
            )
            summary = summarize(rows, sequence, sport, dataset)
            summary["representatives"] = representatives(rows)
            summary.update({
                "frames_replayed": frames, "seconds": round(duration, 1),
                "excluded": excluded, "reproduce_endpoint_max_abs_err": {
                    "g0_vs_stored": reproduce["g0"], "g1_vs_stored": reproduce["g1"]},
            })
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False)[:1000], flush=True)
    dest = Path(args.out)
    dest.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print("saved", dest)


if __name__ == "__main__":
    main()
