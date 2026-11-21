"""Screen shared output gaps in two existing tracker result sets.

This is a diagnostic, not a GT-based failure certification. It reads only
files whose resolved paths remain inside this workspace, uses one CPU thread,
and does not run tracking, training, detectors, or evaluation on ground truth.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from pathlib import Path
import json
import hashlib
from collections import defaultdict
import numpy as np

def safe(root, rel):
    p = (root / rel).resolve()
    p.relative_to(root.resolve())
    return p

def load_rows(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    a = a[np.all(np.isfinite(a[:, :6]), axis=1)]
    a = a[(a[:, 4] > 0) & (a[:, 5] > 0)]
    by_frame, by_id = defaultdict(dict), defaultdict(dict)
    for r in a:
        f, tid = int(r[0]), int(r[1])
        b = r[2:6].astype(float)
        by_frame[f][tid] = b
        by_id[tid][f] = b
    return by_frame, by_id

def iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[:2] + a[2:], b[:2] + b[2:])
    inter = float(np.maximum(hi - lo, 0).prod())
    return inter / max(float(a[2:].prod() + b[2:].prod()) - inter, 1e-9)

def center(b):
    return b[:2] + b[2:] / 2

def velocity(rows, frames):
    f = np.asarray(frames, dtype=float)
    c = np.stack([center(rows[int(x)]) for x in f])
    z = f - f.mean()
    return (z[:, None] * (c - c.mean(0))).sum(0) / max(float((z*z).sum()), 1e-9)

def consecutive_runs(frames):
    if not frames:
        return []
    groups, current = [], [frames[0]]
    for f in frames[1:]:
        if f == current[-1] + 1:
            current.append(f)
        else:
            groups.append(current)
            current = [f]
    groups.append(current)
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="outputs/3. track/sportsmot_val_0.80_kf")
    ap.add_argument("--improved", default="outputs/3. track/sportsmot_val_0.80_mamba")
    ap.add_argument("--out-dir", default="outputs/revision_failure_search_common")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    kd, md = safe(root, args.baseline), safe(root, args.improved)
    out = safe(root, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates, counts, sources = [], [], []
    for kp in sorted(kd.glob("*.txt")):
        kp.resolve().relative_to(root)
        mp = md / kp.name
        if not mp.is_file():
            continue
        mp.resolve().relative_to(root)
        kframes, ktracks = load_rows(kp)
        mframes, mtracks = load_rows(mp)
        n_before = len(candidates)
        sources.append({
            "sequence": kp.stem,
            "kf_sha256": hashlib.sha256(kp.read_bytes()).hexdigest(),
            "mtu_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
        })
        for kid, kr in ktracks.items():
            fs = sorted(kr)
            for left, right in zip(fs, fs[1:]):
                if not 3 <= right - left - 1 <= 40:
                    continue
                before = list(range(left - 4, left + 1))
                after = list(range(right, right + 5))
                if not all(f in kr for f in before + after):
                    continue
                # Same MTU ID must spatially agree with the KF track at both ends.
                ml, mr = mframes.get(left, {}), mframes.get(right, {})
                mids = [mid for mid in ml.keys() & mr.keys()
                        if iou(kr[left], ml[mid]) >= 0.5 and iou(kr[right], mr[mid]) >= 0.5]
                for mid in mids:
                    tr = mtracks[mid]
                    if not all(f in tr for f in before + after):
                        continue
                    support = np.mean([iou(kr[f], tr[f]) for f in before + after])
                    if support < 0.6:
                        continue
                    shared_absent = [f for f in range(left+1, right) if f not in tr]
                    for run in consecutive_runs(shared_absent):
                        if len(run) < 3:
                            continue
                        v0, v1 = velocity(kr, before), velocity(kr, after)
                        p0, p1 = center(kr[left]), center(kr[right])
                        dt = right - left
                        h = max((kr[left][3] + kr[right][3]) / 2, 1)
                        deviation = float(np.linalg.norm(p1 - (p0 + dt * v0)) / h)
                        speed0, speed1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
                        angle = None
                        if min(speed0, speed1) > 0.5:
                            angle = float(np.degrees(np.arccos(np.clip(
                                float(v0 @ v1) / (speed0 * speed1), -1, 1))))
                        overlap = max((iou(kr[left], b) for tid, b in kframes[left].items()
                                       if tid != kid), default=0)
                        # Rank only; neither GT visibility nor true target motion is known.
                        score = min(deviation, 3) + min(len(run), 15) / 30
                        if angle is not None:
                            score += min(angle, 120) / 120
                        score -= overlap
                        candidates.append({
                            "sequence": kp.stem,
                            "kf_id": kid, "mtu_id": mid,
                            "last_reported_before": left,
                            "first_reported_after": right,
                            "shared_absence_start": run[0],
                            "shared_absence_end": run[-1],
                            "shared_absence_length": len(run),
                            "suggested_frames": [left-2, run[0], run[len(run)//2], right],
                            "suggested_view_range": [max(1,left-15), right+15],
                            "endpoint_pair_mean_iou": round(float(support), 4),
                            "endpoint_cv_deviation_in_box_heights": round(deviation, 4),
                            "estimated_heading_change_degrees": None if angle is None else round(angle, 2),
                            "before_overlap_with_other_kf_boxes": round(float(overlap), 4),
                            "rank_score": round(score, 4),
                            "status": "candidate_only_no_GT_or_image_confirmation",
                        })
        counts.append({"sequence": kp.stem, "candidate_count": len(candidates)-n_before})
        print("screened", kp.stem, "candidates", len(candidates)-n_before, flush=True)
    candidates.sort(key=lambda c: c["rank_score"], reverse=True)
    payload = {
        "method": "Same paired IDs agree spatially in 5 frames before and 5 frames after a shared reporting gap.",
        "limitation": "No ground truth or images accessed. Gaps may reflect occlusion, exits, cuts, filtering, or errors. Motion statistics use reported tracker boxes, not GT or latent predictions.",
        "baseline": str(kd), "improved": str(md),
        "sequences": len(counts), "candidate_count": len(candidates),
        "candidates": candidates, "per_sequence": counts, "sources": sources,
    }
    (out / "shared_gap_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({"sequences":len(counts), "candidates":len(candidates),
                                "top":candidates[:12], "output":str(out)}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
