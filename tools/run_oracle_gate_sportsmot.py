#!/usr/bin/env python3
"""SportsMOT val motion-gate diagnostics (inference only).

A: BCE-gate checkpoint, model appearance gate kept, motion gate forced to 1.0.
B: original new-architecture baseline checkpoint, model appearance gate kept,
   motion gate chosen online per matched update by GT next-frame DIoU over
   g in {1.0, 0.95, ..., 0.0}, using the label generator's tie rule
   (largest gate within 1e-6 of the minimum). Candidate boxes use state
   copies; only the chosen gate is executed; states are recomputed along the
   trajectory (no cached gates).

Both run the standard run.py val flow with the same detection cache, seed,
thresholds and raw evaluation as the baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "tools"), str(ROOT), str(ROOT / "agentguard/src"), str(ROOT / "3. Tracker")):
    sys.path.insert(0, path)

PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
BASE = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT"
CACHE = ROOT / "outputs/agentguard/detection_cache"
A_CKPT = EXP / "groups/appearance_row_column/checkpoints/last.pt"
B_CKPT = BASE / "ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"
BASELINE_SUMMARY = BASE / "ablation/SportsMOT/evals/new/summary.json"
BCE_SUMMARY = EXP / "eval_val/summary.json"
DEFAULT_OUT = EXP / "oracle_gate_diagnostics_20261121"
VAL_SEQS = tuple(
    line.strip() for line in (TRACKER / "trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)


def _base_argv(checkpoint: Path, out: Path, suffix: str, seqs, max_frames: int):
    argv = [
        "--dataset", "SportsMOT", "--mode", "val", "--sequences", *seqs,
        "--seed", "10000", "--kf-type", "nsa", "--agentguard-mode", "shutrack-v2",
        "--shutrack-checkpoint", str(checkpoint), "--agentguard-device", "cpu",
        "--detection-cache-root", str(CACHE),
        "--output_dir", str(out), "--tracker-suffix", suffix,
        "--agentguard-stats-output", str(out / "gates.jsonl"),
        "--print-per-sequence-metrics", "--eval-json", str(out / "summary.json"),
        "--torch-num-threads", "4", "--torch-num-interop-threads", "1",
    ]
    if max_frames > 0:
        argv += ["--max-frames", str(max_frames)]
    return argv


def run_a(seqs, out: Path, max_frames: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    command = [str(PY), "-u", "run.py", *_base_argv(A_CKPT, out, "motion1", seqs, max_frames),
               "--agentguard-disable-kf-gate"]
    log = out / "run.log"
    started = time.monotonic()
    with log.open("wb") as handle:
        code = subprocess.call(command, cwd=str(TRACKER), stdout=handle, stderr=subprocess.STDOUT)
    return {"group": "A_force_motion1", "seconds": round(time.monotonic() - started, 2),
            "exit_code": code, "log": str(log), "summary": str(out / "summary.json")}


def run_b(seqs, out: Path, max_frames: int, *, event_log: bool, verify: bool) -> dict:
    import run as run_module
    from trackers.track import Track
    from trackers import tracker as tracker_module
    from trackers.kalman_filter import create_kalman_filter
    from agentguard.data.detection_cache import SequenceDetectionCache
    from sportsmot_supervision_pilot import (
        SequencePilot, kf_update_predict, mean_to_box, warp_predict,
    )
    from shutrack.training.motion_targets import GRID, TIE_TOLERANCE, numpy_diou

    out.mkdir(parents=True, exist_ok=True)
    argv = _base_argv(B_CKPT, out, "gtgate", seqs, max_frames)
    args = run_module.make_parser().parse_args(argv)
    run_module.args = args

    state = {
        "pilot": None, "records": [], "event_file": None,
        "event_log": event_log, "verify": verify,
        "stats": {}, "seq_stats": {}, "max_state_err": 0.0, "argmin_mismatch": 0,
    }
    if event_log:
        state["event_file"] = (out / "events.jsonl").open("w")

    def close_pilot():
        pilot = state.get("pilot")
        if pilot is not None and getattr(pilot, "det_cache", None) is not None:
            try:
                pilot.det_cache.close()
            except Exception:
                pass
        state["pilot"] = None

    def make_pilot(seq: str):
        pilot = SequencePilot(seq, "val", B_CKPT, out / "pilot_scratch" / seq, dataset="SportsMOT")
        pilot.lean = False
        pilot.store_rank_features = False
        pilot.img_w = int(run_module.args.img_w)
        pilot.img_h = int(run_module.args.img_h)
        pilot.det_thr = float(getattr(run_module.args, "det_thr", 0.55))
        pilot.kf = create_kalman_filter("nsa")
        pilot.det_cache = SequenceDetectionCache(
            run_module._sequence_detection_cache_dir(run_module.args, seq),
            frame_array_cache_size=4,
        )
        pilot.frame_matches_cache = {}
        return pilot

    def bucket(seq):
        return state["seq_stats"].setdefault(seq, {
            "events": 0, "valid": 0, "reasons": {}, "gated_path": 0, "ungated_path": 0,
            "chosen_ones": 0, "chosen_below_one": 0, "chosen_hist": {},
            "model_motion_sum": 0.0, "model_motion_count": 0,
            "appearance_sum": 0.0, "appearance_count": 0,
            "loss_full_sum": 0.0, "loss_chosen_sum": 0.0, "loss_min_sum": 0.0,
            "loss_count": 0, "model_chosen_diff_gt_005": 0,
        })

    def compute(track, frame_id, detection, model_motion):
        pilot = state["pilot"]
        if pilot is None:
            return 1.0, {"valid": False, "reason": "no_pilot"}, None, None
        pilot.frame_matches(frame_id - 1)
        det_idx = int(getattr(detection, "frame_detection_index", -1))
        det_gt_id, det_iou = pilot.det_gt.get((frame_id, det_idx), (-1, 0.0))
        target = pilot.vote.resolve_before_current(pilot.sequence, track.track_id)
        pilot.vote.add_current_observation(pilot.sequence, track.track_id, int(det_gt_id))
        info = {
            "sequence": pilot.sequence, "frame": int(frame_id),
            "track_id": int(track.track_id), "det_index": det_idx,
            "det_gt_id": int(det_gt_id), "det_gt_iou": float(det_iou),
            "target_gt_id": None if target is None else int(target),
            "model_motion_gate": None if model_motion is None else float(model_motion),
        }
        chosen = 1.0
        mean_final = cov_final = None
        reason = None
        if track.mean is None or track.covariance is None:
            reason = "no_state"
        else:
            warp_next = pilot.warps.get(frame_id + 1)
            gt_t1 = None if target is None else pilot.gt.get(frame_id + 1, {}).get(int(target))
            if warp_next is None:
                reason = "no_warp"
            elif target is None:
                reason = "no_target"
            elif gt_t1 is None:
                reason = "no_gt_t1"
            else:
                prior = np.asarray(track.mean, dtype=np.float64).copy()
                cov_prior = np.asarray(track.covariance, dtype=np.float64).copy()
                measurement = np.asarray(detection.cxcywh, dtype=np.float64)
                score = float(detection.score)
                mean_full, cov_full = kf_update_predict(
                    prior, cov_prior, measurement, score, warp_next, pilot.kf
                )
                losses = []
                for gate in GRID:
                    mean_g = prior + gate * (mean_full - prior)
                    cov_g = (1.0 - gate) * cov_prior + gate * cov_full
                    pred_mean, _ = warp_predict(mean_g, cov_g, warp_next, pilot.kf)
                    box = mean_to_box(pred_mean)
                    losses.append(float(1.0 - numpy_diou(
                        np.asarray([box]), np.asarray([gt_t1])
                    )[0]))
                losses = np.asarray(losses)
                best = float(losses.min())
                eligible = losses <= best + TIE_TOLERANCE
                index = int(eligible.argmax())
                chosen = float(GRID[index])
                if state["verify"] and float(GRID[int(np.argmin(losses))]) != chosen:
                    # argmin and the tie rule may differ only on exact ties; the
                    # rule must still select the largest gate within tolerance.
                    pass
                info.update({
                    "valid": True, "chosen": chosen,
                    "loss_full": float(losses[0]), "loss_zero": float(losses[-1]),
                    "loss_chosen": float(losses[index]), "min_loss": best,
                })
                if event_log:
                    info["losses"] = losses.tolist()
                if chosen == 1.0:
                    mean_final, cov_final = mean_full, cov_full
                elif chosen == 0.0:
                    mean_final, cov_final = prior, cov_prior
                else:
                    mean_final = prior + chosen * (mean_full - prior)
                    cov_final = (1.0 - chosen) * cov_prior + chosen * cov_full
                    cov_final = 0.5 * (cov_final + cov_final.T)
        if reason is not None:
            info.update({"valid": False, "reason": reason, "chosen": 1.0})
        return chosen, info, mean_final, cov_final

    def record(seq, info, path):
        stats = bucket(seq)
        stats["events"] += 1
        stats[path] += 1
        if info.get("valid"):
            stats["valid"] += 1
            stats["chosen_hist"][str(info["chosen"])] = stats["chosen_hist"].get(str(info["chosen"]), 0) + 1
            if info["chosen"] == 1.0:
                stats["chosen_ones"] += 1
            else:
                stats["chosen_below_one"] += 1
            for key, field in (("loss_full", "loss_full_sum"),
                               ("loss_chosen", "loss_chosen_sum"),
                               ("min_loss", "loss_min_sum")):
                stats[field] += info[key]
            stats["loss_count"] += 1
            if info.get("model_motion_gate") is not None:
                stats["model_motion_sum"] += info["model_motion_gate"]
                stats["model_motion_count"] += 1
                if abs(info["model_motion_gate"] - info["chosen"]) > 0.05:
                    stats["model_chosen_diff_gt_005"] += 1
        else:
            reason = info.get("reason", "unknown")
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        if info.get("appearance_gate") is not None:
            stats["appearance_sum"] += info["appearance_gate"]
            stats["appearance_count"] += 1
        if state["event_file"] is not None:
            state["event_file"].write(json.dumps(info, ensure_ascii=False) + "\n")

    original_init = tracker_module.Tracker.__init__
    original_ug = Track.update_with_gates
    original_u = Track.update

    def patched_init(self, args, vid_name):
        close_pilot()
        state["pilot"] = make_pilot(vid_name)
        state["current_seq"] = vid_name
        return original_init(self, args, vid_name)

    def patched_ug(self, frame_id, detection, motion_gate, appearance_gate):
        pilot = state["pilot"]
        chosen, info, mean_final, cov_final = compute(self, frame_id, detection, motion_gate)
        info["appearance_gate"] = float(appearance_gate)
        result = original_ug(self, frame_id, detection, chosen, appearance_gate)
        if state["verify"] and mean_final is not None:
            err = float(np.max(np.abs(np.asarray(self.mean) - mean_final)))
            state["max_state_err"] = max(state["max_state_err"], err)
            info["state_err_mean"] = err
        record(pilot.sequence if pilot else "?", info, "gated_path")
        if state["event_log"]:
            state["event_file"].flush()
        return result

    def patched_u(self, frame_id, detection):
        pilot = state["pilot"]
        chosen, info, mean_final, cov_final = compute(self, frame_id, detection, None)
        info["appearance_gate"] = 1.0
        result = original_ug(self, frame_id, detection, chosen, 1.0)
        if state["verify"] and mean_final is not None:
            err = float(np.max(np.abs(np.asarray(self.mean) - mean_final)))
            state["max_state_err"] = max(state["max_state_err"], err)
            info["state_err_mean"] = err
        record(pilot.sequence if pilot else "?", info, "ungated_path")
        if state["event_log"]:
            state["event_file"].flush()
        return result

    Tracker = tracker_module.Tracker
    Track.update_with_gates = patched_ug
    Track.update = patched_u
    Tracker.__init__ = patched_init
    started = time.monotonic()
    try:
        run_module.run()
        code = 0
    except Exception as exc:  # noqa: BLE001
        code = 1
        (out / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    finally:
        Track.update_with_gates = original_ug
        Track.update = original_u
        Tracker.__init__ = original_init
        close_pilot()
        if state["event_file"] is not None:
            state["event_file"].close()

    coverage = {"group": "B_gt_oracle", "max_state_error": state["max_state_err"],
                "verification_mismatch": state["argmin_mismatch"],
                "sequences": {}, "totals": {}}
    totals = {"events": 0, "valid": 0, "reasons": {}, "gated_path": 0, "ungated_path": 0,
              "chosen_ones": 0, "chosen_below_one": 0, "loss_full_sum": 0.0,
              "loss_chosen_sum": 0.0, "loss_min_sum": 0.0, "loss_count": 0,
              "model_motion_sum": 0.0, "model_motion_count": 0,
              "appearance_sum": 0.0, "appearance_count": 0,
              "model_chosen_diff_gt_005": 0, "chosen_hist": {}}
    for seq, stats in state["seq_stats"].items():
        item = dict(stats)
        count = max(1, item["loss_count"])
        item.update({
            "loss_full_mean": item["loss_full_sum"] / count,
            "loss_chosen_mean": item["loss_chosen_sum"] / count,
            "loss_min_mean": item["loss_min_sum"] / count,
            "model_motion_mean": (item["model_motion_sum"] / item["model_motion_count"]
                                  if item["model_motion_count"] else None),
            "appearance_mean": (item["appearance_sum"] / item["appearance_count"]
                                if item["appearance_count"] else None),
        })
        coverage["sequences"][seq] = item
        for key in ("events", "valid", "gated_path", "ungated_path", "chosen_ones",
                    "chosen_below_one", "loss_full_sum", "loss_chosen_sum", "loss_min_sum",
                    "loss_count", "model_motion_sum", "model_motion_count",
                    "appearance_sum", "appearance_count", "model_chosen_diff_gt_005"):
            totals[key] += item[key]
        for reason, count_r in item["reasons"].items():
            totals["reasons"][reason] = totals["reasons"].get(reason, 0) + count_r
        for gate, count_g in item["chosen_hist"].items():
            totals["chosen_hist"][gate] = totals["chosen_hist"].get(gate, 0) + count_g
    count = max(1, totals["loss_count"])
    totals.update({
        "loss_full_mean": totals["loss_full_sum"] / count,
        "loss_chosen_mean": totals["loss_chosen_sum"] / count,
        "loss_min_mean": totals["loss_min_sum"] / count,
        "model_motion_mean": (totals["model_motion_sum"] / totals["model_motion_count"]
                              if totals["model_motion_count"] else None),
        "appearance_mean": (totals["appearance_sum"] / totals["appearance_count"]
                            if totals["appearance_count"] else None),
    })
    coverage["totals"] = totals
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n")
    return {"group": "B_gt_oracle", "seconds": round(time.monotonic() - started, 2),
            "exit_code": code, "summary": str(out / "summary.json"),
            "coverage": str(out / "coverage.json")}


def verify_a(out: Path) -> dict:
    gates = out / "gates.jsonl"
    if not gates.is_file():
        return {"a_ok": False, "reason": "missing gates.jsonl"}
    rows = [json.loads(line) for line in gates.read_text().splitlines() if line.strip()]
    motion = [r.get("shutrack_avg_motion_gate") for r in rows if r.get("shutrack_avg_motion_gate") is not None]
    appearance = [r.get("shutrack_avg_appearance_gate") for r in rows if r.get("shutrack_avg_appearance_gate") is not None]
    return {"a_ok": bool(motion) and all(abs(m - 1.0) < 1e-6 for m in motion),
            "sequences": len(rows),
            "motion_gate_min": min(motion) if motion else None,
            "motion_gate_max": max(motion) if motion else None,
            "appearance_gate_min": min(appearance) if appearance else None,
            "appearance_gate_max": max(appearance) if appearance else None}


def compare(out: Path, seqs) -> dict:
    paths = {"baseline": BASELINE_SUMMARY, "bce": BCE_SUMMARY,
             "A_force_motion1": out / "A_force_motion1/summary.json",
             "B_gt_oracle": out / "B_gt_oracle/summary.json"}
    data = {}
    for name, path in paths.items():
        if path.is_file():
            payload = json.loads(path.read_text())
            data[name] = {"combined": payload.get("combined"),
                          "per_sequence": payload.get("per_sequence")}
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
    rows = {}
    for name, payload in data.items():
        rows[name] = {"combined": {k: payload["combined"].get(k) for k in keys},
                      "per_sequence": {seq: {k: (payload.get("per_sequence", {}).get(seq) or {}).get(k)
                                             for k in keys}
                                       for seq in seqs}}
    report = {"rows": rows, "deltas": {}}
    if "bce" in rows and "A_force_motion1" in rows:
        report["deltas"]["A_minus_BCE"] = {
            k: (rows["A_force_motion1"]["combined"][k] - rows["bce"]["combined"][k])
            if None not in (rows["A_force_motion1"]["combined"][k], rows["bce"]["combined"][k]) else None
            for k in keys}
    if "baseline" in rows and "B_gt_oracle" in rows:
        report["deltas"]["B_minus_baseline"] = {
            k: (rows["B_gt_oracle"]["combined"][k] - rows["baseline"]["combined"][k])
            if None not in (rows["B_gt_oracle"]["combined"][k], rows["baseline"]["combined"][k]) else None
            for k in keys}
    (out / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# SportsMOT val 运动门控诊断", "", "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "bce", "A_force_motion1", "B_gt_oracle"):
        if name not in rows:
            continue
        c = rows[name]["combined"]
        def fmt(k):
            v = c.get(k)
            if v is None:
                return "-"
            return f"{v:.0f}" if k == "IDSW" else f"{v*100:.3f}"
        lines.append(f"| {name} | " + " | ".join(fmt(k) for k in keys) + " |")
    if "A_minus_BCE" in report["deltas"]:
        d = report["deltas"]["A_minus_BCE"]
        lines.append("| Δ A−BCE | " + " | ".join(
            f"{d[k]*100:+.3f} pp" if k != "IDSW" and d[k] is not None else
            (f"{d[k]:+.0f}" if d[k] is not None else "-") for k in keys) + " |")
    if "B_minus_baseline" in report["deltas"]:
        d = report["deltas"]["B_minus_baseline"]
        lines.append("| Δ B−baseline | " + " | ".join(
            f"{d[k]*100:+.3f} pp" if k != "IDSW" and d[k] is not None else
            (f"{d[k]:+.0f}" if d[k] is not None else "-") for k in keys) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("A", "B", "both"), default="both")
    parser.add_argument("--sequences", nargs="*", default=list(VAL_SEQS))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--event-log", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"started_at": time.strftime("%F %T"), "sequences": list(args.sequences),
              "max_frames": args.max_frames, "results": {}}
    if args.mode in ("A", "both"):
        report["results"]["A"] = run_a(args.sequences, out / "A_force_motion1", args.max_frames)
        report["verification_A"] = verify_a(out / "A_force_motion1")
    if args.mode in ("B", "both"):
        report["results"]["B"] = run_b(
            args.sequences, out / "B_gt_oracle", args.max_frames,
            event_log=args.event_log, verify=args.verify,
        )
    report["comparison"] = compare(out, args.sequences)
    report["finished_at"] = time.strftime("%F %T")
    (out / "diagnostics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"finished_at": report["finished_at"],
                      "deltas": report["comparison"]["deltas"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
