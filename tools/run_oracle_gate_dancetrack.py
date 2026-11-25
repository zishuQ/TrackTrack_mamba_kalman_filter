#!/usr/bin/env python3
"""DanceTrack val motion-gate diagnostic baseline (inference only).

Both groups use the original full EMA appearance update (appearance gate 1.0):
  A: motion gate fixed at 1.0.
  B: motion gate chosen online per matched update by next-frame GT DIoU over
     g in {1.0, ..., 0.0}, using the label generator's tie rule.

The GT selection logic is the one verified on SportsMOT (SequencePilot identity
vote, NSA update/predict with GMC warp, largest gate within 1e-6 of minimum).
Annotations come from the project data/DanceTrack directory; GMC files use the
dancetrack-000N naming; detection/ReID caches are the existing ones.
"""
from __future__ import annotations

import argparse
import json
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
DATA_DIR = ROOT / "data"
CACHE = ROOT / "outputs/agentguard/detection_cache"
DUMMY_CKPT = (
    ROOT / "outputs/agentguard/experiments/shutrack_sportsmot_cost_trainval_formal_20261122"
    / "SportsMOT/runs/groups/appearance_row_column/checkpoints/epoch_200.pt"
)
EXP = ROOT / "outputs/agentguard/experiments/dancetrack_motion_diag_20261122"
VAL_SEQS = tuple(
    line.strip()
    for line in (TRACKER / "trackeval/seqmap/dancetrack/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)


def _base_argv(out: Path, suffix: str, seqs, max_frames: int):
    argv = [
        "--dataset", "DanceTrack", "--data_dir", str(DATA_DIR) + "/",
        "--mode", "val", "--sequences", *seqs,
        "--seed", "10000", "--kf-type", "nsa", "--agentguard-mode", "shutrack-v2",
        "--shutrack-checkpoint", str(DUMMY_CKPT), "--agentguard-device", "cpu",
        "--detection-cache-root", str(CACHE),
        "--output_dir", str(out), "--tracker-suffix", suffix,
        "--agentguard-stats-output", str(out / "gates.jsonl"),
        "--print-per-sequence-metrics", "--eval-json", str(out / "summary.json"),
        "--torch-num-threads", "4", "--torch-num-interop-threads", "1",
        # Original EMA appearance update in both groups; motion handled below.
        "--agentguard-disable-ema-gate", "--agentguard-disable-kf-gate",
    ]
    if max_frames > 0:
        argv += ["--max-frames", str(max_frames)]
    return argv


def run_a(seqs, out: Path, max_frames: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    command = [str(PY), "-u", "run.py", *_base_argv(out, "motion1", seqs, max_frames)]
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
    argv = _base_argv(out, "gtgate", seqs, max_frames)
    args = run_module.make_parser().parse_args(argv)
    run_module.args = args

    state = {
        "pilot": None, "event_file": None, "event_log": event_log, "verify": verify,
        "seq_stats": {}, "max_state_err": 0.0,
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
        pilot = SequencePilot(seq, "val", DUMMY_CKPT, out / "pilot_scratch" / seq,
                              dataset="DanceTrack")
        pilot.lean = False
        pilot.store_rank_features = False
        pilot.img_w = int(run_module.args.img_w)
        pilot.img_h = int(run_module.args.img_h)
        pilot.det_thr = float(getattr(run_module.args, "det_thr", 0.60))
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
            "loss_full_sum": 0.0, "loss_chosen_sum": 0.0, "loss_min_sum": 0.0,
            "loss_count": 0, "harmful_count": 0, "useful_count": 0,
            "saved_harmful_sum": 0.0, "added_useful_sum": 0.0, "net_delta_sum": 0.0,
            "suppress_count": 0, "appearance_sum": 0.0, "appearance_count": 0,
        })

    def compute(track, frame_id, detection):
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
            "appearance_gate": 1.0,
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

    def record(seq, info):
        stats = bucket(seq)
        stats["events"] += 1
        stats["gated_path"] += 1
        if info.get("valid"):
            stats["valid"] += 1
            chosen = info["chosen"]
            stats["chosen_hist"][str(chosen)] = stats["chosen_hist"].get(str(chosen), 0) + 1
            if chosen == 1.0:
                stats["chosen_ones"] += 1
            else:
                stats["chosen_below_one"] += 1
                stats["suppress_count"] += 1
            for key, field in (("loss_full", "loss_full_sum"),
                               ("loss_chosen", "loss_chosen_sum"),
                               ("min_loss", "loss_min_sum")):
                stats[field] += info[key]
            stats["loss_count"] += 1
            loss_full, loss_chosen, loss_zero = (
                info["loss_full"], info["loss_chosen"], info["loss_zero"])
            if loss_full > loss_zero + 1e-6:
                stats["harmful_count"] += 1
                stats["saved_harmful_sum"] += loss_full - loss_chosen
            else:
                stats["useful_count"] += 1
                stats["added_useful_sum"] += loss_chosen - loss_full
            stats["net_delta_sum"] += loss_chosen - loss_full
        else:
            reason = info.get("reason", "unknown")
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        stats["appearance_sum"] += 1.0
        stats["appearance_count"] += 1
        if state["event_file"] is not None:
            state["event_file"].write(json.dumps(info, ensure_ascii=False) + "\n")

    original_init = tracker_module.Tracker.__init__
    original_ug = Track.update_with_gates

    def patched_init(self, args, vid_name):
        close_pilot()
        state["pilot"] = make_pilot(vid_name)
        state["current_seq"] = vid_name
        return original_init(self, args, vid_name)

    def patched_ug(self, frame_id, detection, motion_gate, appearance_gate):
        pilot = state["pilot"]
        chosen, info, mean_final, cov_final = compute(self, frame_id, detection)
        result = original_ug(self, frame_id, detection, chosen, 1.0)
        if state["verify"] and mean_final is not None:
            err = float(np.max(np.abs(np.asarray(self.mean) - mean_final)))
            state["max_state_err"] = max(state["max_state_err"], err)
            info["state_err_mean"] = err
        record(pilot.sequence if pilot else "?", info)
        if state["event_file"]:
            state["event_file"].flush()
        return result

    Track.update_with_gates = patched_ug
    tracker_module.Tracker.__init__ = patched_init
    started = time.monotonic()
    try:
        run_module.run()
        code = 0
    except Exception as exc:  # noqa: BLE001
        code = 1
        (out / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    finally:
        Track.update_with_gates = original_ug
        tracker_module.Tracker.__init__ = original_init
        close_pilot()
        if state["event_file"] is not None:
            state["event_file"].close()

    coverage = {"group": "B_gt_oracle", "max_state_error": state["max_state_err"],
                "sequences": {}, "totals": {}}
    totals = {
        "events": 0, "valid": 0, "reasons": {}, "gated_path": 0,
        "chosen_ones": 0, "chosen_below_one": 0, "chosen_hist": {},
        "loss_full_sum": 0.0, "loss_chosen_sum": 0.0, "loss_min_sum": 0.0, "loss_count": 0,
        "harmful_count": 0, "useful_count": 0, "saved_harmful_sum": 0.0,
        "added_useful_sum": 0.0, "net_delta_sum": 0.0, "suppress_count": 0,
        "appearance_sum": 0.0, "appearance_count": 0,
    }
    for seq, stats in state["seq_stats"].items():
        item = dict(stats)
        count = max(1, item["loss_count"])
        item.update({
            "loss_full_mean": item["loss_full_sum"] / count,
            "loss_chosen_mean": item["loss_chosen_sum"] / count,
            "loss_min_mean": item["loss_min_sum"] / count,
            "harmful_fraction": item["harmful_count"] / count,
            "saved_harmful_mean": (item["saved_harmful_sum"] / item["harmful_count"]
                                   if item["harmful_count"] else None),
            "added_useful_mean": (item["added_useful_sum"] / item["useful_count"]
                                  if item["useful_count"] else None),
            "net_delta_mean": item["net_delta_sum"] / count,
            "suppression_fraction": item["suppress_count"] / count,
        })
        coverage["sequences"][seq] = item
        for key in totals:
            if key in ("reasons", "chosen_hist"):
                continue
            totals[key] += item[key]
        for reason, value in item["reasons"].items():
            totals["reasons"][reason] = totals["reasons"].get(reason, 0) + value
        for gate, value in item["chosen_hist"].items():
            totals["chosen_hist"][gate] = totals["chosen_hist"].get(gate, 0) + value
    count = max(1, totals["loss_count"])
    totals.update({
        "loss_full_mean": totals["loss_full_sum"] / count,
        "loss_chosen_mean": totals["loss_chosen_sum"] / count,
        "loss_min_mean": totals["loss_min_sum"] / count,
        "harmful_fraction": totals["harmful_count"] / count,
        "saved_harmful_mean": (totals["saved_harmful_sum"] / totals["harmful_count"]
                               if totals["harmful_count"] else None),
        "added_useful_mean": (totals["added_useful_sum"] / totals["useful_count"]
                              if totals["useful_count"] else None),
        "net_delta_mean": totals["net_delta_sum"] / count,
        "suppression_fraction": totals["suppress_count"] / count,
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
    motion = [r.get("shutrack_avg_motion_gate") for r in rows
              if r.get("shutrack_avg_motion_gate") is not None]
    appearance = [r.get("shutrack_avg_appearance_gate") for r in rows
                  if r.get("shutrack_avg_appearance_gate") is not None]
    return {"a_ok": bool(motion) and all(abs(m - 1.0) < 1e-6 for m in motion)
                  and all(abs(a - 1.0) < 1e-6 for a in appearance),
            "sequences": len(rows),
            "motion_gate_min": min(motion) if motion else None,
            "motion_gate_max": max(motion) if motion else None,
            "appearance_gate_min": min(appearance) if appearance else None,
            "appearance_gate_max": max(appearance) if appearance else None}


def compare(out: Path, seqs) -> dict:
    paths = {"A_force_motion1": out / "A_force_motion1/summary.json",
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
                                             for k in keys} for seq in seqs}}
    report = {"rows": rows, "deltas": {}}
    if "A_force_motion1" in rows and "B_gt_oracle" in rows:
        report["deltas"]["B_minus_A"] = {
            k: (rows["B_gt_oracle"]["combined"][k] - rows["A_force_motion1"]["combined"][k])
            if None not in (rows["B_gt_oracle"]["combined"][k],
                            rows["A_force_motion1"]["combined"][k]) else None
            for k in keys}
    coverage_path = out / "B_gt_oracle/coverage.json"
    if coverage_path.is_file():
        report["coverage"] = json.loads(coverage_path.read_text())
    (out / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# DanceTrack val 运动门控诊断", "",
             "两组均使用原 EMA 外观更新（外观 gate=1）；A 运动 gate 固定 1，B 用下一帧 GT 现场选门。", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("A_force_motion1", "B_gt_oracle"):
        if name not in rows:
            continue
        c = rows[name]["combined"]
        def fmt(k):
            v = c.get(k)
            return "-" if v is None else (f"{v:.0f}" if k == "IDSW" else f"{v*100:.3f}")
        lines.append(f"| {name} | " + " | ".join(fmt(k) for k in keys) + " |")
    if "B_minus_A" in report["deltas"]:
        d = report["deltas"]["B_minus_A"]
        lines.append("| Δ B−A | " + " | ".join(
            f"{d[k]*100:+.3f} pp" if k != "IDSW" and d[k] is not None else
            (f"{d[k]:+.0f}" if d[k] is not None else "-") for k in keys) + " |")
    totals = (report.get("coverage") or {}).get("totals") or {}
    if totals:
        lines += ["", "## GT 选门统计（匹配事件）", "",
                  f"- 有效事件：{int(totals.get('valid', 0))}；抑制（chosen<1）："
                  f"{totals.get('suppression_fraction', 0)*100:.1f}%。",
                  f"- 有害更新比例：{totals.get('harmful_fraction', 0)*100:.1f}%；"
                  f"抑制收益（有害事件）：{totals.get('saved_harmful_mean')}；"
                  f"误伤代价（有益事件）：{totals.get('added_useful_mean')}；"
                  f"净变化：{totals.get('net_delta_mean')}。",
                  f"- full/chosen/best 损失均值：{totals.get('loss_full_mean')} / "
                  f"{totals.get('loss_chosen_mean')} / {totals.get('loss_min_mean')}。"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("A", "B", "both"), default="both")
    parser.add_argument("--sequences", nargs="*", default=list(VAL_SEQS))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out", default=str(EXP))
    parser.add_argument("--event-log", action="store_true")
    parser.add_argument("--verify", action="store_true")
    options = parser.parse_args()
    out = Path(options.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {"sequences": len(options.sequences), "max_frames": options.max_frames}
    if options.mode in ("A", "both"):
        result["A"] = run_a(options.sequences, out / "A_force_motion1", options.max_frames)
        result["A_verify"] = verify_a(out / "A_force_motion1")
    if options.mode in ("B", "both"):
        result["B"] = run_b(options.sequences, out / "B_gt_oracle", options.max_frames,
                            event_log=options.event_log, verify=options.verify)
    if options.mode == "both":
        result["comparison"] = compare(out, options.sequences)
    (out / "run_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparison"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
