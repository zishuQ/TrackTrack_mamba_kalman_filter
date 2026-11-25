#!/usr/bin/env python3
"""Summarize SportsMOT D-supervision midpoint checkpoint evaluations."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

EXP = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/sportsmot_d_midpoints_20261122")
BASE = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT")
EPOCHS = (40, 60, 80, 100, 120, 160, 200)
KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
VAL = [line.strip() for line in (Path("/home/shang/workspace/TrackTrack/3. Tracker/trackeval/seqmap/sportsmot/val.txt")).read_text().splitlines()
       if line.strip() and line.strip() != "name"]


def load_summary(epoch):
    path = (BASE / "evals/new/summary.json") if epoch == 200 else (EXP / f"evals/e{epoch:03d}/summary.json")
    return json.loads(path.read_text()) if path.is_file() else None


def load_gates(epoch):
    path = (BASE / "evals/new/gates.jsonl") if epoch == 200 else (EXP / f"evals/e{epoch:03d}/gates.jsonl")
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    motion = [r["shutrack_avg_motion_gate"] for r in rows if r.get("shutrack_avg_motion_gate") is not None]
    appearance = [r["shutrack_avg_appearance_gate"] for r in rows if r.get("shutrack_avg_appearance_gate") is not None]
    return {"rows": len(rows), "motion": motion, "appearance": appearance}


def stats(values):
    if not values:
        return None
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values),
            "median": statistics.median(values)}


def main():
    learning = {r["epoch"]: r["lr"] for r in
                (json.loads(line) for line in (BASE / "groups/appearance_row_column/metrics.jsonl").read_text().splitlines() if line.strip())}
    summaries = {e: load_summary(e) for e in EPOCHS}
    summaries = {e: s for e, s in summaries.items() if s is not None}
    gates = {e: load_gates(e) for e in summaries}
    baseline = summaries[200]
    best_epoch = max(summaries, key=lambda e: summaries[e]["combined"]["HOTA"])
    payload = {"epochs": {}, "best_epoch": best_epoch, "learning_rate": learning,
               "per_sequence": {}}
    for e, summary in summaries.items():
        combined = {k: summary["combined"].get(k) for k in KEYS}
        item = {"combined": combined, "lr": learning.get(e)}
        deltas = [((summary["per_sequence"].get(seq) or {}).get("HOTA", 0) -
                   (baseline["per_sequence"].get(seq) or {}).get("HOTA", 0)) * 100
                  for seq in VAL
                  if (summary["per_sequence"].get(seq) or {}).get("HOTA") is not None
                  and (baseline["per_sequence"].get(seq) or {}).get("HOTA") is not None]
        item["vs_e200"] = {"mean_pp": statistics.mean(deltas),
                           "se_pp": statistics.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else None,
                           "improved": sum(v > 0 for v in deltas), "n": len(deltas),
                           "min_pp": min(deltas), "max_pp": max(deltas)}
        if gates[e]:
            item["gates"] = {"motion": stats(gates[e]["motion"]),
                             "appearance": stats(gates[e]["appearance"])}
        payload["epochs"][str(e)] = item
    for seq in VAL:
        payload["per_sequence"][seq] = {
            str(e): ((summaries[e]["per_sequence"].get(seq) or {}).get("HOTA"))
            for e in summaries}
    (EXP / "midpoint_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = ["# SportsMOT 原 D 监督（train 45 / val 45）中期 checkpoint 对比", "",
             "复用原 e200 评估配置：检测缓存、GMC、阈值、seed 10000、45 条 val raw。", "",
             "| 轮次 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | LR |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in sorted(summaries):
        c = summaries[e]["combined"]
        lr = learning.get(e)
        lr_text = "-" if lr is None else f"{lr:.3g}"
        lines.append(f"| {e} | " + " | ".join(
            f"{c[k]*100:.3f}" if k != "IDSW" else f"{c[k]:.0f}" for k in KEYS) + f" | {lr_text} |")
    lines += ["", "| 轮次 | 运动 gate (min/mean/max) | 外观 gate (min/mean/max) | vs e200 逐序列均值 | 改善数 |",
              "|---:|---|---|---:|---:|"]
    for e in sorted(summaries):
        item = payload["epochs"][str(e)]
        g = item.get("gates")
        if g:
            gate_text = (f"{g['motion']['min']:.4f}/{g['motion']['mean']:.4f}/{g['motion']['max']:.4f} | "
                         f"{g['appearance']['min']:.4f}/{g['appearance']['mean']:.4f}/{g['appearance']['max']:.4f}")
        else:
            gate_text = "- | -"
        d = item["vs_e200"]
        lines.append(f"| {e} | {gate_text} | {d['mean_pp']:+.3f}pp | {d['improved']}/{d['n']} |")
    lines += ["", f"**最佳轮次：e{best_epoch}**，HOTA {summaries[best_epoch]['combined']['HOTA']*100:.3f}，"
                  f"LR {learning.get(best_epoch)}。", "",
              "## 逐序列 HOTA", "",
              "| 序列 | " + " | ".join(f"e{e}" for e in sorted(summaries)) + " | 最佳−e200 (pp) |",
              "|---|" + "---:|" * len(summaries) + "---:|"]
    for seq in VAL:
        row = payload["per_sequence"][seq]
        values = [row.get(str(e)) for e in sorted(summaries)]
        best_value = summaries[best_epoch]["per_sequence"].get(seq, {}).get("HOTA")
        base_value = baseline["per_sequence"].get(seq, {}).get("HOTA")
        delta = "-" if best_value is None or base_value is None else f"{(best_value-base_value)*100:+.3f}"
        lines.append("| " + seq + " | " + " | ".join(
            "-" if v is None else f"{v*100:.3f}" for v in values) + f" | {delta} |")
    (EXP / "midpoint_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
