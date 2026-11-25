#!/usr/bin/env python3
"""Summarize MOT20/MOT17 D-supervision midpoint checkpoint evaluations."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")


def stats(values):
    if not values:
        return None
    return {"min": min(values), "mean": sum(values) / len(values),
            "max": max(values), "median": statistics.median(values)}


def gate_stats(path):
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    motion = [r["shutrack_avg_motion_gate"] for r in rows
              if r.get("shutrack_avg_motion_gate") is not None]
    appearance = [r["shutrack_avg_appearance_gate"] for r in rows
                  if r.get("shutrack_avg_appearance_gate") is not None]
    return {"motion": stats(motion), "appearance": stats(appearance), "rows": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-gates", required=True)
    parser.add_argument("--baseline-epoch", type=int, required=True)
    parser.add_argument("--epochs", required=True,
                        help="comma-separated midpoint epochs; baseline epoch is added automatically")
    args = parser.parse_args()
    exp = Path(args.exp)
    epochs = sorted({int(x) for x in args.epochs.split(",") if x.strip()} | {args.baseline_epoch})
    learning = {r["epoch"]: r["lr"] for r in
                (json.loads(line) for line in Path(args.metrics).read_text().splitlines()
                 if line.strip())}
    summaries, gates = {}, {}
    for epoch in epochs:
        if epoch == args.baseline_epoch:
            summary_path = Path(args.baseline_summary)
            gate_path = Path(args.baseline_gates)
        else:
            summary_path = exp / f"evals/e{epoch:03d}/summary.json"
            gate_path = exp / f"evals/e{epoch:03d}/gates.jsonl"
        if not summary_path.is_file():
            continue
        summaries[epoch] = json.loads(summary_path.read_text())
        gates[epoch] = gate_stats(gate_path)
    baseline = summaries[args.baseline_epoch]
    seqs = list(baseline["per_sequence"])
    best = max(summaries, key=lambda e: summaries[e]["combined"]["HOTA"])
    payload = {"dataset": args.dataset, "best_epoch": best,
               "baseline_epoch": args.baseline_epoch, "learning_rate": learning,
               "epochs": {}, "per_sequence": {}}
    for epoch, summary in summaries.items():
        deltas = [(summary["per_sequence"][seq]["HOTA"] - baseline["per_sequence"][seq]["HOTA"]) * 100
                  for seq in seqs]
        payload["epochs"][str(epoch)] = {
            "combined": {k: summary["combined"].get(k) for k in KEYS},
            "lr": learning.get(epoch),
            "gates": gates[epoch],
            "vs_baseline": {"mean_pp": statistics.mean(deltas),
                            "improved": sum(v > 0 for v in deltas), "n": len(deltas)},
        }
    for seq in seqs:
        payload["per_sequence"][seq] = {
            str(e): (summaries[e]["per_sequence"].get(seq) or {}).get("HOTA") for e in summaries}
    (exp / "midpoint_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [f"# {args.dataset} 原 D 监督中期 checkpoint 对比", "",
             "复用原末轮评估配置：检测缓存、GMC、阈值、seed 10000、raw 评测。", "",
             "| 轮次 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | LR |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for epoch in sorted(summaries):
        c = summaries[epoch]["combined"]
        lr = learning.get(epoch)
        lines.append(f"| {epoch} | " + " | ".join(
            f"{c[k]*100:.3f}" if k != "IDSW" else f"{c[k]:.0f}" for k in KEYS) +
            f" | {'-' if lr is None else f'{lr:.3g}'} |")
    lines += ["", "| 轮次 | 运动 gate (min/mean/max) | 外观 gate (min/mean/max) | vs 末轮逐序列均值 | 改善数 |",
              "|---:|---|---|---:|---:|"]
    for epoch in sorted(summaries):
        g = gates.get(epoch)
        gate_text = ("-" if not g else
                     f"{g['motion']['min']:.4f}/{g['motion']['mean']:.4f}/{g['motion']['max']:.4f} | "
                     f"{g['appearance']['min']:.4f}/{g['appearance']['mean']:.4f}/{g['appearance']['max']:.4f}")
        d = payload["epochs"][str(epoch)]["vs_baseline"]
        lines.append(f"| {epoch} | {gate_text} | {d['mean_pp']:+.3f}pp | {d['improved']}/{d['n']} |")
    lines += ["", f"**最佳轮次：e{best}**，HOTA {summaries[best]['combined']['HOTA']*100:.3f}，"
                  f"LR {learning.get(best)}。", "",
              "## 逐序列 HOTA", "",
              "| 序列 | " + " | ".join(f"e{e}" for e in sorted(summaries)) + " |",
              "|---|" + "---:|" * len(summaries)]
    for seq in seqs:
        row = payload["per_sequence"][seq]
        lines.append("| " + seq + " | " + " | ".join(
            "-" if row.get(str(e)) is None else f"{row[str(e)]*100:.3f}" for e in sorted(summaries)) + " |")
    (exp / "midpoint_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
