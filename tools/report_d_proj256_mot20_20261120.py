#!/usr/bin/env python3
"""D-64 vs D-256 (appearance projection width) report for MOT20."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")
FORMAT = {"HOTA": ".4f", "MOTA": ".4f", "IDF1": ".4f", "DetA": ".4f", "AssA": ".4f",
          "IDSW": ".0f", "Frag": ".0f"}


def load_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_speed(path: Path):
    if not path.is_file():
        return None
    values = []
    for line in path.read_text().splitlines():
        text = line.strip()
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values[-1] if values else None


def peak_rss_mb(path: Path):
    if not path.is_file():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(match.group(1)) / 1024 if match else None


def peak_gpu_mb(path: Path):
    if not path.is_file():
        return None
    values = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                values.append(float(parts[-1]))
            except ValueError:
                continue
    return max(values) if values else None


def agg_gates(records):
    motion_obs = sum(r.get("shutrack_motion_gate_observations", 0) for r in records)
    app_obs = sum(r.get("shutrack_appearance_gate_observations", 0) for r in records)
    return {
        "motion": sum(r.get("shutrack_avg_motion_gate", 0) * r.get("shutrack_motion_gate_observations", 0)
                      for r in records) / motion_obs if motion_obs else None,
        "appearance": sum(r.get("shutrack_avg_appearance_gate", 0) * r.get("shutrack_appearance_gate_observations", 0)
                          for r in records) / app_obs if app_obs else None,
        "motion_range": (min((r.get("shutrack_min_motion_gate", 1.0) for r in records), default=None),
                         max((r.get("shutrack_max_motion_gate", 0.0) for r in records), default=None)),
        "appearance_range": (min((r.get("shutrack_min_appearance_gate", 1.0) for r in records), default=None),
                             max((r.get("shutrack_max_appearance_gate", 0.0) for r in records), default=None)),
        "recovered": sum(r.get("shutrack_recovered_track_count", 0) for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="outputs/agentguard/experiments/shutrack_d_proj256_mot20_20261120")
    parser.add_argument("--baseline", default="outputs/agentguard/experiments/shutrack_bd_mot_fast_20261120/MOT20")
    args = parser.parse_args()
    exp = Path(args.exp)
    baseline = Path(args.baseline)

    base_summary = load_json(baseline / "evals/val_mot20d/summary.json")
    new_summary = load_json(exp / "evals/val_d256/summary.json")
    base_gates = read_jsonl(baseline / "evals/val_mot20d/gates.jsonl")
    new_gates = read_jsonl(exp / "evals/val_d256/gates.jsonl")
    base_cfg = load_json(baseline / "groups/appearance_row_column/config.json")
    new_cfg = load_json(exp / "groups/appearance_row_column/config.json")
    new_metrics = [r for r in read_jsonl(exp / "groups/appearance_row_column/metrics.jsonl") if "epoch" in r]
    base_metrics = [r for r in read_jsonl(baseline / "groups/appearance_row_column/metrics.jsonl") if "epoch" in r]
    new_complete = load_json(exp / "groups/appearance_row_column/complete.json")
    base_complete = load_json(baseline / "groups/appearance_row_column/complete.json")

    import torch
    base_payload = torch.load(baseline / "groups/appearance_row_column/checkpoints/epoch_100.pt",
                              map_location="cpu", weights_only=False)
    new_payload = torch.load(exp / "groups/appearance_row_column/checkpoints/epoch_100.pt",
                             map_location="cpu", weights_only=False)

    base_comb = base_summary.get("combined") or {}
    new_comb = new_summary.get("combined") or {}
    base_per = base_summary.get("per_sequence") or {}
    new_per = new_summary.get("per_sequence") or {}

    def row(name, values):
        return "| " + name + " | " + " | ".join(
            format(values.get(k, 0), FORMAT[k]) for k in METRICS
        ) + " |"

    def delta_row(name, values, base):
        cells = []
        for key in METRICS:
            diff = values[key] - base[key]
            cells.append(f"{diff * 100:+.2f} pp" if key not in ("IDSW", "Frag") else f"{diff:+.0f}")
        return "| " + name + " | " + " | ".join(cells) + " |"

    def steady(rows, n=20):
        tail = [r["epoch_seconds"] for r in rows[-n:]]
        return sum(tail) / len(tail) if tail else None

    lines: list[str] = []
    lines.append("# MOT20 D-64 vs D-256 外观投影容量实验（2026-11-20）")
    lines.append("")
    lines.append("唯一改动：`appearance_proj_dim` 64 → 256；原始 ReID 2048 维、d_model 128、"
                 "gate 头 128→64→1、运动分支与全部 D 监督保持不变。")
    lines.append("表中指标为 0–1 比例；Δ 列以百分点（pp）表示，IDSW/Frag 为计数差。")
    lines.append("")
    lines.append("## 1. 结构、参数量与 checkpoint")
    lines.append("")
    base_params = base_payload.get("parameter_counts") or {}
    new_params = new_payload.get("parameter_counts") or {}
    lines.append(f"- D-64：appearance_proj_dim={base_cfg.get('appearance_proj_dim', 64)}，"
                 f"总参数={int(base_params.get('total', 0)):,}"
                 f"（payload parameter_counts）")
    lines.append(f"- D-256：appearance_proj_dim={new_cfg.get('appearance_proj_dim')}，"
                 f"总参数={int(new_params.get('total', 0)):,}")
    lines.append(f"- D-256 checkpoint：`{exp / 'groups/appearance_row_column/checkpoints/epoch_100.pt'}`")
    lines.append(f"- D-64 checkpoint：`{baseline / 'groups/appearance_row_column/checkpoints/epoch_100.pt'}`")
    lines.append(f"- D-256 训练配置：epochs={new_cfg.get('epochs')}, total_steps={new_cfg.get('total_steps')}, "
                 f"warmup={new_cfg.get('warmup_steps')}, lr={new_cfg.get('lr')}, "
                 f"column_weight={new_cfg.get('column_weight')}, seed={new_cfg.get('seed')}, "
                 f"sequences={new_cfg.get('sequence_names')}")
    lines.append("")
    lines.append("## 2. MOT20 汇总（raw，MOT20-01 + MOT20-05）")
    lines.append("")
    lines.append("| 组 | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(row("D-64", base_comb))
    lines.append(row("D-256", new_comb))
    lines.append(delta_row("D-256 − D-64", new_comb, base_comb))
    lines.append("")
    lines.append("## 3. 逐序列")
    lines.append("")
    lines.append("| 序列 | HOTA 64 | HOTA 256 | ΔHOTA (pp) | IDF1 64 | IDF1 256 | AssA 64 | AssA 256 | IDSW 64/256 | Frag 64/256 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for sequence in sorted(set(base_per) & set(new_per)):
        b, n = base_per[sequence], new_per[sequence]
        lines.append(
            f"| {sequence} | {b['HOTA']:.4f} | {n['HOTA']:.4f} | {(n['HOTA'] - b['HOTA']) * 100:+.2f} | "
            f"{b['IDF1']:.4f} | {n['IDF1']:.4f} | {b['AssA']:.4f} | {n['AssA']:.4f} | "
            f"{int(round(b['IDSW']))}/{int(round(n['IDSW']))} | "
            f"{int(round(b['Frag']))}/{int(round(n['Frag']))} |"
        )
    lines.append("")
    lines.append("## 4. 门分布（val 推理）")
    lines.append("")
    lines.append("| 组 | 运动门均值 | 运动门范围 | 外观门均值 | 外观门范围 | 恢复轨迹计数 |")
    lines.append("|---|---|---|---|---|---|")
    for name, records in (("D-64", base_gates), ("D-256", new_gates)):
        agg = agg_gates(records)
        if not records:
            continue
        lines.append(
            f"| {name} | {agg['motion']:.4f} | [{agg['motion_range'][0]:.4f}, {agg['motion_range'][1]:.4f}] | "
            f"{agg['appearance']:.4f} | [{agg['appearance_range'][0]:.4f}, {agg['appearance_range'][1]:.4f}] | "
            f"{int(round(agg['recovered']))} |"
        )
    lines.append("")
    lines.append("## 5. 训练速度、显存与验证速度")
    lines.append("")
    for name, cfg, complete, metrics, log_path in (
        ("D-64", base_cfg, base_complete, base_metrics, baseline.parent / "logs/eval_MOT20_mot20d.log"),
        ("D-256", new_cfg, new_complete, new_metrics, exp / "logs/eval_d256.log"),
    ):
        st = steady(metrics)
        steps = 111
        lines.append(
            f"- {name}：训练 {complete.get('seconds', 0):.0f} s；稳定阶段末 20 轮均值 {st:.2f} s/轮"
            f"（{st * 1000 / steps:.1f} ms/step，{steps} 步/轮）"
            if st else f"- {name}：训练完成信息缺失"
        )
    rss = peak_rss_mb(exp / "logs/train.log")
    gpu = peak_gpu_mb(exp / "logs/gpu_memory.csv")
    if rss:
        lines.append(f"- D-256 训练峰值主机内存：{rss:.0f} MB" + (f"；峰值显存：{gpu:.0f} MB" if gpu else ""))
    speed64 = last_speed(baseline.parent / "logs/eval_MOT20_mot20d.log")
    speed256 = last_speed(exp / "logs/eval_d256.log")
    if speed64 and speed256:
        lines.append(
            f"- 验证跟踪速度（跟踪总帧数/总耗时）：D-64 {speed64:.2f} 帧/秒，"
            f"D-256 {speed256:.2f} 帧/秒。D-64 数值来自历史评估日志，不是与 D-256 严格同条件测速。"
        )
    if new_metrics:
        cold = [r["epoch_seconds"] for r in new_metrics[:4]]
        lines.append(f"- D-256 冷启动前 4 轮：{', '.join(f'{v:.1f}' for v in cold)} s")
    lines.append("")
    lines.append("## 6. 身份关联案例")
    lines.append("")
    for sequence in sorted(set(base_per) & set(new_per)):
        b, n = base_per[sequence], new_per[sequence]
        lines.append(
            f"- {sequence}: ΔHOTA={((n['HOTA'] - b['HOTA']) * 100):+.2f} pp，"
            f"ΔAssA={((n['AssA'] - b['AssA']) * 100):+.2f} pp，ΔIDF1={((n['IDF1'] - b['IDF1']) * 100):+.2f} pp，"
            f"IDSW {int(round(b['IDSW']))}→{int(round(n['IDSW']))}，"
            f"Frag {int(round(b['Frag']))}→{int(round(n['Frag']))}。"
        )
    lines.append("- 本实验只改变外观投影宽度；任何变化都不能归因于 column 监督本身，"
                 "也不能把轨迹重新激活当作身份纠错。")
    lines.append("")
    lines.append("## 7. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("bash tools/run_d_proj256_mot20_20261120.sh")
    lines.append("bash tools/run_d_proj256_mot20_eval_20261120.sh")
    lines.append(".venv/bin/python tools/report_d_proj256_mot20_20261120.py")
    lines.append("```")
    lines.append("")

    out = exp / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", out)


if __name__ == "__main__":
    main()
