#!/usr/bin/env python3
"""MOT20/MOT17 B vs D paired-ablation report (custom video-level split)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")
FORMAT = {"HOTA": ".4f", "MOTA": ".4f", "IDF1": ".4f", "DetA": ".4f", "AssA": ".4f",
          "IDSW": ".0f", "Frag": ".0f"}
DATASETS = {
    "MOT20": {
        "train": ("MOT20-02", "MOT20-03"),
        "val": ("MOT20-01", "MOT20-05"),
        "suffix": ("mot20b", "mot20d"),
        "epochs": 100,
    },
    "MOT17": {
        "train": ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                  "MOT17-11-FRCNN", "MOT17-13-FRCNN"),
        "val": ("MOT17-09-FRCNN", "MOT17-10-FRCNN"),
        "suffix": ("mot17b", "mot17d"),
        "epochs": 200,
    },
}


def load_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(value, key):
    if value is None:
        return "–"
    return format(value, FORMAT[key])


def delta_cell(value, base, key):
    diff = value - base
    return f"{diff * 100:+.2f} pp" if key not in ("IDSW", "Frag") else f"{diff:+.0f}"


def aggregate_gates(records):
    motion_obs = sum(r.get("shutrack_motion_gate_observations", 0) for r in records)
    app_obs = sum(r.get("shutrack_appearance_gate_observations", 0) for r in records)
    return {
        "sequences": len(records),
        "motion_mean": (
            sum(r.get("shutrack_avg_motion_gate", 0.0) * r.get("shutrack_motion_gate_observations", 0)
                for r in records) / motion_obs if motion_obs else None
        ),
        "appearance_mean": (
            sum(r.get("shutrack_avg_appearance_gate", 0.0) * r.get("shutrack_appearance_gate_observations", 0)
                for r in records) / app_obs if app_obs else None
        ),
        "motion_min": min((r.get("shutrack_min_motion_gate", 1.0) for r in records), default=None),
        "motion_max": max((r.get("shutrack_max_motion_gate", 0.0) for r in records), default=None),
        "appearance_min": min((r.get("shutrack_min_appearance_gate", 1.0) for r in records), default=None),
        "appearance_max": max((r.get("shutrack_max_appearance_gate", 0.0) for r in records), default=None),
        "recovered": sum(r.get("shutrack_recovered_track_count", 0) for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="outputs/agentguard/experiments/shutrack_bd_mot_20261120")
    args = parser.parse_args()
    exp = Path(args.exp)

    lines: list[str] = []
    lines.append("# MOT20 / MOT17 B/D 配对实验报告（2026-11-20）")
    lines.append("")
    lines.append("固定视频级划分，验证视频不参与训练、归一化统计或标签阈值。")
    lines.append("")
    lines.append("表中指标为 0–1 比例；Δ 列以百分点（pp）表示，IDSW/Frag 为计数差。")
    lines.append("")

    for dataset, spec in DATASETS.items():
        root = exp / dataset
        groups = {}
        for group in ("appearance_rank", "appearance_row_column"):
            directory = root / "groups" / group
            groups[group] = {
                "config": load_json(directory / "config.json") if (directory / "config.json").is_file() else {},
                "complete": load_json(directory / "complete.json") if (directory / "complete.json").is_file() else {},
                "last": next(iter(reversed(read_jsonl(directory / "metrics.jsonl"))), {}),
            }
        evals = {}
        for suffix in spec["suffix"]:
            out = root / "evals" / f"val_{suffix}"
            summary_path = out / "summary.json"
            evals[suffix] = {
                "summary": load_json(summary_path) if summary_path.is_file() else {},
                "gates": read_jsonl(out / "gates.jsonl"),
            }
        b_key, d_key = spec["suffix"]
        b_sum, d_sum = evals[b_key]["summary"], evals[d_key]["summary"]
        b_comb, d_comb = b_sum.get("combined") or {}, d_sum.get("combined") or {}
        b_per, d_per = b_sum.get("per_sequence") or {}, d_sum.get("per_sequence") or {}
        b_config = groups["appearance_rank"]["config"]
        d_config = groups["appearance_row_column"]["config"]
        data_root = Path(b_config.get("data_root") or root / "data")

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("### 划分与数据")
        lines.append("")
        lines.append(f"- 训练：{', '.join(spec['train'])}")
        lines.append(f"- 验证：{', '.join(spec['val'])}")
        lines.append(
            f"- 训练配置：epochs={b_config.get('epochs')}, total_steps={b_config.get('total_steps')}, "
            f"warmup_steps={b_config.get('warmup_steps')}, shards={b_config.get('shards')}, "
            f"full_passes={b_config.get('full_passes')}, batch={b_config.get('batch_size')}, "
            f"lr={b_config.get('lr')}, seed={b_config.get('seed')}"
        )
        lines.append(
            f"- 初始化：B={b_config.get('init_mode')} ({b_config.get('init')}), "
            f"D={d_config.get('init_mode')} ({d_config.get('init')}), "
            f"归一化={b_config.get('normalization_source')}; "
            f"taus={b_config.get('taus')}"
        )
        if data_root.is_dir():
            lines.append("")
            lines.append("| 训练序列 | 样本 | rank 有效 | column 有效 |")
            lines.append("|---|---|---|---|")
            for sequence in spec["train"]:
                summary_path = data_root / sequence / "summary.json"
                if not summary_path.is_file():
                    continue
                summary = load_json(summary_path)
                samples = summary.get("samples", 0) or 1
                rank = summary.get("rank_valid", 0)
                column = (summary.get("column_counts") or {}).get("writes", 0)
                lines.append(
                    f"| {sequence} | {samples} | {rank} ({rank / samples:.1%}) | "
                    f"{column} ({column / samples:.1%}) |"
                )
        validation_path = root / "data_validation.json"
        if validation_path.is_file():
            validation = load_json(validation_path)
            lines.append("")
            lines.append(
                f"- 数据校验：data_ok={validation.get('data_ok')}, 样本={validation.get('samples')}, "
                f"键错配={validation.get('key_mismatch')}, 索引错配={validation.get('index_mismatch')}, "
                f"column 帧错配={validation.get('column_mismatch')}, 非有限值={validation.get('finite_bad')}"
            )
        lines.append("")
        lines.append("### Val 汇总（raw）")
        lines.append("")
        lines.append("| 组 | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag |")
        lines.append("|---|---|---|---|---|---|---|---|")
        if b_comb:
            lines.append("| B | " + " | ".join(fmt(b_comb.get(k), k) for k in METRICS) + " |")
        if d_comb:
            lines.append("| D | " + " | ".join(fmt(d_comb.get(k), k) for k in METRICS) + " |")
        if b_comb and d_comb:
            lines.append("| D−B | " + " | ".join(delta_cell(d_comb[k], b_comb[k], k) for k in METRICS) + " |")
        lines.append("")
        lines.append("### 逐序列")
        lines.append("")
        lines.append("| 序列 | HOTA B | HOTA D | ΔHOTA (pp) | IDF1 B | IDF1 D | AssA B | AssA D | IDSW B | IDSW D | Frag B | Frag D |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for sequence in sorted(set(b_per) & set(d_per)):
            b_row, d_row = b_per[sequence], d_per[sequence]
            diff = d_row["HOTA"] - b_row["HOTA"]
            rows.append((sequence, b_row, d_row, diff))
        rows.sort(key=lambda item: item[3], reverse=True)
        for sequence, b_row, d_row, diff in rows:
            lines.append(
                f"| {sequence} | {b_row['HOTA']:.4f} | {d_row['HOTA']:.4f} | {diff * 100:+.2f} | "
                f"{b_row['IDF1']:.4f} | {d_row['IDF1']:.4f} | "
                f"{b_row['AssA']:.4f} | {d_row['AssA']:.4f} | "
                f"{int(round(b_row['IDSW']))} | {int(round(d_row['IDSW']))} | "
                f"{int(round(b_row['Frag']))} | {int(round(d_row['Frag']))} |"
            )
        lines.append("")
        changed = [row for row in rows if abs(row[3]) > 0.001]
        lines.append(
            f"- |ΔHOTA|>0.001 的序列：{len(changed)}/{len(rows)}；"
            + ("、".join(
                f"{row[0]} {row[3] * 100:+.2f} pp（IDSW {int(round(row[1]['IDSW']))}→{int(round(row[2]['IDSW']))}，"
                f"Frag {int(round(row[1]['Frag']))}→{int(round(row[2]['Frag']))}）"
                for row in changed
            ) if changed else "无")
        )
        if rows:
            diffs = np.array([row[3] for row in rows])
            lines.append(
                f"- ΔHOTA：改善 {int((diffs > 0).sum())} 条、回退 {int((diffs < 0).sum())} 条、"
                f"其余 {int((diffs == 0).sum())} 条数值一致。"
            )
        lines.append("")
        lines.append("### 门分布（val 推理）")
        lines.append("")
        lines.append("| 组 | 序列数 | 运动门均值 | 运动门范围 | 外观门均值 | 外观门范围 | 恢复轨迹数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for label, suffix in (("B", b_key), ("D", d_key)):
            agg = aggregate_gates(evals[suffix]["gates"])
            if agg["sequences"]:
                lines.append(
                    f"| {label} | {agg['sequences']} | {agg['motion_mean']:.4f} | "
                    f"[{agg['motion_min']:.4f}, {agg['motion_max']:.4f}] | "
                    f"{agg['appearance_mean']:.4f} | "
                    f"[{agg['appearance_min']:.4f}, {agg['appearance_max']:.4f}] | "
                    f"{int(round(agg['recovered']))} |"
                )
        analysis_path = root / "gate_analysis.json"
        if analysis_path.is_file():
            analysis = load_json(analysis_path)
            lines.append("")
            lines.append("### 训练数据上的门变化（竞争 vs 普通事件）")
            lines.append("")
            lines.append("| 组 | 竞争事件数 | 竞争事件外观门 | 普通事件外观门 | 竞争事件运动门 |")
            lines.append("|---|---|---|---|---|")
            for name, entry in analysis.get("groups", {}).items():
                lines.append(
                    f"| {name} | {entry['valid_events']} | {entry['appearance_gate_valid']:.4f} | "
                    f"{entry['appearance_gate_other']:.4f} | {entry['motion_gate_valid']:.4f} |"
                )
            if "D" in analysis.get("groups", {}) and "B" in analysis.get("groups", {}):
                b_entry = analysis["groups"]["B"]
                d_entry = analysis["groups"]["D"]
                lines.append(
                    f"- D−B 竞争事件外观门：{d_entry['appearance_gate_valid'] - b_entry['appearance_gate_valid']:+.4f}；"
                    f"普通事件：{d_entry['appearance_gate_other'] - b_entry['appearance_gate_other']:+.4f}"
                )
        lines.append("")
        lines.append("### 耗时与训练速度")
        lines.append("")
        for group in ("appearance_rank", "appearance_row_column"):
            seconds = groups[group]["complete"].get("seconds")
            rows = [r for r in read_jsonl(root / "groups" / group / "metrics.jsonl") if "epoch" in r]
            if seconds:
                lines.append(f"- {group} 训练总耗时：{seconds:.0f} s")
            if rows:
                cold = [r["epoch_seconds"] for r in rows[:4]]
                steady = [r["epoch_seconds"] for r in rows[4:8]]
                tail = [r["epoch_seconds"] for r in rows[-20:]]
                lines.append(
                    f"  - 前 4 轮（冷启动）：{', '.join(f'{v:.1f}' for v in cold)} s；"
                    f"第 5–8 轮：{', '.join(f'{v:.1f}' for v in steady)} s；"
                    f"末 20 轮均值：{sum(tail) / len(tail):.2f} s/轮"
                )
        lines.append("")
        lines.append("### 复现命令")
        lines.append("")
        lines.append("```bash")
        init = f"--init {b_config.get('init')}" if b_config.get("init") else "--from-scratch"
        lines.append(
            f"bash tools/run_bd_mot_20261120.sh  # {dataset} 部分："
            f"B/D {init}, {b_config.get('epochs')} shard-epochs, batch {b_config.get('batch_size')}, "
            f"lr {b_config.get('lr')}"
        )
        lines.append("```")
        lines.append("")

    eval_log = exp / "logs" / "eval_driver.log"
    if eval_log.is_file():
        lines.append("## 评估耗时")
        lines.append("")
        for line in eval_log.read_text().splitlines():
            if "eval done in" in line:
                lines.append(f"- {line}")
        lines.append("")

    report = exp / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", report)


if __name__ == "__main__":
    main()
