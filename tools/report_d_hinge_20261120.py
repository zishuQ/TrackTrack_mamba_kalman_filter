#!/usr/bin/env python3
"""Three-way B / D / D-hinge comparison report for SportsMOT val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOCUS = (
    "v_BgwzTUxJaeU_c012",
    "v_0kUtTtmLaJA_c005",
    "v_4r8QL_wglzQ_c001",
    "v_9MHDmAMxO5I_c003",
)
METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")
FORMAT = {"HOTA": ".4f", "MOTA": ".4f", "IDF1": ".4f", "DetA": ".4f", "AssA": ".4f",
          "IDSW": ".0f", "Frag": ".0f"}


def load_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell(value, key, base=None, signed=False) -> str:
    if value is None:
        return "–"
    text = format(value, FORMAT[key])
    if base is not None:
        diff = value - base
        text += f" ({diff:+.4f})" if key not in ("IDSW", "Frag") else f" ({diff:+.0f})"
    return text


def aggregate_gates(records: list[dict]) -> dict:
    motion_obs = sum(r.get("shutrack_motion_gate_observations", 0) for r in records)
    appearance_obs = sum(r.get("shutrack_appearance_gate_observations", 0) for r in records)
    motion_mean = (
        sum(r.get("shutrack_avg_motion_gate", 0.0) * r.get("shutrack_motion_gate_observations", 0)
            for r in records) / motion_obs if motion_obs else None
    )
    appearance_mean = (
        sum(r.get("shutrack_avg_appearance_gate", 0.0) * r.get("shutrack_appearance_gate_observations", 0)
            for r in records) / appearance_obs if appearance_obs else None
    )
    return {
        "sequences": len(records),
        "motion_mean": motion_mean,
        "appearance_mean": appearance_mean,
        "motion_min": min((r.get("shutrack_min_motion_gate", 1.0) for r in records), default=None),
        "motion_max": max((r.get("shutrack_max_motion_gate", 0.0) for r in records), default=None),
        "appearance_min": min((r.get("shutrack_min_appearance_gate", 1.0) for r in records), default=None),
        "appearance_max": max((r.get("shutrack_max_appearance_gate", 0.0) for r in records), default=None),
        "recovered": sum(r.get("shutrack_recovered_track_count", 0) for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bd-exp", default="outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120")
    parser.add_argument("--hinge-exp", default="outputs/agentguard/experiments/shutrack_d_hinge_sportsmot_20261120")
    args = parser.parse_args()
    bd = Path(args.bd_exp)
    hinge = Path(args.hinge_exp)

    summaries = {
        "B": load_json(bd / "evals/val_b/summary.json"),
        "D": load_json(bd / "evals/val_d/summary.json"),
        "H": load_json(hinge / "evals/val_hinge/summary.json"),
    }
    gates = {
        "B": read_jsonl(bd / "evals/val_b/gates.jsonl"),
        "D": read_jsonl(bd / "evals/val_d/gates.jsonl"),
        "H": read_jsonl(hinge / "evals/val_hinge/gates.jsonl"),
    }
    metrics_rows = [row for row in read_jsonl(hinge / "groups/appearance_row_column/metrics.jsonl") if "epoch" in row]
    complete = load_json(hinge / "groups/appearance_row_column/complete.json")
    config = load_json(hinge / "groups/appearance_row_column/config.json")
    preflight = load_json(hinge / "preflight.json") if (hinge / "preflight.json").is_file() else {}
    scan = load_json(bd / "d_hinge_margin_scan.json") if (bd / "d_hinge_margin_scan.json").is_file() else {}

    combined = {name: data.get("combined") or {} for name, data in summaries.items()}
    per = {name: data.get("per_sequence") or {} for name, data in summaries.items()}

    lines: list[str] = []
    lines.append("# SportsMOT B / D / D-hinge 三组对比报告（2026-11-20）")
    lines.append("")
    lines.append("D-hinge 唯一改动：column penalty 由 `softplus((cost-rival)/tau)` 改为 "
                 "`relu((cost-rival)/tau)`；其余数据、标签、归一化、基座 loss 与训练配置不变。")
    lines.append("")
    lines.append("## 1. 训练")
    lines.append("")
    lines.append(
        f"- 配置：column_hinge={config.get('column_hinge')}, column_weight={config.get('column_weight')}, "
        f"epochs={config.get('epochs')}, total_steps={config.get('total_steps')}, "
        f"warmup_steps={config.get('warmup_steps')}, seed={config.get('seed')}, "
        f"data_root=`{config.get('data_root')}`"
    )
    lines.append(
        f"- 完成：{complete.get('finished_at')}，耗时 {complete.get('seconds'):.0f} s，"
        f"global_step={complete.get('global_step')}"
    )
    if metrics_rows:
        valid_counts = [r.get("column_valid_count", 0) for r in metrics_rows]
        active_fracs = [r.get("column_active_fraction") for r in metrics_rows if r.get("column_active_fraction") is not None]
        total_valid = sum(valid_counts)
        weighted_active = (
            sum(r.get("column_active_fraction", 0.0) * r.get("column_valid_count", 0) for r in metrics_rows)
            / max(total_valid, 1)
        )
        lines.append(
            f"- 训练期 column：累计有效事件 {total_valid}；加权平均产生梯度比例 {weighted_active:.4f}；"
            f"首/末 epoch active_fraction={active_fracs[0]:.4f}/{active_fracs[-1]:.4f}"
            if active_fracs else
            f"- 训练期 column：累计有效事件 {total_valid}"
        )
        last = metrics_rows[-1]
        lines.append(
            f"- 最后 epoch {last.get('epoch')}：loss={last.get('loss'):.6f}, "
            f"column_loss={last.get('column_loss')}, "
            f"appearance_gate={last.get('appearance_gate_mean'):.4f}, "
            f"motion_gate={last.get('motion_gate_mean'):.4f}"
        )
    lines.append("")
    lines.append("## 2. 训练前验证")
    lines.append("")
    if preflight:
        lines.append(f"- 预检整体通过：{preflight.get('all_ok')}")
        for sequence, result in preflight.get("per_sequence", {}).items():
            lines.append(
                f"- {sequence}：valid={result['valid_events']}, won={result['won_events']}, "
                f"active={result['active_events']}, 获胜事件惩罚/梯度最大值="
                f"{result['won_penalty_max']:.1e}/{result['won_appearance_grad_max']:.1e}, "
                f"失败事件梯度有限={result['active_grad_finite']}, "
                f"非零比例={result['active_grad_nonzero_fraction']}, "
                f"运动门梯度最大值={result['column_motion_grad_max']:.1e}, "
                f"softplus 参考一致={result['softplus_reference_equal']}"
            )
    if scan:
        lines.append(
            f"- 全量 margin 扫描（原 D 权重）：45 条 train、{scan['valid']} 个有效事件，"
            f"active {scan['active']}（{scan['active_fraction']:.2%}），"
            f"margin 范围 [{scan['margin_min']:.2f}, {scan['margin_max']:.2f}]。"
        )
    lines.append("")
    lines.append("## 3. Val 汇总（45 条，raw）")
    lines.append("")
    lines.append("| 组 | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ("B", "D", "H"):
        row = combined[name]
        lines.append("| " + name + " | " + " | ".join(cell(row.get(k), k) for k in METRICS) + " |")
    for name, base in (("H−B", "B"), ("H−D", "D")):
        row = combined["H"]
        base_row = combined[base]
        cells = []
        for key in METRICS:
            diff = row[key] - base_row[key]
            cells.append(f"{diff:+.4f}" if key not in ("IDSW", "Frag") else f"{diff:+.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 4. 逐序列对比（按 H−B 排序）")
    lines.append("")
    lines.append("| 序列 | HOTA B | HOTA D | HOTA H | ΔH−B | ΔH−D | IDSW B/D/H |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = []
    for sequence in sorted(set(per["B"]) & set(per["D"]) & set(per["H"])):
        rows.append((
            sequence, per["B"][sequence], per["D"][sequence], per["H"][sequence],
            per["H"][sequence]["HOTA"] - per["B"][sequence]["HOTA"],
            per["H"][sequence]["HOTA"] - per["D"][sequence]["HOTA"],
        ))
    rows.sort(key=lambda item: item[4], reverse=True)
    for sequence, b_row, d_row, h_row, dh_b, dh_d in rows:
        lines.append(
            f"| {sequence} | {b_row['HOTA']:.4f} | {d_row['HOTA']:.4f} | {h_row['HOTA']:.4f} | "
            f"{dh_b:+.4f} | {dh_d:+.4f} | "
            f"{int(round(b_row['IDSW']))}/{int(round(d_row['IDSW']))}/{int(round(h_row['IDSW']))} |"
        )
    lines.append("")
    lines.append("## 5. 重点复查序列")
    lines.append("")
    lines.append("| 序列 | HOTA B/D/H | ΔH−B | ΔH−D | IDF1 B/D/H | AssA B/D/H | IDSW B/D/H | Frag B/D/H |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sequence in FOCUS:
        if sequence not in per["B"]:
            continue
        b_row, d_row, h_row = per["B"][sequence], per["D"][sequence], per["H"][sequence]
        lines.append(
            f"| {sequence} | {b_row['HOTA']:.4f}/{d_row['HOTA']:.4f}/{h_row['HOTA']:.4f} | "
            f"{h_row['HOTA'] - b_row['HOTA']:+.4f} | {h_row['HOTA'] - d_row['HOTA']:+.4f} | "
            f"{b_row['IDF1']:.4f}/{d_row['IDF1']:.4f}/{h_row['IDF1']:.4f} | "
            f"{b_row['AssA']:.4f}/{d_row['AssA']:.4f}/{h_row['AssA']:.4f} | "
            f"{int(round(b_row['IDSW']))}/{int(round(d_row['IDSW']))}/{int(round(h_row['IDSW']))} | "
            f"{int(round(b_row['Frag']))}/{int(round(d_row['Frag']))}/{int(round(h_row['Frag']))} |"
        )
    lines.append("")
    lines.append("## 6. Val 门分布（模型预测门）")
    lines.append("")
    lines.append("| 组 | 序列数 | 运动门均值 | 运动门范围 | 外观门均值 | 外观门范围 | 恢复轨迹数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("B", "D", "H"):
        agg = aggregate_gates(gates[name])
        lines.append(
            f"| {name} | {agg['sequences']} | {agg['motion_mean']:.4f} | "
            f"[{agg['motion_min']:.4f}, {agg['motion_max']:.4f}] | "
            f"{agg['appearance_mean']:.4f} | "
            f"[{agg['appearance_min']:.4f}, {agg['appearance_max']:.4f}] | "
            f"{int(round(agg['recovered']))} |"
        )
    lines.append("")
    lines.append("## 7. 耗时")
    lines.append("")
    lines.append(f"- D-hinge 训练：{complete.get('seconds', 0):.0f} s")
    eval_log = hinge / "logs" / "eval_driver.log"
    if eval_log.is_file():
        for line in eval_log.read_text().splitlines():
            if "eval done in" in line:
                lines.append(f"- 评估：{line.split('|')[-1].strip()}")
    lines.append("")

    diffs_hb = np.array([row[4] for row in rows])
    changed = [row for row in rows if abs(row[4]) > 0.001]
    lines.append("## 8. 建议")
    lines.append("")
    lines.append(
        f"H−B 的逐序列 ΔHOTA：改善 {int((diffs_hb > 0).sum())}、回退 {int((diffs_hb < 0).sum())}、"
        f"完全相同 {int((diffs_hb == 0).sum())}；|Δ|>0.001 的序列 {len(changed)} 条。"
    )
    h_b = combined["H"]["HOTA"] - combined["B"]["HOTA"]
    h_d = combined["H"]["HOTA"] - combined["D"]["HOTA"]
    lines.append(
        f"汇总 HOTA：H 相对 B {h_b:+.4f}，相对 D {h_d:+.4f}。"
    )
    if h_b >= 0 and h_d > 0:
        lines.append("hinge 改动在两个基线上都不劣于对照，可考虑保留。")
    elif h_b < 0 and h_d <= 0:
        lines.append("hinge 改动未带来增益，建议不保留，继续沿用原 softplus D 或 B。")
    else:
        lines.append("hinge 改动表现混合（在一个基线上更优、另一个更差），建议结合其训练信号稀疏性谨慎保留。")
    lines.append("")

    out = hinge / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", out)


if __name__ == "__main__":
    main()
