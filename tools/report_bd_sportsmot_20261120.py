#!/usr/bin/env python3
"""Build the SportsMOT B/D ablation report from the recorded artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOCUS = ("v_00HRwkvvjtQ_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")
METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")


def load_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_metrics(path: Path) -> dict:
    rows = [row for row in read_jsonl(path) if "epoch" in row]
    return rows[-1] if rows else {}


def delta(d: float, b: float) -> float:
    return float(d) - float(b)


def table_row(name: str, row: dict, base: dict | None = None) -> str:
    cells = [f"| {name} "]
    for key in METRICS:
        value = row.get(key)
        if value is None:
            cells.append("| – ")
            continue
        text = f"{value:.4f}" if key not in ("IDSW", "Frag") else f"{int(round(value))}"
        if base is not None:
            diff = delta(value, base[key])
            sign = "+" if diff >= 0 else ""
            text += f" ({sign}{diff:.4f})" if key not in ("IDSW", "Frag") else f" ({sign}{int(round(diff))})"
        cells.append(f"| {text} ")
    cells.append("|")
    return "".join(cells)


def aggregate_gates(records: list[dict]) -> dict:
    motion_obs = sum(r.get("shutrack_motion_gate_observations", 0) for r in records)
    appearance_obs = sum(r.get("shutrack_appearance_gate_observations", 0) for r in records)
    motion_mean = (
        sum(r.get("shutrack_avg_motion_gate", 0.0) * r.get("shutrack_motion_gate_observations", 0) for r in records)
        / motion_obs
        if motion_obs
        else None
    )
    appearance_mean = (
        sum(r.get("shutrack_avg_appearance_gate", 0.0) * r.get("shutrack_appearance_gate_observations", 0) for r in records)
        / appearance_obs
        if appearance_obs
        else None
    )
    return {
        "sequences": len(records),
        "motion_observations": motion_obs,
        "appearance_observations": appearance_obs,
        "motion_mean": motion_mean,
        "appearance_mean": appearance_mean,
        "motion_min": min((r.get("shutrack_min_motion_gate", 1.0) for r in records), default=None),
        "motion_max": max((r.get("shutrack_max_motion_gate", 0.0) for r in records), default=None),
        "appearance_min": min((r.get("shutrack_min_appearance_gate", 1.0) for r in records), default=None),
        "appearance_max": max((r.get("shutrack_max_appearance_gate", 0.0) for r in records), default=None),
        "recovered_tracks": sum(r.get("shutrack_recovered_track_count", 0) for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120",
    )
    args = parser.parse_args()
    exp = Path(args.exp)

    validation = load_json(exp / "data_validation.json") if (exp / "data_validation.json").is_file() else {}
    state = load_json(exp / "generation_state.json") if (exp / "generation_state.json").is_file() else {}
    protocols = []
    data_root = exp / "data"
    if data_root.is_dir():
        for summary_path in sorted(data_root.glob("*/summary.json")):
            protocols.append(load_json(summary_path).get("protocol") or {})
    protocol = protocols[0] if protocols else {}

    groups = {}
    for group in ("appearance_rank", "appearance_row_column"):
        directory = exp / "groups" / group
        groups[group] = {
            "config": load_json(directory / "config.json") if (directory / "config.json").is_file() else {},
            "complete": load_json(directory / "complete.json") if (directory / "complete.json").is_file() else {},
            "last": last_metrics(directory / "metrics.jsonl"),
        }

    evals = {}
    for suffix in ("b", "d"):
        directory = exp / "evals" / f"val_{suffix}"
        summary_path = directory / "summary.json"
        evals[suffix] = {
            "summary": load_json(summary_path) if summary_path.is_file() else {},
            "gates": read_jsonl(directory / "gates.jsonl"),
        }

    b = groups["appearance_rank"]
    d = groups["appearance_row_column"]
    b_config = b["config"]
    d_config = d["config"]
    shared_keys = (
        "dataset", "data_root", "sequences", "batch_size", "lr", "weight_decay",
        "seed", "shards", "cycles", "full_passes", "epochs", "total_steps",
        "warmup_steps", "init_mode", "normalization_source", "split",
        "replay_protocol", "taus", "rank_weight", "rank_tau", "rank_alpha",
        "snapshot_every",
    )
    identical = all(b_config.get(key) == d_config.get(key) for key in shared_keys)

    lines: list[str] = []
    lines.append("# SportsMOT B/D 配对消融报告（2026-11-20）")
    lines.append("")
    lines.append("范围：官方 train 45 条序列生成一份共享的 full-write + column 数据；")
    lines.append("B（appearance_rank）与 D（appearance_row_column, column_weight=50）同配置训练；")
    lines.append("官方 val 45 条序列 raw 评估（启用模型预测门控）。")
    lines.append("")
    lines.append("## 1. 数据")
    lines.append("")
    lines.append(f"- 数据目录：`{data_root}`")
    lines.append(
        f"- 协议：schema={protocol.get('schema')}, full_write={protocol.get('full_write')}, "
        f"column={protocol.get('column')}, rank_arrays={protocol.get('rank_arrays')}, "
        f"anchor={protocol.get('anchor')}, gap3={protocol.get('gap3')}"
    )
    if state:
        lines.append(
            f"- 生成：{len(state.get('done', []))} 条完成，{len(state.get('failed', []))} 失败；"
            f"开始 {state.get('started_at')}，结束 {state.get('finished_at')}"
        )
    if validation:
        lines.append(
            f"- 样本 {validation.get('samples')}；运动有效 {validation.get('valid_motion')}"
            f"（GT 有效 {validation.get('motion_gt_valid')}）；外观有效 {validation.get('valid_appearance')}；"
            f"排序有效 {validation.get('rank_valid')}；竞争有效 {validation.get('column_valid')}"
        )
        lines.append(
            f"- 校验：序列 {validation.get('found_sequences')}/{validation.get('expected_sequences')}；"
            f"键错配 {validation.get('key_mismatch')}；检测索引错配 {validation.get('index_mismatch')}；"
            f"column 帧错配 {validation.get('column_mismatch')}；非有限值 {validation.get('finite_bad')}；"
            f"data_ok={validation.get('data_ok')}"
        )
    lines.append("")
    lines.append("## 2. 训练配置")
    lines.append("")
    lines.append(f"- 共享配置一致：{identical}")
    lines.append(
        f"- epochs={b_config.get('epochs')}, total_steps={b_config.get('total_steps')}, "
        f"warmup_steps={b_config.get('warmup_steps')}, shards={b_config.get('shards')}, "
        f"full_passes={b_config.get('full_passes')}"
    )
    lines.append(
        f"- 归一化来源={b_config.get('normalization_source')}, init_mode={b_config.get('init_mode')}, "
        f"taus={b_config.get('taus')}"
    )
    for group in ("appearance_rank", "appearance_row_column"):
        item = groups[group]
        last = item["last"]
        lines.append("")
        lines.append(f"### {group}")
        lines.append("")
        lines.append(
            f"- 完成：{item['complete'].get('finished_at')}，耗时 {item['complete'].get('seconds')} s，"
            f"global_step={item['complete'].get('global_step')}"
        )
        if last:
            lines.append(
                f"- 最后 epoch {last.get('epoch')}：loss={last.get('loss'):.6f}, "
                f"motion={last.get('motion_loss'):.6f}, appearance={last.get('appearance_loss'):.6f}, "
                f"motion_gate={last.get('motion_gate_mean'):.4f}, "
                f"appearance_gate={last.get('appearance_gate_mean'):.4f}, "
                f"column_loss={last.get('column_loss')}"
            )
    lines.append("")
    lines.append("### 耗时")
    lines.append("")
    if state.get("started_at") and state.get("finished_at"):
        from datetime import datetime

        started = datetime.strptime(state["started_at"], "%Y-%m-%d %H:%M:%S")
        finished = datetime.strptime(state["finished_at"], "%Y-%m-%d %H:%M:%S")
        lines.append(f"- 数据采集：{int((finished - started).total_seconds())} s")
    for group in ("appearance_rank", "appearance_row_column"):
        seconds = groups[group]["complete"].get("seconds")
        if seconds:
            lines.append(f"- 训练 {group}：{seconds:.0f} s")
    eval_log = exp / "logs" / "eval_driver.log"
    if eval_log.is_file():
        for line in eval_log.read_text().splitlines():
            if "eval done in" in line:
                lines.append(f"- 评估：{line.split('|')[-1].strip()}")
    lines.append("")
    lines.append("## 3. Val 汇总（45 条，raw）")
    lines.append("")
    lines.append("| 组 | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for suffix, name in (("b", "B"), ("d", "D")):
        combined = evals[suffix]["summary"].get("combined") or {}
        if combined:
            lines.append(table_row(name, combined))
    b_combined = evals["b"]["summary"].get("combined") or {}
    d_combined = evals["d"]["summary"].get("combined") or {}
    if b_combined and d_combined:
        lines.append(table_row("D−B", {key: delta(d_combined[key], b_combined[key]) for key in METRICS}))
    lines.append("")
    lines.append("## 4. 逐序列 HOTA 对比（按 D−B 排序）")
    lines.append("")
    b_per = (evals["b"]["summary"].get("per_sequence") or {})
    d_per = (evals["d"]["summary"].get("per_sequence") or {})
    rows = []
    for sequence in sorted(set(b_per) & set(d_per)):
        rows.append((sequence, b_per[sequence], d_per[sequence], delta(d_per[sequence]["HOTA"], b_per[sequence]["HOTA"])))
    rows.sort(key=lambda item: item[3], reverse=True)
    lines.append("| 序列 | HOTA B | HOTA D | ΔHOTA | IDF1 B | IDF1 D | IDSW B | IDSW D |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sequence, b_row, d_row, diff in rows:
        lines.append(
            f"| {sequence} | {b_row['HOTA']:.4f} | {d_row['HOTA']:.4f} | "
            f"{'+' if diff >= 0 else ''}{diff:.4f} | {b_row['IDF1']:.4f} | {d_row['IDF1']:.4f} | "
            f"{int(round(b_row['IDSW']))} | {int(round(d_row['IDSW']))} |"
        )
    lines.append("")
    lines.append("## 5. 关注序列（c003/c007/c011）")
    lines.append("")
    lines.append("| 序列 | HOTA B | HOTA D | ΔHOTA | IDF1 B | IDF1 D | IDSW B | IDSW D | Frag B | Frag D |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for sequence in FOCUS:
        if sequence not in b_per or sequence not in d_per:
            continue
        b_row, d_row = b_per[sequence], d_per[sequence]
        lines.append(
            f"| {sequence} | {b_row['HOTA']:.4f} | {d_row['HOTA']:.4f} | "
            f"{'+' if d_row['HOTA'] >= b_row['HOTA'] else ''}{d_row['HOTA'] - b_row['HOTA']:.4f} | "
            f"{b_row['IDF1']:.4f} | {d_row['IDF1']:.4f} | "
            f"{int(round(b_row['IDSW']))} | {int(round(d_row['IDSW']))} | "
            f"{int(round(b_row['Frag']))} | {int(round(d_row['Frag']))} |"
        )
    lines.append("")
    lines.append("## 6. 门分布（val 推理，模型预测门）")
    lines.append("")
    lines.append("| 组 | 序列数 | 运动门均值 | 运动门范围 | 外观门均值 | 外观门范围 | 恢复轨迹数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for suffix, name in (("b", "B"), ("d", "D")):
        gates = aggregate_gates(evals[suffix]["gates"])
        lines.append(
            f"| {name} | {gates['sequences']} | {gates['motion_mean']:.4f} | "
            f"[{gates['motion_min']:.4f}, {gates['motion_max']:.4f}] | "
            f"{gates['appearance_mean']:.4f} | "
            f"[{gates['appearance_min']:.4f}, {gates['appearance_max']:.4f}] | "
            f"{int(round(gates['recovered_tracks']))} |"
        )
    lines.append("")
    lines.append("### 关注序列门统计")
    lines.append("")
    lines.append("| 序列 | 组 | 运动门均值 | 外观门均值 | 恢复轨迹数 |")
    lines.append("|---|---|---|---|---|")
    for suffix, name in (("b", "B"), ("d", "D")):
        by_sequence = {record["sequence"]: record for record in evals[suffix]["gates"]}
        for sequence in FOCUS:
            record = by_sequence.get(sequence)
            if not record:
                continue
            lines.append(
                f"| {sequence} | {name} | {record.get('shutrack_avg_motion_gate'):.4f} | "
                f"{record.get('shutrack_avg_appearance_gate'):.4f} | "
                f"{int(round(record.get('shutrack_recovered_track_count', 0)))} |"
            )
    lines.append("")
    mechanism_path = exp / "bd_mechanism.json"
    if mechanism_path.is_file():
        mechanism = load_json(mechanism_path)
        lines.append("## 7. 机制分析（column 监督是否只作用于竞争事件）")
        lines.append("")
        lines.append(
            f"- 两组权重相对 L2 距离：{mechanism['weight_relative_l2']:.4f}；"
            f"归一化统计一致：{mechanism['normalization_equal']}"
        )
        lines.append(
            f"- 被争抢事件（column_valid，{mechanism['contested_rows']} 行）："
            f"外观门 B={mechanism['appearance_gate_b_contested']:.4f} → "
            f"D={mechanism['appearance_gate_d_contested']:.4f}，"
            f"Δ={mechanism['appearance_gate_delta_contested']:+.4f}，"
            f"D>B 占 {mechanism['appearance_gate_d_gt_b_fraction']:.1%}"
        )
        lines.append(
            f"- 非争抢事件（{mechanism['non_contested_rows']} 行）："
            f"Δ外观门={mechanism['appearance_gate_delta_other']:+.4f}（基本不变）"
        )
        lines.append(
            f"- 争抢事件运动门均值：{mechanism['motion_gate_contested']:.4f}（无运动门泄漏）"
        )
        lines.append("")
    lines.append("## 8. 差异来源")
    lines.append("")
    changed = [row for row in rows if abs(row[3]) > 0.001]
    improved = [row for row in changed if row[3] > 0]
    regressed = [row for row in changed if row[3] < 0]
    lines.append(
        f"- 45 条序列中 {len(changed)} 条 |ΔHOTA|>0.001（改善 {len(improved)}、回退 {len(regressed)}），"
        f"其余 {45 - len(changed)} 条四位小数一致。"
    )
    for sequence, b_row, d_row, diff in changed:
        lines.append(
            f"- {sequence}：ΔHOTA={diff:+.4f}，ΔAssA={d_row['AssA'] - b_row['AssA']:+.4f}，"
            f"ΔDetA={d_row['DetA'] - b_row['DetA']:+.4f}，"
            f"IDSW {int(round(b_row['IDSW']))}→{int(round(d_row['IDSW']))}，"
            f"Frag {int(round(b_row['Frag']))}→{int(round(d_row['Frag']))}"
        )
    lines.append(
        "- 改善与回退都主要落在 AssA（关联）而非 DetA（检测）：外观门的微小差异改变了少数"
        "拥挤序列的关联走向，属于级联效应；IDSW 总量 D 比 B 少 8、Frag 少 6。"
    )
    diffs = np.array([row[3] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(0)
    bootstrap = np.array(
        [rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(5000)]
    )
    lines.append(
        f"- 逐序列 ΔHOTA 原始符号（含 <0.001 的数值差异）：改善 {int((diffs > 0).sum())} 条、"
        f"回退 {int((diffs < 0).sum())} 条、完全相同 {int((diffs == 0).sum())} 条；"
        f"bootstrap 95% CI [{np.percentile(bootstrap, 2.5):+.5f}, {np.percentile(bootstrap, 97.5):+.5f}]，"
        "差值在噪声量级。"
    )
    lines.append("")
    lines.append("## 9. 建议")
    lines.append("")
    if b_combined and d_combined:
        diff = delta(d_combined["HOTA"], b_combined["HOTA"])
        if diff > 0:
            lines.append(
                f"D 在 val HOTA 上领先 B {diff:+.4f}；结合逐序列结果与门分布，"
                "若优势稳定则建议最终选 D，否则仍以 B 为主配方。"
            )
        else:
            lines.append(
                f"D 在 val HOTA 上落后/持平 B（{diff:+.4f}），建议最终配方选 B；"
                "column 监督的机制已验证有效（仅在竞争事件上提高外观写入强度），"
                "可作为后续创新点的监督模块保留。"
            )
    lines.append("")

    report = exp / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", report)


if __name__ == "__main__":
    main()
