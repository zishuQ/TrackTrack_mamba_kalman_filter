#!/usr/bin/env python3
"""Summarize the 4-D speed group against the joint and centre-speed groups."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
OLD = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
MID = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119"
NEW = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_speed4_20261119"
VAL_MAP = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
GROUPS = ("joint", "motion_diou_speed", "motion_diou_speed4")
FOCUS = (
    "v_9MHDmAMxO5I_c003",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c011",
)


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def parse_logs() -> dict:
    logs = {
        "joint": OLD / "evals/motion_diou_appearance_score_val.log",
        "motion_diou_speed": MID / "evals/motion_diou_speed_val.log",
        "motion_diou_speed4": NEW / "evals/motion_diou_speed4_val.log",
    }
    command = [
        sys.executable,
        str(WORKTREE / "tools/parse_val_logs.py"),
        "--out",
        str(NEW / "evals/parsed_all.json"),
    ]
    for name, path in logs.items():
        command += ["--log", f"{name}={path}"]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return json.loads((NEW / "evals/parsed_all.json").read_text())


def result_dirs() -> dict[str, Path]:
    return {
        "joint": OLD
        / "evals/motion_diou_appearance_score_val/sportsmot_val_0.80_motion_diou_appearance_score_val_shutrack_v2",
        "motion_diou_speed": MID
        / "evals/motion_diou_speed_val/sportsmot_val_0.80_motion_diou_speed_val_shutrack_v2",
        "motion_diou_speed4": NEW
        / "evals/motion_diou_speed4_val/sportsmot_val_0.80_motion_diou_speed4_val_shutrack_v2",
    }


def run_clear() -> dict:
    command = [
        sys.executable,
        str(WORKTREE / "tools/clear_id_analysis.py"),
        "--sequences-file",
        str(VAL_MAP),
        "--out",
        str(NEW / "evals/clear_id_all.json"),
    ]
    for name, directory in result_dirs().items():
        command += ["--result", f"{name}={directory}"]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return json.loads((NEW / "evals/clear_id_all.json").read_text())


def aggregate_runs(clear: dict) -> dict[str, dict]:
    summary = {}
    for name in GROUPS:
        durations = []
        recovered = []
        for sequence, groups in clear["sequences"].items():
            payload = groups.get(name, {})
            for item in payload.get("wrong_identity_runs", []):
                durations.append(item["duration"])
                if item["duration"] >= 5:
                    recovered.append(bool(item.get("recovered")))
        array = np.asarray(durations, dtype=np.float64)
        long_array = array[array >= 5] if array.size else array
        summary[name] = {
            "wrong_frames_total": int(array.sum()) if array.size else 0,
            "runs": int(array.size),
            "runs_ge5": int(long_array.size),
            "runs_ge20": int((array >= 20).sum()) if array.size else 0,
            "runs_ge100": int((array >= 100).sum()) if array.size else 0,
            "median_long_duration": float(np.median(long_array)) if long_array.size else None,
            "recovered_fraction_ge5": float(np.mean(recovered)) if recovered else None,
        }
    return summary


def focus_clear(clear: dict) -> dict:
    focus = {}
    for sequence in FOCUS:
        entry = {}
        for name in GROUPS:
            runs = clear["sequences"].get(sequence, {}).get(name, {}).get(
                "wrong_identity_runs", []
            )
            entry[name] = [run_item for run_item in runs if run_item["duration"] >= 5]
        focus[sequence] = entry
    return focus


def paired_gap() -> dict:
    path = NEW / "evals/paired_gap.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def validation() -> dict:
    path = NEW / "speed4_validation.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def fmt(value, digits=4):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def main() -> None:
    (NEW / "evals").mkdir(parents=True, exist_ok=True)
    parsed = parse_logs()
    clear = run_clear()
    aggregate = aggregate_runs(clear)
    focus = focus_clear(clear)
    gap = paired_gap()
    valid = validation()

    combined = {name: parsed[name]["combined"] for name in GROUPS}
    per_sequence = {name: parsed[name]["per_sequence"] for name in GROUPS}
    joint = combined["joint"]
    center = combined["motion_diou_speed"]
    speed4 = combined["motion_diou_speed4"]

    lines = ["# 四维速度实验（vx/vy/vw/vh）", ""]
    lines.append("| group | HOTA | AssA | IDF1 | DetA | MOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name in GROUPS:
        row = combined[name]
        lines.append(
            f"| {name} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | "
            f"{row['DetA']:.4f} | {row['MOTA']:.4f} |"
        )
    lines.append("")
    lines.append("## 相对联合组与中心速度组的差值")
    lines.append("")
    lines.append("| group | vs | ΔHOTA | ΔAssA | ΔIDF1 | ΔDetA |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("motion_diou_speed", "motion_diou_speed4"):
        for base_name, base in (("joint", joint), ("motion_diou_speed", center)):
            if name == base_name:
                continue
            row = combined[name]
            lines.append(
                f"| {name} | {base_name} | {row['HOTA']-base['HOTA']:+.4f} | "
                f"{row['AssA']-base['AssA']:+.4f} | {row['IDF1']-base['IDF1']:+.4f} | "
                f"{row['DetA']-base['DetA']:+.4f} |"
            )
    lines.append("")

    lines.append("## 数据与梯度核对")
    lines.append("")
    data = valid.get("data", {})
    batch = valid.get("batch", {})
    if data:
        lines.append(
            f"- 数据对齐：{data.get('sequences')} 条序列、{data.get('samples')} 个样本；"
            f"行/标签/中心端点/GT 中心/mask 不一致序列数 "
            f"{data.get('row_mismatch')}/{data.get('label_mismatch')}/"
            f"{data.get('center_endpoint_mismatch')}/"
            f"{data.get('gt_center_mismatch')}/{data.get('mask_mismatch')}。"
        )
        lines.append(
            f"- 宽高标签：相邻 mask {data.get('size_adj_valid')}，中心差分 mask "
            f"{data.get('size_central_valid')}，共同事件 {data.get('size_shared')}；"
            f"共同事件上两种标签平均绝对差 (vw,vh) = "
            f"{fmt((data.get('size_label_mean_abs_diff') or [None])[0])}/"
            f"{fmt((data.get('size_label_mean_abs_diff') or [None, None])[1])}。"
        )
        lines.append(
            f"- 端点插值误差：4D 速度 {data.get('velocity4_mid_error_max'):.2e}，"
            f"框 {data.get('box_mid_error_max'):.2e}。"
        )
    if batch:
        lines.append(
            f"- 真实 batch 梯度：框 DIoU 运动门控梯度均值 {batch.get('box_grad_abs_mean'):.2e}，"
            f"中心 L1（权重 50）{batch.get('center_grad_abs_mean_at_weight'):.2e}，"
            f"宽高 L1（单位权重）{batch.get('size_grad_abs_mean_unit'):.2e}；"
            f"参数梯度建议宽高系数（α=0.5）{batch.get('suggested_size_weight_alpha50_param'):.1f}。"
        )
        lines.append(
            f"- size_weight=0 与中心速度组：loss/门控梯度/参数梯度最大差 "
            f"{batch.get('zero_equiv_loss_max'):.2e}/"
            f"{batch.get('zero_equiv_grad_max'):.2e}/"
            f"{batch.get('zero_equiv_param_max'):.2e}，"
            f"完全一致={batch.get('zero_equivalence_exact')}。"
        )
        lines.append(
            f"- 宽高调整空间：GT 宽高速度平均幅值 {batch.get('size_target_abs_mean'):.5f}，"
            f"端点差 |g1-g0| = {batch.get('size_delta_abs_mean'):.5f}；"
            f"宽高 L1 g0/g1 = {batch.get('size_at_g0'):.5f}/{batch.get('size_at_g1'):.5f}。"
        )
    lines.append("")

    lines.append("## 逐序列 HOTA")
    lines.append("")
    for name in ("motion_diou_speed", "motion_diou_speed4"):
        for base_name, base_seq in (
            ("joint", per_sequence["joint"]),
            ("motion_diou_speed", per_sequence["motion_diou_speed"]),
        ):
            if name == base_name:
                continue
            items = sorted(
                (
                    (per_sequence[name][seq]["HOTA"] - base_seq[seq]["HOTA"], seq)
                    for seq in base_seq
                ),
                reverse=True,
            )
            improved = sum(1 for value, _ in items if value > 0)
            regressed = sum(1 for value, _ in items if value < 0)
            lines.append(
                f"{name} 相对 {base_name}：改善 {improved}/45，回退 {regressed}/45；"
                f"最大涨幅 {items[0][1]} {items[0][0]:+.4f}，最大跌幅 {items[-1][1]} {items[-1][0]:+.4f}。"
            )
        lines.append("")
    lines.append("| sequence | joint HOTA/AssA | center HOTA/AssA | speed4 HOTA/AssA | Δ vs center |")
    lines.append("|---|---|---|---|---|")
    for sequence in sorted(per_sequence["joint"]):
        lines.append(
            f"| {sequence} | {per_sequence['joint'][sequence]['HOTA']:.4f}/"
            f"{per_sequence['joint'][sequence]['AssA']:.4f} | "
            f"{per_sequence['motion_diou_speed'][sequence]['HOTA']:.4f}/"
            f"{per_sequence['motion_diou_speed'][sequence]['AssA']:.4f} | "
            f"{per_sequence['motion_diou_speed4'][sequence]['HOTA']:.4f}/"
            f"{per_sequence['motion_diou_speed4'][sequence]['AssA']:.4f} | "
            f"{per_sequence['motion_diou_speed4'][sequence]['HOTA'] - per_sequence['motion_diou_speed'][sequence]['HOTA']:+.4f} |"
        )
    lines.append("")

    lines.append("## 45 序列身份错误持续时间（TrackEval CLEAR，前帧匹配优先）")
    lines.append("")
    lines.append("| group | wrong frames | runs ≥5 | ≥20 | ≥100 | median ≥5 | recovered ≥5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in GROUPS:
        row = aggregate[name]
        lines.append(
            f"| {name} | {row['wrong_frames_total']} | {row['runs_ge5']} | "
            f"{row['runs_ge20']} | {row['runs_ge100']} | "
            f"{fmt(row['median_long_duration'], 1)} | {fmt(row['recovered_fraction_ge5'], 3)} |"
        )
    lines.append("")

    lines.append("## 关注序列长身份错误（≥5 帧）")
    lines.append("")
    for sequence in FOCUS:
        lines.append(f"### {sequence}")
        lines.append("")
        lines.append("| group | gt | track | start | end | duration | recovered |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in GROUPS:
            for item in focus[sequence][name]:
                lines.append(
                    f"| {name} | gt{item['gt_id']} | {item['reference_track']}->{item['track_id']} | "
                    f"{item['start']} | {item['end']} | {item['duration']} | "
                    f"{item.get('recovered')} |"
                )
        lines.append("")

    lines.append("## 配对漏检诊断（同一最后匹配事件、同一无检测区间）")
    lines.append("")
    if gap:
        for gap_length in ("1", "3", "5"):
            lines.append(f"### 漏检 {gap_length} 帧后的预测")
            lines.append("")
            lines.append("| group | n | centre error | size error | IoU |")
            lines.append("|---|---|---|---|---|")
            for name in GROUPS:
                entry = gap.get("groups", {}).get(name, {}).get(gap_length)
                if not entry:
                    continue
                lines.append(
                    f"| {name} | {entry['center']['count']} | "
                    f"{fmt(entry['center']['mean'])} | {fmt(entry['size']['mean'])} | "
                    f"{fmt(entry['iou']['mean'])} |"
                )
            lines.append("")
            lines.append("| vs joint | Δcentre (SE) | Δsize (SE) | ΔIoU (SE) |")
            lines.append("|---|---|---|---|")
            for name in GROUPS[1:]:
                entry = gap.get("paired_vs", {}).get(name, {}).get(gap_length)
                if not entry:
                    continue
                lines.append(
                    f"| {name} | {entry['center']['mean']:+.5f} ({entry['center']['stderr']:.5f}) | "
                    f"{entry['size']['mean']:+.5f} ({entry['size']['stderr']:.5f}) | "
                    f"{entry['iou']['mean']:+.4f} ({entry['iou']['stderr']:.4f}) |"
                )
            lines.append("")
    else:
        lines.append("配对诊断尚未生成。")
        lines.append("")

    text = "\n".join(lines) + "\n"
    (NEW / "REPORT.md").write_text(text)
    summary = {
        "combined": combined,
        "delta": {
            name: {
                base_name: {
                    key: combined[name][key] - base[key]
                    for key in ("HOTA", "AssA", "IDF1", "DetA", "MOTA")
                }
                for base_name, base in (
                    ("joint", joint),
                    ("motion_diou_speed", center),
                )
                if name != base_name
            }
            for name in ("motion_diou_speed", "motion_diou_speed4")
        },
        "clear_runs": aggregate,
        "focus_clear": focus,
        "paired_gap": gap,
        "validation": valid,
        "per_sequence": per_sequence,
    }
    (NEW / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)


if __name__ == "__main__":
    main()
