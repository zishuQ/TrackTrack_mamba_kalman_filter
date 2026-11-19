#!/usr/bin/env python3
"""Summarize the appearance-rank and motion-speed groups against the joint group."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
OLD = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
NEW = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119"
VAL_MAP = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
GROUPS = ("joint", "appearance_rank", "motion_diou_speed")
FOCUS = (
    "v_9MHDmAMxO5I_c003",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c011",
)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def parse_logs() -> dict:
    logs = {
        "joint": OLD / "evals" / "motion_diou_appearance_score_val.log",
        "appearance_rank": NEW / "evals" / "appearance_rank_val.log",
        "motion_diou_speed": NEW / "evals" / "motion_diou_speed_val.log",
    }
    command = [
        sys.executable,
        str(WORKTREE / "tools/parse_val_logs.py"),
        "--out",
        str(NEW / "evals" / "parsed_all.json"),
    ]
    for name, path in logs.items():
        command += ["--log", f"{name}={path}"]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return json.loads((NEW / "evals" / "parsed_all.json").read_text())


def result_dirs() -> dict[str, Path]:
    return {
        "joint": OLD
        / "evals/motion_diou_appearance_score_val/sportsmot_val_0.80_motion_diou_appearance_score_val_shutrack_v2",
        "appearance_rank": NEW
        / "evals/appearance_rank_val/sportsmot_val_0.80_appearance_rank_val_shutrack_v2",
        "motion_diou_speed": NEW
        / "evals/motion_diou_speed_val/sportsmot_val_0.80_motion_diou_speed_val_shutrack_v2",
    }


def run_clear() -> dict:
    command = [
        sys.executable,
        str(WORKTREE / "tools/clear_id_analysis.py"),
        "--sequences-file",
        str(VAL_MAP),
        "--out",
        str(NEW / "evals" / "clear_id_all.json"),
    ]
    for name, directory in result_dirs().items():
        command += ["--result", f"{name}={directory}"]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return json.loads((NEW / "evals" / "clear_id_all.json").read_text())


def aggregate_runs(clear: dict) -> dict[str, dict]:
    summary = {}
    for name in GROUPS:
        durations = []
        recovered = []
        per_sequence = {}
        for sequence, groups in clear["sequences"].items():
            payload = groups.get(name, {})
            runs = payload.get("wrong_identity_runs", [])
            sequence_frames = 0
            for item in runs:
                durations.append(item["duration"])
                if item["duration"] >= 5:
                    recovered.append(bool(item.get("recovered")))
                sequence_frames += item["duration"]
            per_sequence[sequence] = sequence_frames
        array = np.asarray(durations, dtype=np.float64)
        long_array = array[array >= 5] if array.size else array
        summary[name] = {
            "wrong_frames_total": int(array.sum()) if array.size else 0,
            "runs": int(array.size),
            "runs_ge5": int(long_array.size),
            "runs_ge20": int((array >= 20).sum()) if array.size else 0,
            "runs_ge50": int((array >= 50).sum()) if array.size else 0,
            "runs_ge100": int((array >= 100).sum()) if array.size else 0,
            "mean_long_duration": float(long_array.mean()) if long_array.size else None,
            "median_long_duration": float(np.median(long_array)) if long_array.size else None,
            "max_duration": float(array.max()) if array.size else None,
            "recovered_fraction_ge5": (
                float(np.mean(recovered)) if recovered else None
            ),
            "per_sequence_wrong_frames": per_sequence,
        }
    return summary


def focus_clear(clear: dict) -> dict:
    focus = {}
    for sequence in FOCUS:
        payloads = clear["sequences"].get(sequence, {})
        entry = {}
        for name in GROUPS:
            payload = payloads.get(name, {})
            errors = payload.get("identity_errors", [])
            runs = payload.get("wrong_identity_runs", [])
            entry[name] = {
                "identity_errors": errors,
                "wrong_runs_ge5": [run for run in runs if run["duration"] >= 5],
            }
        focus[sequence] = entry
    return focus


def gap_summary() -> dict:
    path = NEW / "evals" / "gap_center_error.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    output = {}
    for name, payload in data.items():
        output[name] = {"combined": payload.get("combined", {})}
        focus_entries = {}
        for sequence in FOCUS:
            entry = payload.get("sequences", {}).get(sequence)
            if not entry:
                continue
            focus_entries[sequence] = {}
            for index in ("1", "3", "5"):
                values = entry["errors"][index]
                if values:
                    array = np.asarray(values, dtype=np.float64)
                    focus_entries[sequence][index] = {
                        "count": int(array.size),
                        "mean": float(array.mean()),
                        "median": float(np.median(array)),
                    }
                else:
                    focus_entries[sequence][index] = {"count": 0}
        output[name]["focus"] = focus_entries
    return output


def main() -> None:
    (NEW / "evals").mkdir(parents=True, exist_ok=True)
    parsed = parse_logs()
    clear = run_clear()
    aggregate = aggregate_runs(clear)
    focus = focus_clear(clear)
    gaps = gap_summary()

    combined = {name: parsed[name]["combined"] for name in GROUPS}
    per_sequence = {name: parsed[name]["per_sequence"] for name in GROUPS}
    joint = combined["joint"]

    lines = ["# 直接外观监督与速度监督结果", ""]
    lines.append("| group | HOTA | AssA | IDF1 | DetA | MOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name in GROUPS:
        row = combined[name]
        lines.append(
            f"| {name} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | "
            f"{row['DetA']:.4f} | {row['MOTA']:.4f} |"
        )
    lines.append("")
    lines.append("## 相对联合组的差值")
    lines.append("")
    lines.append("| group | ΔHOTA | ΔAssA | ΔIDF1 | ΔDetA | ΔMOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("appearance_rank", "motion_diou_speed"):
        row = combined[name]
        lines.append(
            f"| {name} | {row['HOTA']-joint['HOTA']:+.4f} | {row['AssA']-joint['AssA']:+.4f} | "
            f"{row['IDF1']-joint['IDF1']:+.4f} | {row['DetA']-joint['DetA']:+.4f} | "
            f"{row['MOTA']-joint['MOTA']:+.4f} |"
        )
    lines.append("")

    configs = {}
    for name in ("appearance_rank", "motion_diou_speed"):
        path = NEW / "groups" / name / "config.json"
        if path.is_file():
            configs[name] = json.loads(path.read_text())
    validation_path = NEW / "rank_speed_validation.json"
    validation = (
        json.loads(validation_path.read_text()) if validation_path.is_file() else {}
    )
    if configs:
        lines.append("## 超参数与真实 batch 测量")
        lines.append("")
        rank_config = configs.get("appearance_rank", {})
        speed_config = configs.get("motion_diou_speed", {})
        lines.append(
            f"- appearance_rank: rank_weight={rank_config.get('rank_weight')}，"
            f"tau={rank_config.get('rank_tau')}，alpha={rank_config.get('rank_alpha')}"
        )
        lines.append(
            f"- motion_diou_speed: speed_weight={speed_config.get('speed_weight')}"
        )
        totals = validation.get("totals", {})
        if totals:
            lines.append(
                f"- 排序有效样本 {totals.get('rank_valid_appearance_rows')}"
                f"/{totals.get('rank_rows')}，产生梯度比例 "
                f"{totals.get('rank_grad_nonzero_fraction'):.3f}，"
                f"外观 logit 梯度均值 {totals.get('rank_grad_abs_mean', 0):.2e}，"
                f"参数梯度范数对齐系数 {totals.get('suggested_rank_weight_param_norm', 0):.1f}，"
                f"模板三档一致误差 {totals.get('template_parity_max', 0):.1e}"
            )
            lines.append(
                f"- 速度有效样本 {totals.get('speed_grad_nonzero')}"
                f"/{totals.get('speed_rows')}，产生梯度比例 "
                f"{totals.get('speed_grad_nonzero_fraction'):.3f}，"
                f"速度/框运动门控梯度比 {totals.get('speed_to_box_grad_ratio', 0):.4f}，"
                f"25% 对齐建议系数 {totals.get('suggested_speed_weight_25pct', 0):.1f}"
            )
        lines.append("")

    lines.append("## 45 序列身份错误持续时间（首轨为准）")
    lines.append("")
    lines.append("| group | wrong frames | runs | ≥5 | ≥20 | ≥50 | ≥100 | median ≥5 | max | recovered ≥5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in GROUPS:
        row = aggregate[name]
        def fmt(value):
            if value is None:
                return "—"
            if isinstance(value, (int,)):
                return str(value)
            value = float(value)
            return f"{value:.3f}" if abs(value) < 1.0 else f"{value:.1f}"
        lines.append(
            f"| {name} | {row['wrong_frames_total']} | {row['runs']} | {row['runs_ge5']} | "
            f"{row['runs_ge20']} | {row['runs_ge50']} | {row['runs_ge100']} | "
            f"{fmt(row['median_long_duration'])} | {fmt(row['max_duration'])} | "
            f"{fmt(row['recovered_fraction_ge5'])} |"
        )
    lines.append("")
    for name in ("appearance_rank", "motion_diou_speed"):
        row = aggregate[name]
        base = aggregate["joint"]
        lines.append(
            f"{name} 相对联合组：wrong frames {row['wrong_frames_total']-base['wrong_frames_total']:+d}，"
            f"≥20 帧错误 {row['runs_ge20']-base['runs_ge20']:+d}，"
            f"≥100 帧错误 {row['runs_ge100']-base['runs_ge100']:+d}。"
        )
    lines.append("")

    lines.append("## 逐序列 HOTA（相对联合组）")
    lines.append("")
    for name in ("appearance_rank", "motion_diou_speed"):
        items = sorted(
            (
                (per_sequence[name][seq]["HOTA"] - per_sequence["joint"][seq]["HOTA"], seq)
                for seq in per_sequence["joint"]
            ),
            reverse=True,
        )
        improved = sum(1 for value, _ in items if value > 0)
        regressed = sum(1 for value, _ in items if value < 0)
        lines.append(
            f"{name}：改善 {improved}/45，回退 {regressed}/45；"
            f"最大涨幅 {items[0][1]} {items[0][0]:+.4f}；最大跌幅 {items[-1][1]} {items[-1][0]:+.4f}。"
        )
        lines.append("")
        lines.append(f"| sequence | joint HOTA/AssA | {name} HOTA/AssA | ΔHOTA |")
        lines.append("|---|---|---|---|")
        for _, sequence in items:
            row = per_sequence[name][sequence]
            base = per_sequence["joint"][sequence]
            lines.append(
                f"| {sequence} | {base['HOTA']:.4f}/{base['AssA']:.4f} | "
                f"{row['HOTA']:.4f}/{row['AssA']:.4f} | {row['HOTA']-base['HOTA']:+.4f} |"
            )
        lines.append("")

    lines.append("## 关注序列：长身份错误（≥5 帧，CLEAR IoU≥0.5）")
    lines.append("")
    for sequence in FOCUS:
        lines.append(f"### {sequence}")
        lines.append("")
        lines.append("| group | gt | track | start | end | duration | recovered |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in GROUPS:
            for run_item in focus[sequence][name]["wrong_runs_ge5"]:
                lines.append(
                    f"| {name} | gt{run_item['gt_id']} | {run_item['reference_track']}->{run_item['track_id']} | "
                    f"{run_item['start']} | {run_item['end']} | {run_item['duration']} | "
                    f"{run_item.get('recovered')} |"
                )
        lines.append("")

    if gaps:
        lines.append("## 连续漏检后的中心预测误差（归一化，越低越好）")
        lines.append("")
        lines.append("| group | miss 1 | miss 3 | miss 5 |")
        lines.append("|---|---|---|---|")
        for name in gaps:
            cells = []
            for index in ("1", "3", "5"):
                entry = gaps[name]["combined"].get(index, {})
                cells.append(
                    f"{entry.get('mean', float('nan')):.4f} (n={entry.get('count', 0)})"
                    if entry.get("count")
                    else "—"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        for sequence in FOCUS:
            rows = []
            for name in gaps:
                entry = gaps[name].get("focus", {}).get(sequence)
                if not entry:
                    continue
                cells = []
                for index in ("1", "3", "5"):
                    item = entry.get(index, {})
                    cells.append(
                        f"{item.get('mean', float('nan')):.4f} (n={item.get('count', 0)})"
                        if item.get("count")
                        else "—"
                    )
                rows.append(f"| {sequence} | {name} | " + " | ".join(cells) + " |")
            if rows:
                lines.append("| sequence | group | miss 1 | miss 3 | miss 5 |")
                lines.append("|---|---|---|---|---|")
                lines.extend(rows)
                lines.append("")
    else:
        lines.append("## 连续漏检后的中心预测误差")
        lines.append("")
        lines.append("gap 诊断尚未生成。")
        lines.append("")

    text = "\n".join(lines) + "\n"
    (NEW / "REPORT.md").write_text(text)
    summary = {
        "combined": combined,
        "delta_vs_joint": {
            name: {
                key: combined[name][key] - joint[key]
                for key in ("HOTA", "AssA", "IDF1", "DetA", "MOTA")
            }
            for name in ("appearance_rank", "motion_diou_speed")
        },
        "clear_runs": aggregate,
        "focus_clear": focus,
        "gap_center_error": gaps,
        "per_sequence": {
            "joint": per_sequence["joint"],
            "appearance_rank": per_sequence["appearance_rank"],
            "motion_diou_speed": per_sequence["motion_diou_speed"],
        },
    }
    (NEW / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)


if __name__ == "__main__":
    main()
