#!/usr/bin/env python3
"""Summarize the joint supervision group against control and both singles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/shang/workspace/TrackTrack/outputs/agentguard/experiments"
    "/shutrack_a_supervision_trainval_20261118"
)
GROUPS = ("e200", "control", "motion_diou", "appearance_score", "motion_diou_appearance_score")
JOINT = "motion_diou_appearance_score"
FOCUS = (
    "v_0kUtTtmLaJA_c006",
    "v_0kUtTtmLaJA_c008",
    "v_00HRwkvvjtQ_c008",
    "v_9MHDmAMxO5I_c003",
)


def main() -> None:
    parsed = json.loads((ROOT / "evals" / "parsed_all.json").read_text())
    clear = {}
    clear_path = ROOT / "evals" / "clear_id.json"
    if clear_path.is_file():
        clear = json.loads(clear_path.read_text())

    combined = {g: parsed[g]["combined"] for g in GROUPS if g in parsed}
    per_sequence = {g: parsed[g]["per_sequence"] for g in GROUPS if g in parsed}
    joint = combined[JOINT]
    control = combined["control"]
    motion = combined["motion_diou"]
    appearance = combined["appearance_score"]
    best_single_name = max(("motion_diou", "appearance_score"), key=lambda g: combined[g]["HOTA"])
    best_single = combined[best_single_name]

    lines = ["# 联合监督组结果", ""]
    lines.append("| group | HOTA | AssA | IDF1 | DetA | MOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name in GROUPS:
        if name not in combined:
            continue
        row = combined[name]
        lines.append(
            f"| {name} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | "
            f"{row['DetA']:.4f} | {row['MOTA']:.4f} |"
        )
    lines.append("")
    deltas = {}
    for name in ("control", "motion_diou", "appearance_score"):
        row = combined[name]
        deltas[name] = {key: joint[key] - row[key] for key in ("HOTA", "AssA", "IDF1", "DetA", "MOTA")}
    lines.append("## 联合组相对各组的差值")
    lines.append("")
    lines.append("| vs | ΔHOTA | ΔAssA | ΔIDF1 | ΔDetA | ΔMOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name, values in deltas.items():
        lines.append(
            f"| {name} | {values['HOTA']:+.4f} | {values['AssA']:+.4f} | "
            f"{values['IDF1']:+.4f} | {values['DetA']:+.4f} | {values['MOTA']:+.4f} |"
        )
    lines.append("")
    lines.append(
        f"最佳单项为 {best_single_name}（HOTA {best_single['HOTA']:.4f}）；"
        f"联合组 ΔHOTA {joint['HOTA'] - best_single['HOTA']:+.4f}，"
        f"ΔAssA {joint['AssA'] - best_single['AssA']:+.4f}，"
        f"ΔDetA {joint['DetA'] - best_single['DetA']:+.4f}。"
        f"相对运动单项 ΔDetA {joint['DetA'] - motion['DetA']:+.4f}。"
    )
    lines.append("")

    base = per_sequence["control"]
    joint_seq = per_sequence[JOINT]
    items = sorted(
        ((joint_seq[s]["HOTA"] - base[s]["HOTA"], s) for s in base), reverse=True
    )
    improved = sum(1 for value, _ in items if value > 0)
    regressed = sum(1 for value, _ in items if value < 0)
    lines.append("## 逐序列（相对 control）")
    lines.append("")
    lines.append(
        f"joint 相对 control：HOTA 改善 {improved}/45，回退 {regressed}/45；"
        f"最大涨幅 {items[0][1]} {items[0][0]:+.4f}；最大跌幅 {items[-1][1]} {items[-1][0]:+.4f}。"
    )
    lines.append("")
    lines.append("### joint 相对 control 回退的序列")
    lines.append("")
    lines.append("| sequence | ΔHOTA | ΔAssA | joint HOTA/AssA |")
    lines.append("|---|---|---|---|")
    for value, sequence in reversed(items):
        if value >= 0:
            continue
        lines.append(
            f"| {sequence} | {value:+.4f} | "
            f"{joint_seq[sequence]['AssA'] - base[sequence]['AssA']:+.4f} | "
            f"{joint_seq[sequence]['HOTA']:.4f}/{joint_seq[sequence]['AssA']:.4f} |"
        )
    lines.append("")

    single_compare = {}
    for name in ("motion_diou", "appearance_score"):
        seq_rows = sorted(
            (
                (joint_seq[s]["HOTA"] - per_sequence[name][s]["HOTA"], s)
                for s in joint_seq
            ),
            reverse=True,
        )
        gains = [(value, s) for value, s in seq_rows if value > 0]
        losses = [(value, s) for value, s in seq_rows if value < 0]
        single_compare[name] = {
            "improved": len(gains),
            "regressed": len(losses),
            "equal": len(seq_rows) - len(gains) - len(losses),
            "top_gains": [
                {"sequence": s, "delta": value} for value, s in seq_rows[:5] if value > 0
            ],
            "top_losses": [
                {"sequence": s, "delta": value} for value, s in seq_rows[-5:][::-1] if value < 0
            ],
        }
        lines.append(
            f"joint 相对 {name}：HOTA 改善 {len(gains)}/45，回退 {len(losses)}/45，持平 "
            f"{len(seq_rows) - len(gains) - len(losses)}/45。"
        )
        if gains:
            lines.append(
                "  主要增益序列："
                + "；".join(f"{s} {v:+.4f}" for v, s in seq_rows[:5])
            )
        if losses:
            lines.append(
                "  主要回退序列："
                + "；".join(f"{s} {v:+.4f}" for v, s in seq_rows[-5:][::-1])
            )
        lines.append("")

    lines.append("| sequence | control HOTA/AssA | motion_diou | appearance_score | joint |")
    lines.append("|---|---|---|---|---|")
    for sequence in FOCUS:
        cells = []
        for name in ("control", "motion_diou", "appearance_score", JOINT):
            row = per_sequence[name][sequence]
            cells.append(f"{row['HOTA']:.4f}/{row['AssA']:.4f}")
        lines.append(f"| {sequence} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 关注序列相对 control 的差值")
    lines.append("")
    lines.append("| sequence | motion_diou ΔHOTA/ΔAssA | appearance_score ΔHOTA/ΔAssA | joint ΔHOTA/ΔAssA |")
    lines.append("|---|---|---|---|")
    for sequence in FOCUS:
        cells = []
        for name in ("motion_diou", "appearance_score", JOINT):
            row = per_sequence[name][sequence]
            cells.append(
                f"{row['HOTA'] - base[sequence]['HOTA']:+.4f}/"
                f"{row['AssA'] - base[sequence]['AssA']:+.4f}"
            )
        lines.append(f"| {sequence} | " + " | ".join(cells) + " |")
    lines.append("")

    c003 = {}
    if clear:
        for name in GROUPS:
            payload = clear.get("sequences", {}).get("v_9MHDmAMxO5I_c003", {}).get(name)
            if not payload:
                continue
            spans = [
                span
                for span in payload.get("identity_errors", [])
                if span["gt_id"] in (3, 4) and span["wrong_from"] >= 100
            ]
            c003[name] = spans
    lines.append("## c003 GT3/GT4 互换（CLEAR IoU≥0.5）")
    lines.append("")
    if c003:
        for name, spans in c003.items():
            text = ", ".join(
                f"gt{span['gt_id']} @{span['wrong_from']} dur={span['wrong_duration']}"
                for span in spans
            ) or "无长期互换"
            lines.append(f"- {name}: {text}")
    else:
        lines.append("CLEAR 结果尚未生成。")
    lines.append("")

    text = "\n".join(lines) + "\n"
    (ROOT / "JOINT_REPORT.md").write_text(text)
    report_path = ROOT / "REPORT.md"
    if report_path.is_file():
        report = report_path.read_text()
        marker = "# 联合监督组结果"
        if marker in report:
            report = report[: report.index(marker)].rstrip() + "\n"
        report_path.write_text(report + "\n" + text)
    summary = {
        "combined": combined,
        "joint_vs": deltas,
        "best_single": best_single_name,
        "joint_minus_best_single": {
            key: joint[key] - best_single[key] for key in ("HOTA", "AssA", "IDF1", "DetA", "MOTA")
        },
        "per_sequence_vs_control": {
            "improved": improved,
            "regressed": regressed,
            "best": {"sequence": items[0][1], "delta": items[0][0]},
            "worst": {"sequence": items[-1][1], "delta": items[-1][0]},
        },
        "per_sequence_vs_single": single_compare,
        "focus": {
            sequence: {
                name: per_sequence[name][sequence] for name in
                ("control", "motion_diou", "appearance_score", JOINT)
            }
            for sequence in FOCUS
        },
        "c003_clear": c003,
    }
    (ROOT / "joint_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)


if __name__ == "__main__":
    main()
