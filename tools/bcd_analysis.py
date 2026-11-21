#!/usr/bin/env python3
"""Summarize the B/C/D supervision comparison."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
NEW = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_bcd_20261119"
VAL_MAP = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
GROUPS = ("appearance_rank", "appearance_rank_speed4", "appearance_row_column")
LABELS = {
    "appearance_rank": "B: rank",
    "appearance_rank_speed4": "C: rank + 4D speed",
    "appearance_row_column": "D: rank + contention",
}
FOCUS = ("v_9MHDmAMxO5I_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")


def run(command: list[str]):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def parse_logs() -> dict:
    command = [
        sys.executable,
        str(WORKTREE / "tools/parse_val_logs.py"),
        "--out",
        str(NEW / "evals/parsed_all.json"),
    ]
    for name in GROUPS:
        command += ["--log", f"{name}={NEW / 'evals' / f'{name}_val.log'}"]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return json.loads((NEW / "evals/parsed_all.json").read_text())


def run_clear() -> dict:
    command = [
        sys.executable,
        str(WORKTREE / "tools/clear_id_analysis.py"),
        "--sequences-file",
        str(VAL_MAP),
        "--out",
        str(NEW / "evals/clear_id_all.json"),
    ]
    for name in GROUPS:
        command += [
            "--result",
            f"{name}={NEW / 'evals' / f'{name}_val' / f'sportsmot_val_0.80_{name}_val_shutrack_v2'}",
        ]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return json.loads((NEW / "evals/clear_id_all.json").read_text())


def aggregate_runs(clear: dict) -> dict:
    summary = {}
    for name in GROUPS:
        durations = []
        recovered = []
        for sequence, groups in clear["sequences"].items():
            for item in groups.get(name, {}).get("wrong_identity_runs", []):
                durations.append(item["duration"])
                if item["duration"] >= 5:
                    recovered.append(bool(item.get("recovered")))
        array = np.asarray(durations, dtype=np.float64)
        long_array = array[array >= 5] if array.size else array
        summary[name] = {
            "wrong_frames": int(array.sum()) if array.size else 0,
            "runs_ge5": int(long_array.size),
            "runs_ge20": int((array >= 20).sum()) if array.size else 0,
            "runs_ge100": int((array >= 100).sum()) if array.size else 0,
            "median_long": float(np.median(long_array)) if long_array.size else None,
            "recovered": float(np.mean(recovered)) if recovered else None,
        }
    return summary


def focus_runs(clear: dict) -> dict:
    out = {}
    for sequence in FOCUS:
        out[sequence] = {
            name: [
                run_item
                for run_item in clear["sequences"].get(sequence, {}).get(name, {}).get(
                    "wrong_identity_runs", []
                )
                if run_item["duration"] >= 5
            ]
            for name in GROUPS
        }
    return out


def wall_times() -> dict:
    path = NEW / "logs" / "train_driver.log"
    if not path.is_file():
        return {}
    times = {}
    for line in path.read_text().splitlines():
        match = re.search(r"\| (\w+): done in (\d+)s", line)
        if match:
            times[match.group(1)] = int(match.group(2))
    return times


def validation() -> dict:
    path = NEW / "bcd_validation.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def main() -> None:
    (NEW / "evals").mkdir(parents=True, exist_ok=True)
    parsed = parse_logs()
    clear = run_clear()
    aggregate = aggregate_runs(clear)
    focus = focus_runs(clear)
    valid = validation()
    times = wall_times()
    combined = {name: parsed[name]["combined"] for name in GROUPS}
    per_sequence = {name: parsed[name]["per_sequence"] for name in GROUPS}
    base = combined["appearance_rank"]

    lines = ["# B/C/D 监督组合对比（trainval 开发实验）", ""]
    lines.append("| group | HOTA | AssA | IDF1 | DetA | MOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name in GROUPS:
        row = combined[name]
        lines.append(
            f"| {LABELS[name]} | {row['HOTA']:.4f} | {row['AssA']:.4f} | "
            f"{row['IDF1']:.4f} | {row['DetA']:.4f} | {row['MOTA']:.4f} |"
        )
    lines.append("")
    lines.append("## 相对 B 的差值")
    lines.append("")
    lines.append("| group | ΔHOTA | ΔAssA | ΔIDF1 | ΔDetA | ΔMOTA |")
    lines.append("|---|---|---|---|---|---|")
    for name in GROUPS[1:]:
        row = combined[name]
        lines.append(
            f"| {LABELS[name]} | {row['HOTA']-base['HOTA']:+.4f} | "
            f"{row['AssA']-base['AssA']:+.4f} | {row['IDF1']-base['IDF1']:+.4f} | "
            f"{row['DetA']-base['DetA']:+.4f} | {row['MOTA']-base['MOTA']:+.4f} |"
        )
    lines.append("")
    lines.append("## 设置与损失定义")
    lines.append("")
    lines.append(
        "- 数据：schema 3、同一次 e200 回放（column probe），90 条序列 / 593,673 样本；"
        "rank 有效 109,360，速度有效 581,462，竞争监督有效 115,205 事件（覆盖 90 条序列）。"
    )
    lines.append(
        "- B `appearance_rank`：运动 GT 框 DIoU + 外观模板排序"
        "（rank_weight=18，tau=0.1，alpha=0.95），无有效排序对时回退 appearance_safe 软标签。"
    )
    lines.append(
        "- C `appearance_rank_speed4`：B + 中心速度 L1（权重 50）+ 宽高速度 L1（权重 14）。"
    )
    lines.append(
        "- D `appearance_row_column`：B + 下一帧外观争抢损失（column_weight=50）。"
        "column_offsets/column_rival 由同一次回放的 ColumnPilot 在实际下一帧关联调用中记录："
        "column_offset 为轨迹到本身份唯一检测的非外观代价（raw cost − 0.5·cos + 来源惩罚），"
        "column_rival 为竞争轨迹到同一检测的最优最终代价；loss = softplus((clamp(offset + 0.5·d(T_g, positive)) − rival)/tau)，"
        "其中 T_g 是用模型外观 gate 对写前模板做线上 EMA 的结果。"
    )
    lines.append(
        "- 与排序监督的区别：排序用未来 5 帧内的同 GT 正例与同帧竞争检测做余弦排序；"
        "column 用下一帧真实匹配池、本身份检测与竞争轨迹的完整代价（含竞争模板与运动项），只在被争抢事件上生效、无回退项。"
    )
    lines.append(
        "- column 权重：数据侧默认 1.0 时外观 gate 梯度均值 9.6e-8（约为 rank 的 0.5%，100% 有效事件产生非零梯度）；"
        "按此前惯例取 rank 外观梯度的 ~25% 定为 50（实测 4.8e-6）。"
    )
    lines.append("")

    data = valid.get("data", {})
    batch = valid.get("batch", {})
    lines.append("## 数据与真实 batch 验证")
    lines.append("")
    if data:
        lines.append(
            f"- 数据：{data.get('sequences')} 条序列、{data.get('samples')} 样本；"
            f"rank 有效 {data.get('rank_valid')}，速度有效 {data.get('speed_valid')}，"
            f"竞争监督有效 {data.get('column_valid')} 事件，覆盖 {data.get('column_sequences')} 条序列；"
            f"column actual_mismatch={data.get('actual_mismatch')}，"
            f"事件键抽查 {data.get('column_key_checked')} 条、错配 {data.get('column_key_mismatch')}。"
        )
    if batch:
        c = batch.get("C", {})
        d = batch.get("D", {})
        lines.append(
            f"- C 真实 batch：rank 有效 {c.get('rank_valid')}，速度有效 {c.get('speed_valid')}；"
            f"运动框梯度 {c.get('grad_motion_box'):.2e}，速度梯度 {c.get('grad_motion_speed'):.2e}，"
            f"外观排序梯度 {c.get('grad_appearance_rank'):.2e}；有限={c.get('finite')}。"
        )
        lines.append(
            f"- D 真实 batch：竞争监督有效 {d.get('column_valid')}；column loss "
            f"{np.mean(d.get('column_losses', [0.0])):.4f}，rank loss "
            f"{np.mean(d.get('rank_losses', [0.0])):.4f}；外观 gate 梯度 "
            f"{d.get('grad_appearance_column'):.2e}（权重 1）/ "
            f"{d.get('grad_appearance_column_weight50'):.2e}（权重 50），"
            f"非零比例 {d.get('column_grad_nonzero_fraction'):.2f}。"
        )
    if times:
        lines.append(
            "- 训练耗时："
            + "，".join(f"{LABELS.get(name, name)} {seconds}s" for name, seconds in times.items())
        )
    lines.append("")

    lines.append("## 逐序列 HOTA")
    lines.append("")
    for name in GROUPS[1:]:
        items = sorted(
            ((per_sequence[name][seq]["HOTA"] - per_sequence["appearance_rank"][seq]["HOTA"], seq)
             for seq in per_sequence["appearance_rank"]),
            reverse=True,
        )
        improved = sum(1 for value, _ in items if value > 0)
        regressed = sum(1 for value, _ in items if value < 0)
        lines.append(
            f"{LABELS[name]} 相对 B：改善 {improved}/45，回退 {regressed}/45；"
            f"最大涨幅 {items[0][1]} {items[0][0]:+.4f}；最大跌幅 {items[-1][1]} {items[-1][0]:+.4f}。"
        )
    lines.append("")
    lines.append("| sequence | B HOTA/AssA | C HOTA/AssA | D HOTA/AssA | ΔC | ΔD |")
    lines.append("|---|---|---|---|---|---|")
    for sequence in sorted(per_sequence["appearance_rank"]):
        b = per_sequence["appearance_rank"][sequence]
        c = per_sequence["appearance_rank_speed4"][sequence]
        d = per_sequence["appearance_row_column"][sequence]
        lines.append(
            f"| {sequence} | {b['HOTA']:.4f}/{b['AssA']:.4f} | "
            f"{c['HOTA']:.4f}/{c['AssA']:.4f} | {d['HOTA']:.4f}/{d['AssA']:.4f} | "
            f"{c['HOTA']-b['HOTA']:+.4f} | {d['HOTA']-b['HOTA']:+.4f} |"
        )
    lines.append("")

    lines.append("## 45 序列身份错误（TrackEval CLEAR，前帧匹配优先）")
    lines.append("")
    lines.append("| group | wrong frames | runs ≥5 | ≥20 | ≥100 | median ≥5 | recovered ≥5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in GROUPS:
        row = aggregate[name]
        lines.append(
            f"| {LABELS[name]} | {row['wrong_frames']} | {row['runs_ge5']} | "
            f"{row['runs_ge20']} | {row['runs_ge100']} | "
            f"{row['median_long']:.1f} | {row['recovered']:.3f} |"
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
                    f"| {LABELS[name]} | gt{item['gt_id']} | {item['reference_track']}->{item['track_id']} | "
                    f"{item['start']} | {item['end']} | {item['duration']} | {item.get('recovered')} |"
                )
        lines.append("")

    lines.append("## 权重与复现")
    lines.append("")
    lines.append(
        f"- 权重：`{NEW}/groups/{{group}}/checkpoints/epoch_040.pt`"
        "（三组均有 epoch_020、epoch_040、last；对照组 B 也在同批 schema 3 数据上重训）。"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/shang/workspace/TrackTrack")
    lines.append(f"NEW={NEW}")
    lines.append('.venv/bin/python tools/generate_trainval_paired.py --root "$NEW" --column-probe')
    lines.append('.venv/bin/python tools/verify_bcd.py')
    lines.append('bash tools/run_train_bcd.sh > "$NEW/logs/train_driver.log" 2>&1')
    lines.append('bash tools/run_eval_bcd.sh > "$NEW/logs/eval_driver.log" 2>&1')
    lines.append('.venv/bin/python tools/bcd_analysis.py')
    lines.append("```")
    lines.append("")

    text = "\n".join(lines) + "\n"
    (NEW / "REPORT.md").write_text(text)
    summary = {
        "combined": combined,
        "delta_vs_B": {
            name: {
                key: combined[name][key] - base[key]
                for key in ("HOTA", "AssA", "IDF1", "DetA", "MOTA")
            }
            for name in GROUPS[1:]
        },
        "clear_runs": aggregate,
        "focus_runs": focus,
        "validation": valid,
        "wall_times_seconds": times,
        "per_sequence": per_sequence,
    }
    (NEW / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)


if __name__ == "__main__":
    main()
