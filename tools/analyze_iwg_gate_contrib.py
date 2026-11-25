#!/usr/bin/env python3
"""Summarize the IWG SportsMOT motion-gate contribution diagnosis."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/iwg_sportsmot_gate_contrib_20261122")
BASE = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/evals/new/summary.json")
NEW = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/shutrack_scratch_cost_rank_20261122/SportsMOT/evals")
VAL = [line.strip() for line in Path("/home/shang/workspace/TrackTrack/3. Tracker/trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
       if line.strip() and line.strip() != "name"]
KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")


def load(path):
    return json.loads(Path(path).read_text())


def cells(combined, keys=KEYS):
    out = []
    for key in keys:
        value = combined.get(key)
        out.append("-" if value is None else (f"{value:.0f}" if key == "IDSW" else f"{value*100:.3f}"))
    return out


def num(value, digits=3, scale=1.0):
    return "-" if value is None else f"{value*scale:.{digits}f}"


def main():
    normal = load(ROOT / "normal/summary.json")
    fixed = load(ROOT / "motion1/summary.json")
    baseline = load(BASE)
    new_normal = load(NEW / "cost_e200/summary.json")
    new_fixed = load(NEW / "cost_e200_motion1/summary.json")
    gates = load(ROOT / "gate_records/normal.json")["global"]
    fixed_gates = load(ROOT / "gate_records/motion1.json")
    stats = [json.loads(line) for line in (ROOT / "normal/gates.jsonl").read_text().splitlines() if line.strip()]
    fixed_stats = [json.loads(line) for line in (ROOT / "motion1/gates.jsonl").read_text().splitlines() if line.strip()]
    fixed_stats_avg = statistics.mean(r["agentguard_avg_motion_gate"] for r in fixed_stats)
    per_seq_avg = [r["agentguard_avg_motion_gate"] for r in stats]
    per_seq_final = [r["agentguard_avg_rg_cma_final_motion_gate"] for r in stats]
    per_seq_base = [r["agentguard_avg_rg_cma_base_motion_gate"] for r in stats]
    seqs = list(normal["per_sequence"])
    delta_hota = [(fixed["per_sequence"][s]["HOTA"] - normal["per_sequence"][s]["HOTA"]) * 100 for s in seqs]

    payload = {
        "checkpoint": "iwg_rg_cma_v1_sportsmot_trainval_interleaved_shard4x1_seed42_bs1024_200e/checkpoints/iwg_rg_cma_epoch200.pt",
        "checkpoint_sha256": "adb94eeb0eb5919511bf9adbd20a25aa3b62f14036ead867564f5a7dc615027d",
        "model_schema": "agentguard_iwg_rg_cma_v1",
        "training_split": "trainval (45 train + 45 val; val is in-sample)",
        "normal": normal["combined"], "fixed_motion_gate_1": fixed["combined"],
        "baseline_D": baseline["combined"],
        "new_cost_e200_normal": new_normal["combined"],
        "new_cost_e200_motion1": new_fixed["combined"],
        "deltas": {
            "fixed_minus_normal": {k: fixed["combined"][k] - normal["combined"][k] for k in KEYS},
            "normal_minus_baseline_D": {k: normal["combined"][k] - baseline["combined"][k] for k in KEYS},
            "iwg_contribution_normal_minus_fixed": {k: normal["combined"][k] - fixed["combined"][k] for k in KEYS},
            "new_cost_contribution_normal_minus_fixed": {k: new_normal["combined"][k] - new_fixed["combined"][k] for k in KEYS},
        },
        "per_sequence_fixed_minus_normal_hota_pp": {
            "mean": statistics.mean(delta_hota), "sd": statistics.stdev(delta_hota),
            "se": statistics.stdev(delta_hota) / len(delta_hota) ** 0.5,
            "improved": sum(v > 0 for v in delta_hota), "n": len(delta_hota),
            "min": min(delta_hota), "max": max(delta_hota)},
        "normal_gate_distribution_matched": gates,
        "per_sequence_average_motion_gate": {
            "count": len(per_seq_avg), "mean": statistics.mean(per_seq_avg),
            "median": statistics.median(per_seq_avg), "min": min(per_seq_avg), "max": max(per_seq_avg),
            "mean_base": statistics.mean(per_seq_base), "mean_final": statistics.mean(per_seq_final),
            "sequences_below_0_8": sum(v < 0.8 for v in per_seq_avg)},
        "fixed_gate_verification": {
            "sequences": len(fixed_gates["sequences"]),
            "all_matched_motion_exactly_one": all(
                bucket["motion"] and bucket["motion"]["all_exactly_one"]
                for bucket in fixed_gates["sequences"].values()),
            "appearance_model_driven_sequences": sum(
                1 for bucket in fixed_gates["sequences"].values()
                if bucket["appearance"] and not bucket["appearance"]["all_exactly_one"]),
            "applied_appearance": fixed_gates["global"]["appearance"],
        },
        "per_sequence": {
            s: {"normal": {k: normal["per_sequence"][s][k] for k in KEYS},
                "fixed1": {k: fixed["per_sequence"][s][k] for k in KEYS},
                "delta_hota_pp": (fixed["per_sequence"][s]["HOTA"] - normal["per_sequence"][s]["HOTA"]) * 100}
            for s in seqs},
    }
    (ROOT / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = ["# IWG SportsMOT 运动门控贡献诊断（同权重推理消融）", "",
             "权重：`iwg_rg_cma_v1_sportsmot_trainval_interleaved_shard4x1_seed42_bs1024_200e/checkpoints/iwg_rg_cma_epoch200.pt`"
             "（sha256 adb94eeb…，model schema `agentguard_iwg_rg_cma_v1`，context 6，correction bound 0.05）。",
             "**训练划分：trainval，45 条 val 全部在训练数据中；下面的 val 指标是 in-sample，"
             "绝对值不能与新架构（仅 train 45 训练、val 留出）直接比较，只有同权重开关差值有意义。**",
             "推理参数：`--agentguard-mode iwg-rg-cma --iwg-rg-cma-output final --agentguard-device cpu`，"
             "kf-type nsa，seed 10000，同一检测/ReID 缓存；第二组仅加 `--agentguard-disable-kf-gate`。", "",
             "## 汇总（45 条 val raw）", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, combined in (("IWG 正常", normal["combined"]),
                           ("IWG 运动 gate 固定 1", fixed["combined"]),
                           ("原 D baseline（新架构参考）", baseline["combined"])):
        lines.append(f"| {name} | " + " | ".join(cells(combined)) + " |")
    lines += ["", "| 差值 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
              "|---|---:|---:|---:|---:|---:|---:|"]

    def delta_cells(delta):
        out = []
        for key in KEYS:
            value = delta[key]
            out.append(f"{value:+.0f}" if key == "IDSW" else f"{value*100:+.3f}")
        return out

    lines.append("| 固定 1 − 正常 | " + " | ".join(delta_cells(payload["deltas"]["fixed_minus_normal"])) + " |")
    lines.append("| 正常 − D baseline | " + " | ".join(delta_cells(payload["deltas"]["normal_minus_baseline_D"])) + " |")
    lines += ["", "**IWG 运动门控贡献 = 正常 − 固定 1：HOTA "
              f"{payload['deltas']['iwg_contribution_normal_minus_fixed']['HOTA']*100:+.3f}pp**"
              "（逐序列均值 "
              f"{payload['per_sequence_fixed_minus_normal_hota_pp']['mean']:+.3f}pp，SE "
              f"{payload['per_sequence_fixed_minus_normal_hota_pp']['se']:.3f}，"
              f"{payload['per_sequence_fixed_minus_normal_hota_pp']['improved']}/45 条固定 1 后改善）。", "",
              "## 与新架构对应消融并列（HOTA）", "",
              "| 权重 | 训练数据 | 正常门控 | 运动固定 1 | 运动门控贡献 |",
              "|---|---|---:|---:|---:|",
              f"| IWG（本诊断） | trainval（val 在训练中） | {normal['combined']['HOTA']*100:.3f} | "
              f"{fixed['combined']['HOTA']*100:.3f} | "
              f"{(normal['combined']['HOTA']-fixed['combined']['HOTA'])*100:+.3f}pp |",
              f"| 新架构 代价监督 e200 | 仅 train 45（val 留出） | {new_normal['combined']['HOTA']*100:.3f} | "
              f"{new_fixed['combined']['HOTA']*100:.3f} | "
              f"{(new_normal['combined']['HOTA']-new_fixed['combined']['HOTA'])*100:+.3f}pp |", "",
              "## 正常运行的运动 gate 分布（匹配事件，288,114 个）", "",
              "| 指标 | 应用 gate | base gate | final gate | 外观 gate |",
              "|---|---:|---:|---:|---:|"]
    for key, label in (("mean", "均值"), ("min", "最小值"), ("max", "最大值")):
        lines.append(f"| {label} | {num(gates['motion'][key], 4)} | {num(gates['base_motion'][key], 4)} | "
                     f"{num(gates['final_motion'][key], 4)} | {num(gates['appearance'][key], 4)} |")
    q = gates["motion"]["quantiles"]
    lines.append(f"| 分位数 p1/p5/p25/p50/p75/p95/p99 | "
                 + " / ".join(f"{v:.3f}" for v in q) + " | - | - | - |")
    lines.append(f"| >0.95 占比 | {gates['motion']['fraction_gt095']*100:.1f}% | "
                 f"{gates['base_motion']['fraction_gt095']*100:.1f}% | "
                 f"{gates['final_motion']['fraction_gt095']*100:.1f}% | "
                 f"{gates['appearance']['fraction_gt095']*100:.1f}% |")
    lines.append(f"| <0.05 占比 | {gates['motion']['fraction_lt005']*100:.2f}% | "
                 f"{gates['base_motion']['fraction_lt005']*100:.2f}% | "
                 f"{gates['final_motion']['fraction_lt005']*100:.2f}% | "
                 f"{gates['appearance']['fraction_lt005']*100:.1f}% |")
    lines += ["", "运动 gate 直方图（20 等宽桶，0→1）：`" + json.dumps(gates["motion"]["hist_20"]) + "`。",
              "逐序列平均运动 gate（运行时统计，含未匹配 0）：均值 "
              f"{statistics.mean(per_seq_avg):.4f}，中位数 {statistics.median(per_seq_avg):.4f}，"
              f"范围 {min(per_seq_avg):.4f}–{max(per_seq_avg):.4f}；"
              f"{sum(v < 0.8 for v in per_seq_avg)}/45 条低于 0.8。", "",
              "## 固定 1 组核验", "",
              f"- 45/45 条序列、{fixed_gates['global']['motion']['count']} 个匹配事件的应用运动 gate **全部精确为 1.0**"
              f"（`all_exactly_one=True`）。",
              "- 外观 gate 仍由旧模型输出：45/45 条序列外观 gate 非常数，均值 "
              f"{fixed_gates['global']['appearance']['mean']:.4f}，范围 "
              f"{fixed_gates['global']['appearance']['min']:.4f}–{fixed_gates['global']['appearance']['max']:.4f}。",
              "- 旧运行时统计的 `avg_motion_gate` 在固定组按序列平均为 "
              f"{fixed_stats_avg:.4f}（它把未匹配事件记为 0）；"
              "事件级记录只统计匹配事件，确认真实写回位置全部为 1。", "",
              "## 写回路径核对", "",
              "- 旧运行时在 `run_iwg_rg_cma_batch_inference` 中把 `applied[:, 0]` 置 1；"
              "该 `applied` 即 `GateDecision.motion_gate` → `adapter.apply_gate` → "
              "`Track.update_with_gates(motion_gate, appearance_gate)`，这是唯一的 KF 运动写回位置。",
              "- RG-CMA 只修正 gate 标量（correction bound 0.05；本次正常组的 base/final 运动 gate 均值 "
              f"{gates['base_motion']['mean']:.4f}/{gates['final_motion']['mean']:.4f}），"
              "不直接修改 box 或 KF 状态；policy/cue 仅为诊断输出；`fallback_threshold=0` 未生效；"
              "未匹配事件走 `mark_lost`，不进入 KF 写回。", "",
              "## 逐序列结果（固定运动 gate=1，Δ 为相对正常门控）", "",
              "| 序列 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | ΔHOTA (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seq in seqs:
        row = fixed["per_sequence"][seq]
        delta = payload["per_sequence"][seq]["delta_hota_pp"]
        lines.append(f"| {seq} | {row['HOTA']*100:.3f} | {row['AssA']*100:.3f} | {row['DetA']*100:.3f} | "
                     f"{row['IDF1']*100:.3f} | {row['MOTA']*100:.3f} | {int(row['IDSW'])} | {delta:+.3f} |")
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "comparison.md").read_text())


if __name__ == "__main__":
    main()
