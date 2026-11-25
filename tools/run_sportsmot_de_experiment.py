#!/usr/bin/env python3
"""SportsMOT D/E: dropout 0.05 and motion_harm_weight 2.

Same protocol as A/B/C: official train 45 / val 45, query_film_bilinear from
scratch, all parameters, seed 42, one shard with the whole pool resident in
RAM, 20 full epochs (old e80 sample budget), cosine planned over 50 epochs,
warmup 80, stop LR 6.5899e-5.

  D: dropout 0.05, motion_harm_weight 1
  E: dropout 0.05, motion_harm_weight 2

Checkpoints at epochs 10/15/20 get a full 45-sequence raw val evaluation and a
motion-gate target analysis. A/B/C results are loaded as references.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sportsmot_abc_experiment as abc  # noqa: E402

DE_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_de_20261123"
ABC_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_abc_20261123"
abc.EXP = DE_EXP
abc.LOGS = DE_EXP / "logs"
abc.EVALS = DE_EXP / "evals"
abc.GATES = DE_EXP / "gate_targets"
abc.STATE_PATH = DE_EXP / "state.json"
abc.PREFLIGHT_PATH = DE_EXP / "preflight.json"
abc.RUNS = {
    "D": {"dropout": 0.05, "motion_harm_weight": 1.0},
    "E": {"dropout": 0.05, "motion_harm_weight": 2.0},
}
RUNS = abc.RUNS
EPOCHS = (20,)
SCORE_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")


def ref_group(name):
    return ABC_EXP / "runs" / name / "groups/appearance_row_column"


def ref_checkpoint(name, epoch):
    return ref_group(name) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def load_epoch(exp, name, epoch):
    base = exp / "evals" / f"{name}_e{epoch:03d}"
    summary = json.loads((base / "summary.json").read_text())
    gates_path = base / "gates.jsonl"
    target_path = exp / "gate_targets" / f"{name}_e{epoch:03d}.json"
    return {
        "combined": summary["combined"],
        "per_sequence": summary["per_sequence"],
        "gates": abc.gate_stats(gates_path) if gates_path.is_file() else None,
        "gate_target": (json.loads(target_path.read_text())
                        if target_path.is_file() else None),
    }


def delta(values, reference):
    return {key: values.get(key, 0.0) - reference.get(key, 0.0) for key in SCORE_KEYS}


def per_sequence_change(per, reference):
    changes = {sequence: per[sequence]["HOTA"] - reference[sequence]["HOTA"]
               for sequence in per if sequence in reference}
    improved = sum(1 for value in changes.values() if value > 0.005)
    degraded = sum(1 for value in changes.values() if value < -0.005)
    ranked = sorted(changes.items(), key=lambda item: item[1])
    return {"improved": improved, "degraded": degraded,
            "unchanged": len(changes) - improved - degraded,
            "worst": ranked[:5], "best": ranked[-5:][::-1]}


def training_stats(name):
    group = abc.group_dir(name)
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    log_lines = (group / "training.log").read_text().splitlines()
    preload = {}
    verification = {}
    orders = []
    for line in log_lines:
        if line.startswith("preload: "):
            preload = json.loads(line[len("preload: "):])
        if line.startswith("verification: "):
            verification = json.loads(line[len("verification: "):])
        if line.startswith("epoch order: "):
            orders.append(json.loads(line[len("epoch order: "):])["sequences"])
    last = metrics[-1]
    return {
        "preload": preload, "verification": verification,
        "epoch_seconds_mean": sum(row["epoch_seconds"] for row in metrics) / len(metrics),
        "rss_peak_gb": last.get("rss_peak_gb"),
        "cuda_peak_allocated_mb": last.get("cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": last.get("cuda_peak_reserved_mb"),
        "grad_norms": {row["epoch"]: {
            "motion": row.get("grad_norm_motion_shared"),
            "appearance": row.get("grad_norm_appearance_shared")}
            for row in metrics if row.get("grad_norm_motion_shared") is not None},
        "motion_harm_fraction": last.get("motion_harm_fraction"),
        "motion_harm_weight_share": last.get("motion_harm_weight_share"),
        "epoch_orders": orders, "last_metric": last,
    }


def analyze():
    hist_summary = json.loads(abc.HIST_SUMMARY.read_text())
    hist_combined = hist_summary["combined"]
    hist_per = hist_summary["per_sequence"]
    runs = {}
    for name in RUNS:
        runs[name] = {epoch: load_epoch(DE_EXP, name, epoch) for epoch in EPOCHS}
    reference = {}
    for name in ("A", "B", "C"):
        reference[name] = {epoch: load_epoch(ABC_EXP, name, epoch) for epoch in EPOCHS}
    training = {name: training_stats(name) for name in RUNS}
    abc_training = json.loads((ABC_EXP / "report.json").read_text())["training"]

    summary = {
        "generated_at": abc.now(), "historical_e80": hist_combined,
        "training": {name: {key: value for key, value in stats.items()
                           if key not in ("epoch_orders", "verification")}
                     for name, stats in training.items()},
        "verification": {
            "initial_parameter_sha256": {name: stats["verification"].get("initial_parameter_sha256")
                                         for name, stats in training.items()},
            "initial_parameters_identical": len({
                stats["verification"].get("initial_parameter_sha256")
                for stats in training.values()}) == 1,
            "dropout_modules": {name: stats["verification"].get("dropout_modules")
                                for name, stats in training.items()},
            "optimizer_param_numel": {name: stats["verification"].get("optimizer_param_numel")
                                      for name, stats in training.items()},
        },
        "reference_training": abc_training,
        "runs": {},
        "reference": {},
    }
    order_d = training["D"]["epoch_orders"]
    order_e = training["E"]["epoch_orders"]
    summary["data_order_D_E_identical"] = order_d == order_e
    abc_a_orders = []
    for line in (ref_group("A") / "training.log").read_text().splitlines():
        if line.startswith("epoch order: "):
            abc_a_orders.append(json.loads(line[len("epoch order: "):])["sequences"])
    summary["data_order_vs_A_identical"] = order_d == abc_a_orders
    for name in RUNS:
        summary["runs"][name] = {}
        for epoch in EPOCHS:
            entry = runs[name][epoch]
            summary["runs"][name][str(epoch)] = {
                "combined": entry["combined"],
                "delta_vs_hist_e80": delta(entry["combined"], hist_combined),
                "gates": entry["gates"],
                "gate_target": entry["gate_target"],
                "delta_vs_A": delta(entry["combined"], reference["A"][epoch]["combined"]),
                "per_sequence_vs_A": per_sequence_change(
                    entry["per_sequence"], reference["A"][epoch]["per_sequence"]),
                "per_sequence_vs_hist_e80": per_sequence_change(
                    entry["per_sequence"], hist_per),
            }
        if name == "E":
            for epoch in EPOCHS:
                summary["runs"][name][str(epoch)]["delta_vs_D"] = delta(
                    runs[name][epoch]["combined"], runs["D"][epoch]["combined"])
                summary["runs"][name][str(epoch)]["per_sequence_vs_D"] = per_sequence_change(
                    runs[name][epoch]["per_sequence"], runs["D"][epoch]["per_sequence"])
                summary["runs"][name][str(epoch)]["delta_vs_C"] = delta(
                    runs[name][epoch]["combined"], reference["C"][epoch]["combined"])
                summary["runs"][name][str(epoch)]["per_sequence_vs_C"] = per_sequence_change(
                    runs[name][epoch]["per_sequence"], reference["C"][epoch]["per_sequence"])
    for name in ("A", "B", "C"):
        summary["reference"][name] = {
            str(epoch): {"combined": reference[name][epoch]["combined"],
                         "gates": reference[name][epoch]["gates"],
                         "gate_target": reference[name][epoch]["gate_target"]}
            for epoch in EPOCHS}
    summary["reference"]["hist_e80"] = {"combined": hist_combined}

    candidates = []
    for name in RUNS:
        candidates.append((summary["runs"][name]["20"]["combined"]["HOTA"],
                           f"{name}_e020", abc.checkpoint(name, 20)))
    for name in ("A", "B", "C"):
        candidates.append((reference[name][20]["combined"]["HOTA"],
                           f"{name}_e020", ref_checkpoint(name, 20)))
    best_hota, best_label, best_ckpt = max(candidates, key=lambda item: item[0])
    best_de = max(
        ((summary["runs"][name]["20"]["combined"]["HOTA"], f"{name}_e020",
          abc.checkpoint(name, 20)) for name in RUNS), key=lambda item: item[0])
    summary["best_overall"] = {"label": best_label, "hota": best_hota,
                               "checkpoint": str(best_ckpt)}
    summary["best_de"] = {"label": best_de[1], "hota": best_de[0],
                          "checkpoint": str(best_de[2])}
    report = render(summary)
    (DE_EXP / "report.md").write_text(report + "\n")
    (DE_EXP / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)


def render(summary):
    ref = summary["reference"]
    hist = summary["historical_e80"]
    training = summary["training"]
    ref_training = summary["reference_training"]

    def val_row(label, combined):
        return (f"| {label} | {combined['HOTA']*100:.3f} | {combined['AssA']*100:.3f} | "
                f"{combined['DetA']*100:.3f} | {combined['IDF1']*100:.3f} | "
                f"{combined['MOTA']*100:.3f} | {combined['IDSW']:.0f} | "
                f"{combined['Frag']:.0f} |")

    lines = ["# SportsMOT D/E：dropout 0.05 与 motion_harm_weight 2", "",
             "协议与 A/B/C 相同：官方 train 45 / val 45，query_film_bilinear 从头全参数，"
             "seed 42，shards=1 全量驻留，20 个全量轮（6,112,620 样本，等于旧 e80），"
             "余弦按 50 轮规划、warmup 80、终点 LR 6.5899e-5。", "",
             "评估清单：D/E 仅对最终 epoch_020 做 45 条 val raw 评估和门控目标分析；"
             "e10/e15 仅保存 checkpoint。已完成的中期评估产物保留但不纳入比较。", "",
             "## 训练与资源", "",
             "| 组 | Dropout | harm | 加载 s | 平均轮时 s | RSS 峰值 GiB | CUDA 峰值 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stats in training.items():
        lines.append(f"| {name} | {RUNS[name]['dropout']} | {RUNS[name]['motion_harm_weight']} | "
                     f"{stats['preload'].get('seconds', '-')} | "
                     f"{stats['epoch_seconds_mean']:.1f} | {stats['rss_peak_gb']} | "
                     f"{stats['cuda_peak_allocated_mb']} |")
    for name in ("A", "B", "C"):
        stats = ref_training[name]
        lines.append(f"| {name}（参考） | {stats['dropout']} | {stats['motion_harm_weight']} | "
                     f"{stats['preload'].get('seconds', '-')} | "
                     f"{stats['epoch_seconds_mean']:.1f} | {stats['rss_peak_gb']} | "
                     f"{stats['cuda_peak_allocated_mb']} |")

    lines += ["", "## Val（45 条 raw）", "",
              "| 组 | 轮次 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("D", "E"):
        for epoch in EPOCHS:
            entry = summary["runs"][name][str(epoch)]
            lines.append(val_row(f"{name} e{epoch}", entry["combined"]))
    for name in ("A", "B", "C"):
        for epoch in EPOCHS:
            lines.append(val_row(f"{name} e{epoch} (参考)", ref[name][str(epoch)]["combined"]))
    lines.append(val_row("历史 e80", hist))

    lines += ["", "## D − A：dropout 0.05 的效果（同轮次）", "",
              "| 轮次 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | 逐序列改善/退化/不变 |",
              "|---:|---:|---:|---:|---:|---|"]
    for epoch in EPOCHS:
        entry = summary["runs"]["D"][str(epoch)]
        d = entry["delta_vs_A"]
        change = entry["per_sequence_vs_A"]
        lines.append(f"| e{epoch} | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                     f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | "
                     f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")
    lines += ["", "## 三档 dropout 对照（A=0，D=0.05，B=0.1）", "",
              "| 轮次 | ΔHOTA D−A | ΔHOTA B−A |", "|---:|---:|---:|"]
    for epoch in EPOCHS:
        d = summary["runs"]["D"][str(epoch)]["delta_vs_A"]["HOTA"] * 100
        b = (ref["B"][str(epoch)]["combined"]["HOTA"] - ref["A"][str(epoch)]["combined"]["HOTA"]) * 100
        lines.append(f"| e{epoch} | {d:+.3f} | {b:+.3f} |")

    lines += ["", "## E − D：harm_weight 2 的效果（同轮次）", "",
              "| 轮次 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW | 逐序列改善/退化/不变 |",
              "|---:|---:|---:|---:|---:|---:|---|"]
    for epoch in EPOCHS:
        entry = summary["runs"]["E"][str(epoch)]
        d = entry["delta_vs_D"]
        change = entry["per_sequence_vs_D"]
        lines.append(f"| e{epoch} | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                     f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
                     f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")

    lines += ["", "## 三档 harm 对照（A/D=1，E=2，C=8）", "",
              "| 轮次 | HOTA A | HOTA E | HOTA C | ΔHOTA E−A | ΔHOTA C−A |",
              "|---:|---:|---:|---:|---:|---:|"]
    for epoch in EPOCHS:
        a = ref["A"][str(epoch)]["combined"]["HOTA"] * 100
        e = summary["runs"]["E"][str(epoch)]["combined"]["HOTA"] * 100
        c = ref["C"][str(epoch)]["combined"]["HOTA"] * 100
        lines.append(f"| e{epoch} | {a:.3f} | {e:.3f} | {c:.3f} | {e-a:+.3f} | {c-a:+.3f} |")

    lines += ["", "## 运动/外观 gate 分布（跟踪时）", "",
              "| 组 | 轮次 | motion min/mean/max | appearance min/mean/max |",
              "|---|---:|---|---|"]
    for name in ("D", "E"):
        for epoch in EPOCHS:
            g = summary["runs"][name][str(epoch)]["gates"]
            if g:
                lines.append(f"| {name} | e{epoch} | {g['motion_min']:.4f}/{g['motion_mean']:.4f}/"
                             f"{g['motion_max']:.4f} | {g['appearance_min']:.4f}/"
                             f"{g['appearance_mean']:.4f}/{g['appearance_max']:.4f} |")
    for name in ("A", "B", "C"):
        g = ref[name]["20"]["gates"]
        if g:
            lines.append(f"| {name}（e20 参考） | e20 | {g['motion_min']:.4f}/"
                         f"{g['motion_mean']:.4f}/{g['motion_max']:.4f} | "
                         f"{g['appearance_min']:.4f}/{g['appearance_mean']:.4f}/"
                         f"{g['appearance_max']:.4f} |")

    lines += ["", "## 门控目标分析（D/E 与 harm8 参考）", "",
              "| 组 | 轮次 | 有害 gate | 有害节省/事件 | 有益 gate | 有益误伤/事件 | 净 DIoU 变化 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("D", "E"):
        for epoch in EPOCHS:
            target = summary["runs"][name][str(epoch)]["gate_target"]["new"]
            lines.append(f"| {name} | e{epoch} | {target['harmful_gate_mean']:.4f} | "
                         f"{target['harmful_savings_mean']:.6f} | "
                         f"{target['useful_gate_mean']:.4f} | "
                         f"{target['useful_damage_mean']:.6f} | {target['net_loss_change']:+.6f} |")
    for name in ("A", "C"):
        for epoch in EPOCHS:
            target = ref[name][str(epoch)]["gate_target"]["new"]
            lines.append(f"| {name}（参考） | e{epoch} | {target['harmful_gate_mean']:.4f} | "
                         f"{target['harmful_savings_mean']:.6f} | "
                         f"{target['useful_gate_mean']:.4f} | "
                         f"{target['useful_damage_mean']:.6f} | {target['net_loss_change']:+.6f} |")

    lines += ["", "## E 与 harm8（C）：节省/误伤如何变化", "",
              "| 轮次 | 节省 E | 节省 C | 误伤 E | 误伤 C | 净损失 E | 净损失 C |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for epoch in EPOCHS:
        e = summary["runs"]["E"][str(epoch)]["gate_target"]["new"]
        c = ref["C"][str(epoch)]["gate_target"]["new"]
        lines.append(f"| e{epoch} | {e['harmful_savings_mean']:.6f} | "
                     f"{c['harmful_savings_mean']:.6f} | {e['useful_damage_mean']:.6f} | "
                     f"{c['useful_damage_mean']:.6f} | {e['net_loss_change']:+.6f} | "
                     f"{c['net_loss_change']:+.6f} |")

    lines += ["", "## 共享主干梯度范数（首 batch，epoch 1/5/10/15/20）", "",
              "| 组 | epoch | motion | appearance |", "|---|---:|---:|---:|"]
    for name, stats in training.items():
        for epoch, values in sorted(stats["grad_norms"].items()):
            lines.append(f"| {name} | {epoch} | {values['motion']:.4f} | "
                         f"{values['appearance']:.4f} |")

    lines += ["", "## 逐序列（e20）", ""]
    for name in ("D", "E"):
        change = summary["runs"][name]["20"]["per_sequence_vs_A"]
        lines.append(f"- {name} vs A：改善 {change['improved']} / 退化 {change['degraded']} / "
                     f"不变 {change['unchanged']}；最好 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["best"][:3])
                     + "；最差 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["worst"][:3]))
    change = summary["runs"]["E"]["20"]["per_sequence_vs_D"]
    lines.append(f"- E vs D：改善 {change['improved']} / 退化 {change['degraded']} / "
                 f"不变 {change['unchanged']}；最好 "
                 + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["best"][:3])
                 + "；最差 "
                 + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["worst"][:3]))
    change = summary["runs"]["E"]["20"]["per_sequence_vs_C"]
    lines.append(f"- E vs C（harm2 vs harm8）：改善 {change['improved']} / 退化 {change['degraded']} / "
                 f"不变 {change['unchanged']}")

    verification = summary["verification"]
    lines += ["", "## 核验", "",
              f"- D/E 初始参数哈希一致：{verification['initial_parameters_identical']} "
              f"({verification['initial_parameter_sha256']['D'][:16]}…)",
              f"- 优化器参数数：{verification['optimizer_param_numel']}",
              f"- Dropout 模块：{json.dumps(verification['dropout_modules']['D'], ensure_ascii=False)}",
              f"- D/E 数据顺序一致：{summary['data_order_D_E_identical']}；"
              f"与 A 一致：{summary['data_order_vs_A_identical']}"]
    best = summary["best_overall"]
    best_de = summary["best_de"]
    lines += ["", "## 最佳配置与 checkpoint", "",
              f"- 本轮最佳：{best_de['label']}，val HOTA {best_de['hota']*100:.3f}，"
              f"checkpoint `{best_de['checkpoint']}`",
              f"- 五组合计最佳：{best['label']}，val HOTA {best['hota']*100:.3f}，"
              f"checkpoint `{best['checkpoint']}`",
              "", "单 seed 结果；小差异需要 seed 复现。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "probe", "train", "eval", "analyze"),
                        default="all")
    args = parser.parse_args()
    if args.stage == "probe":
        abc.probe()
        return
    DE_EXP.mkdir(parents=True, exist_ok=True)
    abc.LOGS.mkdir(parents=True, exist_ok=True)
    state = abc.load_state()
    if args.stage in ("all", "train"):
        print(json.dumps({"active_jobs": abc.active_jobs(),
                          "mem_available_gib": abc.mem_available_gib(),
                          "gpu": abc.gpu_free_mib()}, indent=1), flush=True)
        if not abc.PREFLIGHT_PATH.is_file():
            abc.run("probe", [abc.PY, "-u", str(Path(__file__).resolve()), "--stage", "probe"])
        preflight = json.loads(abc.PREFLIGHT_PATH.read_text())
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        for name in RUNS:
            abc.train_run(name, RUNS[name], preflight, state)
            verification = abc.verify_run(name, RUNS[name], preflight)
            state["runs"].setdefault(name, {})["verify"] = {
                "status": "complete", "at": abc.now(),
                "model_param_numel": verification["verification"].get("model_param_numel"),
                "initial_parameter_sha256": verification["verification"].get("initial_parameter_sha256"),
            }
            abc.save_state(state)
    if args.stage in ("all", "eval"):
        for name in RUNS:
            for epoch in EPOCHS:
                abc.eval_run(name, epoch, state)
                abc.gate_run(name, epoch, state)
    if args.stage in ("all", "analyze"):
        analyze()


if __name__ == "__main__":
    main()
