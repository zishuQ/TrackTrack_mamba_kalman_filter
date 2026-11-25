#!/usr/bin/env python3
"""SportsMOT four-group mechanism experiments (packed replay v2).

Baseline = existing E_e20 (one-step motion DIoU, no augmentation).
E_e100 is retained separately as a historical reference.
G1-3 = three-frame continuous motion loss, no augmentation.
G1-5 = five-frame continuous motion loss, no augmentation.
G2-3 = three-frame history warmup + anchor one-step target, MoTTrack augmentation.
G3-3 = three-frame continuous loss + MoTTrack augmentation.

All groups: official train 45, query_film_bilinear from scratch, dropout 0.05,
motion_harm_weight 2, seed 42, shards=1 with the pool resident in RAM,
50-epoch cosine plan (16,050 steps), stopped at epoch 20 (6,420 steps),
warmup 80. The learning rate at the stopping point is about 6.59e-5.

Only the final epoch 20 is evaluated (45-sequence val raw + gate analysis).
Formal trainval training / test / submission is NOT started automatically;
the report waits for the user's decision.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sportsmot_abc_experiment as abc  # noqa: E402

abc.HEAVY = set(abc.HEAVY) | {"train_window_groups.py"}

# The old run used incompatible input normalization. Keep all its artifacts.
SOURCE_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_mechanisms_20261124"
MECH_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_mechanisms_e20_20261124"
BASELINE_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_de_20261123"
DATA = SOURCE_EXP / "data"
E100_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_e100_20261123"
abc.EXP = MECH_EXP
abc.LOGS = MECH_EXP / "logs"
abc.EVALS = MECH_EXP / "evals"
abc.GATES = MECH_EXP / "gate_targets"
abc.STATE_PATH = MECH_EXP / "state.json"
abc.PREFLIGHT_PATH = MECH_EXP / "preflight.json"
RUNS = {
    "G1-3": {"window_frames": 3, "window_supervision": "full", "augment": "none"},
    "G1-5": {"window_frames": 5, "window_supervision": "full", "augment": "none"},
    "G2-3": {"window_frames": 3, "window_supervision": "anchor", "augment": "mottrack"},
    "G3-3": {"window_frames": 3, "window_supervision": "full", "augment": "mottrack"},
}
COMMON = {
    "planned_epochs": 50, "stop_epoch": 20, "snapshot_every": 5,
    "batch_size": 1024, "micro_batch": 1024,
    "lr": 1e-4, "weight_decay": 1e-4, "seed": 42, "rank_weight": 18.0,
    "rank_alpha": 0.95, "rank_tau": 0.1, "column_weight": 50.0,
    "motion_harm_weight": 2.0, "dropout": 0.05, "warmup_steps": 80,
    "grad_norm_every": 5,
}
EVAL_EPOCH = 20
SCORE_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")


def group_dir(name):
    return MECH_EXP / "runs" / name / "groups/appearance_row_column"


def checkpoint(name, epoch):
    return group_dir(name) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def data_steps():
    samples = 0
    steps = 0
    for path in sorted(DATA.iterdir()):
        if not (path / "samples.npz").is_file():
            continue
        import numpy as np
        with np.load(path / "samples.npz", allow_pickle=False) as archive:
            count = int(archive["frame"].shape[0])
        samples += count
        steps += math.ceil(count / COMMON["batch_size"])
    planned_steps = steps * COMMON["planned_epochs"]
    actual_steps = steps * COMMON["stop_epoch"]
    warmup = min(COMMON["warmup_steps"], planned_steps)

    def lr_at(step):
        if step < warmup:
            return COMMON["lr"] * (step + 1) / warmup
        progress = (step - warmup) / max(1, planned_steps - warmup)
        return COMMON["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))

    return {"sequences": len(list(DATA.glob("*/samples.npz"))), "samples": samples,
            "steps_per_epoch": steps, "planned_epochs": COMMON["planned_epochs"],
            "stop_epoch": COMMON["stop_epoch"], "total_steps": planned_steps,
            "actual_steps": actual_steps, "warmup_steps": warmup,
            "lr_start": COMMON["lr"], "lr_at_stop": lr_at(actual_steps),
            "lr_end": lr_at(actual_steps)}


def preflight():
    payload = {**data_steps(), "implementation": "packed_replay_v2",
               "association_context": "frozen_replay_candidates",
               "g2_policy": "fixed_write_noise_history_anchor_only",
               "created_at": abc.now(),
               "mem_available_gib": round(abc.mem_available_gib(), 2),
               "gpu": abc.gpu_free_mib()}
    MECH_EXP.mkdir(parents=True, exist_ok=True)
    (MECH_EXP / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def train_command(name, spec, pre):
    command = [
        abc.PY, "-u", "tools/train_window_groups.py",
        "--data-root", str(DATA), "--groups-root", str(MECH_EXP / "runs" / name / "groups"),
        "--max-epochs", str(COMMON["planned_epochs"]),
        "--stop-after-epoch", str(COMMON["stop_epoch"]),
        "--snapshot-every", str(COMMON["snapshot_every"]),
        "--batch-size", str(COMMON["batch_size"]),
        "--micro-batch", str(COMMON["micro_batch"]),
        "--lr", str(COMMON["lr"]), "--weight-decay", str(COMMON["weight_decay"]),
        "--seed", str(COMMON["seed"]),
        "--rank-weight", str(COMMON["rank_weight"]),
        "--rank-alpha", str(COMMON["rank_alpha"]),
        "--rank-tau", str(COMMON["rank_tau"]),
        "--column-weight", str(COMMON["column_weight"]),
        "--motion-harm-weight", str(COMMON["motion_harm_weight"]),
        "--dropout", str(COMMON["dropout"]),
        "--preload-features", "--warmup-steps", str(COMMON["warmup_steps"]),
        "--grad-norm-every", str(COMMON["grad_norm_every"]),
        "--window-frames", str(spec["window_frames"]),
        "--window-supervision", spec["window_supervision"],
        "--augment", spec["augment"], "--device", "cuda",
    ]
    if spec["augment"] == "mottrack":
        command += ["--augment-pos-std", "1", "--augment-size-std", "1",
                    "--augment-miss", "0.004", "--augment-burst", "0",
                    "--augment-continue", "0.4"]
    last = group_dir(name) / "checkpoints/last.pt"
    if last.is_file() and not checkpoint(name, EVAL_EPOCH).is_file():
        command.append("--resume")
    return command


def train_run(name, spec, pre, state):
    if checkpoint(name, EVAL_EPOCH).is_file():
        print(f"{name}: e{EVAL_EPOCH} exists; skip", flush=True)
        return
    abc.check_resources()
    started = time.time()
    abc.run(f"train_{name}_e{EVAL_EPOCH}", train_command(name, spec, pre))
    state["runs"].setdefault(name, {})["train"] = {
        "status": "complete", "seconds": round(time.time() - started, 1)}
    abc.save_state(state)


def verify(name, spec, pre):
    import torch
    group = group_dir(name)
    config = json.loads((group / "config.json").read_text())
    expected = {"window_frames": spec["window_frames"],
                "window_supervision": spec["window_supervision"],
                "augment": spec["augment"], "epochs": COMMON["planned_epochs"],
                "stop_after_epoch": COMMON["stop_epoch"],
                "total_steps": pre["total_steps"], "warmup_steps": pre["warmup_steps"],
                "dropout": COMMON["dropout"],
                "motion_harm_weight": COMMON["motion_harm_weight"],
                "batch_size": COMMON["batch_size"], "seed": COMMON["seed"]}
    mismatched = {key: {"expected": value, "actual": config.get(key)}
                  for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"{name}: config mismatch {mismatched}")
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(metrics) != COMMON["stop_epoch"] or int(metrics[-1]["epoch"]) != COMMON["stop_epoch"]:
        raise RuntimeError(f"{name}: metrics endpoint {len(metrics)}/{metrics[-1].get('epoch')}")
    last = metrics[-1]
    if int(last["global_step"]) != pre["actual_steps"]:
        raise RuntimeError(f"{name}: global_step {last['global_step']}")
    if abs(float(last["lr"]) - pre["lr_at_stop"]) > 1e-9:
        raise RuntimeError(f"{name}: lr {last['lr']} != {pre['lr_at_stop']}")
    payload = torch.load(checkpoint(name, EVAL_EPOCH), map_location="cpu", weights_only=False)
    model_config = payload.get("model_config") or {}
    training_config = payload.get("training_config") or {}
    reload_ok = (
        int(payload.get("epoch", -1)) == EVAL_EPOCH
        and int(payload.get("global_step", -1)) == pre["actual_steps"]
        and float(model_config.get("dropout", -1)) == COMMON["dropout"]
        and training_config.get("window_frames") == spec["window_frames"]
        and training_config.get("window_supervision") == spec["window_supervision"]
        and training_config.get("augment") == spec["augment"]
        and training_config.get("stop_after_epoch") == COMMON["stop_epoch"]
        and payload.get("rng_state") is not None)
    if not reload_ok:
        raise RuntimeError(f"{name}: checkpoint reload verification failed")
    return {"last": last, "metrics": metrics, "reload_ok": reload_ok,
            "config": config}


def load_epoch(exp, name, epoch):
    base = exp / "evals" / f"{name}_e{epoch:03d}"
    summary = json.loads((base / "summary.json").read_text())
    gates_path = base / "gates.jsonl"
    target_path = exp / "gate_targets" / f"{name}_e{epoch:03d}.json"
    return {"combined": summary["combined"], "per_sequence": summary["per_sequence"],
            "gates": abc.gate_stats(gates_path) if gates_path.is_file() else None,
            "gate_target": json.loads(target_path.read_text())
            if target_path.is_file() else None}


def delta(values, reference):
    return {key: values.get(key, 0) - reference.get(key, 0) for key in SCORE_KEYS}


def per_change(per, reference):
    changes = {seq: per[seq]["HOTA"] - reference[seq]["HOTA"]
               for seq in per if seq in reference}
    improved = sum(1 for v in changes.values() if v > 0.005)
    degraded = sum(1 for v in changes.values() if v < -0.005)
    ranked = sorted(changes.items(), key=lambda item: item[1])
    return {"improved": improved, "degraded": degraded,
            "unchanged": len(changes) - improved - degraded,
            "best": ranked[-5:][::-1], "worst": ranked[:5]}


def analyze():
    baseline = load_epoch(BASELINE_EXP, "E", 20)
    strong_baseline = load_epoch(BASELINE_EXP, "D", 20)
    historical = load_epoch(E100_EXP, "E", 100)
    runs = {name: load_epoch(MECH_EXP, name, EVAL_EPOCH) for name in RUNS}
    state = abc.load_state()
    training = {}
    for name in RUNS:
        group = group_dir(name)
        metrics = [json.loads(line) for line in
                   (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
        training[name] = {
            "epoch_seconds_mean": sum(r["epoch_seconds"] for r in metrics) / len(metrics),
            "epoch_seconds_total": sum(r["epoch_seconds"] for r in metrics),
            "rss_peak_gb": metrics[-1].get("rss_peak_gb"),
            "cuda_peak_allocated_mb": metrics[-1].get("cuda_peak_allocated_mb"),
            "cuda_peak_reserved_mb": metrics[-1].get("cuda_peak_reserved_mb"),
            "augmented_missing_last": metrics[-1].get("augmented_missing"),
            "grad_norms": {r["epoch"]: {"motion": r.get("grad_norm_motion_shared"),
                                        "appearance": r.get("grad_norm_appearance_shared")}
                           for r in metrics if r.get("grad_norm_motion_shared") is not None},
            "state": state["runs"].get(name, {}),
        }
    summary = {"generated_at": abc.now(), "preflight": json.loads(
        (MECH_EXP / "preflight.json").read_text()), "runs": {}, "training": training,
        "baseline": {"combined": baseline["combined"], "gates": baseline["gates"],
                     "gate_target": baseline["gate_target"]},
        "strong_baseline_harm1": {"combined": strong_baseline["combined"]},
        "historical_e100": {"combined": historical["combined"]},
        "deltas": {}, "per_sequence": {}}
    for name, entry in runs.items():
        summary["runs"][name] = {
            "combined": entry["combined"], "gates": entry["gates"],
            "gate_target": entry["gate_target"],
            "delta_vs_baseline": delta(entry["combined"], baseline["combined"]),
            "per_sequence_vs_baseline": per_change(
                entry["per_sequence"], baseline["per_sequence"]),
        }
    for label, first, second in (
            ("G1-5_vs_G1-3", "G1-5", "G1-3"),
            ("G2-3_vs_G1-3", "G2-3", "G1-3"),
            ("G3-3_vs_G1-3", "G3-3", "G1-3"),
            ("G3-3_vs_G2-3", "G3-3", "G2-3")):
        summary["deltas"][label] = delta(runs[first]["combined"], runs[second]["combined"])
        summary["per_sequence"][label] = per_change(
            runs[first]["per_sequence"], runs[second]["per_sequence"])
    a, b = training["G1-5"], training["G1-3"]
    summary["cost_delta_G1_5_vs_G1_3"] = {
        "epoch_seconds": a["epoch_seconds_mean"] - b["epoch_seconds_mean"],
        "total_seconds": a["epoch_seconds_total"] - b["epoch_seconds_total"],
        "rss_peak_gb": (a["rss_peak_gb"] or 0) - (b["rss_peak_gb"] or 0),
        "cuda_peak_allocated_mb": (a["cuda_peak_allocated_mb"] or 0)
        - (b["cuda_peak_allocated_mb"] or 0),
    }
    candidates = [("E_e20_baseline", {"combined": baseline["combined"]})] + list(runs.items())
    label, entry = max(candidates, key=lambda item: tuple(
        item[1]["combined"][metric] for metric in ("HOTA", "AssA", "IDF1")))
    summary["best"] = {"label": label, "hota": entry["combined"]["HOTA"]}
    report = render(summary)
    (MECH_EXP / "report.md").write_text(report + "\n")
    (MECH_EXP / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)


def render(summary):
    pre = summary["preflight"]
    baseline = summary["baseline"]
    historical = summary["historical_e100"]["combined"]
    runs = summary["runs"]
    groups = list(RUNS)
    specs = {"G1-3": (3, "full", "none"), "G1-5": (5, "full", "none"),
             "G2-3": (3, "anchor", "mottrack"), "G3-3": (3, "full", "mottrack")}
    lines = ["# SportsMOT 四组机制实验（e20 预算）", "",
             f"预算：train 45，{pre['samples']} 样本/轮，{pre['steps_per_epoch']} 步/轮；"
             f"{pre['planned_epochs']} 轮余弦计划，实际训练 {pre['stop_epoch']} 轮 = "
             f"{pre['actual_steps']} 步（计划 {pre['total_steps']} 步），warmup "
             f"{pre['warmup_steps']}，e20 末 LR {pre['lr_at_stop']:.4e}；"
             f"实现 {pre.get('implementation','packed_replay_v2')}。", "",
             "| 组 | dropout | harm | 窗口 | 监督 | 增强 | 轮时 s | 总训练 s | RSS GiB | CUDA MiB |",
             "|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for name in groups:
        stats = summary["training"][name]
        window, supervision, augment = specs[name]
        lines.append(f"| {name} | 0.05 | 2 | {window} | {supervision} | {augment} | "
                     f"{stats['epoch_seconds_mean']:.1f} | "
                     f"{stats['epoch_seconds_total']:.0f} | {stats['rss_peak_gb']} | "
                     f"{stats['cuda_peak_allocated_mb']} |")
    cost = summary["cost_delta_G1_5_vs_G1_3"]
    lines += ["", f"G1-5 相对 G1-3 的额外成本：轮时 {cost['epoch_seconds']:+.1f} s/轮，"
                  f"20 轮合计 {cost['total_seconds']:+.0f} s，"
                  f"RSS 峰值 {cost['rss_peak_gb']:+.3f} GiB，"
                  f"CUDA 峰值 {cost['cuda_peak_allocated_mb']:+.1f} MiB。",
              "", "## Val e20（45 条 raw）", "",
              "| 组 | 预算 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag | ΔHOTA vs E_e20 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def row(label, budget, combined, delta_value=None):
        return (f"| {label} | {budget} | {combined['HOTA']*100:.3f} | "
                f"{combined['AssA']*100:.3f} | {combined['DetA']*100:.3f} | "
                f"{combined['IDF1']*100:.3f} | {combined['MOTA']*100:.3f} | "
                f"{combined['IDSW']:.0f} | {combined['Frag']:.0f} | "
                f"{'-' if delta_value is None else f'{delta_value*100:+.3f}'} |")
    lines.append(row("D_e20（harm1参照）", "20 轮 / 6420 步", summary["strong_baseline_harm1"]["combined"]))
    lines.append(row("E_e20（harm2基线）", "20 轮 / 6420 步", baseline["combined"]))
    for name in groups:
        lines.append(row(name, "20 轮 / 6420 步", runs[name]["combined"],
                         runs[name]["delta_vs_baseline"]["HOTA"]))
    lines.append(row("E_e100（历史参考）", "100 轮 / 32100 步", historical))
    lines += ["", "D_e20（harm1）HOTA83.660高于本轮四组最佳G1-3（harm2）的83.517。"
              "三帧连续harm1已单独补测；同harm比较用于判断连续监督作用。"
              "目前指标变化反映整套全参数训练方案，运动与外观gate均变化，运动分支的独立贡献尚未隔离。"]
    lines += ["", "## G1-5 vs G1-3（额外两帧）", "",
              "| 对比 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW | 逐序列改善/退化/不变 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    d = summary["deltas"]["G1-5_vs_G1-3"]
    change = summary["per_sequence"]["G1-5_vs_G1-3"]
    lines.append(f"| G1-5 − G1-3 | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                 f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
                 f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")
    lines += ["", "## 各组与基线、机制之间的差值", "",
              "| 对比 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW | 逐序列改善/退化/不变 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name in groups:
        d = runs[name]["delta_vs_baseline"]
        change = runs[name]["per_sequence_vs_baseline"]
        lines.append(f"| {name} − E_e20 | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                     f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
                     f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")
    for label in ("G2-3_vs_G1-3", "G3-3_vs_G1-3", "G3-3_vs_G2-3"):
        d = summary["deltas"][label]
        change = summary["per_sequence"][label]
        lines.append(f"| {label.replace('_',' ')} | {d['HOTA']*100:+.3f} | "
                     f"{d['AssA']*100:+.3f} | {d['DetA']*100:+.3f} | "
                     f"{d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
                     f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")
    lines += ["", "## 门控分布（跟踪时，e20）", "",
              "以下 min/mean/max 是 45 个序列各自平均 gate 的最小值、等权均值和最大值。", "",
              "| 组 | motion min/mean/max | appearance min/mean/max |",
              "|---|---|---|"]
    for label, entry in [("E_e20（基线）", baseline)] + [(name, runs[name]) for name in groups]:
        g = entry["gates"]
        lines.append(f"| {label} | {g['motion_min']:.4f}/{g['motion_mean']:.4f}/"
                     f"{g['motion_max']:.4f} | {g['appearance_min']:.4f}/"
                     f"{g['appearance_mean']:.4f}/{g['appearance_max']:.4f} |")
    lines += ["", "## 门控目标（e20）", "",
              "| 组 | 有害 gate | 有益 gate | 分离度 | 节省/事件 | 误伤/事件 | 净 DIoU 变化 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in [("E_e20（基线）", baseline)] + [(name, runs[name]) for name in groups]:
        target = entry["gate_target"]["new"]
        lines.append(f"| {label} | {target['harmful_gate_mean']:.4f} | "
                     f"{target['useful_gate_mean']:.4f} | "
                     f"{target['useful_gate_mean']-target['harmful_gate_mean']:.4f} | "
                     f"{target['harmful_savings_mean']:.6f} | "
                     f"{target['useful_damage_mean']:.6f} | {target['net_loss_change']:+.6f} |")
    lines += ["", "## 低频梯度范数（运动/外观对共享主干）", ""]
    for name in groups:
        stats = summary["training"][name]["grad_norms"]
        points = [p for p in (1, 5, 10, 20) if p in stats]
        lines.append(f"- {name}: " + "；".join(
            f"e{p} {stats[p]['motion']:.5f}/{stats[p]['appearance']:.4f}" for p in points))
    lines += ["", "## 结论", "",
              f"- 最高 val HOTA：{summary['best']['label']} "
              f"{summary['best']['hota']*100:.3f}。单 seed；选优按 HOTA，"
              "并列依次看 AssA/IDF1。",
              "- 本轮止于 train45 的 e20 训练、val45 评估与报告；"
              "正式 trainval 训练、test/post 与提交等待用户决定。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "train", "eval", "analyze", "formal", "plan"), default="all")
    parser.add_argument("--runs", nargs="*", choices=tuple(RUNS), default=list(RUNS))
    args = parser.parse_args()
    if args.stage == "plan":
        print(json.dumps({"experiment":str(MECH_EXP),"data":str(DATA),
                          "runs":RUNS,"common":COMMON,"final_only_epoch":20,
                          "formal_after_selection":False},indent=2))
        return
    MECH_EXP.mkdir(parents=True, exist_ok=True)
    abc.LOGS.mkdir(parents=True, exist_ok=True)
    pre = json.loads((MECH_EXP / "preflight.json").read_text()) \
        if (MECH_EXP / "preflight.json").is_file() else preflight()
    print(json.dumps(pre, indent=2, sort_keys=True), flush=True)
    state = abc.load_state()
    if args.stage in ("all", "train"):
        for name in args.runs:
            train_run(name, RUNS[name], pre, state)
            state["runs"].setdefault(name, {})["verify"] = verify(name, RUNS[name], pre)
            abc.save_state(state)
    if args.stage in ("all", "eval"):
        for name in args.runs:
            abc.eval_run(name, EVAL_EPOCH, state)
            abc.gate_run(name, EVAL_EPOCH, state)
    if args.stage in ("all", "analyze"):
        analyze()
    if args.stage == "formal":
        import run_sportsmot_mechanism_formal
        run_sportsmot_mechanism_formal.run(sys.modules[__name__])


if __name__ == "__main__":
    main()


