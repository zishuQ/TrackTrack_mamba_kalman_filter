#!/usr/bin/env python3
"""SportsMOT E configuration, 100 full epochs from scratch.

dropout=0.05, motion_harm_weight=2, query_film_bilinear, seed 42, shards=1 with
the whole pool resident in RAM, batch 1024.  100-epoch cosine from 1e-4 to 0
with a 160-step warmup (0.4984% of 32,100 steps).  Only the final epoch 100 is
evaluated: one 45-sequence raw val pass and one gate-target analysis.
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

E100_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_e100_20261123"
DE_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_de_20261123"
E30_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_de_e30_20261123"
ABC_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_abc_20261123"
abc.EXP = E100_EXP
abc.LOGS = E100_EXP / "logs"
abc.EVALS = E100_EXP / "evals"
abc.GATES = E100_EXP / "gate_targets"
abc.STATE_PATH = E100_EXP / "state.json"
abc.PREFLIGHT_PATH = E100_EXP / "preflight.json"
abc.RUNS = {"E": {"dropout": 0.05, "motion_harm_weight": 2.0}}
NAME = "E"
SPEC = abc.RUNS[NAME]
TOTAL_EPOCHS = 100
EVAL_EPOCH = 100
SCORE_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")


def preflight():
    base = json.loads((DE_EXP / "preflight.json").read_text())
    steps = base["steps_per_epoch"]
    total = steps * TOTAL_EPOCHS
    warm = round(0.005 * total)
    samples = base["samples"] * TOTAL_EPOCHS

    def lr_at(step):
        if step < warm:
            return 1e-4 * (step + 1) / warm
        progress = (step - warm) / (total - warm)
        return 1e-4 * 0.5 * (1.0 + math.cos(math.pi * progress))

    payload = {
        "created_at": abc.now(), "train_sequences": base["train_sequences"],
        "val_sequences": base["val_sequences"], "samples_per_epoch": base["samples"],
        "steps_per_epoch": steps, "total_epochs": TOTAL_EPOCHS,
        "total_steps": total, "total_samples": samples,
        "warmup_steps": warm, "warmup_fraction": warm / total,
        "lr_start": 1e-4, "lr_end": lr_at(total - 1),
        "lr_at": {str(epoch): lr_at(epoch * steps - 1)
                  for epoch in (1, 20, 30, 50, 75, 100)},
        "mem_available_gib": round(abc.mem_available_gib(), 2),
        "gpu": abc.gpu_free_mib(),
    }
    E100_EXP.mkdir(parents=True, exist_ok=True)
    (E100_EXP / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def checkpoint(epoch):
    return abc.group_dir(NAME) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def train_command(preflight_payload):
    groups_root = E100_EXP / "runs" / NAME / "groups"
    command = [
        abc.PY, "-u", "tools/train_supervision_groups.py",
        "--group", "appearance_row_column",
        "--dataset", "SportsMOT", "--split", "train", "--from-scratch",
        "--data-root", str(abc.DATA), "--groups-root", str(groups_root),
        "--shards", "1", "--max-epochs", str(TOTAL_EPOCHS), "--snapshot-every", "5",
        "--batch-size", "1024", "--lr", "1e-4", "--weight-decay", "1e-4",
        "--seed", "42", "--rank-weight", "18", "--rank-alpha", "0.95",
        "--rank-tau", "0.1", "--column-weight", "50",
        "--motion-harm-weight", str(SPEC["motion_harm_weight"]),
        "--dropout", str(SPEC["dropout"]),
        "--warmup-steps", str(preflight_payload["warmup_steps"]),
        "--grad-norm-every", "5", "--device", "cuda",
    ]
    last = abc.group_dir(NAME) / "checkpoints/last.pt"
    if last.is_file() and not checkpoint(EVAL_EPOCH).is_file():
        command.append("--resume")
    return command


def train_run(preflight_payload, state):
    if checkpoint(EVAL_EPOCH).is_file():
        print(f"epoch_{EVAL_EPOCH} exists; skip training", flush=True)
        return
    abc.check_resources()
    started = time.time()
    abc.run(f"train_{NAME}_e{EVAL_EPOCH}", train_command(preflight_payload))
    state["runs"].setdefault(NAME, {})["train"] = {
        "status": "complete", "seconds": round(time.time() - started, 1)}
    abc.save_state(state)


def verify(preflight_payload):
    import torch
    group = abc.group_dir(NAME)
    config = json.loads((group / "config.json").read_text())
    expected = {
        "dropout": SPEC["dropout"], "motion_harm_weight": SPEC["motion_harm_weight"],
        "epochs": TOTAL_EPOCHS, "total_steps": preflight_payload["total_steps"],
        "warmup_steps": preflight_payload["warmup_steps"], "shards": 1,
        "batch_size": 1024, "seed": 42, "snapshot_every": 5,
        "grad_norm_every": 5,
        "architecture_variant": "query_film_bilinear", "appearance_proj_dim": 256,
    }
    mismatched = {key: {"expected": value, "actual": config.get(key)}
                  for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"config mismatch {mismatched}")
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(metrics) != TOTAL_EPOCHS or int(metrics[-1]["epoch"]) != TOTAL_EPOCHS:
        raise RuntimeError(f"metrics endpoints {len(metrics)}/{metrics[-1].get('epoch')}")
    last = metrics[-1]
    if int(last["global_step"]) != preflight_payload["total_steps"]:
        raise RuntimeError(f"global_step {last['global_step']}")
    if float(last["lr"]) > 1e-9:
        raise RuntimeError(f"final lr {last['lr']} not near zero")
    log_text = (group / "training.log").read_text()
    verification = {}
    for line in log_text.splitlines():
        if line.startswith("verification: "):
            verification = json.loads(line[len("verification: "):])
    if verification.get("model_param_numel") != verification.get("optimizer_param_numel"):
        raise RuntimeError("optimizer does not hold every parameter")
    reference_init = None
    ref_log = DE_EXP / "runs/E/groups/appearance_row_column/training.log"
    for line in ref_log.read_text().splitlines():
        if line.startswith("verification: "):
            reference_init = json.loads(line[len("verification: "):]).get("initial_parameter_sha256")
    initial_matches = reference_init == verification.get("initial_parameter_sha256")
    payload = torch.load(checkpoint(EVAL_EPOCH), map_location="cpu", weights_only=False)
    model_config = payload.get("model_config") or {}
    training_config = payload.get("training_config") or {}
    scheduler = payload.get("scheduler_state_dict") or {}
    reload_ok = (
        int(payload.get("epoch", -1)) == EVAL_EPOCH
        and int(payload.get("global_step", -1)) == preflight_payload["total_steps"]
        and model_config.get("architecture_variant") == "query_film_bilinear"
        and int(model_config.get("appearance_proj_dim", 0)) == 256
        and float(model_config.get("dropout", -1)) == SPEC["dropout"]
        and float(training_config.get("motion_harm_weight", -1)) == SPEC["motion_harm_weight"]
        and payload.get("rng_state") is not None
        and int(scheduler.get("last_epoch", -1)) == preflight_payload["total_steps"])
    if not reload_ok:
        raise RuntimeError("checkpoint reload verification failed")
    return {"config": config, "metrics": metrics, "last": last,
            "verification": verification, "initial_matches_E20": initial_matches,
            "checkpoint_config": model_config, "reload_ok": reload_ok}


def load_epoch(exp, name, epoch):
    base = exp / "evals" / f"{name}_e{epoch:03d}"
    summary = json.loads((base / "summary.json").read_text())
    gates_path = base / "gates.jsonl"
    target_path = exp / "gate_targets" / f"{name}_e{epoch:03d}.json"
    return {
        "combined": summary["combined"], "per_sequence": summary["per_sequence"],
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


def analyze():
    e100 = load_epoch(E100_EXP, NAME, 100)
    e20 = load_epoch(DE_EXP, NAME, 20)
    e30 = load_epoch(E30_EXP, NAME, 30)
    a20 = load_epoch(ABC_EXP, "A", 20)
    d20 = load_epoch(DE_EXP, "D", 20)
    group = abc.group_dir(NAME)
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    state = abc.load_state()

    def read_metrics(exp):
        path = exp / "runs" / NAME / "groups/appearance_row_column/metrics.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    de_metrics = read_metrics(DE_EXP)
    e30_metrics = read_metrics(E30_EXP)
    combined_metrics = de_metrics + e30_metrics

    def reference(rows, steps, cosine, warmup, lr_end):
        return {
            "epochs": len(rows), "steps": steps, "samples": len(rows) * 305631,
            "cosine": cosine, "warmup": warmup, "lr_end": lr_end,
            "epoch_mean": sum(row["epoch_seconds"] for row in rows) / len(rows),
            "train_seconds": sum(row["epoch_seconds"] for row in rows),
            "rss_peak_gb": max(row.get("rss_peak_gb") or 0 for row in rows),
            "cuda_peak_allocated_mb": max(row.get("cuda_peak_allocated_mb") or 0 for row in rows),
        }

    summary = {
        "generated_at": abc.now(),
        "preflight": json.loads((E100_EXP / "preflight.json").read_text()),
        "training": {
            "epoch_seconds_mean": sum(row["epoch_seconds"] for row in metrics) / len(metrics),
            "seconds_total": state["runs"].get(NAME, {}).get("train", {}).get("seconds"),
            "rss_peak_gb": metrics[-1].get("rss_peak_gb"),
            "cuda_peak_allocated_mb": metrics[-1].get("cuda_peak_allocated_mb"),
            "curve": [{key: row.get(key) for key in
                       ("epoch", "global_step", "lr", "loss", "motion_loss",
                        "appearance_loss", "motion_gate_mean", "appearance_gate_mean",
                        "epoch_seconds", "grad_norm_motion_shared",
                        "grad_norm_appearance_shared")} for row in metrics],
            "state": state["runs"].get(NAME, {}),
        },
        "reference_training": {
            "E_e20": reference(de_metrics, 6420, 50, 80, 6.589917097997974e-05),
            "E_e30": reference(combined_metrics, 9630, 50, 80, 3.4848798925095794e-05),
        },
        "runs": {
            "E_e20": {"combined": e20["combined"], "gates": e20["gates"],
                      "gate_target": e20["gate_target"]},
            "E_e30": {"combined": e30["combined"], "gates": e30["gates"],
                      "gate_target": e30["gate_target"]},
            "E_e100": {"combined": e100["combined"], "gates": e100["gates"],
                       "gate_target": e100["gate_target"]},
            "A_e20": {"combined": a20["combined"]},
            "D_e20": {"combined": d20["combined"]},
        },
        "delta_e100_vs_e20": delta(e100["combined"], e20["combined"]),
        "delta_e100_vs_e30": delta(e100["combined"], e30["combined"]),
        "per_sequence_e100_vs_e20": per_sequence_change(e100["per_sequence"], e20["per_sequence"]),
        "per_sequence_e100_vs_e30": per_sequence_change(e100["per_sequence"], e30["per_sequence"]),
    }
    report = render(summary)
    (E100_EXP / "report.md").write_text(report + "\n")
    (E100_EXP / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)


def gate_delta(target):
    return target["useful_gate_mean"] - target["harmful_gate_mean"]


def render(summary):
    pre = summary["preflight"]
    training = summary["training"]
    runs = summary["runs"]
    lines = ["# SportsMOT E 配置 100 轮从头训练", "",
             f"官方 train 45 / val 45；dropout 0.05，motion_harm_weight 2；"
             f"query_film_bilinear 从头全参数，seed 42；shards=1 全量驻留，batch 1024。",
             f"预算：{pre['total_epochs']} 个全量轮 × {pre['steps_per_epoch']} 步 = "
             f"{pre['total_steps']} 步，{pre['total_samples']:,} 样本；"
             f"warmup {pre['warmup_steps']} 步（{pre['warmup_fraction']*100:.4f}%）；"
             f"LR 1e-4 → {pre['lr_end']:.3e}（余弦到 0）。", "",
             "## 训练成本", "",
             "| 组 | 轮数 | 总步数 | 总样本 | 计划余弦 | warmup | 终点 LR | 平均轮时 s | 总训练 s | RSS 峰值 GiB | CUDA 峰值 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| E_e100 | 100 | {pre['total_steps']} | {pre['total_samples']:,} | 100 轮 | "
                 f"{pre['warmup_steps']} | {pre['lr_end']:.3e} | "
                 f"{training['epoch_seconds_mean']:.1f} | "
                 f"{sum(row['epoch_seconds'] for row in training['curve']):.1f} | "
                 f"{training['rss_peak_gb']} | {training['cuda_peak_allocated_mb']} |")
    lines.append("| E_e30（参考） | 30 | 9,630 | 9,168,930 | 50 轮 | 80 | 3.4849e-05 | "
                 f"{summary['reference_training']['E_e30']['epoch_mean']:.1f} | "
                 f"{summary['reference_training']['E_e30']['train_seconds']:.1f} | "
                 f"{summary['reference_training']['E_e30']['rss_peak_gb']} | "
                 f"{summary['reference_training']['E_e30']['cuda_peak_allocated_mb']} |")
    lines.append("| E_e20（参考） | 20 | 6,420 | 6,112,620 | 50 轮 | 80 | 6.5899e-05 | "
                 f"{summary['reference_training']['E_e20']['epoch_mean']:.1f} | "
                 f"{summary['reference_training']['E_e20']['train_seconds']:.1f} | "
                 f"{summary['reference_training']['E_e20']['rss_peak_gb']} | "
                 f"{summary['reference_training']['E_e20']['cuda_peak_allocated_mb']} |")

    lines += ["", "## 训练走势（选取轮次）", "",
              "| epoch | step | loss | motion | appearance | motion gate | appearance gate | lr |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in training["curve"]:
        if int(row["epoch"]) in (1, 5, 10, 20, 30, 50, 75, 100):
            lines.append(f"| {row['epoch']} | {row['global_step']} | {row['loss']:.4f} | "
                         f"{row['motion_loss']:.4f} | {row['appearance_loss']:.4f} | "
                         f"{row['motion_gate_mean']:.4f} | {row['appearance_gate_mean']:.4f} | "
                         f"{row['lr']:.4e} |")

    lines += ["", "## Val（45 条 raw）", "",
              "| 组 | 训练预算 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    budgets = {"E_e20": "20 轮 / 6,420 步 / 余弦 50", "E_e30": "30 轮 / 9,630 步 / 余弦 50",
               "E_e100": "100 轮 / 32,100 步 / 余弦 100", "A_e20": "20 轮 / 6,420 步 / 余弦 50",
               "D_e20": "20 轮 / 6,420 步 / 余弦 50"}
    for label in ("E_e20", "E_e30", "E_e100", "A_e20", "D_e20"):
        c = runs[label]["combined"]
        lines.append(f"| {label} | {budgets[label]} | {c['HOTA']*100:.3f} | "
                     f"{c['AssA']*100:.3f} | {c['DetA']*100:.3f} | {c['IDF1']*100:.3f} | "
                     f"{c['MOTA']*100:.3f} | {c['IDSW']:.0f} | {c['Frag']:.0f} |")

    lines += ["", "## 延长到 100 轮的效果", "",
              "| 对比 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW | 逐序列改善/退化/不变 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for label, dkey, ckey in (
            ("E_e100 − E_e20", "delta_e100_vs_e20", "per_sequence_e100_vs_e20"),
            ("E_e100 − E_e30", "delta_e100_vs_e30", "per_sequence_e100_vs_e30")):
        d = summary[dkey]
        change = summary[ckey]
        lines.append(f"| {label} | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                     f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
                     f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")

    lines += ["", "## 门控目标（E 各预算）", "",
              "| 版本 | 有害 gate | 有益 gate | 分离度 | 节省/事件 | 误伤/事件 | 净 DIoU 变化 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ("E_e20", "E_e30", "E_e100"):
        target = runs[label]["gate_target"]["new"]
        lines.append(f"| {label} | {target['harmful_gate_mean']:.4f} | "
                     f"{target['useful_gate_mean']:.4f} | {gate_delta(target):.4f} | "
                     f"{target['harmful_savings_mean']:.6f} | "
                     f"{target['useful_damage_mean']:.6f} | {target['net_loss_change']:+.6f} |")

    lines += ["", "## 门控分布（跟踪时）", "",
              "| 版本 | motion min/mean/max | appearance min/mean/max |",
              "|---|---|---|"]
    for label in ("E_e20", "E_e30", "E_e100"):
        g = runs[label]["gates"]
        lines.append(f"| {label} | {g['motion_min']:.4f}/{g['motion_mean']:.4f}/"
                     f"{g['motion_max']:.4f} | {g['appearance_min']:.4f}/"
                     f"{g['appearance_mean']:.4f}/{g['appearance_max']:.4f} |")

    lines += ["", "## 低频梯度范数（首 batch，每 5 轮）", ""]
    for row in training["curve"]:
        if int(row["epoch"]) in (1, 5, 10, 20, 30, 50, 75, 100) and row.get("grad_norm_motion_shared") is not None:
            lines.append(f"- e{row['epoch']}: motion {row['grad_norm_motion_shared']:.6f}，"
                         f"appearance {row['grad_norm_appearance_shared']:.4f}")

    lines += ["", "## 逐序列（e100 vs e20 / e30）", ""]
    for label, key in (("e100 vs e20", "per_sequence_e100_vs_e20"),
                       ("e100 vs e30", "per_sequence_e100_vs_e30")):
        change = summary[key]
        lines.append(f"- {label}: 最好 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["best"][:3])
                     + "；最差 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["worst"][:3]))

    verification = training["state"].get("verify") or {}
    init = verification.get("verification") or {}
    lines += ["", "## 核验", "",
              f"- 初始参数与 E_e20 一致：{verification.get('initial_matches_E20')}",
              f"- Dropout 模块：{json.dumps(init.get('dropout_modules'), ensure_ascii=False)}",
              f"- 参数/优化器：{init.get('model_param_numel')} / "
              f"{init.get('optimizer_param_numel')}",
              f"- 终点 checkpoint 重载：{verification.get('reload_ok')}；"
              f"epoch/step/LR：{training['curve'][-1]['epoch']} / "
              f"{training['curve'][-1]['global_step']} / {training['curve'][-1]['lr']:.3e}"]
    best = max(((runs[label]["combined"]["HOTA"], label) for label in
                ("E_e20", "E_e30", "E_e100", "A_e20", "D_e20")), key=lambda item: item[0])
    lines += ["", "## 最佳版本", "",
              f"- {best[1]}：val HOTA {best[0]*100:.3f}",
              f"- E_e100 checkpoint：`{checkpoint(100)}`",
              "", "单 seed 结果；不同预算之间的差异需结合 LR 调度解读。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "train", "eval", "analyze"), default="all")
    args = parser.parse_args()
    E100_EXP.mkdir(parents=True, exist_ok=True)
    abc.LOGS.mkdir(parents=True, exist_ok=True)
    preflight_payload = json.loads((E100_EXP / "preflight.json").read_text()) \
        if (E100_EXP / "preflight.json").is_file() else preflight()
    print(json.dumps(preflight_payload, indent=2, sort_keys=True), flush=True)
    state = abc.load_state()
    if args.stage in ("all", "train"):
        print(json.dumps({"active_jobs": abc.active_jobs(),
                          "mem_available_gib": abc.mem_available_gib(),
                          "gpu": abc.gpu_free_mib()}, indent=1), flush=True)
        train_run(preflight_payload, state)
        state["runs"].setdefault(NAME, {})["verify"] = verify(preflight_payload)
        abc.save_state(state)
    if args.stage in ("all", "eval"):
        abc.eval_run(NAME, EVAL_EPOCH, state)
        abc.gate_run(NAME, EVAL_EPOCH, state)
    if args.stage in ("all", "analyze"):
        analyze()


if __name__ == "__main__":
    main()
