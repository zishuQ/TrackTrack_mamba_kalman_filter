#!/usr/bin/env python3
"""SportsMOT D/E continuation: epoch 20 -> 30 on the original 50-epoch cosine.

Resumes model/optimizer/scheduler (and RNG state when the source checkpoint has
it) from each run's epoch_020.pt in a new experiment directory, keeping the
original artifacts untouched. Only the final epoch 30 checkpoint is evaluated:
one 45-sequence val raw pass and one gate-target analysis per group.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sportsmot_abc_experiment as abc  # noqa: E402

DE_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_de_20261123"
CONT_EXP = ROOT / "outputs/agentguard/experiments/sportsmot_de_e30_20261123"
abc.EXP = CONT_EXP
abc.LOGS = CONT_EXP / "logs"
abc.EVALS = CONT_EXP / "evals"
abc.GATES = CONT_EXP / "gate_targets"
abc.STATE_PATH = CONT_EXP / "state.json"
abc.PREFLIGHT_PATH = CONT_EXP / "preflight.json"

RESUME_EPOCH = 20
STOP_EPOCH = 30
EVAL_EPOCH = 30
RUNS = {
    "D": {"dropout": 0.05, "motion_harm_weight": 1.0,
          "source": DE_EXP / "runs/D/groups/appearance_row_column/checkpoints/epoch_020.pt"},
    "E": {"dropout": 0.05, "motion_harm_weight": 2.0,
          "source": DE_EXP / "runs/E/groups/appearance_row_column/checkpoints/epoch_020.pt"},
}
SCORE_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")


def preflight():
    base = json.loads((DE_EXP / "preflight.json").read_text())
    steps = base["steps_per_epoch"]
    total = base["total_steps"]
    warm = base["warmup_steps"]
    start_step = RESUME_EPOCH * steps
    stop_step = STOP_EPOCH * steps
    ratio = 0.5 * (1.0 + math.cos(math.pi * (stop_step - warm) / (total - warm)))
    payload = {**base,
               "resume_epoch": RESUME_EPOCH, "resume_step": start_step,
               "target_epoch": STOP_EPOCH, "target_step": stop_step,
               "lr_at_resume": (0.5 * (1.0 + math.cos(
                   math.pi * (start_step - warm) / (total - warm)))) * base["lr_start"],
               "lr_at_target": ratio * base["lr_start"], "lr_ratio_at_target": ratio}
    CONT_EXP.mkdir(parents=True, exist_ok=True)
    (CONT_EXP / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def checkpoint(name, epoch):
    return abc.group_dir(name) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def prepare(name, spec):
    group = abc.group_dir(name)
    ckpt_dir = group / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"
    if checkpoint(name, STOP_EPOCH).is_file():
        return "done"
    if last.is_file():
        return "resume"
    if not spec["source"].is_file():
        raise SystemExit(f"{name}: missing source checkpoint {spec['source']}")
    shutil.copyfile(spec["source"], last)
    return "seeded"


def train_command(name, spec, preflight_payload):
    groups_root = CONT_EXP / "runs" / name / "groups"
    return [
        abc.PY, "-u", "tools/train_stop_after.py",
        "--group", "appearance_row_column",
        "--dataset", "SportsMOT", "--split", "train", "--from-scratch",
        "--data-root", str(abc.DATA), "--groups-root", str(groups_root),
        "--shards", "1", "--max-epochs", str(preflight_payload["planned_epochs"]),
        "--snapshot-every", "5", "--stop-after-epoch", str(STOP_EPOCH),
        "--batch-size", "1024", "--lr", "1e-4", "--weight-decay", "1e-4",
        "--seed", "42", "--rank-weight", "18", "--rank-alpha", "0.95",
        "--rank-tau", "0.1", "--column-weight", "50",
        "--motion-harm-weight", str(spec["motion_harm_weight"]),
        "--dropout", str(spec["dropout"]),
        "--preload-features", "--warmup-steps", str(preflight_payload["warmup_steps"]),
        "--grad-norm-every", "5", "--device", "cuda", "--resume",
    ]


def verify(name, spec, preflight_payload):
    import torch
    group = abc.group_dir(name)
    config = json.loads((group / "config.json").read_text())
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    last = metrics[-1]
    log_text = (group / "training.log").read_text()
    driver_log = abc.LOGS / f"train_{name}_e{STOP_EPOCH}.log"
    driver_text = driver_log.read_text() if driver_log.is_file() else ""
    resumed = (f"resume from epoch {RESUME_EPOCH} step {preflight_payload['resume_step']}"
               in driver_text)
    if not resumed:
        raise RuntimeError(f"{name}: resume line not found")
    if int(last["epoch"]) != STOP_EPOCH or int(last["global_step"]) != preflight_payload["target_step"]:
        raise RuntimeError(f"{name}: endpoint {last['epoch']}/{last['global_step']} "
                           f"!= {STOP_EPOCH}/{preflight_payload['target_step']}")
    if abs(float(last["lr"]) - preflight_payload["lr_at_target"]) > 1e-9:
        raise RuntimeError(f"{name}: lr {last['lr']} != {preflight_payload['lr_at_target']}")
    verification = {}
    for line in log_text.splitlines():
        if line.startswith("verification: "):
            verification = json.loads(line[len("verification: "):])
    if verification.get("model_param_numel") != verification.get("optimizer_param_numel"):
        raise RuntimeError(f"{name}: optimizer does not hold every parameter")
    payload = torch.load(checkpoint(name, STOP_EPOCH), map_location="cpu", weights_only=False)
    model_config = payload.get("model_config") or {}
    training_config = payload.get("training_config") or {}
    scheduler = payload.get("scheduler_state_dict") or {}
    reload_ok = (
        int(payload.get("epoch", -1)) == STOP_EPOCH
        and int(payload.get("global_step", -1)) == preflight_payload["target_step"]
        and model_config.get("architecture_variant") == "query_film_bilinear"
        and int(model_config.get("appearance_proj_dim", 0)) == 256
        and float(model_config.get("dropout", -1)) == spec["dropout"]
        and float(training_config.get("motion_harm_weight", -1)) == spec["motion_harm_weight"]
        and "optimizer_state_dict" in payload
        and payload.get("rng_state") is not None
        and int(scheduler.get("last_epoch", -1)) == preflight_payload["target_step"])
    if not reload_ok:
        raise RuntimeError(f"{name}: checkpoint reload verification failed")
    return {
        "resumed": resumed,
        "rng_restored": "resume rng state: restored" in driver_text,
        "epochs_trained": [int(row["epoch"]) for row in metrics],
        "epoch_seconds": [row["epoch_seconds"] for row in metrics],
        "last": last,
        "verification": verification,
        "checkpoint_config": model_config,
        "reload_ok": reload_ok,
        "config": config,
    }


def train_run(name, spec, preflight_payload, state):
    if checkpoint(name, STOP_EPOCH).is_file():
        print(f"{name}: epoch_{STOP_EPOCH} exists; skip training", flush=True)
        return
    mode = prepare(name, spec)
    print(f"{name}: {mode} from {spec['source']}", flush=True)
    abc.check_resources()
    started = time.time()
    abc.run(f"train_{name}_e{STOP_EPOCH}", train_command(name, spec, preflight_payload))
    state["runs"].setdefault(name, {})["train"] = {
        "status": "complete", "seconds": round(time.time() - started, 1),
        "resume_mode": mode}
    abc.save_state(state)


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


def training_curve(name):
    rows = []
    for exp in (DE_EXP, CONT_EXP):
        path = exp / "runs" / name / "groups/appearance_row_column/metrics.jsonl"
        if path.is_file():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def analyze():
    runs = {name: {"20": load_epoch(DE_EXP, name, 20),
                   "30": load_epoch(CONT_EXP, name, 30)} for name in RUNS}
    curves = {name: training_curve(name) for name in RUNS}
    state = abc.load_state()
    summary = {"generated_at": abc.now(),
               "preflight": json.loads((CONT_EXP / "preflight.json").read_text()),
               "runs": {}, "training": {}}
    for name in RUNS:
        e20, e30 = runs[name]["20"], runs[name]["30"]
        summary["runs"][name] = {
            "e20": {"combined": e20["combined"], "gates": e20["gates"],
                    "gate_target": e20["gate_target"]},
            "e30": {"combined": e30["combined"], "gates": e30["gates"],
                    "gate_target": e30["gate_target"]},
            "delta_e30_vs_e20": delta(e30["combined"], e20["combined"]),
            "per_sequence_e30_vs_e20": per_sequence_change(
                e30["per_sequence"], e20["per_sequence"]),
        }
        curve = curves[name]
        summary["training"][name] = {
            "epochs": len(curve),
            "curve": [{key: row.get(key) for key in
                       ("epoch", "global_step", "lr", "loss", "motion_loss",
                        "appearance_loss", "epoch_seconds", "rss_peak_gb",
                        "cuda_peak_allocated_mb")} for row in curve],
            "continuation_seconds": sum(row["epoch_seconds"] for row in curve
                                        if RESUME_EPOCH < int(row["epoch"]) <= STOP_EPOCH),
            "state": state["runs"].get(name, {}),
            "last": curve[-1],
        }
    summary["runs"]["E"]["delta_e30_vs_D"] = delta(
        runs["E"]["30"]["combined"], runs["D"]["30"]["combined"])
    summary["runs"]["E"]["per_sequence_e30_vs_D"] = per_sequence_change(
        runs["E"]["30"]["per_sequence"], runs["D"]["30"]["per_sequence"])
    best = max(((summary["runs"][name]["e30"]["combined"]["HOTA"], f"{name}_e030",
                 str(checkpoint(name, 30))) for name in RUNS), key=lambda item: item[0])
    summary["best_e30"] = {"label": best[1], "hota": best[0], "checkpoint": best[2]}
    report = render(summary)
    (CONT_EXP / "report.md").write_text(report + "\n")
    (CONT_EXP / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)


def gate_delta(target):
    return target["useful_gate_mean"] - target["harmful_gate_mean"]


def render(summary):
    lines = ["# SportsMOT D/E 续训：epoch 20 → 30", "",
             f"从各组 epoch_020.pt 完整恢复模型/优化器/调度器，沿用 50 轮余弦；"
             f"全量步数 {summary['preflight']['resume_step']} → "
             f"{summary['preflight']['target_step']}，LR "
             f"{summary['preflight']['lr_at_resume']:.4e} → "
             f"{summary['preflight']['lr_at_target']:.4e}。"
             "仅评估 e30；e20 结果复用。", "",
             "## 续训与资源", "",
             "| 组 | 恢复 | 21–30 轮合计 s | 平均轮时 s | RSS 峰值 GiB | CUDA 峰值 MiB |",
             "|---|---|---|---:|---:|---:|"]
    for name in RUNS:
        training = summary["training"][name]
        last = training["last"]
        curve = [row for row in training["curve"] if RESUME_EPOCH < int(row["epoch"]) <= STOP_EPOCH]
        mean = training["continuation_seconds"] / max(1, len(curve))
        state = (training["state"] or {}).get("train") or {}
        lines.append(f"| {name} | {state.get('resume_mode', '-')} | "
                     f"{training['continuation_seconds']:.1f} | {mean:.1f} | "
                     f"{last.get('rss_peak_gb')} | {last.get('cuda_peak_allocated_mb')} |")

    lines += ["", "## 训练走势（loss / LR）", "",
              "| 组 | epoch | step | loss | motion | appearance | lr |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in RUNS:
        curve = summary["training"][name]["curve"]
        for row in curve:
            if int(row["epoch"]) in (1, 5, 10, 15, 20, 21, 25, 30):
                lines.append(f"| {name} | {row['epoch']} | {row['global_step']} | "
                             f"{row['loss']:.4f} | {row['motion_loss']:.4f} | "
                             f"{row['appearance_loss']:.4f} | {row['lr']:.6e} |")

    lines += ["", "## Val（45 条 raw）", "",
              "| 组 | 轮次 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in RUNS:
        for epoch in ("e20", "e30"):
            c = summary["runs"][name][epoch]["combined"]
            lines.append(f"| {name} | {epoch} | {c['HOTA']*100:.3f} | {c['AssA']*100:.3f} | "
                         f"{c['DetA']*100:.3f} | {c['IDF1']*100:.3f} | {c['MOTA']*100:.3f} | "
                         f"{c['IDSW']:.0f} | {c['Frag']:.0f} |")
    lines += ["", "## 延长训练的效果（e30 − e20）", "",
              "| 组 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW | 逐序列改善/退化/不变 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name in RUNS:
        d = summary["runs"][name]["delta_e30_vs_e20"]
        change = summary["runs"][name]["per_sequence_e30_vs_e20"]
        lines.append(f"| {name} | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                     f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
                     f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")

    lines += ["", "## 门控目标：e20 vs e30", "",
              "| 组 | 轮次 | 有害 gate | 有益 gate | 分离度 | 节省/事件 | 误伤/事件 | 净 DIoU 变化 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in RUNS:
        for epoch in ("e20", "e30"):
            target = summary["runs"][name][epoch]["gate_target"]["new"]
            lines.append(f"| {name} | {epoch} | {target['harmful_gate_mean']:.4f} | "
                         f"{target['useful_gate_mean']:.4f} | {gate_delta(target):.4f} | "
                         f"{target['harmful_savings_mean']:.6f} | "
                         f"{target['useful_damage_mean']:.6f} | {target['net_loss_change']:+.6f} |")

    d = summary["runs"]["E"]["delta_e30_vs_D"]
    change = summary["runs"]["E"]["per_sequence_e30_vs_D"]
    lines += ["", "## E_e30 与 D_e30（同预算 harm2 vs harm1）", "",
              f"- ΔHOTA {d['HOTA']*100:+.3f}，ΔAssA {d['AssA']*100:+.3f}，"
              f"ΔDetA {d['DetA']*100:+.3f}，ΔIDF1 {d['IDF1']*100:+.3f}，ΔIDSW {d['IDSW']:+.0f}",
              f"- 逐序列：{change['improved']} 改善 / {change['degraded']} 退化 / "
              f"{change['unchanged']} 不变",
              "", "## 门控分布（跟踪时）", "",
              "| 组 | 轮次 | motion min/mean/max | appearance min/mean/max |",
              "|---|---:|---|---|"]
    for name in RUNS:
        for epoch in ("e20", "e30"):
            g = summary["runs"][name][epoch]["gates"]
            lines.append(f"| {name} | {epoch} | {g['motion_min']:.4f}/{g['motion_mean']:.4f}/"
                         f"{g['motion_max']:.4f} | {g['appearance_min']:.4f}/"
                         f"{g['appearance_mean']:.4f}/{g['appearance_max']:.4f} |")

    lines += ["", "## 逐序列（e30 vs e20）", ""]
    for name in RUNS:
        change = summary["runs"][name]["per_sequence_e30_vs_e20"]
        lines.append(f"- {name}: 最好 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["best"][:3])
                     + "；最差 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["worst"][:3]))
    lines += ["", "## 恢复核验", ""]
    for name in RUNS:
        record = summary["training"][name]["state"] or {}
        verify_record = record.get("verify") or {}
        verification = verify_record.get("verification") or {}
        resume_mode = (record.get("train") or {}).get("resume_mode", "-")
        lines.append(f"- {name}: 恢复模式 {resume_mode}；"
                     f"resume 行已确认 {verify_record.get('resumed', False)}；"
                     f"参数/优化器 {verification.get('model_param_numel')} / "
                     f"{verification.get('optimizer_param_numel')}；"
                     f"checkpoint 重载 {verify_record.get('reload_ok', False)}；"
                     f"源 RNG 恢复 {verify_record.get('rng_restored', False)}。")
    lines += ["", f"- 源 e20 checkpoint 未保存 RNG 状态（当时格式没有该字段），"
                  "本次续训的 torch/numpy/python RNG 从 seed=42 重启；"
                  "数据顺序按 (seed, epoch) 确定性生成，与连续训练完全一致，"
                  "差异仅在 dropout 掩码流。e30 checkpoint 起已写入 RNG 状态，后续续训可精确恢复。",
              "",
              f"## 最佳 checkpoint", "",
              f"- {summary['best_e30']['label']}：val HOTA "
              f"{summary['best_e30']['hota']*100:.3f}，"
              f"`{summary['best_e30']['checkpoint']}`",
              "", "单 seed 结果；小差异需 seed 复现。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "train", "eval", "analyze"), default="all")
    args = parser.parse_args()
    CONT_EXP.mkdir(parents=True, exist_ok=True)
    abc.LOGS.mkdir(parents=True, exist_ok=True)
    preflight_payload = json.loads((CONT_EXP / "preflight.json").read_text()) \
        if (CONT_EXP / "preflight.json").is_file() else preflight()
    print(json.dumps(preflight_payload, indent=2, sort_keys=True), flush=True)
    state = abc.load_state()
    if args.stage in ("all", "train"):
        print(json.dumps({"active_jobs": abc.active_jobs(),
                          "mem_available_gib": abc.mem_available_gib(),
                          "gpu": abc.gpu_free_mib()}, indent=1), flush=True)
        for name, spec in RUNS.items():
            train_run(name, spec, preflight_payload, state)
            state["runs"].setdefault(name, {})["verify"] = verify(name, spec, preflight_payload)
            abc.save_state(state)
    if args.stage in ("all", "eval"):
        for name in RUNS:
            abc.eval_run(name, EVAL_EPOCH, state)
            abc.gate_run(name, EVAL_EPOCH, state)
    if args.stage in ("all", "analyze"):
        analyze()


if __name__ == "__main__":
    main()
