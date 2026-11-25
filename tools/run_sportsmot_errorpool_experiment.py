#!/usr/bin/env python3
"""Train one group with GT + real detection error-pool simulated observations.

Observations come from the tracked target's GT box at frame t plus a jointly
sampled 4-D residual from the train45 detection/GT pool.  History KF/evidence
are warmed with fixed full writes; the model is called once at the anchor and
supervises the anchor's t+1 prediction (original one-step objective).

Budget: 50-epoch cosine plan, stop after e20 (6,420 / 16,050 steps, warmup 80,
LR 6.5899e-5).  Only the final e20 is evaluated.
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

EXP = ROOT / "outputs/agentguard/experiments/sportsmot_errorpool_e20_20261124"
DATA = ROOT / "outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data"
POOL = EXP / "pool"
BASELINE_DE = ROOT / "outputs/agentguard/experiments/sportsmot_de_20261123"
abc.EXP = EXP
abc.LOGS = EXP / "logs"
abc.EVALS = EXP / "evals"
abc.GATES = EXP / "gate_targets"
abc.STATE_PATH = EXP / "state.json"
abc.PREFLIGHT_PATH = EXP / "preflight.json"
NAME = "EP"
SPEC = {"window_frames": 1, "window_supervision": "anchor", "augment": "error_pool"}
COMMON = {
    "planned_epochs": 50, "stop_epoch": 20, "snapshot_every": 5,
    "batch_size": 1024, "micro_batch": 1024,
    "lr": 1e-4, "weight_decay": 1e-4, "seed": 42, "rank_weight": 18.0,
    "rank_alpha": 0.95, "rank_tau": 0.1, "column_weight": 50.0,
    "motion_harm_weight": 1.0, "dropout": 0.05, "warmup_steps": 80,
    "grad_norm_every": 5,
}
EVAL_EPOCH = 20
SCORE_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")


def group_dir():
    return EXP / "runs" / NAME / "groups/appearance_row_column"


def checkpoint(epoch):
    return group_dir() / "checkpoints" / f"epoch_{epoch:03d}.pt"


def data_steps():
    import numpy as np
    samples = steps = 0
    for path in sorted(DATA.iterdir()):
        if not (path / "samples.npz").is_file():
            continue
        with np.load(path / "samples.npz", allow_pickle=False) as archive:
            count = int(archive["frame"].shape[0])
        samples += count
        steps += math.ceil(count / COMMON["batch_size"])
    total = steps * COMMON["planned_epochs"]
    actual = steps * COMMON["stop_epoch"]
    warm = min(COMMON["warmup_steps"], total)

    def lr_at(step):
        if step < warm:
            return COMMON["lr"] * (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return COMMON["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))

    pool_report = json.loads((POOL / "pool_report.json").read_text())
    diagnostic = json.loads((POOL / "diagnostic.json").read_text())
    return {"sequences": len(list(DATA.glob("*/samples.npz"))), "samples": samples,
            "steps_per_epoch": steps, "planned_epochs": COMMON["planned_epochs"],
            "stop_epoch": COMMON["stop_epoch"], "total_steps": total,
            "actual_steps": actual, "warmup_steps": warm,
            "lr_start": COMMON["lr"], "lr_at_stop": lr_at(actual),
            "pool": {"pairs": pool_report["pairs"],
                     "coverage": pool_report["pair_coverage"],
                     "relative_std": pool_report["relative"]["std"]},
            "diagnostic": {"harmful_real": diagnostic["harmful_ratio"]["real"],
                           "harmful_sim": diagnostic["harmful_ratio"]["sim"],
                           "endpoint_g0": diagnostic["endpoint_loss"]["real_g0"],
                           "endpoint_g1_real": diagnostic["endpoint_loss"]["real_g1"],
                           "endpoint_g1_sim": diagnostic["endpoint_loss"]["sim_g1"]}}


def preflight():
    payload = {**data_steps(), "implementation": "packed_replay_v2",
               "metric": "shared_training_diou", "created_at": abc.now(),
               "mem_available_gib": round(abc.mem_available_gib(), 2),
               "gpu": abc.gpu_free_mib()}
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def train_command(pre):
    command = [
        abc.PY, "-u", "tools/train_window_groups.py",
        "--data-root", str(DATA), "--groups-root", str(EXP / "runs" / NAME / "groups"),
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
        "--window-frames", str(SPEC["window_frames"]),
        "--window-supervision", SPEC["window_supervision"],
        "--augment", "error_pool", "--error-pool", str(POOL),
        "--device", "cuda",
    ]
    if checkpoint(EVAL_EPOCH).is_file() is False and (group_dir() / "checkpoints/last.pt").is_file():
        command.append("--resume")
    return command


def verify(pre):
    import torch
    group = group_dir()
    config = json.loads((group / "config.json").read_text())
    expected = {"window_frames": 1, "window_supervision": "anchor",
                "augment": "error_pool", "epochs": COMMON["planned_epochs"],
                "stop_after_epoch": COMMON["stop_epoch"],
                "total_steps": pre["total_steps"], "warmup_steps": pre["warmup_steps"],
                "dropout": COMMON["dropout"],
                "motion_harm_weight": COMMON["motion_harm_weight"],
                "batch_size": COMMON["batch_size"], "seed": COMMON["seed"]}
    mismatched = {key: {"expected": value, "actual": config.get(key)}
                  for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"config mismatch {mismatched}")
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(metrics) != COMMON["stop_epoch"] or int(metrics[-1]["epoch"]) != COMMON["stop_epoch"]:
        raise RuntimeError(f"metrics endpoint {len(metrics)}/{metrics[-1].get('epoch')}")
    last = metrics[-1]
    if int(last["global_step"]) != pre["actual_steps"]:
        raise RuntimeError(f"global_step {last['global_step']}")
    if abs(float(last["lr"]) - pre["lr_at_stop"]) > 1e-9:
        raise RuntimeError(f"lr {last['lr']} != {pre['lr_at_stop']}")
    if int(last.get("error_pool_simulated", 0)) <= 0:
        raise RuntimeError("no simulated observations recorded")
    payload = torch.load(checkpoint(EVAL_EPOCH), map_location="cpu", weights_only=False)
    training_config = payload.get("training_config") or {}
    model_config = payload.get("model_config") or {}
    reload_ok = (
        int(payload.get("epoch", -1)) == EVAL_EPOCH
        and int(payload.get("global_step", -1)) == pre["actual_steps"]
        and training_config.get("augment") == "error_pool"
        and training_config.get("error_pool_sha256") == config.get("error_pool_sha256")
        and training_config.get("stop_after_epoch") == COMMON["stop_epoch"]
        and float(model_config.get("dropout", -1)) == COMMON["dropout"]
        and payload.get("rng_state") is not None)
    if not reload_ok:
        raise RuntimeError("checkpoint reload verification failed")
    return {"last": last, "metrics": metrics, "reload_ok": reload_ok, "config": config}


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
    baseline = load_epoch(BASELINE_DE, "D", 20)
    e20 = load_epoch(BASELINE_DE, "E", 20)
    ep = load_epoch(EXP, NAME, EVAL_EPOCH)
    group = group_dir()
    metrics = [json.loads(line) for line in
               (group / "metrics.jsonl").read_text().splitlines() if line.strip()]
    pre = json.loads((EXP / "preflight.json").read_text())
    de_report = json.loads((BASELINE_DE / "report.json").read_text())
    summary = {
        "generated_at": abc.now(), "preflight": pre,
        "baseline_D_e20": {"combined": baseline["combined"], "gates": baseline["gates"],
                           "gate_target": baseline["gate_target"],
                           "training": de_report["training"].get("D")},
        "reference_E_e20": {"combined": e20["combined"], "gates": e20["gates"],
                            "gate_target": e20["gate_target"]},
        "EP": {"combined": ep["combined"], "gates": ep["gates"],
               "gate_target": ep["gate_target"],
               "delta_vs_D_e20": delta(ep["combined"], baseline["combined"]),
               "per_sequence_vs_D_e20": per_change(ep["per_sequence"],
                                                   baseline["per_sequence"]),
               "training": {
                   "epoch_seconds_mean": sum(r["epoch_seconds"] for r in metrics) / len(metrics),
                   "epoch_seconds_total": sum(r["epoch_seconds"] for r in metrics),
                   "rss_peak_gb": metrics[-1].get("rss_peak_gb"),
                   "cuda_peak_allocated_mb": metrics[-1].get("cuda_peak_allocated_mb"),
                   "error_pool_simulated": metrics[-1].get("error_pool_simulated"),
                   "error_pool_fallback": metrics[-1].get("error_pool_fallback"),
                   "grad_norms": {r["epoch"]: {"motion": r.get("grad_norm_motion_shared"),
                                               "appearance": r.get("grad_norm_appearance_shared")}
                                  for r in metrics if r.get("grad_norm_motion_shared") is not None},
                   "state": abc.load_state()["runs"].get(NAME, {}),
               }},
    }
    report = render(summary)
    (EXP / "report.md").write_text(report + "\n")
    (EXP / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)


def render(summary):
    pre = summary["preflight"]
    base = summary["baseline_D_e20"]
    ep = summary["EP"]
    e20 = summary["reference_E_e20"]
    lines = ["# SportsMOT GT＋真实检测误差池训练（e20）", "",
             f"预算：train 45，{pre['samples']} 样本/轮，{pre['steps_per_epoch']} 步/轮；"
             f"50 轮余弦计划，实际 {pre['stop_epoch']} 轮 = {pre['actual_steps']} 步"
             f"（计划 {pre['total_steps']}），warmup {pre['warmup_steps']}，"
             f"e20 末 LR {pre['lr_at_stop']:.4e}。", "",
             f"误差池：{pre['pool']['pairs']:,} 对可靠配对，覆盖 "
             f"{pre['pool']['coverage']*100:.2f}%；诊断（shared_training_diou）"
             f"有害事件比例 真实 {pre['diagnostic']['harmful_real']*100:.2f}% vs 模拟 "
             f"{pre['diagnostic']['harmful_sim']*100:.2f}%；g1−g0 端点增益 "
             f"真实 {pre['diagnostic']['endpoint_g1_real']-pre['diagnostic']['endpoint_g0']:+.4f}"
             f" vs 模拟 {pre['diagnostic']['endpoint_g1_sim']-pre['diagnostic']['endpoint_g0']:+.4f}。"
             "模拟不要求与真实组完全一致。", "",
             "## Val e20（45 条 raw）", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag | ΔHOTA vs D_e20 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def row(label, c, d=None):
        return (f"| {label} | {c['HOTA']*100:.3f} | {c['AssA']*100:.3f} | "
                f"{c['DetA']*100:.3f} | {c['IDF1']*100:.3f} | {c['MOTA']*100:.3f} | "
                f"{c['IDSW']:.0f} | {c['Frag']:.0f} | "
                f"{'-' if d is None else f'{d*100:+.3f}'} |")
    lines.append(row("D_e20（基线 83.660）", base["combined"]))
    lines.append(row("EP（误差池，一步/harm1）", ep["combined"],
                     ep["delta_vs_D_e20"]["HOTA"]))
    lines.append(row("E_e20（参考）", e20["combined"]))
    d = ep["delta_vs_D_e20"]
    change = ep["per_sequence_vs_D_e20"]
    lines += ["", "## EP − D_e20", "",
              "| ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW | 逐序列改善/退化/不变 |",
              "|---:|---:|---:|---:|---:|---|",
              f"| {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | {d['DetA']*100:+.3f} | "
              f"{d['IDF1']*100:+.3f} | {d['IDSW']:+.0f} | "
              f"{change['improved']}/{change['degraded']}/{change['unchanged']} |",
              "", "## 门控分布（跟踪时，e20）", "",
              "| 组 | motion min/mean/max | appearance min/mean/max |",
              "|---|---|---|"]
    for label, entry in (("D_e20", base), ("EP", ep), ("E_e20", e20)):
        g = entry["gates"]
        lines.append(f"| {label} | {g['motion_min']:.4f}/{g['motion_mean']:.4f}/"
                     f"{g['motion_max']:.4f} | {g['appearance_min']:.4f}/"
                     f"{g['appearance_mean']:.4f}/{g['appearance_max']:.4f} |")
    lines += ["", "## 门控目标（e20）", "",
              "| 组 | 有害 gate | 有益 gate | 分离度 | 节省/事件 | 误伤/事件 | 净 DIoU 变化 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in (("D_e20", base), ("EP", ep), ("E_e20", e20)):
        target = entry["gate_target"]["new"]
        lines.append(f"| {label} | {target['harmful_gate_mean']:.4f} | "
                     f"{target['useful_gate_mean']:.4f} | "
                     f"{target['useful_gate_mean']-target['harmful_gate_mean']:.4f} | "
                     f"{target['harmful_savings_mean']:.6f} | "
                     f"{target['useful_damage_mean']:.6f} | {target['net_loss_change']:+.6f} |")
    stats = ep["training"]
    base_training = base.get("training") or {}
    lines += ["", "## 训练成本", "",
              "| 组 | 轮时 s | 20 轮训练 s | RSS GiB | CUDA MiB | 模拟观测 | GT 缺失回退 |",
              "|---|---:|---:|---:|---:|---:|---:|",
              f"| EP | {stats['epoch_seconds_mean']:.1f} | "
              f"{stats['epoch_seconds_total']:.0f} | {stats['rss_peak_gb']} | "
              f"{stats['cuda_peak_allocated_mb']} | {stats['error_pool_simulated']} | "
              f"{stats['error_pool_fallback']} |",
              f"| D_e20（参考） | {base_training.get('epoch_seconds_mean', 0):.1f} | "
              f"{base_training.get('epochs', 20)*base_training.get('epoch_seconds_mean', 0):.0f} | "
              f"{base_training.get('rss_peak_gb')} | "
              f"{base_training.get('cuda_peak_allocated_mb')} | - | - |"]
    lines += ["", "## 低频梯度范数", ""]
    points = [p for p in (1, 5, 10, 20) if p in stats["grad_norms"]]
    lines.append("- EP: " + "；".join(
        f"e{p} {stats['grad_norms'][p]['motion']:.5f}/"
        f"{stats['grad_norms'][p]['appearance']:.4f}" for p in points))
    lines += ["", "## 结论", "",
              f"- EP e20：HOTA {ep['combined']['HOTA']*100:.3f}，相对 D_e20 "
              f"{d['HOTA']*100:+.3f}（单 seed；选优仍以 HOTA，并列看 AssA/IDF1）。",
              f"- checkpoint：`{checkpoint(EVAL_EPOCH)}`",
              "- 本轮只做 train45/e20；正式 trainval、test/post、提交等待用户决定。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "train", "eval", "analyze", "plan"),
                        default="all")
    args = parser.parse_args()
    if args.stage == "plan":
        print(json.dumps({"experiment": str(EXP), "data": str(DATA),
                          "pool": str(POOL), "spec": SPEC, "common": COMMON,
                          "final_only_epoch": EVAL_EPOCH}, indent=2))
        return
    EXP.mkdir(parents=True, exist_ok=True)
    abc.LOGS.mkdir(parents=True, exist_ok=True)
    pre = json.loads((EXP / "preflight.json").read_text()) \
        if (EXP / "preflight.json").is_file() else preflight()
    print(json.dumps(pre, indent=2, sort_keys=True), flush=True)
    state = abc.load_state()
    if args.stage in ("all", "train"):
        if not checkpoint(EVAL_EPOCH).is_file():
            abc.check_resources()
            started = time.time()
            abc.run(f"train_{NAME}_e{EVAL_EPOCH}", train_command(pre))
            state["runs"].setdefault(NAME, {})["train"] = {
                "status": "complete", "seconds": round(time.time() - started, 1)}
            abc.save_state(state)
        state["runs"].setdefault(NAME, {})["verify"] = verify(pre)
        abc.save_state(state)
    if args.stage in ("all", "eval"):
        abc.eval_run(NAME, EVAL_EPOCH, state)
        abc.gate_run(NAME, EVAL_EPOCH, state)
    if args.stage in ("all", "analyze"):
        analyze()


if __name__ == "__main__":
    main()
