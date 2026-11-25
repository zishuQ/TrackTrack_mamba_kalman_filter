#!/usr/bin/env python3
"""Resumable driver for the full-parameter pairwise motion-gate ranking experiment.

Order (serial, each stage resumable):
  labels -> preflight (binding + full-parameter A/B benchmark) -> train A
  (L_base, 20) -> fixed-val diagnostics A -> raw val eval A -> train B
  (L_base+L_rank, 20) -> diagnostics B -> raw val eval B -> report.

Both groups train all parameters (shared encoder, HCE, RSC and both gates)
with the original appearance rank + column losses and weights. They share
init, seed, sampling order, lr and 20 shard-epochs; only the motion rank term
differs. No test submission and no 200-epoch extension.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
BASE = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
COST = ROOT / "outputs/agentguard/experiments/shutrack_motion_cost_bce_20261122/SportsMOT"
REF = ROOT / "outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_motion_rank_allparams_20261122/SportsMOT"
CONFIG = EXP / "config.json"
TARGETS = REF / "targets"
RANKS = ROOT / "outputs/agentguard/experiments/shutrack_motion_rank_20261122/SportsMOT/rank_labels"
DATA90 = ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data"
BASELINE = BASE / "ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"
BASELINE_SUMMARY = BASE / "ablation/SportsMOT/evals/new/summary.json"
COST_SUMMARY = COST / "evals/lambda_0.25_e20/summary.json"
TRAIN_TOOL = ROOT / "tools/train_motion_rank.py"
VERIFY_TOOL = ROOT / "tools/verify_motion_rank.py"
DIAG_TOOL = ROOT / "tools/evaluate_motion_rank_val.py"
GEN_TOOL = ROOT / "tools/generate_motion_rank_labels.py"
VARIANTS = {"A": 0.0, "B": 1.0}
GROUP = "appearance_row_column"
STAGE_EPOCHS = 20
VAL_SEQS = tuple(
    line.strip() for line in (TRACKER / "trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)
HEAVY = {"train_motion_rank.py", "train_supervision_groups.py", "run.py",
         "verify_motion_rank.py", "evaluate_motion_rank_val.py",
         "generate_motion_rank_labels.py", "generate_trainval_paired.py",
         "evaluate_motion_gate_targets.py"}
STATE_PATH = EXP / "state.json"
LOGS = EXP / "logs"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def gpu_free_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()[0]
        return float(out.strip())
    except Exception:
        return 0.0


def mem_available_mib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def active_jobs():
    jobs = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [a.decode(errors="replace") for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        scripts = {Path(arg).name for arg in argv if arg.endswith(".py")}
        if scripts & HEAVY:
            jobs.append({"pid": int(entry.name), "scripts": sorted(scripts & HEAVY)})
    return jobs


class Driver:
    def __init__(self):
        EXP.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(exist_ok=True)
        self.state = (json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file()
                      else {"created_at": now(), "stages": {}, "results": {}})

    def save(self):
        temporary = STATE_PATH.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temporary.replace(STATE_PATH)

    def wait_resources(self, label):
        while True:
            jobs = active_jobs()
            free = gpu_free_mib()
            avail = mem_available_mib()
            if not jobs and free >= 2048 and avail >= 18000:
                return
            print(f"{now()} waiting for resources before {label}: jobs={jobs} "
                  f"gpu_free={free:.0f}MiB mem_available={avail:.0f}MiB", flush=True)
            time.sleep(30)

    def run(self, label, command, cwd=ROOT):
        self.wait_resources(label)
        log = LOGS / f"{label}.log"
        started = time.time()
        entry = self.state["stages"].setdefault(label, {})
        entry.update({"status": "running", "command": [str(x) for x in command],
                      "started_at": now()})
        self.save()
        with log.open("ab") as handle:
            handle.write(f"\n===== {now()} =====\n{command}\n".encode())
            handle.flush()
            code = subprocess.call(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
        entry.update({"finished_at": now(), "seconds": round(time.time() - started, 1),
                      "exit_code": code, "log": str(log),
                      "status": "complete" if code == 0 else "failed"})
        self.save()
        if code:
            raise RuntimeError(f"{label}: exit {code}")
        return log

    def labels(self):
        present = sorted(p.name for p in RANKS.iterdir() if (p / "metadata.json").is_file()) \
            if RANKS.is_dir() else []
        if len(present) >= 90:
            self.state["labels"] = {"sequences": len(present), "reused": True}
            self.save()
            return
        self.run("generate_rank_labels", [
            str(PY), "-u", str(GEN_TOOL), "--data-root", str(DATA90),
            "--target-root", str(TARGETS), "--out", str(RANKS)])

    def preflight(self):
        out = EXP / "preflight_verification.json"
        if out.is_file():
            payload = json.loads(out.read_text())
            if (payload.get("binding", {}).get("keep_rows_match")
                    and payload.get("benchmark")
                    and payload.get("model", {}).get("optimizer_covers_all")):
                self.state["preflight"] = {
                    "model_total": payload["model"]["model_total"],
                    "all_parameters_requires_grad": payload["model"]["all_parameters_requires_grad"],
                    "optimizer_covers_all": payload["model"]["optimizer_covers_all"],
                    "benchmark_A": payload["benchmark"].get("A", {}).get("step_seconds_mean"),
                    "benchmark_B": payload["benchmark"].get("B", {}).get("step_seconds_mean"),
                    "reused": True}
                self.save()
                return payload
        command = [str(PY), "-u", str(VERIFY_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                   "--checkpoint", str(BASELINE), "--rows", "1024",
                   "--steps", "10", "--warmup", "3", "--device", "cuda",
                   "--out", str(out)]
        self.run("preflight_verification", command)
        payload = json.loads(out.read_text())
        self.state["preflight"] = {
            "model_total": payload["model"]["model_total"],
            "all_parameters_requires_grad": payload["model"]["all_parameters_requires_grad"],
            "optimizer_covers_all": payload["model"]["optimizer_covers_all"],
            "benchmark_A": payload["benchmark"].get("A", {}).get("step_seconds_mean"),
            "benchmark_B": payload["benchmark"].get("B", {}).get("step_seconds_mean")}
        self.save()
        return payload

    def checkpoint(self, variant):
        return EXP / "runs" / variant / "groups" / GROUP / "checkpoints/last.pt"

    def train(self, variant):
        weight = VARIANTS[variant]
        label = f"train_{variant}"
        groups = EXP / "runs" / variant / "groups"
        checkpoint = self.checkpoint(variant)
        if checkpoint.is_file():
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", 0)) >= STAGE_EPOCHS:
                self.state["results"].setdefault(variant, {})["train"] = {
                    "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
                    "global_step": int(payload.get("global_step", 0)), "reused": True}
                self.save()
                return checkpoint
        command = [str(PY), "-u", str(TRAIN_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                   "--groups-root", str(groups), "--base-lambda", "0.25",
                   "--rank-weight", f"{weight:g}", "--pair-seed", "20261122",
                   "--stop-after-epoch", str(STAGE_EPOCHS)]
        if checkpoint.is_file():
            command.append("--resume")
        self.run(label, command)
        if not checkpoint.is_file():
            raise RuntimeError(f"{label}: missing checkpoint")
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.state["results"].setdefault(variant, {})["train"] = {
            "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
            "global_step": int(payload.get("global_step", 0))}
        self.save()
        return checkpoint

    def diagnose(self, variant, checkpoint):
        out = EXP / "diagnostics" / f"{variant}.json"
        if out.is_file():
            payload = json.loads(out.read_text())
            if int(payload.get("sequences", 0)) == len(VAL_SEQS):
                self.state["results"].setdefault(variant, {})["diagnostics"] = {
                    "summary": str(out), "reused": True}
                self.save()
                return payload
        command = [str(PY), "-u", str(DIAG_TOOL), "--config", str(CONFIG),
                   "--checkpoint", str(checkpoint), "--target-root", str(TARGETS),
                   "--rank-root", str(RANKS), "--sequence-root", str(DATA90),
                   "--split", "val", "--batch-size", "2048", "--device", "cuda",
                   "--out", str(out)]
        self.run(f"diagnostics_{variant}", command)
        payload = json.loads(out.read_text())
        if int(payload.get("sequences", 0)) != len(VAL_SEQS):
            raise RuntimeError(f"diagnostics_{variant}: wrong sequence count")
        self.state["results"].setdefault(variant, {})["diagnostics"] = {"summary": str(out)}
        self.save()
        return payload

    def evaluate(self, variant, checkpoint):
        label = f"eval_{variant}"
        out = EXP / "evals" / f"{variant}_e{STAGE_EPOCHS}"
        summary = out / "summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text())
            if len(payload.get("per_sequence") or {}) == len(VAL_SEQS):
                self.state["results"].setdefault(variant, {})["eval"] = {
                    "summary": str(summary), "combined": payload.get("combined"), "reused": True}
                self.save()
                return payload
        out.mkdir(parents=True, exist_ok=True)
        command = [str(PY), "-u", "run.py", "--dataset", "SportsMOT", "--mode", "val",
                   "--sequences", *VAL_SEQS, "--seed", "10000", "--kf-type", "nsa",
                   "--agentguard-mode", "shutrack-v2", "--shutrack-checkpoint", str(checkpoint),
                   "--agentguard-device", "cpu", "--detection-cache-root",
                   str(ROOT / "outputs/agentguard/detection_cache"),
                   "--output_dir", str(out), "--tracker-suffix",
                   f"motionrank_{variant}_e{STAGE_EPOCHS}",
                   "--agentguard-stats-output", str(out / "gates.jsonl"),
                   "--print-per-sequence-metrics", "--eval-json", str(summary),
                   "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
        self.run(label, command, cwd=TRACKER)
        payload = json.loads(summary.read_text())
        if len(payload.get("per_sequence") or {}) != len(VAL_SEQS):
            raise RuntimeError(f"{label}: wrong per-sequence count")
        self.state["results"].setdefault(variant, {})["eval"] = {
            "summary": str(summary), "combined": payload.get("combined")}
        self.save()
        return payload


def load_metrics(variant):
    path = EXP / "runs" / variant / "groups" / GROUP / "motion_rank_metrics.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    keys = ("epoch", "rank_pairs", "rank_accuracy", "rank_logit_gap_mean", "rank_gate_gap_mean",
            "rank_pair_weight_mean", "rank_loss_mean", "target_mae", "gate_gt095_fraction",
            "gate_lt005_fraction", "loss_full_mean", "loss_pred_mean", "loss_best_mean",
            "saved_harmful_mean", "added_useful_mean", "net_delta_mean",
            "harmful_count", "harmful_gate_mean", "useful_count", "useful_gate_mean",
            "target1_count", "target1_gate_mean", "target_lt1_count", "target_lt1_gate_mean",
            "appearance_valid", "appearance_target_mean", "appearance_gate_mean",
            "appearance_mae", "appearance_gate_gt095_fraction",
            "appearance_gate_lt005_fraction", "appearance_bce_mean",
            "cuda_peak_mib", "process_peak_rss_gib")
    def select(row):
        return {key: row.get(key) for key in keys if key in row}
    averaged = {}
    for key in ("rank_accuracy", "rank_gate_gap_mean", "net_delta_mean",
                "saved_harmful_mean", "added_useful_mean"):
        values = [r[key] for r in rows[-5:] if r.get(key) is not None]
        averaged[key] = sum(values) / len(values) if values else None
    return {"last": select(rows[-1]), "first": select(rows[0]), "epochs": len(rows),
            "last5_mean": averaged}


def load_train_log(variant):
    path = EXP / "runs" / variant / "groups" / GROUP / "metrics.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    return {"epochs": len(rows),
            "epoch_seconds_mean": sum(r.get("epoch_seconds", 0.0) for r in rows) / len(rows),
            "epoch_seconds_total": sum(r.get("epoch_seconds", 0.0) for r in rows),
            "samples_total": sum(int(r.get("samples", 0)) for r in rows),
            "last_epoch": rows[-1].get("epoch"),
            "last_loss": rows[-1].get("loss"),
            "last_motion_loss": rows[-1].get("motion_loss"),
            "last_appearance_loss": rows[-1].get("appearance_loss"),
            "last_rank_mean": rows[-1].get("rank_mean"),
            "last_column_loss": rows[-1].get("column_loss"),
            "last_motion_gate_mean": rows[-1].get("motion_gate_mean"),
            "last_appearance_gate_mean": rows[-1].get("appearance_gate_mean")}


def _num(value, digits=3, scale=1.0):
    if value is None:
        return "-"
    return f"{value * scale:.{digits}f}"


def _pct(value, digits=2):
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def compare(driver):
    rows = {}
    refs = {"baseline": BASELINE_SUMMARY, "cost_l025_e20": COST_SUMMARY}
    for name, path in refs.items():
        if path.is_file():
            payload = json.loads(path.read_text())
            rows[name] = {"combined": payload.get("combined"),
                          "per_sequence": payload.get("per_sequence")}
    for variant in VARIANTS:
        entry = (driver.state["results"].get(variant) or {}).get("eval") or {}
        path = Path(entry.get("summary", ""))
        if path.is_file():
            payload = json.loads(path.read_text())
            rows[variant] = {"combined": payload.get("combined"),
                             "per_sequence": payload.get("per_sequence")}
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
    summary = {
        name: {"combined": {k: (row["combined"] or {}).get(k) for k in keys},
               "per_sequence": {
                   seq: {k: ((row.get("per_sequence") or {}).get(seq) or {}).get(k) for k in keys}
                   for seq in VAL_SEQS}}
        for name, row in rows.items()
    }
    diagnostics = {"val": {}, "train": {}, "train_log": {}}
    for variant in VARIANTS:
        path = EXP / "diagnostics" / f"{variant}.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            diagnostics["val"][variant] = {key: payload.get(key) for key in
                                           ("epoch", "counts", "gate", "loss", "rank",
                                            "appearance", "per_sequence")}
        diagnostics["train"][variant] = load_metrics(variant)
        diagnostics["train_log"][variant] = load_train_log(variant)
    output = {"comparison": summary, "diagnostics": diagnostics,
              "val_sequences": list(VAL_SEQS)}
    (EXP / "comparison.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    labels = {"baseline": "baseline", "cost_l025_e20": "λ0.25 e20",
              "A": "A: L_base", "B": "B: L_base+L_rank"}
    lines = ["# SportsMOT 运动门控成对排序监督（全参数训练，20 shard-epochs）", "",
             "A/B 两组均为全参数训练（共享编码器、HCE、RSC 与两路门控），保留原外观 rank+column 监督及权重；"
             "唯一差别是 B 增加运动成对排序项。",
             "基线 = new 架构原 checkpoint（83.343）；λ0.25 e20 = 之前仅门控头训练的代价 BCE 对照（83.315），仅供参考。", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "cost_l025_e20", "A", "B"):
        if name not in summary:
            continue
        combined = summary[name]["combined"]
        cells = []
        for k in keys:
            v = combined.get(k)
            cells.append("-" if v is None else (f"{v:.0f}" if k == "IDSW" else f"{v*100:.3f}"))
        lines.append(f"| {labels[name]} | " + " | ".join(cells) + " |")
    if "baseline" in summary:
        lines += ["", "## 逐序列 HOTA 变化（pp）", "",
                  "| 序列 | baseline | A−base | B−base | B−A |", "|---|---:|---:|---:|---:|"]
        improved_a = improved_b = 0
        for seq in VAL_SEQS:
            base = (summary["baseline"]["per_sequence"].get(seq) or {}).get("HOTA")
            a = ((summary.get("A") or {}).get("per_sequence", {}).get(seq) or {}).get("HOTA")
            b = ((summary.get("B") or {}).get("per_sequence", {}).get(seq) or {}).get("HOTA")
            if base is None or a is None or b is None:
                continue
            if a > base + 1e-9:
                improved_a += 1
            if b > base + 1e-9:
                improved_b += 1
            lines.append(f"| {seq} | {base*100:.3f} | {(a-base)*100:+.3f} | "
                         f"{(b-base)*100:+.3f} | {(b-a)*100:+.3f} |")
        lines += ["", f"A 改善序列数 {improved_a}/{len(VAL_SEQS)}；"
                      f"B 改善序列数 {improved_b}/{len(VAL_SEQS)}。"]
    lines += ["", "## 固定 val 诊断", "",
              "| 组 | 有效 | 有益(g=1) | 有害 | keep | suppress | gate 均值 | 有害 gate | 有益 gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("A", "B"):
        d = diagnostics["val"].get(variant)
        if not d:
            continue
        c, g = d["counts"], d["gate"]
        lines.append(f"| {variant} | {int(c['valid'])} | {int(c['target1'])} | "
                     f"{int(c['harmful'])} | {int(c['keep'])} | {int(c['suppress'])} | "
                     f"{_num(g['gate_mean'], 4)} | {_num(g['harmful_mean'], 4)} | "
                     f"{_num(g['useful_mean'], 4)} |")
    lines += ["", "| 组 | full | pred | best | 有害节省/事件 | 有益新增/事件 | 净变化/事件 | rank 对数 | 排序准确率 | logit 差 | gate 差 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("A", "B"):
        d = diagnostics["val"].get(variant)
        if not d:
            continue
        loss, rank = d["loss"], d["rank"]
        lines.append(f"| {variant} | {_num(loss['full_mean'], 3, 1e4)} | "
                     f"{_num(loss['pred_mean'], 3, 1e4)} | {_num(loss['best_mean'], 3, 1e4)} | "
                     f"{_num(loss['saved_harmful_mean'], 3, 1e4)} | "
                     f"{_num(loss['added_useful_mean'], 3, 1e4)} | "
                     f"{_num(loss['net_delta_mean'], 3, 1e4)} | {int(rank['pairs'])} | "
                     f"{_pct(rank['accuracy'])} | {_num(rank['logit_gap_mean'], 3)} | "
                     f"{_num(rank['gate_gap_mean'], 4)} |")
    lines += ["", "损失单位为 1e-4；排序准确率是门控 logit 的成对比较，与 box 实际增益分开看。", "",
              "外观支路（固定 val）", "",
              "| 组 | 有效 | 目标均值 | gate 均值 | MAE | >0.95 | <0.05 | BCE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("A", "B"):
        d = diagnostics["val"].get(variant)
        if not d:
            continue
        a = d.get("appearance") or {}
        lines.append(f"| {variant} | {int(a.get('valid') or 0)} | "
                     f"{_num(a.get('target_mean'), 4)} | {_num(a.get('gate_mean'), 4)} | "
                     f"{_num(a.get('mae'), 4)} | {_pct(a.get('gt095_fraction'))} | "
                     f"{_pct(a.get('lt005_fraction'))} | {_num(a.get('bce_mean'), 4)} |")
    lines += ["", "## 训练期最后一轮诊断", "",
              "| 组 | epoch | rank 对数 | 排序准确率 | logit 差 | gate 差 | 有害 gate | 有益 gate | 净变化/事件 | CUDA 峰值 MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("A", "B"):
        last = (diagnostics["train"].get(variant) or {}).get("last")
        if not last:
            continue
        lines.append(f"| {variant} | {last.get('epoch')} | {int(last.get('rank_pairs') or 0)} | "
                     f"{_pct(last.get('rank_accuracy'))} | "
                     f"{_num(last.get('rank_logit_gap_mean'), 3)} | "
                     f"{_num(last.get('rank_gate_gap_mean'), 4)} | "
                     f"{_num(last.get('harmful_gate_mean'), 4)} | "
                     f"{_num(last.get('useful_gate_mean'), 4)} | "
                     f"{_num(last.get('net_delta_mean'), 3, 1e4)} | "
                     f"{_num(last.get('cuda_peak_mib'), 1, 1)} |")
    lines += ["", "## 两路表现（训练末轮）", "",
              "| 组 | motion loss | appearance loss | rank mean | column loss | motion gate | appearance gate | appearance MAE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("A", "B"):
        last = (diagnostics["train"].get(variant) or {}).get("last") or {}
        log = diagnostics["train_log"].get(variant) or {}
        if not last and not log:
            continue
        lines.append(f"| {variant} | {_num(log.get('last_motion_loss'), 6)} | "
                     f"{_num(log.get('last_appearance_loss'), 6)} | "
                     f"{_num(log.get('last_rank_mean'), 6)} | "
                     f"{_num(log.get('last_column_loss'), 6)} | "
                     f"{_num(log.get('last_motion_gate_mean'), 4)} | "
                     f"{_num(last.get('appearance_gate_mean'), 4)} | "
                     f"{_num(last.get('appearance_mae'), 4)} |")
    lines += ["", "## 训练成本", "",
              "| 组 | epochs | 平均 epoch 秒 | 总样本 | 末轮 loss | 末轮 motion loss |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant in ("A", "B"):
        log = diagnostics["train_log"].get(variant) or {}
        if not log:
            continue
        lines.append(f"| {variant} | {log.get('epochs')} | "
                     f"{_num(log.get('epoch_seconds_mean'), 1, 1)} | "
                     f"{int(log.get('samples_total') or 0)} | "
                     f"{_num(log.get('last_loss'), 6)} | {_num(log.get('last_motion_loss'), 6)} |")
    (EXP / "comparison.md").write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "labels", "preflight", "train_a",
                                            "train_b", "diagnostics", "eval", "report"),
                        default="all")
    args = parser.parse_args()
    driver = Driver()
    if not CONFIG.is_file():
        raise SystemExit(f"missing config {CONFIG}")
    if args.stage in ("all", "labels"):
        driver.labels()
    if args.stage in ("all", "preflight"):
        payload = driver.preflight()
        print(json.dumps({"preflight": payload.get("model", {}),
                          "benchmark": payload.get("benchmark", {})}, indent=2), flush=True)
    if args.stage in ("all", "train_a"):
        checkpoint = driver.train("A")
        driver.diagnose("A", checkpoint)
    if args.stage in ("all", "train_b"):
        checkpoint = driver.train("B")
        driver.diagnose("B", checkpoint)
    if args.stage == "diagnostics":
        for variant in VARIANTS:
            checkpoint = driver.checkpoint(variant)
            if checkpoint.is_file():
                driver.diagnose(variant, checkpoint)
    if args.stage in ("all", "eval"):
        for variant in VARIANTS:
            checkpoint = driver.checkpoint(variant)
            if checkpoint.is_file():
                driver.evaluate(variant, checkpoint)
    if args.stage in ("all", "report"):
        output = compare(driver)
        print(json.dumps({"finished_at": now(),
                          "rows": list(output["comparison"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
