#!/usr/bin/env python3
"""Resumable driver for the cost-weighted motion-gate BCE exploration.

Stages: preflight -> lambda sweep (20 shard-epochs, val) -> promote 1 candidate
plus the lambda=0 control -> 80 shard-epochs + val -> 200 shard-epochs + val.
Heavy jobs run serially; resources are checked before each command.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
BASE = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
REF = ROOT / "outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_motion_cost_bce_20261122/SportsMOT"
CONFIG = EXP / "config.json"
TARGETS = REF / "targets"
BASELINE = BASE / "ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"
TRAIN_TOOL = ROOT / "tools/train_motion_cost_bce.py"
VERIFY_TOOL = ROOT / "tools/verify_motion_cost_bce.py"
VAL_SEQS = tuple(
    line.strip() for line in (TRACKER / "trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)
LAMBDAS = (0.0, 0.25, 1.0, 4.0)
HEAVY = {"train_motion_cost_bce.py", "train_supervision_groups.py", "run.py",
         "generate_trainval_paired.py", "arch_night_test_submit.py",
         "verify_motion_cost_bce.py", "evaluate_motion_gate_targets.py"}
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

    def train(self, lam, stage):
        label = f"train_lambda_{lam:g}_e{stage}"
        groups = EXP / "runs" / f"lambda_{lam:g}" / "groups"
        checkpoint = groups / "appearance_row_column/checkpoints/last.pt"
        if checkpoint.is_file():
            try:
                import torch
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if int(payload.get("epoch", 0)) >= stage:
                    self.state["results"].setdefault(f"lambda_{lam:g}", {})[f"train_{stage}"] = {
                        "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
                        "global_step": int(payload.get("global_step", 0)), "reused": True}
                    self.save()
                    return checkpoint
            except Exception:
                pass
        command = [str(PY), "-u", str(TRAIN_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--groups-root", str(groups),
                   "--lambda-cost", str(lam), "--freeze-motion-head",
                   "--stop-after-epoch", str(stage)]
        if checkpoint.is_file():
            command.append("--resume")
        self.run(label, command)
        if not checkpoint.is_file():
            raise RuntimeError(f"{label}: missing checkpoint")
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.state["results"].setdefault(f"lambda_{lam:g}", {})[f"train_{stage}"] = {
            "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
            "global_step": int(payload.get("global_step", 0))}
        self.save()
        return checkpoint

    def evaluate(self, lam, stage, checkpoint):
        label = f"eval_lambda_{lam:g}_e{stage}"
        out = EXP / "evals" / f"lambda_{lam:g}_e{stage}"
        summary = out / "summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text())
            if len(payload.get("per_sequence") or {}) == len(VAL_SEQS):
                self.state["results"].setdefault(f"lambda_{lam:g}", {})[f"eval_{stage}"] = {
                    "summary": str(summary), "combined": payload.get("combined"), "reused": True}
                self.save()
                return payload
        out.mkdir(parents=True, exist_ok=True)
        command = [str(PY), "-u", "run.py", "--dataset", "SportsMOT", "--mode", "val",
                   "--sequences", *VAL_SEQS, "--seed", "10000", "--kf-type", "nsa",
                   "--agentguard-mode", "shutrack-v2", "--shutrack-checkpoint", str(checkpoint),
                   "--agentguard-device", "cpu", "--detection-cache-root",
                   str(ROOT / "outputs/agentguard/detection_cache"),
                   "--output_dir", str(out), "--tracker-suffix", f"costbce_{lam:g}_e{stage}",
                   "--agentguard-stats-output", str(out / "gates.jsonl"),
                   "--print-per-sequence-metrics", "--eval-json", str(summary),
                   "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
        self.run(label, command, cwd=TRACKER)
        payload = json.loads(summary.read_text())
        if len(payload.get("per_sequence") or {}) != len(VAL_SEQS):
            raise RuntimeError(f"{label}: wrong per-sequence count")
        self.state["results"].setdefault(f"lambda_{lam:g}", {})[f"eval_{stage}"] = {
            "summary": str(summary), "combined": payload.get("combined")}
        self.save()
        return payload

    def preflight(self):
        out = EXP / "preflight_verification.json"
        if out.is_file():
            return json.loads(out.read_text())
        command = [str(PY), "-u", str(VERIFY_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--checkpoint", str(BASELINE),
                   "--lambda-cost", "1.0", "--rows", "1024", "--out", str(out)]
        self.run("preflight_verification", command)
        payload = json.loads(out.read_text())
        self.state["preflight"] = payload
        self.save()
        return payload

    def sweep(self):
        for lam in LAMBDAS:
            checkpoint = self.train(lam, 20)
            self.evaluate(lam, 20, checkpoint)

    def promote(self):
        scores = {}
        for lam in LAMBDAS:
            entry = (self.state["results"].get(f"lambda_{lam:g}") or {}).get("eval_20") or {}
            combined = entry.get("combined") or {}
            if combined.get("HOTA") is not None:
                scores[lam] = combined["HOTA"]
        nonzero = {lam: score for lam, score in scores.items() if lam != 0.0}
        best = max(nonzero, key=nonzero.get) if nonzero else (max(scores, key=scores.get) if scores else 1.0)
        candidates = sorted(set([0.0, float(best)]))
        self.state["promotion"] = {"scores_20": scores, "best_nonzero": best,
                                   "candidates": candidates}
        self.save()
        return candidates

    def stage(self, stage, candidates):
        for lam in candidates:
            checkpoint = self.train(lam, stage)
            self.evaluate(lam, stage, checkpoint)


def compare(driver):
    rows = {}
    refs = {
        "baseline": BASE / "ablation/SportsMOT/evals/new/summary.json",
        "A_force_motion1": REF / "oracle_gate_diagnostics_20261121/A_force_motion1/summary.json",
        "B_gt_oracle": REF / "oracle_gate_diagnostics_20261121/B_gt_oracle/summary.json",
        "bce_previous": REF / "eval_val/summary.json",
    }
    for name, path in refs.items():
        if path.is_file():
            payload = json.loads(path.read_text())
            rows[name] = {"combined": payload.get("combined"),
                          "per_sequence": payload.get("per_sequence")}
    for key, entry in driver.state["results"].items():
        for stage_name, record in entry.items():
            if not stage_name.startswith("eval_") or not record.get("summary"):
                continue
            path = Path(record["summary"])
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            rows[f"{key}_{stage_name}"] = {"combined": payload.get("combined"),
                                           "per_sequence": payload.get("per_sequence")}
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
    summary = {
        name: {"combined": {k: (row["combined"] or {}).get(k) for k in keys},
               "per_sequence": {
                   seq: {k: ((row.get("per_sequence") or {}).get(seq) or {}).get(k) for k in keys}
                   for seq in VAL_SEQS}}
        for name, row in rows.items()
    }
    (EXP / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# SportsMOT 运动门控代价加权 BCE 探索", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    order = list(refs) + [f"lambda_{lam:g}_{stage}" for lam in LAMBDAS
                          for stage in ("eval_20", "eval_80", "eval_200")]
    for name in order:
        if name not in summary:
            continue
        c = summary[name]["combined"]
        def fmt(k):
            v = c.get(k)
            if v is None:
                return "-"
            return f"{v:.0f}" if k == "IDSW" else f"{v*100:.3f}"
        lines.append(f"| {name} | " + " | ".join(fmt(k) for k in keys) + " |")
    (EXP / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "preflight", "sweep", "promote",
                                             "stage80", "stage200", "report"),
                        default="all")
    args = parser.parse_args()
    driver = Driver()
    if not CONFIG.is_file():
        raise SystemExit(f"missing config {CONFIG}")
    if args.stage in ("all", "preflight"):
        print(json.dumps({"preflight": driver.preflight().get("model", {})}, indent=2), flush=True)
    if args.stage in ("all", "sweep"):
        driver.sweep()
    candidates = (driver.state.get("promotion") or {}).get("candidates")
    if not candidates:
        candidates = driver.promote() if driver.state.get("results") else [0.0, 1.0]
    if args.stage in ("all", "promote"):
        candidates = driver.promote()
        print("candidates:", candidates, flush=True)
    if args.stage in ("all", "stage80"):
        driver.stage(80, candidates)
    if args.stage in ("all", "stage200"):
        driver.stage(200, candidates)
    if args.stage in ("all", "report"):
        summary = compare(driver)
        print(json.dumps({"finished_at": now(), "rows": list(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
