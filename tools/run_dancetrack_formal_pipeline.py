#!/usr/bin/env python3
"""DanceTrack formal D-supervision pipeline.

Stages (serial, resumable): collect (40 train sequences, full-write replay,
appearance rank+column labels, lean low-memory pipeline) -> train 200
shard-epochs from scratch (seed 42) -> val normal gates -> val motion gate 1
-> report.  No test submission.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
DATA_DIR = ROOT / "data"
CACHE = ROOT / "outputs/agentguard/detection_cache"
EXP = ROOT / "outputs/agentguard/experiments/dancetrack_formal_d_supervision_20261122"
DATA = EXP / "data"
GROUPS = EXP / "groups"
LOGS = EXP / "logs"
GEN_TOOL = ROOT / "tools/generate_trainval_paired.py"
TRAIN_TOOL = ROOT / "tools/train_supervision_groups.py"
GROUP = "appearance_row_column"
EPOCHS = 200
TRAIN_SEQS = tuple(line.strip() for line in
                   (TRACKER / "trackeval/seqmap/dancetrack/train.txt").read_text().splitlines()
                   if line.strip() and line.strip() != "name")
VAL_SEQS = tuple(line.strip() for line in
                 (TRACKER / "trackeval/seqmap/dancetrack/val.txt").read_text().splitlines()
                 if line.strip() and line.strip() != "name")
HEAVY = {"train_supervision_groups.py", "generate_trainval_paired.py",
         "sportsmot_supervision_pilot.py", "run.py", "run_oracle_gate_dancetrack.py"}
STATE_PATH = EXP / "state.json"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def gpu_free_mib():
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True).splitlines()[0].strip())
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
        for path in (EXP, LOGS):
            path.mkdir(parents=True, exist_ok=True)
        self.state = (json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file()
                      else {"created_at": now(), "stages": {}, "results": {}})

    def save(self):
        tmp = STATE_PATH.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        tmp.replace(STATE_PATH)

    def wait_resources(self, label):
        while True:
            jobs = active_jobs()
            free = gpu_free_mib()
            avail = mem_available_mib()
            if not jobs and free >= 2048 and avail >= 18000:
                return
            print(f"{now()} waiting before {label}: jobs={jobs} gpu_free={free:.0f} "
                  f"mem={avail:.0f}", flush=True)
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

    def collect(self):
        complete = [s for s in TRAIN_SEQS
                    if (DATA / s / "summary.json").is_file()
                    and json.loads((DATA / s / "summary.json").read_text()).get("complete")]
        self.state["results"]["collect"] = {"complete": len(complete),
                                            "total": len(TRAIN_SEQS)}
        self.save()
        if len(complete) == len(TRAIN_SEQS):
            return
        command = [str(PY), "-u", str(GEN_TOOL), "--root", str(EXP),
                   "--dataset", "DanceTrack", "--split", "train",
                   "--full-write", "--column-probe", "--lean"]
        self.run("collect_train", command)

    def train(self):
        group_dir = GROUPS / GROUP
        checkpoint = group_dir / "checkpoints/last.pt"
        target = group_dir / f"checkpoints/epoch_{EPOCHS:03d}.pt"
        if checkpoint.is_file() and target.is_file():
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", 0)) >= EPOCHS:
                self.state["results"]["train"] = {"checkpoint": str(target), "reused": True}
                self.save()
                return
        command = [str(PY), "-u", str(TRAIN_TOOL),
                   "--group", GROUP, "--dataset", "DanceTrack",
                   "--from-scratch", "--split", "train",
                   "--data-root", str(DATA), "--groups-root", str(GROUPS),
                   "--batch-size", "1024", "--lr", "1e-4", "--weight-decay", "1e-4",
                   "--seed", "42", "--shards", "4", "--max-epochs", str(EPOCHS),
                   "--snapshot-every", "20", "--rank-weight", "18",
                   "--rank-alpha", "0.95", "--rank-tau", "0.1",
                   "--column-weight", "50", "--device", "cuda"]
        if checkpoint.is_file():
            command.append("--resume")
        self.run("train_d", command)
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != EPOCHS or not target.is_file():
            raise RuntimeError("DanceTrack training did not finish at epoch 200")
        self.state["results"]["train"] = {"checkpoint": str(target),
                                          "epoch": payload.get("epoch"),
                                          "global_step": payload.get("global_step")}
        self.save()

    def evaluate(self, name, extra):
        group_dir = GROUPS / GROUP
        checkpoint = group_dir / f"checkpoints/epoch_{EPOCHS:03d}.pt"
        out = EXP / "evals" / name
        summary = out / "summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text())
            if len(payload.get("per_sequence") or {}) == len(VAL_SEQS):
                self.state["results"].setdefault("evals", {})[name] = {
                    "summary": str(summary), "combined": payload.get("combined"),
                    "reused": True}
                self.save()
                return payload
        out.mkdir(parents=True, exist_ok=True)
        command = [str(PY), "-u", "run.py", "--dataset", "DanceTrack",
                   "--data_dir", str(DATA_DIR) + "/", "--mode", "val",
                   "--sequences", *VAL_SEQS, "--seed", "10000", "--kf-type", "nsa",
                   "--agentguard-mode", "shutrack-v2",
                   "--shutrack-checkpoint", str(checkpoint),
                   "--agentguard-device", "cpu",
                   "--detection-cache-root", str(CACHE),
                   "--output_dir", str(out), "--tracker-suffix", f"dance_d_e{EPOCHS}_{name}",
                   "--agentguard-stats-output", str(out / "gates.jsonl"),
                   "--print-per-sequence-metrics", "--eval-json", str(summary),
                   "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
        command += extra
        self.run(f"eval_{name}", command, cwd=TRACKER)
        payload = json.loads(summary.read_text())
        if len(payload.get("per_sequence") or {}) != len(VAL_SEQS):
            raise RuntimeError(f"eval_{name}: wrong per-sequence count")
        self.state["results"].setdefault("evals", {})[name] = {
            "summary": str(summary), "combined": payload.get("combined")}
        self.save()
        return payload

    def evals(self):
        self.evaluate("normal", [])
        self.evaluate("motion1", ["--agentguard-disable-kf-gate"])


def report(driver):
    paths = {"normal": EXP / "evals/normal/summary.json",
             "motion1": EXP / "evals/motion1/summary.json"}
    data = {}
    for name, path in paths.items():
        if path.is_file():
            payload = json.loads(path.read_text())
            data[name] = {"combined": payload.get("combined"),
                          "per_sequence": payload.get("per_sequence")}
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
    rows = {}
    for name, payload in data.items():
        rows[name] = {"combined": {k: payload["combined"].get(k) for k in keys},
                      "per_sequence": {seq: {k: (payload.get("per_sequence", {}).get(seq) or {}).get(k)
                                             for k in keys} for seq in VAL_SEQS}}
    gates = {}
    for name in ("normal", "motion1"):
        path = EXP / f"evals/{name}/gates.jsonl"
        if path.is_file():
            stats = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            motion = [r.get("shutrack_avg_motion_gate") for r in stats
                      if r.get("shutrack_avg_motion_gate") is not None]
            appearance = [r.get("shutrack_avg_appearance_gate") for r in stats
                          if r.get("shutrack_avg_appearance_gate") is not None]
            gates[name] = {"rows": len(stats),
                           "motion_min": min(motion) if motion else None,
                           "motion_mean": (sum(motion) / len(motion)) if motion else None,
                           "motion_max": max(motion) if motion else None,
                           "appearance_mean": (sum(appearance) / len(appearance))
                           if appearance else None}
    metrics_path = GROUPS / GROUP / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()
               if line.strip()] if metrics_path.is_file() else []
    group_config = GROUPS / GROUP / "config.json"
    config = json.loads(group_config.read_text()) if group_config.is_file() else {}
    payload = {"rows": rows, "gates": gates,
               "deltas": {}, "config": config,
               "training": {"epochs": len(metrics),
                            "samples_total": sum(r.get("samples", 0) for r in metrics),
                            "seconds_total": sum(r.get("epoch_seconds", 0) for r in metrics),
                            "last": metrics[-1] if metrics else None}}
    if "normal" in rows and "motion1" in rows:
        payload["deltas"]["normal_minus_motion1"] = {
            k: (rows["normal"]["combined"][k] - rows["motion1"]["combined"][k])
            if None not in (rows["normal"]["combined"][k], rows["motion1"]["combined"][k]) else None
            for k in keys}
    (EXP / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = ["# DanceTrack 原 D 监督 baseline 与运动门控贡献", "",
             f"数据：train {len(TRAIN_SEQS)} 条 / val {len(VAL_SEQS)} 条；query_film_bilinear，"
             "from_scratch、seed 42、200 分片轮；运动 GT DIoU + 外观 rank(18)+column(50)。", "",
             "## 训练调度", "",
             f"- sequences={config.get('sequences')}，total_steps={config.get('total_steps')}，"
             f"warmup_steps={config.get('warmup_steps')}，shards={config.get('shards')}。",
             f"- 训练样本 {payload['training']['samples_total']}，耗时 "
             f"{payload['training']['seconds_total']:.0f}s。", "",
             "## val 成绩", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("normal", "motion1"):
        if name not in rows:
            continue
        c = rows[name]["combined"]
        def fmt(k):
            v = c.get(k)
            return "-" if v is None else (f"{v:.0f}" if k == "IDSW" else f"{v*100:.3f}")
        lines.append(f"| {name} | " + " | ".join(fmt(k) for k in keys) + " |")
    if "normal_minus_motion1" in payload["deltas"]:
        d = payload["deltas"]["normal_minus_motion1"]
        lines.append("| 运动门控贡献（normal−motion1） | " + " | ".join(
            f"{d[k]*100:+.3f} pp" if k != "IDSW" and d[k] is not None else
            (f"{d[k]:+.0f}" if d[k] is not None else "-") for k in keys) + " |")
    if gates:
        lines += ["", "## 门控统计", ""]
        for name, g in gates.items():
            lines.append(f"- {name}: 运动 gate min/mean/max = {g['motion_min']:.4f}/"
                         f"{g['motion_mean']:.4f}/{g['motion_max']:.4f}；"
                         f"外观 gate 均值 {g['appearance_mean']:.4f}。")
    if "normal" in rows:
        lines += ["", "| 序列 | 正常 HOTA | 运动固定1 HOTA | Δ (pp) |",
                  "|---|---:|---:|---:|"]
        for seq in VAL_SEQS:
            n = (rows["normal"]["per_sequence"].get(seq) or {}).get("HOTA")
            m = (rows["motion1"]["per_sequence"].get(seq) or {}).get("HOTA")
            if n is None or m is None:
                continue
            lines.append(f"| {seq} | {n*100:.3f} | {m*100:.3f} | {(n-m)*100:+.3f} |")
    (EXP / "report.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "collect", "train", "eval", "report"),
                        default="all")
    args = parser.parse_args()
    driver = Driver()
    if args.stage in ("all", "collect"):
        driver.collect()
    if args.stage in ("all", "train"):
        driver.train()
    if args.stage in ("all", "eval"):
        driver.evals()
    if args.stage in ("all", "report"):
        report(driver)


if __name__ == "__main__":
    main()
