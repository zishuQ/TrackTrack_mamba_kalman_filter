#!/usr/bin/env python3
"""From-scratch 200-shard-epoch cost / cost+rank experiment on SportsMOT.

Protocol: official train 45 -> val 45, query_film_bilinear (1,208,220 params),
from_scratch=true / init=null, seed 42, lr 1e-4, batch 1024, shards 4,
200 epochs, appearance rank+column losses inherited from the original D
baseline. Two groups differ only by the motion pairwise rank term (weight 1).

Stages (serial, resumable):
  labels -> preflight (binding / init equality / full-loss / benchmark)
  -> smoke 1 epoch in a separate dir per group -> train both to 200
  -> fixed-val diagnostics at 20/80/200 -> raw 45-val eval at 80/200 -> report.
No test submission; the original experiment directories are untouched.
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
BASE_CONFIG = BASE / "ablation/SportsMOT/groups/appearance_row_column/config.json"
BASE_SUMMARY = BASE / "ablation/SportsMOT/evals/new/summary.json"
REF = ROOT / "outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_scratch_cost_rank_20261122/SportsMOT"
CONFIG = EXP / "config.json"
TARGETS = REF / "targets"
RANKS = ROOT / "outputs/agentguard/experiments/shutrack_motion_rank_20261122/SportsMOT/rank_labels"
FINETUNE = ROOT / "outputs/agentguard/experiments/shutrack_motion_rank_allparams_20261122/SportsMOT/comparison.json"
TRAIN_TOOL = ROOT / "tools/train_motion_rank.py"
VERIFY_TOOL = ROOT / "tools/verify_scratch_init.py"
DIAG_TOOL = ROOT / "tools/evaluate_motion_rank_val.py"
GEN_TOOL = ROOT / "tools/generate_motion_rank_labels.py"
DATA90 = ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data"
GROUPS = {"cost": 0.0, "cost_rank": 1.0}
GROUP = "appearance_row_column"
EPOCHS = 200
DIAG_STAGES = (20, 80, 200)
EVAL_STAGES = (80, 200)
VAL_SEQS = tuple(
    line.strip() for line in (TRACKER / "trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)
HEAVY = {"train_motion_rank.py", "train_supervision_groups.py", "run.py",
         "verify_scratch_init.py", "evaluate_motion_rank_val.py",
         "generate_motion_rank_labels.py", "generate_trainval_paired.py"}
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
        out = EXP / "preflight_scratch.json"
        if out.is_file():
            payload = json.loads(out.read_text())
            if (payload.get("init", {}).get("equal")
                    and payload.get("init", {}).get("optimizer_covers_all")
                    and all(payload.get("binding", {}).values())):
                self.state["preflight"] = {
                    "init_hash": payload["init"]["hash"],
                    "init_equal": payload["init"]["equal"],
                    "benchmark": {k: payload["benchmark"][k]["step_seconds_mean"]
                                  for k in ("cost", "cost_rank")},
                    "reused": True}
                self.save()
                return payload
        command = [str(PY), "-u", str(VERIFY_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                   "--norm-checkpoint",
                   str(BASE / "ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"),
                   "--rows", "1024", "--steps", "10", "--warmup", "3",
                   "--seed", "42", "--device", "cuda", "--out", str(out)]
        self.run("preflight_scratch", command)
        payload = json.loads(out.read_text())
        self.state["preflight"] = {
            "init_hash": payload["init"]["hash"], "init_equal": payload["init"]["equal"],
            "benchmark": {k: payload["benchmark"][k]["step_seconds_mean"]
                          for k in ("cost", "cost_rank")}}
        self.save()
        return payload

    def checkpoint(self, group, stage=None):
        directory = EXP / "runs" / group / "groups" / GROUP / "checkpoints"
        return directory / (f"epoch_{stage:03d}.pt" if stage else "last.pt")

    def train(self, group, smoke=False):
        weight = GROUPS[group]
        label = f"smoke_{group}" if smoke else f"train_{group}"
        root = EXP / ("smoke" if smoke else "runs") / group / "groups"
        checkpoint = root / GROUP / "checkpoints/last.pt"
        target_epoch = 1 if smoke else EPOCHS
        if checkpoint.is_file():
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", 0)) >= target_epoch:
                self.state["results"].setdefault(group, {})[label] = {
                    "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
                    "reused": True}
                self.save()
                return checkpoint
        command = [str(PY), "-u", str(TRAIN_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                   "--groups-root", str(root), "--base-lambda", "0.25",
                   "--rank-weight", f"{weight:g}", "--pair-seed", "20261122",
                   "--grad-every", "10"]
        if smoke:
            command += ["--stop-after-epoch", "1"]
        if checkpoint.is_file():
            command.append("--resume")
        self.run(label, command)
        if not checkpoint.is_file():
            raise RuntimeError(f"{label}: missing checkpoint")
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.state["results"].setdefault(group, {})[label] = {
            "checkpoint": str(checkpoint), "epoch": int(payload["epoch"]),
            "global_step": int(payload.get("global_step", 0))}
        self.save()
        return checkpoint

    def diagnose(self, group, stage):
        checkpoint = self.checkpoint(group, stage)
        if not checkpoint.is_file():
            raise RuntimeError(f"diagnostics_{group}_e{stage}: missing {checkpoint}")
        out = EXP / "diagnostics" / f"{group}_e{stage}.json"
        if out.is_file():
            payload = json.loads(out.read_text())
            if int(payload.get("sequences", 0)) == len(VAL_SEQS):
                self.state["results"].setdefault(group, {})[f"diagnostics_e{stage}"] = {
                    "summary": str(out), "reused": True}
                self.save()
                return payload
        command = [str(PY), "-u", str(DIAG_TOOL), "--config", str(CONFIG),
                   "--checkpoint", str(checkpoint), "--target-root", str(TARGETS),
                   "--rank-root", str(RANKS), "--sequence-root", str(DATA90),
                   "--split", "val", "--batch-size", "2048", "--device", "cuda",
                   "--out", str(out)]
        self.run(f"diagnostics_{group}_e{stage}", command)
        payload = json.loads(out.read_text())
        if int(payload.get("sequences", 0)) != len(VAL_SEQS):
            raise RuntimeError(f"diagnostics_{group}_e{stage}: wrong sequence count")
        self.state["results"].setdefault(group, {})[f"diagnostics_e{stage}"] = {
            "summary": str(out)}
        self.save()
        return payload

    def evaluate(self, group, stage):
        checkpoint = self.checkpoint(group, stage)
        if not checkpoint.is_file():
            raise RuntimeError(f"eval_{group}_e{stage}: missing {checkpoint}")
        out = EXP / "evals" / f"{group}_e{stage}"
        summary = out / "summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text())
            if len(payload.get("per_sequence") or {}) == len(VAL_SEQS):
                self.state["results"].setdefault(group, {})[f"eval_e{stage}"] = {
                    "summary": str(summary), "combined": payload.get("combined"),
                    "reused": True}
                self.save()
                return payload
        label = f"eval_{group}_e{stage}"
        out.mkdir(parents=True, exist_ok=True)
        command = [str(PY), "-u", "run.py", "--dataset", "SportsMOT", "--mode", "val",
                   "--sequences", *VAL_SEQS, "--seed", "10000", "--kf-type", "nsa",
                   "--agentguard-mode", "shutrack-v2", "--shutrack-checkpoint", str(checkpoint),
                   "--agentguard-device", "cpu", "--detection-cache-root",
                   str(ROOT / "outputs/agentguard/detection_cache"),
                   "--output_dir", str(out), "--tracker-suffix", f"scratch_{group}_e{stage}",
                   "--agentguard-stats-output", str(out / "gates.jsonl"),
                   "--print-per-sequence-metrics", "--eval-json", str(summary),
                   "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
        self.run(label, command, cwd=TRACKER)
        payload = json.loads(summary.read_text())
        if len(payload.get("per_sequence") or {}) != len(VAL_SEQS):
            raise RuntimeError(f"{label}: wrong per-sequence count")
        self.state["results"].setdefault(group, {})[f"eval_e{stage}"] = {
            "summary": str(summary), "combined": payload.get("combined")}
        self.save()
        return payload


def load_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def protocol_audit():
    baseline = json.loads(BASE_CONFIG.read_text()) if BASE_CONFIG.is_file() else {}
    current = json.loads(CONFIG.read_text()) if CONFIG.is_file() else {}
    keys = ("dataset", "data_root", "split", "sequences", "batch_size", "lr",
            "weight_decay", "seed", "shards", "epochs", "total_steps",
            "warmup_steps", "sequence_sampling", "sequence_sampling_power",
            "rank_weight", "rank_alpha", "rank_tau", "column_weight",
            "column_hinge", "appearance_proj_dim", "architecture_variant",
            "snapshot_every", "from_scratch", "init")
    return {key: {"baseline": baseline.get(key), "current": current.get(key),
                  "same": baseline.get(key) == current.get(key)} for key in keys}


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
    if BASE_SUMMARY.is_file():
        payload = json.loads(BASE_SUMMARY.read_text())
        rows["baseline_D"] = {"combined": payload.get("combined"),
                              "per_sequence": payload.get("per_sequence")}
    for group in GROUPS:
        for stage in EVAL_STAGES:
            entry = (driver.state["results"].get(group) or {}).get(f"eval_e{stage}") or {}
            path = Path(entry.get("summary", ""))
            if path.is_file():
                payload = json.loads(path.read_text())
                rows[f"{group}_e{stage}"] = {"combined": payload.get("combined"),
                                             "per_sequence": payload.get("per_sequence")}
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")
    summary = {
        name: {"combined": {k: (row["combined"] or {}).get(k) for k in keys},
               "per_sequence": {
                   seq: {k: ((row.get("per_sequence") or {}).get(seq) or {}).get(k) for k in keys}
                   for seq in VAL_SEQS}}
        for name, row in rows.items()
    }
    diagnostics = {}
    for group in GROUPS:
        for stage in DIAG_STAGES:
            path = EXP / "diagnostics" / f"{group}_e{stage}.json"
            if path.is_file():
                payload = json.loads(path.read_text())
                diagnostics[f"{group}_e{stage}"] = {
                    key: payload.get(key) for key in
                    ("epoch", "counts", "gate", "loss", "rank", "appearance", "per_sequence")}
    initial = {}
    for group in GROUPS:
        path = EXP / "runs" / group / "groups" / GROUP / "initial_state.json"
        if path.is_file():
            initial[group] = json.loads(path.read_text())
    training = {}
    for group in GROUPS:
        group_dir = EXP / "runs" / group / "groups" / GROUP
        training[group] = {
            "metrics": load_jsonl(group_dir / "metrics.jsonl"),
            "motion": load_jsonl(group_dir / "motion_rank_metrics.jsonl"),
        }
    reference = {}
    if FINETUNE.is_file():
        payload = json.loads(FINETUNE.read_text())
        for name in ("cost_l025_e20", "A", "B"):
            if name in payload.get("comparison", {}):
                reference[name] = payload["comparison"][name]["combined"]
    output = {"comparison": summary, "diagnostics": diagnostics,
              "initial_state": initial, "training": training,
              "reference_finetune": reference, "protocol": protocol_audit(),
              "val_sequences": list(VAL_SEQS)}
    (EXP / "comparison.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    lines = ["# SportsMOT 从头训练：代价监督 vs 代价+运动排序（200 分片轮）", "",
             "协议：官方 train 45 训练 / val 45 验证；query_film_bilinear（1,208,220 参数）；"
             "from_scratch=true、init=null、seed 42；lr 1e-4、batch 1024、shards 4、200 epochs；"
             "外观 rank(18, α0.95, τ0.1) + column(50) 监督继承原 D baseline。",
             "原 D baseline 本身即为从头训练（已核对 config：from_scratch=true, init=null, seed 42, "
             "45 条 train, 200 轮），因此无需另补同条件 D 对照。", "",
             "## 主结果（45 条 val raw）", "",
             "| 组 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"baseline_D": "D baseline (原)", "cost_e80": "组1 代价 e80",
              "cost_e200": "组1 代价 e200", "cost_rank_e80": "组2 代价+排序 e80",
              "cost_rank_e200": "组2 代价+排序 e200"}
    for name in ("baseline_D", "cost_e80", "cost_e200", "cost_rank_e80", "cost_rank_e200"):
        if name not in summary:
            continue
        combined = summary[name]["combined"]
        cells = []
        for key in keys:
            value = combined.get(key)
            cells.append("-" if value is None else
                         (f"{value:.0f}" if key == "IDSW" else f"{value*100:.3f}"))
        lines.append(f"| {labels[name]} | " + " | ".join(cells) + " |")
    lines += ["", "原 D baseline 参考 HOTA 83.343。", "",
              "## 微调实验参考（独立协议，不与从头训练直接合并比较）", "",
              "| 实验 | 训练协议 | HOTA | AssA | DetA | IDF1 | IDSW |",
              "|---|---|---:|---:|---:|---:|---:|"]
    ref_labels = {"cost_l025_e20": "λ0.25 仅门控头 20 轮",
                  "A": "全参数 L_base 20 轮（微调）",
                  "B": "全参数 +L_rank 20 轮（微调）"}
    for name, combined in reference.items():
        cells = []
        for key in ("HOTA", "AssA", "DetA", "IDF1", "IDSW"):
            value = combined.get(key)
            cells.append("-" if value is None else
                         (f"{value:.0f}" if key == "IDSW" else f"{value*100:.3f}"))
        lines.append(f"| {ref_labels.get(name, name)} | 从 baseline epoch_200 初始化 | "
                     + " | ".join(cells) + " |")
    if "baseline_D" in summary:
        for group in GROUPS:
            for stage in EVAL_STAGES:
                name = f"{group}_e{stage}"
                if name not in summary:
                    continue
                lines += ["", f"## 逐序列 {labels[name]} vs D baseline", "",
                          "| 序列 | HOTA | AssA | DetA | IDF1 | IDSW | ΔHOTA (pp) |",
                          "|---|---:|---:|---:|---:|---:|---:|"]
                improved = 0
                for seq in VAL_SEQS:
                    row = summary[name]["per_sequence"].get(seq) or {}
                    base = summary["baseline_D"]["per_sequence"].get(seq) or {}
                    if row.get("HOTA") is None or base.get("HOTA") is None:
                        continue
                    delta = (row["HOTA"] - base["HOTA"]) * 100
                    if delta > 1e-9:
                        improved += 1
                    lines.append(f"| {seq} | {_num(row.get('HOTA'), 3, 100)} | "
                                 f"{_num(row.get('AssA'), 3, 100)} | "
                                 f"{_num(row.get('DetA'), 3, 100)} | "
                                 f"{_num(row.get('IDF1'), 3, 100)} | "
                                 f"{_num(row.get('IDSW'), 0, 1)} | {delta:+.3f} |")
                lines += ["", f"改善序列数 {improved}/{len(VAL_SEQS)}。"]
    lines += ["", "## 固定 val 诊断（20/80/200 轮）", "",
              "| 组 | 轮 | 有效 | 有害 | keep | suppress | gate 均值 | 有害 gate | 有益 gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        for stage in DIAG_STAGES:
            d = diagnostics.get(f"{group}_e{stage}")
            if not d:
                continue
            c, g = d["counts"], d["gate"]
            lines.append(f"| {group} | {stage} | {int(c['valid'])} | {int(c['harmful'])} | "
                         f"{int(c['keep'])} | {int(c['suppress'])} | "
                         f"{_num(g['gate_mean'], 4)} | {_num(g['harmful_mean'], 4)} | "
                         f"{_num(g['useful_mean'], 4)} |")
    lines += ["", "| 组 | 轮 | pred−full (e-4) | 有害节省 (e-4) | 有益新增 (e-4) | 净变化 (e-4) | 排序准确率 | logit 差 | gate 差 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        for stage in DIAG_STAGES:
            d = diagnostics.get(f"{group}_e{stage}")
            if not d:
                continue
            loss, rank = d["loss"], d["rank"]
            lines.append(f"| {group} | {stage} | "
                         f"{_num((loss['pred_mean'] or 0) - (loss['full_mean'] or 0), 3, 1e4)} | "
                         f"{_num(loss['saved_harmful_mean'], 3, 1e4)} | "
                         f"{_num(loss['added_useful_mean'], 3, 1e4)} | "
                         f"{_num(loss['net_delta_mean'], 3, 1e4)} | "
                         f"{_pct(rank['accuracy'])} | {_num(rank['logit_gap_mean'], 3)} | "
                         f"{_num(rank['gate_gap_mean'], 4)} |")
    lines += ["", "| 组 | 轮 | 外观有效 | 外观 gate | 外观 MAE | 外观 BCE |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        for stage in DIAG_STAGES:
            d = diagnostics.get(f"{group}_e{stage}")
            if not d:
                continue
            a = d.get("appearance") or {}
            lines.append(f"| {group} | {stage} | {int(a.get('valid') or 0)} | "
                         f"{_num(a.get('gate_mean'), 4)} | {_num(a.get('mae'), 4)} | "
                         f"{_num(a.get('bce_mean'), 4)} |")
    lines += ["", "## 初始化一致性", "",
              "| 组 | init sha256 | seed | from_scratch | 参数张量 | 总参数 |",
              "|---|---|---:|---|---:|---:|"]
    for group in GROUPS:
        info = initial.get(group) or {}
        lines.append(f"| {group} | {str(info.get('sha256'))[:16]} | {info.get('seed')} | "
                     f"{info.get('from_scratch')} | {info.get('parameter_tensors')} | "
                     f"{info.get('total')} |")
    same_init = (len(initial) == 2
                 and len({info.get("sha256") for info in initial.values()}) == 1)
    lines += ["", f"两组件初始参数一致：{'是' if same_init else '否'}。", "",
              "## 训练走势（每 20 轮）", ""]
    for group in GROUPS:
        metric_rows = training.get(group, {}).get("metrics", [])
        motion_rows = training.get(group, {}).get("motion", [])
        if not metric_rows:
            continue
        lines += [f"### {group}", "",
                  "| 轮 | epoch 秒 | loss | motion | appearance | rank mean | column | motion gate | appearance gate |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        by_epoch = {r["epoch"]: r for r in metric_rows}
        motion_by_epoch = {r["epoch"]: r for r in motion_rows}
        for epoch in range(20, EPOCHS + 1, 20):
            row = by_epoch.get(epoch)
            if not row:
                continue
            lines.append(f"| {epoch} | {_num(row.get('epoch_seconds'), 1, 1)} | "
                         f"{_num(row.get('loss'), 4)} | {_num(row.get('motion_loss'), 4)} | "
                         f"{_num(row.get('appearance_loss'), 4)} | "
                         f"{_num(row.get('rank_mean'), 4)} | {_num(row.get('column_loss'), 4)} | "
                         f"{_num(row.get('motion_gate_mean'), 4)} | "
                         f"{_num(row.get('appearance_gate_mean'), 4)} |")
        lines += ["", "| 轮 | 排序准确率 | gate 差 | 净变化 (e-4) | grad hce | grad appearance | grad gate0 | update hce | update appearance |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for epoch in range(20, EPOCHS + 1, 20):
            row = motion_by_epoch.get(epoch)
            if not row:
                continue
            grads = row.get("grad_norm") or {}
            updates = row.get("update_norm") or {}
            lines.append(f"| {epoch} | {_pct(row.get('rank_accuracy'))} | "
                         f"{_num(row.get('rank_gate_gap_mean'), 4)} | "
                         f"{_num(row.get('net_delta_mean'), 3, 1e4)} | "
                         f"{_num(grads.get('hce'), 5)} | "
                         f"{_num(grads.get('appearance_encoder'), 5)} | "
                         f"{_num(grads.get('rsc.gate_heads.0'), 5)} | "
                         f"{_num(updates.get('hce'), 5)} | "
                         f"{_num(updates.get('appearance_encoder'), 5)} |")
    lines += ["", "## 训练成本", "",
              "| 组 | 阶段 | 秒 | 备注 |", "|---|---|---:|---|"]
    for key, entry in sorted(driver.state["stages"].items()):
        lines.append(f"| {key} | {entry.get('status')} | {entry.get('seconds')} | "
                     f"{entry.get('log', '')} |")
    if training:
        peak = {}
        for group in GROUPS:
            rows = training.get(group, {}).get("motion", [])
            if rows:
                peak[group] = (max(r.get("cuda_peak_mib") or 0 for r in rows),
                               max(r.get("process_peak_rss_gib") or 0 for r in rows))
        if peak:
            lines += ["", "| 组 | 训练 CUDA 峰值 MiB | 进程峰值 RSS GiB |",
                      "|---|---:|---:|"]
            for group, (cuda, rss) in peak.items():
                lines.append(f"| {group} | {_num(cuda, 1)} | {_num(rss, 2)} |")
    (EXP / "comparison.md").write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "labels", "preflight", "smoke",
                                            "train", "diagnostics", "eval", "report"),
                        default="all")
    args = parser.parse_args()
    driver = Driver()
    if not CONFIG.is_file():
        raise SystemExit(f"missing config {CONFIG}")
    if args.stage in ("all", "labels"):
        driver.labels()
    if args.stage in ("all", "preflight"):
        payload = driver.preflight()
        print(json.dumps({"init": payload.get("init", {}),
                          "loss": payload.get("loss_on_real_batch", {})}, indent=2), flush=True)
    if args.stage in ("all", "smoke"):
        for group in GROUPS:
            driver.train(group, smoke=True)
    if args.stage in ("all", "train"):
        for group in GROUPS:
            driver.train(group)
    if args.stage in ("all", "diagnostics"):
        for group in GROUPS:
            for stage in DIAG_STAGES:
                driver.diagnose(group, stage)
    if args.stage in ("all", "eval"):
        for group in GROUPS:
            for stage in EVAL_STAGES:
                driver.evaluate(group, stage)
    if args.stage in ("all", "report"):
        output = compare(driver)
        print(json.dumps({"finished_at": now(),
                          "rows": list(output["comparison"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
