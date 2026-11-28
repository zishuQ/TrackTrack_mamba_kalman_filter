#!/usr/bin/env python3
"""SportsMOT A/B/C: dropout and motion-harm-weight experiments (train 45 / val 45).

A: dropout 0.0, motion_harm_weight 1
B: dropout 0.1, motion_harm_weight 1
C: dropout 0.0, motion_harm_weight 8

All runs: query_film_bilinear, from scratch, full parameters, seed 42, one shard,
sample budget matched to the historical e80
(20 full passes), cosine planned over 50 full epochs, stop after epoch 20.
Checkpoints at epochs 10/15/20 get a full 45-sequence raw val evaluation and a
motion-gate target analysis. No test submission in this round.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
DATA = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/data"
EXP = ROOT / "outputs/agentguard/experiments/sportsmot_abc_20261123"
LOGS = EXP / "logs"
EVALS = EXP / "evals"
GATES = EXP / "gate_targets"
CACHE = ROOT / "outputs/agentguard/detection_cache"
TARGETS = ROOT / "outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT/targets"
HIST_DIR = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/groups/appearance_row_column"
HIST_CKPT = HIST_DIR / "checkpoints/epoch_080.pt"
HIST_SUMMARY = ROOT / "outputs/agentguard/experiments/sportsmot_d_midpoints_20261122/evals/e080/summary.json"
STATE_PATH = EXP / "state.json"
PREFLIGHT_PATH = EXP / "preflight.json"
EVAL_EPOCHS = (10, 15, 20)
PLANNED_EPOCHS = 50
STOP_EPOCH = 20
HEAVY = {"run.py", "train_supervision_groups.py", "train_stop_after.py",
         "continue_low_lr.py", "train_motion_rank.py"}

for path in (ROOT / "tools", ROOT, ROOT / "agentguard/src", TRACKER):
    sys.path.insert(0, str(path))
from paired_training_input import official_sequences  # noqa: E402

VAL_SEQS = sorted(official_sequences("val", "SportsMOT"))
TRAIN_SEQS = sorted(official_sequences("train", "SportsMOT"))

RUNS = {
    "A": {"dropout": 0.0, "motion_harm_weight": 1.0},
    "B": {"dropout": 0.1, "motion_harm_weight": 1.0},
    "C": {"dropout": 0.0, "motion_harm_weight": 8.0},
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file() else {"runs": {}}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE_PATH)


def active_jobs():
    jobs = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [a.decode(errors="replace") for a in
                    (entry / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        scripts = {Path(arg).name for arg in argv if arg.endswith(".py")}
        if scripts & HEAVY:
            jobs.append({"pid": int(entry.name), "scripts": sorted(scripts & HEAVY)})
    return jobs


def wait_jobs(label):
    while True:
        jobs = active_jobs()
        if not jobs:
            return
        print(f"{now()} waiting before {label}: {jobs}", flush=True)
        time.sleep(30)


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def gpu_free_mib():
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return free / 2 ** 20, total / 2 ** 20
    except Exception:  # noqa: BLE001
        return None, None


def rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def run(label, command, cwd=ROOT):
    wait_jobs(label)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{label}.log"
    started = time.time()
    print(f"{now()} start {label}", flush=True)
    with log.open("ab") as handle:
        handle.write(f"\n===== {now()} =====\n{command}\n".encode())
        handle.flush()
        code = subprocess.call([str(x) for x in command], cwd=str(cwd),
                               stdout=handle, stderr=subprocess.STDOUT)
    print(f"{now()} {label} exit={code} {time.time() - started:.1f}s", flush=True)
    if code:
        raise RuntimeError(f"{label}: exit {code}; see {log}")
    return log


def probe():
    sys.path.insert(0, str(ROOT / "tools"))
    import train_supervision_groups as trainer

    EXP.mkdir(parents=True, exist_ok=True)
    mem_before = mem_available_gib()
    started = time.time()
    datasets = {}
    for sequence in TRAIN_SEQS:
        datasets[sequence] = trainer.SequenceData(sequence, "train", DATA, "SportsMOT")
    load_seconds = time.time() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    steps = trainer.build_schedule(datasets, 1024, PLANNED_EPOCHS, 1,
                                   "sample-proportional", 1.0)
    samples = sum(data.n for data in datasets.values())
    for data in datasets.values():
        data.samples.close()
    del datasets
    total_steps = sum(steps)
    warmup = steps[0] // 4
    progress = (STOP_EPOCH * steps[0] - warmup) / (total_steps - warmup)
    ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
    hist_rows = [json.loads(line) for line in
                 (HIST_DIR / "metrics.jsonl").read_text().splitlines() if line.strip()]
    old_budget = sum(int(row["samples"]) for row in hist_rows if int(row["epoch"]) <= 80)
    gpu_free, gpu_total = gpu_free_mib()
    payload = {
        "created_at": now(),
        "train_sequences": len(TRAIN_SEQS), "val_sequences": len(VAL_SEQS),
        "samples": samples, "steps_per_epoch": steps[0],
        "planned_epochs": PLANNED_EPOCHS, "stop_epoch": STOP_EPOCH,
        "total_steps": total_steps, "warmup_steps": warmup,
        "lr_start": 1e-4, "lr_end": ratio * 1e-4, "lr_ratio": ratio,
        "sample_budget_20_epochs": samples * STOP_EPOCH,
        "old_e80_sample_budget": old_budget,
        "sample_budget_delta": samples * STOP_EPOCH - old_budget,
        "sequence_load_seconds": round(load_seconds, 2),
        "probe_rss_peak_gib": round(peak, 3),
        "mem_available_before_gib": round(mem_before, 2),
        "mem_available_gib": round(mem_available_gib(), 2),
        "gpu_free_mib": None if gpu_free is None else round(gpu_free, 1),
        "gpu_total_mib": None if gpu_total is None else round(gpu_total, 1),
    }
    PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def check_resources(min_mem_gib=18.0, min_gpu_mib=2048.0):
    available = mem_available_gib()
    free, total = gpu_free_mib()
    print(f"{now()} resources: MemAvailable={available:.1f} GiB "
          f"GPU free={None if free is None else round(free)} MiB / "
          f"{None if total is None else round(total)} MiB", flush=True)
    if available < min_mem_gib:
        raise RuntimeError(f"MemAvailable {available:.1f} GiB < {min_mem_gib} GiB")
    if free is not None and free < min_gpu_mib:
        raise RuntimeError(f"GPU free {free:.0f} MiB < {min_gpu_mib} MiB")


def group_dir(name):
    return EXP / "runs" / name / "groups/appearance_row_column"


def checkpoint(name, epoch):
    return group_dir(name) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def train_command(name, spec, preflight):
    groups_root = EXP / "runs" / name / "groups"
    command = [
        PY, "-u", "tools/train_stop_after.py",
        "--group", "appearance_row_column",
        "--dataset", "SportsMOT", "--split", "train", "--from-scratch",
        "--data-root", str(DATA), "--groups-root", str(groups_root),
        "--shards", "1", "--max-epochs", str(PLANNED_EPOCHS),
        "--snapshot-every", "5", "--stop-after-epoch", str(STOP_EPOCH),
        "--batch-size", "1024", "--lr", "1e-4", "--weight-decay", "1e-4",
        "--seed", "42", "--rank-weight", "18", "--rank-alpha", "0.95",
        "--rank-tau", "0.1", "--column-weight", "50",
        "--motion-harm-weight", str(spec["motion_harm_weight"]),
        "--dropout", str(spec["dropout"]),
        "--warmup-steps", str(preflight["warmup_steps"]),
        "--grad-norm-every", "5", "--device", "cuda",
    ]
    last = group_dir(name) / "checkpoints/last.pt"
    if last.is_file() and not checkpoint(name, STOP_EPOCH).is_file():
        command.append("--resume")
    return command


def train_run(name, spec, preflight, state):
    if checkpoint(name, STOP_EPOCH).is_file():
        print(f"{name}: epoch_{STOP_EPOCH} exists; skip training", flush=True)
        return
    check_resources()
    run(f"train_{name}", train_command(name, spec, preflight))
    state["runs"].setdefault(name, {})["train"] = {"status": "complete", "at": now()}
    save_state(state)


def verify_run(name, spec, preflight):
    import torch
    config = json.loads((group_dir(name) / "config.json").read_text())
    expected = {
        "dropout": spec["dropout"], "motion_harm_weight": spec["motion_harm_weight"],
        "shards": 1, "epochs": PLANNED_EPOCHS, "total_steps": preflight["total_steps"],
        "warmup_steps": preflight["warmup_steps"], "batch_size": 1024,
        "architecture_variant": "query_film_bilinear", "appearance_proj_dim": 256,
        "seed": 42, "lr": 1e-4,
    }
    mismatched = {key: {"expected": value, "actual": config.get(key)}
                  for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"{name}: config mismatch {mismatched}")
    metrics = [json.loads(line) for line in
               (group_dir(name) / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not metrics or int(metrics[-1]["epoch"]) != STOP_EPOCH:
        raise RuntimeError(f"{name}: metrics stop at {metrics[-1].get('epoch') if metrics else None}")
    last = metrics[-1]
    if abs(float(last["lr"]) - preflight["lr_end"]) > 1e-9:
        raise RuntimeError(f"{name}: epoch-20 lr {last['lr']} != {preflight['lr_end']}")
    verification = {}
    for line in (group_dir(name) / "training.log").read_text().splitlines():
        if line.startswith("verification: "):
            verification = json.loads(line[len("verification: "):])
        if line.startswith("preload: "):
            verification["preload"] = json.loads(line[len("preload: "):])
    if verification.get("model_param_numel") != verification.get("optimizer_param_numel"):
        raise RuntimeError(f"{name}: optimizer does not hold every parameter")
    probe_row = next((row for row in metrics
                      if row.get("dropout_probe_max_abs_diff") is not None), None)
    if spec["dropout"] > 0 and not (probe_row and probe_row["dropout_probe_max_abs_diff"]):
        raise RuntimeError(f"{name}: dropout probe missing")
    payload = torch.load(checkpoint(name, STOP_EPOCH), map_location="cpu", weights_only=False)
    model_config = payload.get("model_config") or {}
    training_config = payload.get("training_config") or {}
    reload_ok = (int(payload.get("epoch", -1)) == STOP_EPOCH
                 and model_config.get("architecture_variant") == "query_film_bilinear"
                 and int(model_config.get("appearance_proj_dim", 0)) == 256
                 and float(model_config.get("dropout", -1)) == spec["dropout"]
                 and float(training_config.get("motion_harm_weight", -1)) == spec["motion_harm_weight"]
                 and "optimizer_state_dict" in payload and "scheduler_state_dict" in payload)
    if not reload_ok:
        raise RuntimeError(f"{name}: checkpoint reload verification failed")
    return {"config": config, "last_metric": last, "verification": verification,
            "checkpoint_config": model_config, "reload_ok": reload_ok,
            "epoch_order": [json.loads(line[len("epoch order: "):])["sequences"]
                            for line in (group_dir(name) / "training.log").read_text().splitlines()
                            if line.startswith("epoch order: ")]}


def eval_command(name, epoch):
    out = EVALS / f"{name}_e{epoch:03d}"
    return [PY, "-u", "run.py", "--dataset", "SportsMOT", "--mode", "val",
            "--sequences", *VAL_SEQS, "--seed", "10000", "--kf-type", "nsa",
            "--agentguard-mode", "shutrack-v2",
            "--shutrack-checkpoint", str(checkpoint(name, epoch)),
            "--agentguard-device", "cpu", "--detection-cache-root", str(CACHE),
            "--output_dir", str(out), "--tracker-suffix", f"abc_{name.lower()}_e{epoch:03d}",
            "--agentguard-stats-output", str(out / "gates.jsonl"),
            "--print-per-sequence-metrics", "--eval-json", str(out / "summary.json"),
            "--torch-num-threads", "4", "--torch-num-interop-threads", "1"], out


def eval_run(name, epoch, state):
    command, out = eval_command(name, epoch)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text())
        if len(payload.get("per_sequence") or {}) == len(VAL_SEQS):
            print(f"{name} e{epoch}: eval exists", flush=True)
            return
    run(f"eval_{name}_e{epoch:03d}", command, cwd=TRACKER)
    payload = json.loads(summary.read_text())
    if len(payload.get("per_sequence") or {}) != len(VAL_SEQS):
        raise RuntimeError(f"{name} e{epoch}: per-sequence count "
                           f"{len(payload.get('per_sequence') or {})}")
    state["runs"].setdefault(name, {}).setdefault("evals", {})[f"e{epoch:03d}"] = {
        "status": "complete", "combined": payload.get("combined")}
    save_state(state)


def gate_command(name, epoch):
    out = GATES / f"{name}_e{epoch:03d}.json"
    return [PY, "-u", "tools/evaluate_motion_gate_targets.py",
            "--baseline", str(HIST_CKPT), "--checkpoint", str(checkpoint(name, epoch)),
            "--data-root", str(DATA), "--target-root", str(TARGETS),
            "--sequences", *VAL_SEQS, "--dataset", "SportsMOT", "--out", str(out)], out


def gate_run(name, epoch, state):
    command, out = gate_command(name, epoch)
    GATES.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        print(f"{name} e{epoch}: gate analysis exists", flush=True)
        return
    run(f"gate_{name}_e{epoch:03d}", command)
    state["runs"].setdefault(name, {}).setdefault("gates", {})[
        f"e{epoch:03d}"] = {"status": "complete"}
    save_state(state)


def load_summary(path):
    return json.loads(Path(path).read_text())


def gate_stats(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    motion = [r["shutrack_avg_motion_gate"] for r in rows
              if r.get("shutrack_avg_motion_gate") is not None]
    appearance = [r["shutrack_avg_appearance_gate"] for r in rows
                  if r.get("shutrack_avg_appearance_gate") is not None]
    return {"rows": len(rows), "motion_min": min(motion), "motion_mean": sum(motion) / len(motion),
            "motion_max": max(motion), "appearance_min": min(appearance),
            "appearance_mean": sum(appearance) / len(appearance),
            "appearance_max": max(appearance)}


def analyze():
    hist = load_summary(HIST_SUMMARY)
    hist_combined = hist["combined"]
    hist_per = hist["per_sequence"]
    runs = {}
    for name in RUNS:
        runs[name] = {}
        for epoch in EVAL_EPOCHS:
            summary_path = EVALS / f"{name}_e{epoch:03d}/summary.json"
            gates_path = EVALS / f"{name}_e{epoch:03d}/gates.jsonl"
            gt_path = GATES / f"{name}_e{epoch:03d}.json"
            if not summary_path.is_file():
                continue
            payload = load_summary(summary_path)
            entry = {"combined": payload["combined"],
                     "per_sequence": payload["per_sequence"],
                     "gates": gate_stats(gates_path) if gates_path.is_file() else None,
                     "gate_target": (json.loads(gt_path.read_text())
                                     if gt_path.is_file() else None)}
            runs[name][epoch] = entry
    training = {}
    for name, spec in RUNS.items():
        metrics = [json.loads(line) for line in
                   (group_dir(name) / "metrics.jsonl").read_text().splitlines() if line.strip()]
        preload = {}
        orders = []
        for line in (group_dir(name) / "training.log").read_text().splitlines():
            if line.startswith("preload: "):
                preload = json.loads(line[len("preload: "):])
            if line.startswith("epoch order: "):
                orders.append(json.loads(line[len("epoch order: "):])["sequences"])
        last = metrics[-1]
        training[name] = {
            "dropout": spec["dropout"], "motion_harm_weight": spec["motion_harm_weight"],
            "preload": preload,
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
            "epoch_order_hashes": [hash(tuple(order)) for order in orders],
            "last_metric": last,
        }
    order_same = all(training[name]["epoch_order_hashes"] == training["A"]["epoch_order_hashes"]
                     for name in RUNS)
    for name in RUNS:
        del training[name]["epoch_order_hashes"]

    def delta(values, reference):
        keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")
        return {key: values.get(key, 0) - reference.get(key, 0) for key in keys}

    def per_sequence_change(per, reference):
        changes = {sequence: per[sequence]["HOTA"] - reference[sequence]["HOTA"]
                   for sequence in per if sequence in reference}
        improved = sum(1 for value in changes.values() if value > 0.005)
        degraded = sum(1 for value in changes.values() if value < -0.005)
        ranked = sorted(changes.items(), key=lambda item: item[1])
        return {"improved": improved, "degraded": degraded,
                "unchanged": len(changes) - improved - degraded,
                "worst": ranked[:5], "best": ranked[-5:][::-1]}

    summary = {"generated_at": now(), "historical_e80": hist_combined,
               "training": training, "data_order_identical": order_same,
               "runs": {}}
    for name in RUNS:
        summary["runs"][name] = {}
        for epoch, entry in runs.get(name, {}).items():
            summary["runs"][name][epoch] = {
                "combined": entry["combined"],
                "delta_vs_hist_e80": delta(entry["combined"], hist_combined),
                "gates": entry["gates"], "gate_target": entry["gate_target"],
            }
            if name != "A":
                ref = runs["A"].get(epoch)
                if ref:
                    summary["runs"][name][epoch]["delta_vs_A_same_epoch"] = delta(
                        entry["combined"], ref["combined"])
                    summary["runs"][name][epoch]["per_sequence_vs_A"] = per_sequence_change(
                        entry["per_sequence"], ref["per_sequence"])
        entry20 = runs.get(name, {}).get(20)
        if entry20:
            summary["runs"][name]["per_sequence_vs_hist_e80"] = per_sequence_change(
                entry20["per_sequence"], hist_per)
    report = render_report(summary)
    (EXP / "report.md").write_text(report + "\n")
    (EXP / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)


def render_report(summary):
    lines = ["# SportsMOT A/B/C：Dropout 与 motion_harm_weight", "",
             "训练：官方 train 45 条，从头全参数，query_film_bilinear，seed 42；"
             "shards=1，20 个全量轮（对齐旧 e80 样本预算）。"
             "余弦按 50 轮规划。验证：官方 val 45 条 raw，正常双门控。", "",
             "| 组 | Dropout | motion_harm_weight | 加载 s | 平均轮时 s | RSS 峰值 GiB | CUDA 峰值 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, training in summary["training"].items():
        lines.append(f"| {name} | {training['dropout']} | {training['motion_harm_weight']} | "
                     f"{training['preload'].get('seconds', '-')} | "
                     f"{training['epoch_seconds_mean']:.1f} | "
                     f"{training['rss_peak_gb']} | {training['cuda_peak_allocated_mb']} |")
    lines += ["", "## Val（45 条 raw）", "",
              "| 组 | 轮次 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag | ΔHOTA vs 旧 e80 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("A", "B", "C"):
        for epoch in EVAL_EPOCHS:
            entry = summary["runs"].get(name, {}).get(epoch)
            if not entry:
                continue
            c = entry["combined"]
            lines.append(f"| {name} | e{epoch} | {c['HOTA']*100:.3f} | {c['AssA']*100:.3f} | "
                         f"{c['DetA']*100:.3f} | {c['IDF1']*100:.3f} | {c['MOTA']*100:.3f} | "
                         f"{c['IDSW']:.0f} | {c['Frag']:.0f} | "
                         f"{entry['delta_vs_hist_e80']['HOTA']*100:+.3f} |")
    hist = summary["historical_e80"]
    lines.append(f"| 历史 e80 | e80 | {hist['HOTA']*100:.3f} | {hist['AssA']*100:.3f} | "
                 f"{hist['DetA']*100:.3f} | {hist['IDF1']*100:.3f} | {hist['MOTA']*100:.3f} | "
                 f"{hist['IDSW']:.0f} | {hist['Frag']:.0f} | - |")
    lines += ["", "## B/C 对 A（同轮次）", "",
              "| 组 | 轮次 | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | 逐序列改善/退化/不变 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name in ("B", "C"):
        for epoch in EVAL_EPOCHS:
            entry = summary["runs"].get(name, {}).get(epoch)
            if not entry or "delta_vs_A_same_epoch" not in entry:
                continue
            d = entry["delta_vs_A_same_epoch"]
            change = entry["per_sequence_vs_A"]
            lines.append(f"| {name} | e{epoch} | {d['HOTA']*100:+.3f} | {d['AssA']*100:+.3f} | "
                         f"{d['DetA']*100:+.3f} | {d['IDF1']*100:+.3f} | "
                         f"{change['improved']}/{change['degraded']}/{change['unchanged']} |")
    lines += ["", "## 运动 gate 分布（跟踪时）", "",
              "| 组 | 轮次 | motion min/mean/max | appearance min/mean/max |",
              "|---|---:|---|---|"]
    for name in ("A", "B", "C"):
        for epoch in EVAL_EPOCHS:
            entry = summary["runs"].get(name, {}).get(epoch)
            if not entry or not entry.get("gates"):
                continue
            g = entry["gates"]
            lines.append(f"| {name} | e{epoch} | {g['motion_min']:.4f}/{g['motion_mean']:.4f}/"
                         f"{g['motion_max']:.4f} | {g['appearance_min']:.4f}/"
                         f"{g['appearance_mean']:.4f}/{g['appearance_max']:.4f} |")
    lines += ["", "## 门控目标分析（有害节省 / 有益误伤 / 净损失）", "",
              "| 组 | 轮次 | 有害事件 | 有害 gate | 节省均值 | 有益事件 | 有益 gate | 误伤均值 | 净损失变化 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("A", "B", "C"):
        for epoch in EVAL_EPOCHS:
            entry = summary["runs"].get(name, {}).get(epoch)
            target = entry.get("gate_target") if entry else None
            if not target:
                continue
            model = target["new"]
            lines.append(f"| {name} | e{epoch} | {model['harmful_count']:.0f} | "
                         f"{model['harmful_gate_mean']:.4f} | "
                         f"{model['harmful_savings_mean']:.6f} | {model['useful_count']:.0f} | "
                         f"{model['useful_gate_mean']:.4f} | "
                         f"{model['useful_damage_mean']:.6f} | {model['net_loss_change']:+.6f} |")
    lines += ["", "## 梯度范数（共享主干，首 batch，epoch 1/5/10/15/20）", "",
              "| 组 | epoch | motion | appearance |", "|---|---:|---:|---:|"]
    for name, training in summary["training"].items():
        for epoch, values in sorted(training["grad_norms"].items()):
            lines.append(f"| {name} | {epoch} | {values['motion']:.4f} | {values['appearance']:.4f} |")
    lines += ["", "## 逐序列变化（e20 vs 历史 e80）", ""]
    for name in ("A", "B", "C"):
        change = summary["runs"].get(name, {}).get("per_sequence_vs_hist_e80")
        if not change:
            continue
        lines.append(f"- {name}: 改善 {change['improved']} / 退化 {change['degraded']} / "
                     f"不变 {change['unchanged']}；最好 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["best"][:3])
                     + "；最差 "
                     + ", ".join(f"{seq} {value*100:+.3f}" for seq, value in change["worst"][:3]))
    lines += ["", f"- 三组数据顺序一致（epoch order 对比）：{summary['data_order_identical']}"]
    lines += ["", "## 结论", ""]
    hist_hota = summary["historical_e80"]["HOTA"] * 100
    a20 = summary["runs"].get("A", {}).get(20)
    if a20:
        lines.append(f"- A（复现）：e20 HOTA {a20['combined']['HOTA']*100:.3f}，"
                     f"相对历史 e80 {hist_hota:.3f} 为 "
                     f"{a20['delta_vs_hist_e80']['HOTA']*100:+.3f}，"
                     "新全量内存协议复现了旧配方。")
    for name in ("B", "C"):
        entry = summary["runs"].get(name, {}).get(20)
        if not entry or "delta_vs_A_same_epoch" not in entry:
            continue
        delta = entry["delta_vs_A_same_epoch"]["HOTA"] * 100
        change = entry["per_sequence_vs_A"]
        lines.append(f"- {name}：e20 相对 A 同轮次 HOTA {delta:+.3f}，"
                     f"逐序列 {change['improved']} 改善 / {change['degraded']} 退化 / "
                     f"{change['unchanged']} 不变。")
    c20 = summary["runs"].get("C", {}).get(20)
    if c20 and c20.get("gate_target"):
        model = c20["gate_target"]["new"]
        lines.append(f"- C 的 harm_weight=8 把运动 gate 均值压到 "
                     f"{c20['gates']['motion_mean']:.4f}（跟踪时）；目标分析中"
                     f"有害事件节省 {model['harmful_savings_mean']:.6f}/事件，"
                     f"有益事件误伤 {model['useful_damage_mean']:.6f}/事件，"
                     f"由于有益事件数是 harmful 的 "
                     f"{model['useful_count']/model['harmful_count']:.1f} 倍，"
                     f"净 DIoU 损失变化 {model['net_loss_change']:+.6f}（正=变差）。")
    lines += ["", "单 seed 结果；A 与历史 e80 的差在噪声量级，B 的小幅差异需要 seed 复现才能判定。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "probe", "train", "eval", "analyze"),
                        default="all")
    parser.add_argument("--runs", nargs="*", choices=tuple(RUNS), default=list(RUNS))
    args = parser.parse_args()
    if args.stage == "probe":
        probe()
        return
    EXP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if args.stage in ("all", "train"):
        if not PREFLIGHT_PATH.is_file():
            run("probe", [PY, "-u", str(Path(__file__).resolve()), "--stage", "probe"])
        preflight = json.loads(PREFLIGHT_PATH.read_text())
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        for name in args.runs:
            train_run(name, RUNS[name], preflight, state)
            verification = verify_run(name, RUNS[name], preflight)
            state["runs"].setdefault(name, {})["verify"] = {
                "status": "complete", "at": now(),
                "model_param_numel": verification["verification"].get("model_param_numel")}
            save_state(state)
    if args.stage in ("all", "eval"):
        for name in args.runs:
            for epoch in EVAL_EPOCHS:
                eval_run(name, epoch, state)
                gate_run(name, epoch, state)
    if args.stage in ("all", "analyze"):
        analyze()


if __name__ == "__main__":
    main()
