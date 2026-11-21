#!/usr/bin/env python3
"""Resumable driver for the SHUTrack architecture night run.

Tasks are executed in dependency order. Each task records its command,
timestamps, exit code and artifacts in TASK_STATE.json. A task marked complete
is only skipped after its completion predicate re-validates the artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
MOT20_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT20/data"
MOT17_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT17/data"
SM_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data"
OLD_MOT20_CKPT = ROOT / "outputs/agentguard/experiments/shutrack_d_proj256_mot20_20261120/groups/appearance_row_column/checkpoints/epoch_100.pt"
OLD_SM_CKPT = ROOT / "outputs/agentguard/experiments/shutrack_d_proj256_sportsmot_20261120/groups/appearance_row_column/checkpoints/epoch_200.pt"
OLD_MOT20_EVAL = ROOT / "outputs/agentguard/experiments/shutrack_d_proj256_mot20_20261120/evals/val_d256"
OLD_SM_EVAL = ROOT / "outputs/agentguard/experiments/shutrack_d_proj256_sportsmot_20261120/evals/val_d256"
MOT20_TRAIN = ("MOT20-02", "MOT20-03")
MOT17_ABL_TRAIN = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                   "MOT17-11-FRCNN", "MOT17-13-FRCNN")
MOT20_VAL = ("MOT20-01", "MOT20-05")
MOT17_VAL = ("MOT17-09-FRCNN", "MOT17-10-FRCNN")
SM_VAL = tuple(
    line.strip()
    for line in (ROOT / "3. Tracker/trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)
SM_TRAIN_VIEW = tuple(
    sorted(p.name for p in SM_DATA.iterdir() if (p / "samples.npz").is_file())
)
STATE_PATH = EXP / "TASK_STATE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _proc_rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


class Driver:
    def __init__(self):
        EXP.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file() else {
            "experiment_root": str(EXP), "created_at": now(), "tasks": {}
        }
        self.tasks = {}
        self.logs = EXP / "logs"
        self.logs.mkdir(exist_ok=True)

    def save(self):
        STATE_PATH.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")

    def task(self, name, deps=(), fn=None):
        self.tasks[name] = {"deps": tuple(deps), "fn": fn}

    def task_state(self, name):
        return self.state["tasks"].setdefault(name, {"status": "pending"})

    def run_command(self, label, command, cwd=ROOT, env=None):
        log_path = self.logs / f"{label}.log"
        started = now()
        wall = time.time()
        entry = self.task_state(label)
        if entry.get("pid"):
            previous = int(entry["pid"])
            if os.path.isdir(f"/proc/{previous}"):
                try:
                    cmdline = Path(f"/proc/{previous}/cmdline").read_bytes().decode(
                        errors="ignore"
                    )
                except OSError:
                    cmdline = ""
                parts = command if isinstance(command, list) else [command]
                needles = [str(part) for part in parts if str(part).endswith(".py")]
                if needles and any(needle in cmdline for needle in needles):
                    raise RuntimeError(
                        f"{label}: previous instance pid {previous} is still running; "
                        "refusing to start a second copy"
                    )
        entry.update({"command": command, "started_at": started, "status": "running"})
        self.save()
        with log_path.open("ab") as handle:
            handle.write(f"\n===== {started} =====\n{command}\n".encode())
            handle.flush()
            proc = subprocess.Popen(
                command, cwd=str(cwd), shell=isinstance(command, str),
                stdout=handle, stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "MALLOC_ARENA_MAX": "2",
                    "MALLOC_TRIM_THRESHOLD_": "131072",
                    "MALLOC_MMAP_THRESHOLD_": "131072",
                    **(env or {}),
                },
                start_new_session=True,
            )
            entry["pid"] = proc.pid
            self.save()
            stop = threading.Event()
            watchdog = {"killed": False, "reason": None}
            max_rss = float(os.environ.get("ARCH_MAX_RSS_GB", "16"))
            min_avail = float(os.environ.get("ARCH_MIN_AVAIL_GB", "2.0"))

            def watch():
                low_streak = 0
                while not stop.wait(5):
                    rss = _proc_rss_gb(proc.pid)
                    avail = _mem_available_gb()
                    if rss > max_rss or avail < min_avail:
                        low_streak += 1
                        if low_streak >= 3:
                            watchdog["killed"] = True
                            watchdog["reason"] = f"rss={rss:.1f}GB avail={avail:.1f}GB"
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            break
                    else:
                        low_streak = 0

            thread = threading.Thread(target=watch, daemon=True)
            thread.start()
            code = proc.wait()
            stop.set()
            thread.join(timeout=1)
            if watchdog["killed"]:
                entry["memory_watchdog"] = watchdog["reason"]
        entry.update({
            "finished_at": now(),
            "seconds": round(time.time() - wall, 3),
            "exit_code": code,
            "log": str(log_path),
        })
        entry["status"] = "complete" if code == 0 else "failed"
        self.save()
        return code

    def run(self):
        remaining = dict(self.tasks)
        failures = set()
        while remaining:
            progressed = False
            for name, spec in list(remaining.items()):
                deps = spec["deps"]
                if any(dep in failures for dep in deps):
                    continue
                if not all(self.task_state(dep).get("status") == "complete" for dep in deps):
                    continue
                entry = self.task_state(name)
                try:
                    spec["fn"](self, name)
                    if entry.get("status") != "complete":
                        raise RuntimeError(f"{name}: task did not mark itself complete")
                except Exception as exc:  # noqa: BLE001
                    entry.setdefault("errors", []).append(str(exc))
                    entry["status"] = "failed"
                    entry["finished_at"] = now()
                    self.save()
                    failures.add(name)
                    print(f"[{now()}] FAILED {name}: {exc}", flush=True)
                else:
                    print(f"[{now()}] DONE {name}", flush=True)
                del remaining[name]
                progressed = True
                break
            if not progressed:
                blocked = {name: spec["deps"] for name, spec in remaining.items()}
                self.state["blocked"] = blocked
                self.save()
                print(f"[{now()}] BLOCKED tasks: {sorted(remaining)}", flush=True)
                break
        self.state["last_run_at"] = now()
        self.save()
        return 0 if not failures else 1


def mark(driver, name, **fields):
    entry = driver.task_state(name)
    entry.update(fields)
    entry["status"] = "complete"
    entry["finished_at"] = entry.get("finished_at") or now()
    driver.save()


# ---------------------------------------------------------------------------
# Data tasks


def generate_sequences(driver, name, root, dataset, split, sequences):
    import zipfile
    from paired_collection_io import inspect_shard

    protocol = {"schema": 3, "full_write": True, "anchor": False,
                "column": True, "gap3": False, "rank_arrays": True,
                "teacher_sha256": None}
    if dataset != "SportsMOT":
        protocol["dataset"] = dataset
    missing = []
    for seq in sequences:
        directory = Path(root) / "data" / seq
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            missing.append(seq)
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{name}: unreadable summary for {seq}: {exc}") from exc
        if summary.get("protocol") != protocol:
            raise RuntimeError(f"{name}: incompatible replay protocol for {seq}")
        if not summary.get("complete"):
            missing.append(seq)
            continue
        try:
            inspect_shard(directory, protocol, repair_counts=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            missing.append(seq)
    if missing:
        command = [
            str(PY), "-u", "tools/generate_trainval_paired.py",
            "--root", str(root), "--dataset", dataset, "--split", split,
            "--full-write", "--column-probe", "--lean", "--only", *missing,
        ]
        code = driver.run_command(name, command)
        if code:
            raise RuntimeError(f"{name}: generation failed")
    checked = {}
    for seq in sequences:
        checked[seq] = inspect_shard(Path(root) / "data" / seq, protocol,
                                     repair_counts=True)
    mark(driver, name, sequences=list(sequences), collection_headers=checked)


def validate_pool(driver, name, data_root, dataset, split, sequences):
    out = EXP / "reports" / f"{name}_validation.json"
    command = [
        str(PY), "tools/verify_final_data.py",
        "--data-root", str(data_root), "--dataset", dataset, "--split", split,
        "--expect-full-write", "--require-column",
        "--sequences", *sequences, "--out", str(out),
    ]
    code = driver.run_command(name, command)
    if code:
        raise RuntimeError(f"{name}: validation failed")
    payload = json.loads(out.read_text())
    if not payload.get("data_ok"):
        raise RuntimeError(f"{name}: data_ok false")
    mark(driver, name, validation=str(out), samples=payload.get("samples"))


def task_data_mot20(d, name):
    generate_sequences(d, name, ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT20",
                       "MOT20", "train", ("MOT20-01", "MOT20-05"))


def task_data_mot17(d, name):
    generate_sequences(d, name, ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT17",
                       "MOT17", "train", ("MOT17-09-FRCNN", "MOT17-10-FRCNN"))


def task_data_sportsmot(d, name):
    generate_sequences(d, name, ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120",
                       "SportsMOT", "trainval", SM_VAL)


def task_validate_mot20(d, name):
    validate_pool(d, name, MOT20_DATA, "MOT20", "train", ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"))


def task_validate_mot17(d, name):
    validate_pool(d, name, MOT17_DATA, "MOT17", "train",
                  ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                   "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"))


def task_validate_sportsmot(d, name):
    validate_pool(d, name, SM_DATA, "SportsMOT", "trainval", tuple(sorted(SM_TRAIN_VIEW + SM_VAL)))


def make_view(path: Path, sequence_dirs):
    path.mkdir(parents=True, exist_ok=True)
    for sequence, source in sequence_dirs.items():
        target = path / sequence
        if target.is_symlink() or target.exists():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            raise RuntimeError(f"view path already exists differently: {target}")
        target.symlink_to(source, target_is_directory=True)


def task_views(d, name):
    make_view(EXP / "ablation/MOT20/data",
              {seq: MOT20_DATA / seq for seq in MOT20_TRAIN})
    make_view(EXP / "ablation/MOT17/data",
              {seq: MOT17_DATA / seq for seq in MOT17_ABL_TRAIN})
    make_view(EXP / "ablation/SportsMOT/data",
              {seq: SM_DATA / seq for seq in SM_TRAIN_VIEW})
    mark(d, name)


# ---------------------------------------------------------------------------
# Training / evaluation


def train_task(driver, name, dataset, group, data_root, groups_root, epochs, lr,
               sequences=None, init=None, architecture="legacy", proj=256,
               shards=4, sampling="sample-proportional", split="train"):
    group_dir = Path(groups_root) / group
    complete = group_dir / "complete.json"
    config_path = group_dir / "config.json"
    last = group_dir / "checkpoints" / "last.pt"
    if complete.is_file():
        payload = _check_checkpoint(last, architecture, proj)
        mark(driver, name, checkpoint=str(last), global_step=payload["global_step"],
             note="already complete", sha256=sha256_file(last))
        return
    command = [
        str(PY), "-u", "tools/train_supervision_groups.py",
        "--dataset", dataset, "--group", group,
        "--architecture-variant", architecture, "--appearance-proj-dim", str(proj),
        "--data-root", str(data_root), "--groups-root", str(groups_root),
        "--split", split, "--device", "cuda", "--batch-size", "1024",
        "--lr", str(lr), "--weight-decay", "1e-4", "--seed", "42",
        "--max-epochs", str(epochs), "--snapshot-every", "20",
        "--rank-weight", "18", "--rank-tau", "0.1", "--rank-alpha", "0.95",
        "--shards", str(shards), "--sequence-sampling", sampling,
    ]
    if sequences:
        command += ["--sequences", *sequences]
    if init:
        command += ["--init", str(init)]
    else:
        command += ["--from-scratch"]
    if group == "appearance_row_column":
        command += ["--column-weight", "50"]
    if last.is_file():
        command += ["--resume"]
    code = driver.run_command(name, command)
    if code:
        raise RuntimeError(f"{name}: training failed")
    if not complete.is_file():
        raise RuntimeError(f"{name}: complete.json missing")
    payload = _check_checkpoint(last, architecture, proj)
    config = json.loads(config_path.read_text())
    if int(payload["global_step"]) <= 0:
        raise RuntimeError(f"{name}: invalid global_step")
    if config.get("architecture_variant", "legacy") != architecture:
        raise RuntimeError(f"{name}: config architecture mismatch")
    if int(config.get("appearance_proj_dim", 64)) != proj:
        raise RuntimeError(f"{name}: config projection mismatch")
    mark(driver, name, checkpoint=str(last), global_step=payload["global_step"],
         seconds=payload.get("seconds"), config=str(config_path),
         sha256=sha256_file(last))


def _check_checkpoint(path, architecture, proj):
    import torch
    if not Path(path).is_file():
        raise RuntimeError(f"missing checkpoint {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config") or {}
    if config.get("architecture_variant", "legacy") != architecture:
        raise RuntimeError(f"{path}: architecture {config.get('architecture_variant')} != {architecture}")
    if int(config.get("appearance_proj_dim", 64)) != int(proj):
        raise RuntimeError(f"{path}: projection {config.get('appearance_proj_dim')} != {proj}")
    return payload


def eval_task(driver, name, dataset, checkpoint, sequences, mode, suffix, out_dir=None):
    out = Path(out_dir) if out_dir else EXP / "ablation" / dataset / "evals" / suffix
    summary = out / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text())
        if len(payload.get("per_sequence") or {}) == len(sequences):
            mark(driver, name, summary=str(summary), reused=True)
            return
    command = [
        str(PY), "-u", "run.py", "--dataset", dataset, "--mode", mode,
        "--sequences", *sequences, "--seed", "10000", "--kf-type", "nsa",
        "--agentguard-mode", "shutrack-v2", "--shutrack-checkpoint", str(checkpoint),
        "--agentguard-device", "cpu",
        "--detection-cache-root", str(ROOT / "outputs/agentguard/detection_cache"),
        "--output_dir", str(out), "--tracker-suffix", suffix,
        "--agentguard-stats-output", str(out / "gates.jsonl"),
        "--print-per-sequence-metrics", "--eval-json", str(summary),
        "--torch-num-threads", "4", "--torch-num-interop-threads", "1",
    ]
    code = driver.run_command(name, command, cwd=ROOT / "3. Tracker")
    if code:
        raise RuntimeError(f"{name}: evaluation failed")
    payload = json.loads(summary.read_text())
    if len(payload.get("per_sequence") or {}) != len(sequences):
        raise RuntimeError(f"{name}: wrong per-sequence count")
    mark(driver, name, summary=str(summary), checkpoint=str(checkpoint))


def reuse_old_ablation(driver, name, dataset, checkpoint, summary_dir, expected):
    if not checkpoint.is_file():
        raise RuntimeError(f"{name}: missing OLD checkpoint {checkpoint}")
    summary = summary_dir / "summary.json"
    if not summary.is_file():
        raise RuntimeError(f"{name}: missing OLD eval {summary}")
    payload = json.loads(summary.read_text())
    if len(payload.get("per_sequence") or {}) != expected:
        raise RuntimeError(f"{name}: OLD eval count mismatch")
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("model_config") or {}
    if config.get("appearance_proj_dim") != 256 or config.get("architecture_variant", "legacy") != "legacy":
        raise RuntimeError(f"{name}: OLD checkpoint config mismatch: {config}")
    mark(driver, name, checkpoint=str(checkpoint), summary=str(summary),
         sha256=sha256_file(checkpoint))


def task_recipe(driver, name):
    iwg = ROOT / "outputs/agentguard/experiments"
    recipe = {
        "created_at": now(),
        "provenance": {
            "MOT20": str(iwg / "iwg_rg_cma_sqrt_size_mot20_bound005_context6_seed42_bs1024_shard4x1_100e/provenance/train.command.txt"),
            "MOT17": str(iwg / "iwg_rg_cma_mot20s4e100_to_mot17_finetune50_lr1e5_bs1024/checkpoints/training_summary.json"),
            "SportsMOT": str(iwg / "iwg_rg_cma_v1_sportsmot_trainval_interleaved_shard4x1_seed42_bs1024_200e/provenance/train.command.txt"),
            "auxiliary": [
                str(iwg / "iwg_rg_cma_best_base_final_raw_20260725/provenance"),
                str(iwg / "shutrack_a_old_recipe_20261117/plan.json"),
            ],
        },
        "budget_mapping": {
            "MOT20": {
                "data": "official train MOT20-01/02/03/05",
                "init": "from-scratch",
                "shards": 4, "cycles": 25, "epochs": 100,
                "sequence_sampling": "sqrt-size", "sampling_power": 0.5,
                "lr": 1e-4, "batch_size": 1024, "weight_decay": 1e-4, "seed": 42,
                "warmup": "first shard-epoch actual steps", "grad_clip": 1.0,
            },
            "SportsMOT": {
                "data": "official train+val 90 sequences",
                "init": "from-scratch",
                "shards": 4, "cycles": 50, "epochs": 200,
                "sequence_sampling": "sample-proportional",
                "sampling_source": "historical command has no --sequence-sampling flag (default)",
                "lr": 1e-4, "batch_size": 1024, "weight_decay": 1e-4, "seed": 42,
                "warmup": "first shard-epoch actual steps", "grad_clip": 1.0,
            },
            "MOT17": {
                "data": "official train 7 FRCNN/private sequences",
                "init": "formal NEW-MOT20 final weights, weights-only, fresh optimizer/scheduler, refit MOT17 normalization and taus",
                "shards": 1, "cycles": 50, "epochs": 50,
                "sequence_sampling": "sample-proportional (full data, single shard)",
                "lr": 1e-5, "batch_size": 1024, "weight_decay": 1e-4, "seed": 42,
                "warmup": "first shard-epoch actual steps", "grad_clip": 1.0,
            },
        },
        "trainer_extension": {
            "shards_parameterized": True,
            "sequence_sampling": ["sample-proportional", "sqrt-size"],
            "sqrt_size_definition": "IWG exact quota allocation: weight ~ count**0.5, exact phase budget, sampling with replacement",
            "default_equivalence": "4-shard sample-proportional schedule, order and row indices verified identical to the pre-extension trainer",
            "batch_composition_difference": "sqrt-size batches are sequence-contiguous; IWG shuffled the flat sampled index list globally",
        },
        "actual_data": {
            "MOT20": str(MOT20_DATA), "MOT17": str(MOT17_DATA), "SportsMOT": str(SM_DATA),
        },
        "known_differences": [
            "Historical optimizer-step counts (e.g. MOT17 4,600) come from the IWG packed datasets; this round recomputes steps from the schema-3 paired shards.",
            "SportsMOT formal val is a training-scene diagnostic, not an independent validation set.",
            "The new architecture is a different model family; budgets are transferred, not IWG weights or losses.",
        ],
    }
    path = EXP / "FINAL_RECIPE.json"
    path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    mark(driver, name, path=str(path))


def task_test_submit(driver, name):
    script = ROOT / "tools/arch_night_test_submit.py"
    if not script.is_file():
        raise RuntimeError(f"{name}: missing {script}")
    code = driver.run_command(name, [str(PY), str(script)])
    if code:
        raise RuntimeError(f"{name}: test/submit failed")
    results = EXP / "submissions/SUBMISSION_RESULTS.json"
    if not results.is_file():
        raise RuntimeError(f"{name}: missing submission results")
    payload = json.loads(results.read_text())
    missing = [ds for ds in ("MOT20", "MOT17", "SportsMOT") if ds not in payload]
    if missing:
        raise RuntimeError(f"{name}: missing datasets {missing}")
    failed = [ds for ds in ("MOT20", "MOT17", "SportsMOT")
              if (payload.get(ds) or {}).get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{name}: submission failed for {failed}")
    mark(driver, name, results=str(results))


def task_test_submit_dataset(driver, name, dataset):
    script = ROOT / "tools/arch_night_test_submit.py"
    if not script.is_file():
        raise RuntimeError(f"{name}: missing {script}")
    code = driver.run_command(name, [str(PY), str(script), "--datasets", dataset])
    if code:
        raise RuntimeError(f"{name}: test/submit failed")
    results = EXP / "submissions/SUBMISSION_RESULTS.json"
    if not results.is_file():
        raise RuntimeError(f"{name}: missing submission results")
    payload = json.loads(results.read_text())
    entry = payload.get(dataset) or {}
    if entry.get("status") == "failed":
        raise RuntimeError(f"{name}: {dataset} failed: {entry.get('error')}")
    submission = entry.get("submission") or {}
    if submission.get("submission_id") is None:
        raise RuntimeError(f"{name}: no submission id for {dataset}")
    mark(driver, name, results=str(results), dataset_status=entry.get("status"),
         submission_id=submission.get("submission_id"), scores=submission.get("scores"))


def task_report(driver, name, kind):
    script = ROOT / "tools/arch_night_reports.py"
    if not script.is_file():
        raise RuntimeError(f"{name}: missing report script")
    code = driver.run_command(name, [str(PY), str(script), kind])
    if code:
        raise RuntimeError(f"{name}: report failed")
    expected = EXP / "reports" / ("ABLATION_REPORT.md" if kind == "ablation" else "FINAL_REPORT.md")
    if not expected.is_file():
        raise RuntimeError(f"{name}: missing {expected}")
    mark(driver, name, report=str(expected))


def build_driver(stage_a_only=False) -> Driver:
    d = Driver()
    if stage_a_only:
        # The ablation pools (MOT20-02/03, the five MOT17 train sequences and
        # the 45 SportsMOT train sequences) already exist and were accepted by
        # the B/D experiments; no generation or formal-pool validation is
        # needed before Stage A. Raw val evaluation reads detection caches,
        # not paired data, so missing val-sequence data does not block it.
        d.task("views", (), task_views)
    else:
        # Data
        d.task("data_mot20", (), task_data_mot20)
        d.task("data_mot17", (), task_data_mot17)
        d.task("data_sportsmot", (), task_data_sportsmot)
        d.task("validate_mot20", ("data_mot20",), task_validate_mot20)
        d.task("validate_mot17", ("data_mot17",), task_validate_mot17)
        d.task("validate_sportsmot", ("data_sportsmot",), task_validate_sportsmot)
        d.task("views", ("validate_mot20", "validate_mot17", "validate_sportsmot"), task_views)

    # Stage A reuse of OLD D-256
    d.task("ablation_mot20_old", ("views",), lambda drv, name: reuse_old_ablation(
        drv, name, "MOT20", OLD_MOT20_CKPT, OLD_MOT20_EVAL, 2))
    d.task("ablation_sportsmot_old", ("views",), lambda drv, name: reuse_old_ablation(
        drv, name, "SportsMOT", OLD_SM_CKPT, OLD_SM_EVAL, 45))

    # Stage A NEW training
    d.task("ablation_mot20_new_train", ("views",), lambda drv, name: train_task(
        drv, name, "MOT20", "appearance_row_column",
        EXP / "ablation/MOT20/data", EXP / "ablation/MOT20/groups",
        100, "1e-4", sequences=MOT20_TRAIN,
        architecture="query_film_bilinear", proj=256))
    d.task("ablation_mot20_new_eval", ("ablation_mot20_new_train",), lambda drv, name: eval_task(
        drv, name, "MOT20",
        EXP / "ablation/MOT20/groups/appearance_row_column/checkpoints/epoch_100.pt",
        MOT20_VAL, "all", "new"))

    d.task("ablation_sportsmot_new_train", ("views",), lambda drv, name: train_task(
        drv, name, "SportsMOT", "appearance_row_column",
        EXP / "ablation/SportsMOT/data", EXP / "ablation/SportsMOT/groups",
        200, "1e-4", architecture="query_film_bilinear", proj=256))
    d.task("ablation_sportsmot_new_eval", ("ablation_sportsmot_new_train",), lambda drv, name: eval_task(
        drv, name, "SportsMOT",
        EXP / "ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt",
        SM_VAL, "val", "new"))

    # MOT17 ablation: OLD init from OLD MOT20, NEW init from NEW MOT20.
    d.task("ablation_mot17_old_train", ("ablation_mot20_old", "views"), lambda drv, name: train_task(
        drv, name, "MOT17", "appearance_row_column",
        EXP / "ablation/MOT17/data", EXP / "ablation/MOT17/groups_old",
        200, "1e-5", sequences=MOT17_ABL_TRAIN,
        init=OLD_MOT20_CKPT, architecture="legacy", proj=256))
    d.task("ablation_mot17_old_eval", ("ablation_mot17_old_train",), lambda drv, name: eval_task(
        drv, name, "MOT17",
        EXP / "ablation/MOT17/groups_old/appearance_row_column/checkpoints/epoch_200.pt",
        MOT17_VAL, "all", "old"))
    d.task("ablation_mot17_new_train", ("ablation_mot20_new_train", "views"), lambda drv, name: train_task(
        drv, name, "MOT17", "appearance_row_column",
        EXP / "ablation/MOT17/data", EXP / "ablation/MOT17/groups_new",
        200, "1e-5", sequences=MOT17_ABL_TRAIN,
        init=EXP / "ablation/MOT20/groups/appearance_row_column/checkpoints/epoch_100.pt",
        architecture="query_film_bilinear", proj=256))
    d.task("ablation_mot17_new_eval", ("ablation_mot17_new_train",), lambda drv, name: eval_task(
        drv, name, "MOT17",
        EXP / "ablation/MOT17/groups_new/appearance_row_column/checkpoints/epoch_200.pt",
        MOT17_VAL, "all", "new"))

    if stage_a_only:
        return d

    # Stage B: final budgets transferred from the recorded IWG recipes.
    d.task("recipe", (
        "ablation_mot20_new_eval", "ablation_sportsmot_new_eval",
        "ablation_mot17_old_eval", "ablation_mot17_new_eval",
    ), task_recipe)
    d.task("final_mot20_train", ("recipe", "validate_mot20"), lambda drv, name: train_task(
        drv, name, "MOT20", "appearance_row_column",
        MOT20_DATA, EXP / "final/MOT20/groups", 100, "1e-4",
        architecture="query_film_bilinear", proj=256, shards=4, sampling="sqrt-size"))
    d.task("final_sportsmot_train", ("recipe", "validate_sportsmot"), lambda drv, name: train_task(
        drv, name, "SportsMOT", "appearance_row_column",
        SM_DATA, EXP / "final/SportsMOT/groups", 200, "1e-4", split="trainval",
        architecture="query_film_bilinear", proj=256, shards=4,
        sampling="sample-proportional"))
    d.task("final_sportsmot_val", ("final_sportsmot_train",), lambda drv, name: eval_task(
        drv, name, "SportsMOT",
        EXP / "final/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt",
        SM_VAL, "val", "final_diag",
        out_dir=EXP / "final/SportsMOT/evals/diag_val"))
    d.task("final_mot17_train", (
        "recipe", "validate_mot17", "final_mot20_train",
    ), lambda drv, name: train_task(
        drv, name, "MOT17", "appearance_row_column",
        MOT17_DATA, EXP / "final/MOT17/groups", 50, "1e-5",
        init=EXP / "final/MOT20/groups/appearance_row_column/checkpoints/last.pt",
        architecture="query_film_bilinear", proj=256, shards=1,
        sampling="sample-proportional"))

    d.task("test_submit", (
        "final_mot20_train", "final_sportsmot_train", "final_sportsmot_val",
        "final_mot17_train",
    ), task_test_submit)
    d.task("report_ablation", (
        "ablation_mot20_old", "ablation_mot20_new_eval",
        "ablation_sportsmot_old", "ablation_sportsmot_new_eval",
        "ablation_mot17_old_eval", "ablation_mot17_new_eval",
    ), lambda drv, name: task_report(drv, name, "ablation"))
    d.task("report_final", ("test_submit", "report_ablation"),
           lambda drv, name: task_report(drv, name, "final"))
    return d


def build_mainline_driver() -> Driver:
    """User-ordered main line: SportsMOT -> MOT20 -> MOT17, each followed by
    test/post/package and its own Codabench submission.

    Reuses the full driver's task definitions (data checks, formal-pool
    validation, ablation reuse, recipe, final training configs) but replaces
    the combined submission task with one task per dataset and reorders the
    DAG so SportsMOT runs first.
    """
    d = build_driver()
    d.tasks.pop("test_submit", None)
    d.tasks["test_submit_sportsmot"] = {
        "deps": ("final_sportsmot_train", "final_sportsmot_val"),
        "fn": lambda drv, name: task_test_submit_dataset(drv, name, "SportsMOT"),
    }
    d.tasks["test_submit_mot20"] = {
        "deps": ("final_mot20_train",),
        "fn": lambda drv, name: task_test_submit_dataset(drv, name, "MOT20"),
    }
    d.tasks["test_submit_mot17"] = {
        "deps": ("final_mot17_train",),
        "fn": lambda drv, name: task_test_submit_dataset(drv, name, "MOT17"),
    }
    d.tasks["report_final"]["deps"] = (
        "test_submit_sportsmot", "test_submit_mot20", "test_submit_mot17",
        "report_ablation",
    )
    order = [
        "data_mot20", "data_mot17", "data_sportsmot",
        "validate_mot20", "validate_mot17", "validate_sportsmot", "views",
        "ablation_mot20_old", "ablation_sportsmot_old",
        "ablation_mot20_new_train", "ablation_mot20_new_eval",
        "ablation_sportsmot_new_train", "ablation_sportsmot_new_eval",
        "ablation_mot17_old_train", "ablation_mot17_old_eval",
        "ablation_mot17_new_train", "ablation_mot17_new_eval",
        "recipe",
        "final_sportsmot_train", "final_sportsmot_val", "test_submit_sportsmot",
        "final_mot20_train", "test_submit_mot20",
        "final_mot17_train", "test_submit_mot17",
        "report_ablation", "report_final",
    ]
    d.tasks = {name: d.tasks[name] for name in order}
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--stage-a", action="store_true",
        help="Run only the NEW-vs-OLD ablation on existing data, skipping "
             "formal data generation and validation (which block Stage B).",
    )
    parser.add_argument(
        "--main-line", action="store_true",
        help="Formal training + test/post/package + per-dataset submission in "
             "the user's order: SportsMOT, MOT20, MOT17.",
    )
    args = parser.parse_args()
    if args.main_line:
        driver = build_mainline_driver()
    else:
        driver = build_driver(stage_a_only=args.stage_a)
    if args.list:
        for name, spec in driver.tasks.items():
            print(name, "<-", ", ".join(spec["deps"]) or "-")
        return
    raise SystemExit(driver.run())


if __name__ == "__main__":
    main()
