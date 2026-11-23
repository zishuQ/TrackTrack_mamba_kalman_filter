#!/usr/bin/env python3
"""MOT17/MOT20 SHUTrack pipeline: ledger, training, eval, test, package, submit.

Every stage is idempotent and resumable:
  - experiments.json (this directory) holds the historical ledger: recipes,
    checkpoints, official scores and submission IDs;
  - new runs write <runs>/<experiment>/state.json with stage status and commands;
  - submit refuses to create a second submission for an experiment already in
    the ledger and polls existing records instead.

Commands default to PRINTING the command (dry run). Add --execute to run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SKILL = Path(__file__).resolve().parent
ROOT = SKILL.parents[2]
assert (ROOT / "3. Tracker").is_dir(), f"unexpected skill location: {SKILL}"
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
CACHE = ROOT / "outputs/agentguard/detection_cache"
RUNS = ROOT / "outputs/agentguard/experiments/mot_pipeline_runs"
LEDGER = SKILL / "experiments.json"
MAKE_PACKAGE = ROOT / "outputs/agentguard/experiments/shutrack_a_submit_20261117/scripts/make_package.py"
SUBMIT_TOOL = ROOT / "tools/codabench_submit.py"
TRAIN_TOOL = ROOT / "tools/train_supervision_groups.py"
CONTINUE_TOOL = ROOT / "tools/continue_low_lr.py"
NON_TERMINAL = ("Submitting", "Running", "Preparing", "Queued", "Pending",
                "Submitted", "Scoring")
HEAVY = {"run.py", "train_supervision_groups.py", "continue_low_lr.py",
         "train_motion_rank.py", "train_stop_after.py"}
TOKENS = (("jojo0", Path.home() / ".config/codabench/token"),
          ("shangqingnie", Path.home() / ".config/codabench/token_edge"))


# ---------------------------------------------------------------- ledger ----
def load_ledger():
    return json.loads(LEDGER.read_text())


def save_ledger(ledger):
    ledger["updated_at"] = time.strftime("%Y-%m-%d")
    if LEDGER.is_file():
        shutil.copyfile(LEDGER, LEDGER.with_suffix(".json.bak"))
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(LEDGER)


def dataset_protocol(ledger, dataset):
    if dataset not in ledger["datasets"]:
        raise SystemExit(f"unknown dataset {dataset}; choose {sorted(ledger['datasets'])}")
    return ledger["datasets"][dataset]["protocol"]


def experiment_dir(name):
    return RUNS / name


def load_state(name):
    path = experiment_dir(name) / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {"stages": {}, "results": {}}


def save_state(name, state):
    path = experiment_dir(name) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


TRAINING_MODES = ("scratch", "finetune", "continue")
TUNABLE = ("mode", "epochs", "shards", "lr", "batch_size", "init", "source",
           "allow_partial_cycles")


def repo_rel(path):
    if path in (None, ""):
        return None
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(ROOT))
        except ValueError:
            return str(candidate)
    return str(candidate)


def canonical_config(dataset, params):
    mode = str(params.get("mode") or params.get("kind"))
    if mode not in TRAINING_MODES:
        raise SystemExit(f"mode must be one of {TRAINING_MODES}")
    config = {"dataset": dataset, "mode": mode,
              "epochs": int(params["epochs"]), "shards": int(params["shards"]),
              "lr": float(params["lr"]),
              "batch_size": int(params.get("batch_size", 1024)),
              "allow_partial_cycles": bool(params.get("allow_partial_cycles", False)),
              "init": None, "source": None}
    if mode == "finetune":
        config["init"] = repo_rel(params.get("init"))
    elif mode == "continue":
        config["source"] = repo_rel(params.get("source"))
    return config


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def experiment_params(entry):
    return {"mode": entry.get("kind"), "epochs": entry.get("epochs"),
            "shards": entry.get("shards"), "lr": entry.get("lr"),
            "batch_size": entry.get("batch_size", 1024),
            "init": entry.get("init") or entry.get("source_checkpoint"),
            "source": entry.get("source") or entry.get("source_checkpoint"),
            "allow_partial_cycles": entry.get("allow_partial_cycles", False)}


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


def wait_jobs(label):
    while True:
        jobs = active_jobs()
        if not jobs:
            return
        print(f"{time.strftime('%H:%M:%S')} waiting before {label}: {jobs}", flush=True)
        time.sleep(30)


def execute(name, stage, command, cwd=ROOT):
    wait_jobs(f"{name}:{stage}")
    state = load_state(name)
    entry = state["stages"].setdefault(stage, {})
    entry.update({"status": "running", "command": [str(x) for x in command],
                  "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    save_state(name, state)
    log_dir = experiment_dir(name) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{stage}.log"
    started = time.time()
    with log.open("ab") as handle:
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{command}\n".encode())
        handle.flush()
        code = subprocess.call([str(x) for x in command], cwd=str(cwd),
                               stdout=handle, stderr=subprocess.STDOUT)
    entry.update({"status": "complete" if code == 0 else "failed",
                  "exit_code": code, "seconds": round(time.time() - started, 1),
                  "log": str(log), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    save_state(name, state)
    if code:
        raise SystemExit(f"{name}:{stage} failed (exit {code}); see {log}")


def run_or_print(name, stage, command, do_execute, cwd=ROOT):
    printable = " ".join(str(x) for x in command)
    print(f"[{name}:{stage}] {printable}", flush=True)
    if not do_execute:
        return
    execute(name, stage, command, cwd=cwd)


# --------------------------------------------------------------- status -----
def cmd_status(args):
    ledger = load_ledger()
    for dataset, entry in ledger["datasets"].items():
        if args.dataset and dataset != args.dataset:
            continue
        best = entry["official_best"]
        print(f"\n== {dataset}  best HOTA {best['hota']:.3f}  "
              f"({best['experiment']}, submission {best['submission_id']}, {best['account']})")
        print(f"   train {entry['protocol']['train_sequences']}")
        print(f"   val   {entry['protocol']['val_sequences']}")
        print(f"   exp  {entry['protocol']['competition']}/{entry['protocol']['phase']}/{entry['protocol']['task']}")
        for name, exp in entry["experiments"].items():
            scores = exp.get("official_scores") or {}
            hota = scores.get("HOTA")
            subs = exp.get("submissions") or {}
            sub_text = ", ".join(f"{acct}:{rec.get('id')}" for acct, rec in subs.items()) or "-"
            ckpt = exp.get("checkpoint")
            ckpt_ok = "" if ckpt is None else ("ok" if (ROOT / ckpt).is_file() else "MISSING")
            hota_text = "-" if hota is None else f"{hota:.3f}"
            print(f"   - {name:24s} {exp.get('kind','?'):20s} HOTA {hota_text:>7s}  "
                  f"sub {sub_text:24s} ckpt {ckpt_ok}")
        if args.json:
            print(json.dumps(entry, indent=2, ensure_ascii=False))


def cmd_verify(args):
    ledger = load_ledger()
    missing = []
    for dataset, entry in ledger["datasets"].items():
        for name, exp in entry["experiments"].items():
            for key in ("checkpoint", "source_checkpoint"):
                path = exp.get(key)
                if path and not (ROOT / path).is_file():
                    missing.append((dataset, name, key, path))
            for key, path in (exp.get("artifacts") or {}).items():
                if not (ROOT / path).exists():
                    missing.append((dataset, name, f"artifacts.{key}", path))
            for key in ("state_dir",):
                path = exp.get(key)
                if path and not (ROOT / path).exists():
                    missing.append((dataset, name, key, path))
    if missing:
        print(json.dumps({"status": "missing", "entries": missing}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "ok", "datasets": sorted(ledger["datasets"])}, indent=2))


# ---------------------------------------------------------------- train -----
def _checkpoint_epoch(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"]), payload


def cmd_train(args):
    ledger = load_ledger()
    protocol = dataset_protocol(ledger, args.dataset)
    if not args.experiment:
        raise SystemExit("--experiment is required")
    name = args.experiment
    if name in ledger["datasets"][args.dataset]["experiments"]:
        raise SystemExit(f"{name}: experiment name already exists in the ledger; choose a new name")
    state = load_state(name)
    final_epoch = args.epochs
    final_ckpt = experiment_dir(name) / "groups/appearance_row_column/checkpoints" / f"epoch_{final_epoch:03d}.pt"
    if final_ckpt.is_file():
        epoch, _ = _checkpoint_epoch(final_ckpt)
        if epoch == final_epoch:
            print(json.dumps({"experiment": name, "status": "train already complete",
                              "checkpoint": str(final_ckpt)}, indent=2), flush=True)
            return
    groups = experiment_dir(name) / "groups"
    base = [str(PY), "-u", str(TRAIN_TOOL),
            "--group", "appearance_row_column", "--dataset", args.dataset,
            "--split", "train", "--data-root", protocol["data_root"],
            "--groups-root", str(groups),
            "--batch-size", str(getattr(args, "batch_size", 1024) or 1024),
            "--lr", str(args.lr),
            "--weight-decay", "1e-4", "--seed", "42", "--shards", str(args.shards),
            "--max-epochs", str(args.epochs), "--snapshot-every", "20",
            "--rank-weight", "18", "--rank-alpha", "0.95", "--rank-tau", "0.1",
            "--column-weight", "50", "--appearance-proj-dim", "256",
            "--architecture-variant", "query_film_bilinear", "--device", "cuda"]
    if getattr(args, "allow_partial_cycles", False):
        base.append("--allow-partial-cycles")
    if args.mode == "continue":
        if not args.source:
            raise SystemExit("--source is required for --mode continue")
        source = ROOT / args.source
        if not source.is_file():
            raise SystemExit(f"missing source checkpoint: {source}")
        source_epoch, _ = _checkpoint_epoch(source)
        command = [str(PY), "-u", str(CONTINUE_TOOL),
                   "--source-checkpoint", str(source),
                   "--stage-epochs", str(args.epochs),
                   "--cumulative-epochs", str(source_epoch + args.epochs),
                   "--provenance", str(experiment_dir(name) / "provenance.json")]
        command += base + ["--init", str(source)]
    elif args.mode == "finetune":
        if not args.init:
            raise SystemExit("--init is required for --mode finetune")
        init = ROOT / args.init
        if not init.is_file():
            raise SystemExit(f"missing init checkpoint: {init}")
        command = base + ["--init", str(init)]
    else:
        command = base + ["--from-scratch"]
    run_or_print(name, "train", command, args.execute)
    if args.execute:
        if not final_ckpt.is_file():
            raise SystemExit(f"training finished without {final_ckpt}")
        epoch, payload = _checkpoint_epoch(final_ckpt)
        state["results"]["train"] = {"checkpoint": str(final_ckpt), "epoch": epoch,
                                     "global_step": payload.get("global_step"),
                                     "sha256": sha256(final_ckpt)}
        save_state(name, state)
        entry = ledger["datasets"][args.dataset]["experiments"].setdefault(name, {})
        entry.update({"kind": args.mode, "epochs": args.epochs, "shards": args.shards,
                      "lr": args.lr, "checkpoint": str(final_ckpt.relative_to(ROOT)),
                      "source_checkpoint": args.source, "init": args.init})
        save_ledger(ledger)


# --------------------------------------------------------------- eval -------
def eval_command(protocol, checkpoint, out, suffix):
    return [str(PY), "-u", "run.py", "--dataset", protocol.get("dataset"),
            "--mode", "all", "--sequences", *protocol["val_sequences"],
            "--seed", "10000", "--kf-type", "nsa",
            "--agentguard-mode", "shutrack-v2",
            "--shutrack-checkpoint", str(checkpoint),
            "--agentguard-device", "cpu",
            "--detection-cache-root", protocol["cache_root"],
            "--output_dir", str(out), "--tracker-suffix", suffix,
            "--agentguard-stats-output", str(out / "gates.jsonl"),
            "--print-per-sequence-metrics", "--eval-json", str(out / "summary.json"),
            "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]


def cmd_eval(args):
    ledger = load_ledger()
    protocol = dataset_protocol(ledger, args.dataset)
    protocol = {**protocol, "dataset": args.dataset}
    if not args.experiment:
        raise SystemExit("--experiment is required")
    name = args.experiment
    state = load_state(name)
    checkpoints = []
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint).expanduser().resolve()]
    else:
        if not args.checkpoint_dir:
            raise SystemExit("pass --checkpoint or --checkpoint-dir")
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        for epoch in [int(x) for x in args.epochs.split(",") if x.strip()]:
            path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            if not path.is_file():
                raise SystemExit(f"missing checkpoint: {path}")
            checkpoints.append(path)
    for checkpoint in checkpoints:
        epoch, _ = _checkpoint_epoch(checkpoint)
        out = experiment_dir(name) / "evals" / f"e{epoch:03d}"
        if (out / "summary.json").is_file():
            print(f"[{name}] eval e{epoch:03d} already exists: {out / 'summary.json'}", flush=True)
            continue
        command = eval_command(protocol, checkpoint, out, f"{name}_e{epoch:03d}")
        run_or_print(name, f"eval_e{epoch:03d}", command, args.execute, cwd=TRACKER)
        if args.execute:
            payload = json.loads((out / "summary.json").read_text())
            combined = payload.get("combined") or {}
            state = load_state(name)
            state["results"].setdefault("evals", {})[f"e{epoch:03d}"] = {
                "summary": str(out / "summary.json"), "combined": combined}
            save_state(name, state)
            entry = ledger["datasets"][args.dataset]["experiments"].get(name)
            if entry is not None and combined.get("HOTA") is not None:
                values = entry.setdefault("val_hota", {})
                values[str(epoch)] = round(float(combined["HOTA"]) * 100, 3)
                best_epoch = max(values, key=lambda key: values[key])
                entry["best_epoch"] = int(best_epoch)
                entry["best_val_hota"] = values[best_epoch]
                save_ledger(ledger)


# --------------------------------------------------------------- test -------
def cmd_test(args):
    ledger = load_ledger()
    protocol = dataset_protocol(ledger, args.dataset)
    protocol = {**protocol, "dataset": args.dataset}
    if not args.experiment:
        raise SystemExit("--experiment is required")
    name = args.experiment
    state = load_state(name)
    existing = (state.get("results", {}).get("test") or {})
    post_existing = Path(existing.get("post_dir", ""))
    if (not getattr(args, "force", False) and post_existing.is_dir()
            and len(list(post_existing.glob("*.txt"))) == protocol["expected_test_files"]):
        print(f"[{name}] test already complete: {post_existing}", flush=True)
        return
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint is None:
        exp = ledger["datasets"][args.dataset]["experiments"].get(name) or {}
        if exp.get("checkpoint"):
            checkpoint = ROOT / exp["checkpoint"]
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint for {name}")
    out_dir = experiment_dir(name) / "test/tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(PY), "-u", "run.py", "--dataset", args.dataset,
               "--mode", "test", "--sequences", *protocol["test_sequences"],
               "--seed", "10000", "--kf-type", "nsa",
               "--agentguard-mode", "shutrack-v2",
               "--shutrack-checkpoint", str(checkpoint),
               "--agentguard-device", "cpu",
               "--detection-cache-root", protocol["cache_root"],
               "--output_dir", str(out_dir), "--tracker-suffix", name,
               "--agentguard-stats-output", str(experiment_dir(name) / "test/gates.jsonl"),
               "--use_post", "--skip-eval",
               "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
    run_or_print(name, "test", command, args.execute, cwd=TRACKER)
    if not args.execute:
        return
    posts = [p for p in out_dir.iterdir() if p.is_dir() and p.name.endswith("_post")]
    if len(posts) != 1:
        raise SystemExit(f"{name}: expected one post dir, found {posts}")
    post = posts[0]
    if args.dataset == "MOT17":
        frcnn = sorted(post.glob("*-FRCNN.txt"))
        if len(frcnn) != 7:
            raise SystemExit(f"MOT17: expected 7 FRCNN txt, found {len(frcnn)}")
        for source in frcnn:
            stem = source.name[: -len("-FRCNN.txt")]
            for detector in ("DPM", "SDP"):
                shutil.copyfile(source, post / f"{stem}-{detector}.txt")
    files = sorted(post.glob("*.txt"))
    if len(files) != protocol["expected_test_files"]:
        raise SystemExit(f"{name}: expected {protocol['expected_test_files']} txt, found {len(files)}")
    gates = [json.loads(line) for line in
             (experiment_dir(name) / "test/gates.jsonl").read_text().splitlines() if line.strip()]
    state = load_state(name)
    state["results"]["test"] = {
        "post_dir": str(post), "files": len(files), "gate_rows": len(gates),
        "checkpoint": str(checkpoint)}
    save_state(name, state)


# ------------------------------------------------------------- package ------
def cmd_package(args):
    ledger = load_ledger()
    protocol = dataset_protocol(ledger, args.dataset)
    if not args.experiment:
        raise SystemExit("--experiment is required")
    name = args.experiment
    state = load_state(name)
    packaged = (state.get("results", {}).get("package") or {})
    if (not getattr(args, "force", False) and packaged.get("zip")
            and Path(packaged["zip"]).is_file()
            and state.get("stages", {}).get("validate", {}).get("status") == "complete"):
        print(f"[{name}] package already complete: {packaged['zip']}", flush=True)
        return
    post = Path(args.post_dir).expanduser().resolve() if args.post_dir else None
    if post is None:
        post = Path(state["results"]["test"]["post_dir"])
    zip_path = experiment_dir(name) / "submissions" / f"{args.dataset}.zip"
    command = [str(PY), str(MAKE_PACKAGE), "--source", str(post),
               "--zip", str(zip_path), "--expected-count", str(protocol["expected_test_files"])]
    run_or_print(name, "package", command, args.execute)
    if not args.execute:
        return
    validate = [str(PY), str(SUBMIT_TOOL), "validate", "--zip", str(zip_path),
                "--dataset", protocol["dataset_flag"],
                "--seqinfo-root", protocol["seqinfo_test"]]
    execute(name, "validate", validate)
    state = load_state(name)
    state["results"]["package"] = {"zip": str(zip_path), "sha256": sha256(zip_path)}
    save_state(name, state)


# -------------------------------------------------------------- submit ------
def _curl_json(token, path, timeout=60):
    last = None
    for _ in range(6):
        result = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout),
             "-H", f"Authorization: Token {token}", f"https://www.codabench.org{path}"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                last = result.stdout[:200]
        else:
            last = result.stderr[:200]
        time.sleep(4)
    raise RuntimeError(f"Codabench API unavailable: {path} ({last})")


def _phase_submissions(token, phase):
    payload = _curl_json(token, f"/api/submissions/?phase={phase}")
    return payload if isinstance(payload, list) else payload.get("results", [])


def _poll(token, submission_id, record, timeout=2700):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = _curl_json(token, f"/api/submissions/{submission_id}/")
        status = payload.get("status")
        if status is None and payload.get("id") is None:
            raise RuntimeError(f"cannot read submission {submission_id}")
        print(f"{time.strftime('%H:%M:%S')} {submission_id} status={status}", flush=True)
        if status not in NON_TERMINAL:
            record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return payload
        time.sleep(30)
    raise RuntimeError(f"submission {submission_id} still running after {timeout}s")


def _scores(payload):
    scores = payload.get("scores")
    if isinstance(scores, dict):
        return scores
    if isinstance(scores, list):
        return {item.get("column_key"): item.get("score")
                for item in scores if isinstance(item, dict)}
    return {}


def cmd_submit(args):
    ledger = load_ledger()
    protocol = dataset_protocol(ledger, args.dataset)
    if not args.experiment:
        raise SystemExit("--experiment is required")
    name = args.experiment
    entry = ledger["datasets"][args.dataset]["experiments"].get(name) or {}
    if entry.get("submissions"):
        print(json.dumps({"experiment": name, "already_submitted": entry["submissions"]}, indent=2))
        if args.execute:
            raise SystemExit(f"{name}: ledger already records a submission; refusing duplicate")
        return
    state = load_state(name)
    zip_path = Path(args.zip).expanduser().resolve() if args.zip else Path(
        (state.get("results", {}).get("package") or {}).get("zip", ""))
    if not zip_path.is_file():
        raise SystemExit(f"missing zip: {zip_path}")
    record = experiment_dir(name) / "submissions" / f"{args.dataset}_submission.json"
    state_file = experiment_dir(name) / "submissions" / "submission_state.json"
    substate = json.loads(state_file.read_text()) if state_file.is_file() else {"baseline": {}}
    planned = {"experiment": name, "zip": str(zip_path),
               "accounts": [account for account, path in TOKENS if path.is_file()]}
    print(json.dumps(planned, indent=2))
    if not args.execute:
        print("dry run; add --execute to submit")
        return
    if record.is_file():
        payload = json.loads(record.read_text())
        if payload.get("id") is not None:
            if payload.get("status") in NON_TERMINAL:
                token = None
                for _, token_file in TOKENS:
                    if not token_file.is_file():
                        continue
                    candidate = token_file.read_text().strip()
                    check = _curl_json(candidate, f"/api/submissions/{payload['id']}/")
                    if check.get("status") is not None:
                        token = candidate
                        break
                if token is None:
                    raise SystemExit(f"cannot poll {payload['id']} with either account")
                payload = _poll(token, payload["id"], record)
            print(json.dumps({"submission_id": payload["id"], "status": payload.get("status"),
                              "scores": _scores(payload)}, indent=2))
            self_record(ledger, args.dataset, name,
                        payload.get("owner") or "unknown", payload, record)
            return
    last_error = None
    for account, token_file in TOKENS:
        if not token_file.is_file():
            continue
        token = token_file.read_text().strip()
        can = _curl_json(token, f"/api/can_make_submission/{protocol['phase']}/")
        if not can.get("can"):
            last_error = f"{account}: cannot submit ({can.get('reason')})"
            print(last_error, flush=True)
            continue
        baseline = substate["baseline"].get(account)
        if baseline is None:
            baseline = max([int(s["id"]) for s in _phase_submissions(token, protocol["phase"])
                            if s.get("id")] or [0])
            substate["baseline"][account] = baseline
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(substate, indent=2) + "\n")
        else:
            fresh = [s for s in _phase_submissions(token, protocol["phase"])
                     if int(s.get("id", 0)) > int(baseline)]
            if fresh:
                submission_id = max(int(s["id"]) for s in fresh)
                payload = _poll(token, submission_id, record)
                self_record(ledger, args.dataset, name, account, payload, record)
                return
        command = [str(PY), str(SUBMIT_TOOL), "submit", "--zip", str(zip_path),
                   "--competition", protocol["competition"], "--phase", protocol["phase"],
                   "--task", protocol["task"],
                   "--name", f"shutrack_{name}_{time.strftime('%Y%m%dT%H%M%S')}",
                   "--dataset", protocol["dataset_flag"],
                   "--seqinfo-root", protocol["seqinfo_test"],
                   "--token-file", str(token_file),
                   "--poll", "--poll-interval", "30", "--poll-timeout", "2400",
                   "--save-record", str(record)]
        code = subprocess.call(command)
        if record.is_file():
            payload = json.loads(record.read_text())
            if payload.get("id") is not None:
                if payload.get("status") in NON_TERMINAL:
                    payload = _poll(token, payload["id"], record)
                self_record(ledger, args.dataset, name, account, payload, record)
                return
        fresh = [s for s in _phase_submissions(token, protocol["phase"])
                 if int(s.get("id", 0)) > int(baseline)]
        if fresh:
            submission_id = max(int(s["id"]) for s in fresh)
            payload = _poll(token, submission_id, record)
            self_record(ledger, args.dataset, name, account, payload, record)
            return
        last_error = f"{account}: submit exited {code}"
    raise SystemExit(f"submission failed: {last_error}")


def self_record(ledger, dataset, name, account, payload, record):
    scores = _scores(payload)
    entry = ledger["datasets"][dataset]["experiments"].setdefault(name, {})
    entry.setdefault("submissions", {})[account] = {
        "id": payload.get("id"), "status": payload.get("status"), "scores": scores,
        "record": str(record)}
    official = entry.get("official_scores") or {}
    entry["official_scores"] = scores or official
    save_ledger(ledger)
    state = load_state(name)
    state["results"]["submission"] = {"id": payload.get("id"),
                                      "status": payload.get("status"),
                                      "account": account, "scores": scores}
    save_state(name, state)
    print(json.dumps({"submission_id": payload.get("id"), "status": payload.get("status"),
                      "scores": scores, "account": account}, indent=2), flush=True)


# -------------------------------------------------------------- record ------
def cmd_record(args):
    ledger = load_ledger()
    payload = json.loads(Path(args.submission_json).read_text())
    account = args.account or payload.get("owner") or "unknown"
    self_record(ledger, args.dataset, args.experiment, account, payload,
                Path(args.submission_json))


# -------------------------------------------------------------- propose -----
def _parse_set(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item}")
        key, value = item.split("=", 1)
        if key not in TUNABLE:
            raise SystemExit(f"{key} is not tunable; allowed: {', '.join(TUNABLE)}")
        overrides[key] = value
    return overrides


def cmd_propose(args):
    ledger = load_ledger()
    dataset = args.dataset
    experiments = ledger["datasets"][dataset]["experiments"]
    if args.experiment in experiments:
        raise SystemExit(f"{args.experiment}: experiment already exists")
    base = experiments.get(args.base)
    if not base or base.get("kind") not in TRAINING_MODES:
        raise SystemExit("--base must be a training experiment (scratch/finetune/continue)")
    params = experiment_params(base)
    for key, raw in _parse_set(args.set).items():
        if key in ("epochs", "shards", "batch_size"):
            params[key] = int(raw)
        elif key == "lr":
            params[key] = float(raw)
        elif key == "allow_partial_cycles":
            params[key] = raw.lower() in ("1", "true", "yes", "on")
        else:
            params[key] = raw
    if params["mode"] == "scratch":
        params["init"] = None
        params["source"] = None
    if params["mode"] == "finetune" and not params.get("init"):
        params["init"] = base.get("init")
    if params["mode"] == "continue" and not params.get("source"):
        params["source"] = base.get("source") or base.get("source_checkpoint")
    if dataset == "MOT20" and params["mode"] == "continue":
        params["allow_partial_cycles"] = True
    config = canonical_config(dataset, params)
    if config["mode"] == "finetune" and not config["init"]:
        raise SystemExit("finetune requires --set init=<checkpoint>")
    if config["mode"] == "continue" and not config["source"]:
        raise SystemExit("continue requires --set source=<checkpoint>")
    base_config = canonical_config(dataset, experiment_params(base))
    diff = {key: {"base": base_config.get(key), "new": config.get(key)}
            for key in config if base_config.get(key) != config.get(key)}
    digest = config_hash(config)
    reuse = None
    for name, experiment in experiments.items():
        if experiment.get("config_hash") != digest:
            continue
        checkpoint = experiment.get("checkpoint")
        if checkpoint and (ROOT / checkpoint).is_file():
            reuse = name
            break
    entry = {"kind": config["mode"], "epochs": config["epochs"],
             "shards": config["shards"], "lr": config["lr"],
             "batch_size": config["batch_size"], "init": config["init"],
             "source": config["source"],
             "allow_partial_cycles": config["allow_partial_cycles"],
             "params": params, "config": config, "config_hash": digest,
             "base_experiment": args.base, "change_reason": args.reason,
             "status": "proposed", "reuse_from": reuse,
             "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    experiments[args.experiment] = entry
    save_ledger(ledger)
    print(json.dumps({"experiment": args.experiment, "config": config,
                      "diff_vs_base": diff, "config_hash": digest,
                      "reuse_from": reuse, "reason": args.reason},
                     indent=2, ensure_ascii=False), flush=True)


# ------------------------------------------------------------------ run -----
def _complete(entry, stages):
    state = load_state(entry["name"]) if "name" in entry else {}
    results = state.get("results", {})
    checks = {"train": bool(results.get("train")),
              "test": bool(results.get("test")),
              "package": bool(results.get("package")),
              "submit": bool(results.get("submission"))}
    return all(checks.get(stage, True) for stage in stages)


def cmd_run(args):
    ledger = load_ledger()
    dataset = args.dataset
    name = args.experiment
    entry = ledger["datasets"][dataset]["experiments"].get(name)
    if not entry:
        raise SystemExit(f"{name}: unknown experiment; run propose first")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(stages) - {"train", "test", "package", "submit"}
    if bad:
        raise SystemExit(f"unknown stages: {sorted(bad)}")
    reuse = entry.get("reuse_from")
    print(json.dumps({"experiment": name, "status": entry.get("status"),
                      "reuse_from": reuse, "stages": stages}, indent=2))
    if reuse:
        source = ledger["datasets"][dataset]["experiments"][reuse]
        plan = {}
        if "train" in stages:
            plan["train"] = {"reuse_checkpoint": source.get("checkpoint")}
            if args.execute:
                state = load_state(name)
                state["stages"]["train"] = {"status": "complete", "reused_from": reuse}
                state["results"]["train"] = {"checkpoint": source.get("checkpoint"),
                                             "reused_from": reuse}
                save_state(name, state)
        if "test" in stages:
            post = (source.get("artifacts") or {}).get("post_dir")
            plan["test"] = {"reuse_post": post}
            if args.execute and post and Path(post).is_dir():
                state = load_state(name)
                state["stages"]["test"] = {"status": "complete", "reused_from": reuse}
                state["results"]["test"] = {"post_dir": post, "reused_from": reuse,
                                            "files": len(list(Path(post).glob("*.txt")))}
                save_state(name, state)
        if "package" in stages:
            zip_path = (source.get("artifacts") or {}).get("zip")
            plan["package"] = {"reuse_zip": zip_path}
            if args.execute and zip_path and Path(zip_path).is_file():
                state = load_state(name)
                state["stages"]["package"] = {"status": "complete", "reused_from": reuse}
                state["results"]["package"] = {"zip": zip_path, "sha256": sha256(zip_path),
                                               "reused_from": reuse}
                save_state(name, state)
        if "submit" in stages:
            submissions = source.get("submissions") or {}
            plan["submit"] = {"source_submissions": submissions,
                              "action": "skip duplicate" if submissions
                              else "submit reused artifacts"}
            if args.execute:
                if submissions:
                    first = next(iter(submissions.values()))
                    state = load_state(name)
                    state["results"]["submission"] = {"reused_from": reuse,
                                                      "duplicate": True, **first}
                    save_state(name, state)
                    print("source experiment already submitted; refusing duplicate", flush=True)
                else:
                    command = [str(PY), str(Path(__file__).resolve()), "submit",
                               "--dataset", dataset, "--experiment", name, "--execute"]
                    subprocess.call(command)
        if args.execute:
            entry["status"] = "complete" if _complete({"name": name, **entry}, stages) else "running"
            save_ledger(ledger)
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return
    pending_ok = True
    for stage in stages:
        command = [str(PY), str(Path(__file__).resolve()), stage,
                   "--dataset", dataset, "--experiment", name]
        if stage == "train":
            params = entry.get("params") or experiment_params(entry)
            for key in ("mode", "epochs", "shards", "lr", "batch_size", "init", "source"):
                value = params.get(key)
                if value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
            if params.get("allow_partial_cycles"):
                command.append("--allow-partial-cycles")
        if args.execute:
            command.append("--execute")
        elif stage in ("test", "package", "submit"):
            state = load_state(name)
            checkpoint = (state.get("results", {}).get("train") or {}).get("checkpoint")
            if checkpoint is None and stage != "submit":
                print(f"[{name}:{stage}] pending training; skipping dry-run", flush=True)
                continue
        print(" ".join(command), flush=True)
        if args.execute:
            code = subprocess.call(command)
            if code:
                raise SystemExit(f"{name}:{stage} failed")
            entry["status"] = stage
            save_ledger(ledger)
    if args.execute:
        entry["status"] = "complete"
        save_ledger(ledger)


# --------------------------------------------------------------- report -----
SCORE_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")


def _gate_stats(path):
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.is_file():
        return None
    rows = [json.loads(line) for line in candidate.read_text().splitlines() if line.strip()]
    motion = [r["shutrack_avg_motion_gate"] for r in rows
              if r.get("shutrack_avg_motion_gate") is not None]
    appearance = [r["shutrack_avg_appearance_gate"] for r in rows
                  if r.get("shutrack_avg_appearance_gate") is not None]
    if not motion:
        return None
    return {"rows": len(rows), "motion_min": min(motion),
            "motion_mean": sum(motion) / len(motion), "motion_max": max(motion),
            "appearance_min": min(appearance),
            "appearance_mean": sum(appearance) / len(appearance),
            "appearance_max": max(appearance)}


def _score_delta(scores, reference):
    out = {}
    for key in SCORE_KEYS:
        try:
            out[key] = float(scores[key]) - float(reference[key])
        except (KeyError, TypeError, ValueError):
            out[key] = None
    return out


def cmd_report(args):
    ledger = load_ledger()
    dataset = args.dataset
    name = args.experiment
    entry = ledger["datasets"][dataset]["experiments"].get(name)
    if not entry:
        raise SystemExit(f"{name}: unknown experiment")
    source = ledger["datasets"][dataset]["experiments"].get(entry.get("reuse_from") or "")
    state = load_state(name)
    scores = entry.get("official_scores") or (source or {}).get("official_scores") or {}
    best = ledger["datasets"][dataset]["official_best"]
    base = ledger["datasets"][dataset]["experiments"].get(entry.get("base_experiment") or "")
    base_scores = (base or {}).get("official_scores") or {}
    checkpoint = ((state.get("results", {}).get("train") or {}).get("checkpoint")
                  or entry.get("checkpoint") or (source or {}).get("checkpoint"))
    zip_path = ((state.get("results", {}).get("package") or {}).get("zip")
                or (entry.get("artifacts") or {}).get("zip")
                or (source or {}).get("artifacts", {}).get("zip"))
    gates_path = ((entry.get("artifacts") or {}).get("gates")
                  or (source or {}).get("artifacts", {}).get("gates"))
    gates = _gate_stats(gates_path)
    delta_best = _score_delta(scores, {"HOTA": best["hota"]})
    delta_base = _score_delta(scores, base_scores) if base_scores else {}
    suggestions = []
    if not scores:
        suggestions.append("No official score yet; finish the submit stage.")
    else:
        hota_delta = delta_best.get("HOTA")
        if hota_delta is None:
            suggestions.append("No HOTA comparison available.")
        elif hota_delta >= 0.05:
            suggestions.append("Improves the official best; consider promoting after a seed check.")
        elif hota_delta > 0:
            suggestions.append("Slight improvement over best; verify with another seed before promoting.")
        elif hota_delta > -0.05:
            suggestions.append("Within noise of the official best; keep the best weights unless another metric matters.")
        else:
            suggestions.append("Below the official best; keep the best weights.")
        if (delta_base.get("IDSW") is not None and delta_base["IDSW"] < 0
                and (delta_base.get("HOTA") or 0) < 0):
            suggestions.append("IDSW improved while HOTA dropped; inspect AssA/DetA tradeoffs.")
    if entry.get("reuse_from"):
        suggestions.append(f"Reused historical weights from {entry['reuse_from']}; no new training.")
    report = {"experiment": name, "dataset": dataset,
              "config": entry.get("config") or experiment_params(entry),
              "base_experiment": entry.get("base_experiment"),
              "change_reason": entry.get("change_reason"),
              "config_hash": entry.get("config_hash"),
              "reuse_from": entry.get("reuse_from"),
              "status": entry.get("status"),
              "checkpoint": checkpoint,
              "checkpoint_sha256": sha256(ROOT / checkpoint) if checkpoint and (ROOT / checkpoint).is_file() else None,
              "zip": zip_path,
              "zip_sha256": sha256(ROOT / zip_path) if zip_path and (ROOT / zip_path).is_file() else None,
              "official_scores": scores, "official_best": best,
              "base_scores": base_scores,
              "delta_vs_best": delta_best, "delta_vs_base": delta_base,
              "gates": gates, "state": state, "suggestions": suggestions}
    out_dir = experiment_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [f"# {dataset} experiment: {name}", "",
             f"- base: {entry.get('base_experiment')}  |  status: {entry.get('status')}"
             f"  |  reuse_from: {entry.get('reuse_from')}",
             f"- config: `{json.dumps(report['config'], ensure_ascii=False)}`",
             f"- reason: {entry.get('change_reason') or '-'}",
             f"- checkpoint: `{checkpoint}` (sha256 {str(report['checkpoint_sha256'])[:16]}…)",
             f"- zip: `{zip_path}` (sha256 {str(report['zip_sha256'])[:16]}…)",
             "", "## Official scores", "",
             "| metric | value | vs best | vs base |",
             "|---|---:|---:|---:|"]
    for key in SCORE_KEYS:
        value = scores.get(key)
        def fmt(delta):
            return "-" if delta is None else f"{delta:+.3f}"
        lines.append(f"| {key} | {value if value is not None else '-'} | "
                     f"{fmt(delta_best.get(key))} | {fmt((delta_base or {}).get(key))} |")
    lines += ["", f"Official best HOTA: {best['hota']:.3f} ({best['experiment']}, "
                  f"submission {best['submission_id']}).", ""]
    if gates:
        lines += ["## Gates", "",
                  f"- motion min/mean/max: {gates['motion_min']:.4f}/{gates['motion_mean']:.4f}/"
                  f"{gates['motion_max']:.4f}",
                  f"- appearance min/mean/max: {gates['appearance_min']:.4f}/"
                  f"{gates['appearance_mean']:.4f}/{gates['appearance_max']:.4f}", ""]
    lines += ["## Commands and stages", ""]
    for stage, info in state.get("stages", {}).items():
        command = (" ".join(info.get("command", []))
                   if isinstance(info.get("command"), list) else "")
        reused = info.get("reused_from")
        detail = f"reused_from={reused}" if reused and not command else command
        lines.append(f"- {stage}: {info.get('status')} "
                     f"({info.get('seconds', '-') }s) `{detail}`")
    lines += ["", "## Suggestions", ""] + [f"- {item}" for item in suggestions]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    entry["report"] = {"md": str(out_dir / "report.md"), "json": str(out_dir / "report.json")}
    save_ledger(ledger)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


# ----------------------------------------------------------------- main -----
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show ledger and experiment status")
    status.add_argument("--dataset", choices=("MOT20", "MOT17"))
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    verify = sub.add_parser("verify", help="check ledger artifact paths")
    verify.set_defaults(func=cmd_verify)

    train = sub.add_parser("train", help="build/run a training stage")
    train.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    train.add_argument("--experiment", required=True)
    train.add_argument("--mode", choices=("scratch", "finetune", "continue"), required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--shards", type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--init")
    train.add_argument("--source")
    train.add_argument("--allow-partial-cycles", action="store_true")
    train.add_argument("--execute", action="store_true")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="val-raw evaluation for one or more checkpoints")
    ev.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    ev.add_argument("--experiment", required=True)
    ev.add_argument("--checkpoint")
    ev.add_argument("--checkpoint-dir")
    ev.add_argument("--epochs", default="")
    ev.add_argument("--execute", action="store_true")
    ev.set_defaults(func=cmd_eval)

    test = sub.add_parser("test", help="test inference + post")
    test.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    test.add_argument("--experiment", required=True)
    test.add_argument("--checkpoint")
    test.add_argument("--force", action="store_true", help="rerun even if the post dir is complete")
    test.add_argument("--execute", action="store_true")
    test.set_defaults(func=cmd_test)

    pkg = sub.add_parser("package", help="package + local validation")
    pkg.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    pkg.add_argument("--experiment", required=True)
    pkg.add_argument("--post-dir")
    pkg.add_argument("--force", action="store_true", help="repackage even if already validated")
    pkg.add_argument("--execute", action="store_true")
    pkg.set_defaults(func=cmd_package)

    subp = sub.add_parser("submit", help="submit once with account fallback")
    subp.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    subp.add_argument("--experiment", required=True)
    subp.add_argument("--zip")
    subp.add_argument("--execute", action="store_true")
    subp.set_defaults(func=cmd_submit)

    rec = sub.add_parser("record", help="merge a saved submission JSON into the ledger")
    rec.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    rec.add_argument("--experiment", required=True)
    rec.add_argument("--submission-json", required=True)
    rec.add_argument("--account")
    rec.set_defaults(func=cmd_record)

    prop = sub.add_parser("propose", help="select candidate parameters from a base recipe")
    prop.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    prop.add_argument("--experiment", required=True)
    prop.add_argument("--base", required=True)
    prop.add_argument("--set", nargs="*", default=[])
    prop.add_argument("--reason", required=True)
    prop.set_defaults(func=cmd_propose)

    runner = sub.add_parser("run", help="run or reuse the stages of a proposed experiment")
    runner.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    runner.add_argument("--experiment", required=True)
    runner.add_argument("--stages", default="train,test,package,submit")
    runner.add_argument("--execute", action="store_true")
    runner.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="write report.md/json for an experiment")
    rep.add_argument("--dataset", choices=("MOT20", "MOT17"), required=True)
    rep.add_argument("--experiment", required=True)
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    # Defaults that mirror the verified recipes.
    if args.command == "train":
        defaults = {
            "MOT20": {"scratch": (100, 4, 1e-4), "continue": (50, 4, 1e-5),
                      "finetune": (50, 4, 1e-5)},
            "MOT17": {"finetune": (50, 1, 1e-5), "continue": (50, 1, 1e-6),
                      "scratch": (100, 4, 1e-4)},
        }[args.dataset][args.mode]
        if args.epochs is None:
            args.epochs = defaults[0]
        if args.shards is None:
            args.shards = defaults[1]
        if args.lr is None:
            args.lr = defaults[2]
        if args.mode == "finetune" and not args.init and args.dataset == "MOT17":
            args.init = ("outputs/agentguard/experiments/shutrack_architecture_night_20261120/"
                         "final/MOT20/groups/appearance_row_column/checkpoints/epoch_100.pt")
        if args.mode == "continue" and not args.source:
            args.source = ("outputs/agentguard/experiments/shutrack_architecture_night_20261120/"
                           f"final/{args.dataset}/groups/appearance_row_column/checkpoints/"
                           + ("epoch_100.pt" if args.dataset == "MOT20" else "epoch_050.pt"))
        if args.dataset == "MOT20" and args.mode == "continue":
            args.allow_partial_cycles = True
    args.func(args)


if __name__ == "__main__":
    main()
