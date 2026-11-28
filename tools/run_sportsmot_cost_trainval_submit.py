#!/usr/bin/env python3
"""Formal SportsMOT trainval training + test inference + one Codabench submission.

Protocol: query_film_bilinear, 90 official trainval sequences, from scratch,
seed 42, 200 shard-epochs, motion cost supervision
(L_DIoU + 0.25 * weighted BCE), appearance rank+column losses unchanged.
The schedule (sequence list, quotas, normalization, total_steps, warmup) is
recomputed by the trainer from the 90-sequence data; the smoke stage asserts
the 90-sequence schedule (37000 steps / 185 warmup) instead of the 45-seq one.

Stages (serial, resumable):
  preflight -> smoke -> train (200) -> test (raw+post) -> package -> submit -> report.
The submit stage runs only after local packaging/validation and saves the
submission record; it is authorized for exactly this one result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
CACHE = ROOT / "outputs/agentguard/detection_cache"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_sportsmot_cost_trainval_formal_20261122/SportsMOT"
CONFIG = EXP / "config.json"
TARGETS = ROOT / "outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT/targets"
RANKS = ROOT / "outputs/agentguard/experiments/shutrack_motion_rank_20261122/SportsMOT/rank_labels"
DATA90 = ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data"
NORM_CKPT = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120/final/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"
PRIOR_RECORD = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120/submissions/SportsMOT_submission.json"
TRAIN_TOOL = ROOT / "tools/train_motion_rank.py"
VERIFY_TOOL = ROOT / "tools/verify_scratch_init.py"
MAKE_PACKAGE = ROOT / "tools/package_tracking_results.py"
SUBMIT_TOOL = ROOT / "tools/codabench_submit.py"
SM_SEQMAPP = TRACKER / "trackeval/seqmap/sportsmot/test.txt"
SM_SEQS = tuple(line.strip() for line in SM_SEQMAPP.read_text().splitlines()
                if line.strip() and line.strip() != "name")
SM_LIST = EXP / "logs/sportsmot_test_sequences.txt"
SUBDIR = EXP / "submissions"
GROUP = "appearance_row_column"
EPOCHS = 200
EXPECTED_STEPS = 37000
EXPECTED_WARMUP = 185
HEAVY = {"train_motion_rank.py", "train_supervision_groups.py", "run.py",
         "verify_scratch_init.py", "generate_trainval_paired.py"}
STATE_PATH = EXP / "state.json"
LOGS = EXP / "logs"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_free_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True).splitlines()[0]
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
        SUBDIR.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
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
            print(f"{now()} waiting before {label}: jobs={jobs} gpu_free={free:.0f}MiB "
                  f"mem_available={avail:.0f}MiB", flush=True)
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

    def group_dir(self, root):
        return root / GROUP

    def preflight(self):
        out = EXP / "preflight_scratch.json"
        if out.is_file():
            payload = json.loads(out.read_text())
            if (payload.get("init", {}).get("equal")
                    and payload.get("init", {}).get("optimizer_covers_all")
                    and all(payload.get("binding", {}).values())):
                self.state["preflight"] = {"reused": True,
                                           "init_hash": payload["init"]["hash"]}
                self.save()
                return payload
        command = [str(PY), "-u", str(VERIFY_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                   "--norm-checkpoint", str(NORM_CKPT), "--rows", "1024",
                   "--steps", "10", "--warmup", "3", "--seed", "42",
                   "--device", "cuda", "--out", str(out)]
        self.run("preflight_scratch", command)
        payload = json.loads(out.read_text())
        self.state["preflight"] = {"init_hash": payload["init"]["hash"],
                                   "init_equal": payload["init"]["equal"]}
        self.save()
        return payload

    def schedule_from_log(self, log_path):
        for line in Path(log_path).read_text().splitlines():
            line = line.strip()
            if line.startswith("{") and "\"start\"" in line:
                payload = json.loads(line)
                return {
                    "sequences": payload.get("sequences"),
                    "total_steps": payload.get("total_steps"),
                    "warmup_steps": payload.get("warmup_steps"),
                    "sequence_counts": len(payload.get("sequence_counts") or {}),
                    "data_root": payload.get("data_root"),
                    "split": payload.get("split"),
                }
        raise RuntimeError(f"no start config in {log_path}")

    def smoke(self):
        root = EXP / "smoke/groups"
        checkpoint = self.group_dir(root) / "checkpoints/last.pt"
        if not checkpoint.is_file():
            command = [str(PY), "-u", str(TRAIN_TOOL), "--config", str(CONFIG),
                       "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                       "--groups-root", str(root), "--base-lambda", "0.25",
                       "--rank-weight", "0", "--pair-seed", "20261122",
                       "--grad-every", "10", "--stop-after-epoch", "1"]
            self.run("smoke_train", command)
        schedule = self.schedule_from_log(self.group_dir(root) / "training.log")
        self.state["smoke_schedule"] = schedule
        self.save()
        if (schedule["sequences"] != 90 or schedule["sequence_counts"] != 90
                or schedule["total_steps"] != EXPECTED_STEPS
                or schedule["warmup_steps"] != EXPECTED_WARMUP):
            raise RuntimeError(f"90-sequence schedule mismatch: {schedule}")
        return schedule

    def train(self):
        root = EXP / "runs/groups"
        checkpoint = self.group_dir(root) / "checkpoints/last.pt"
        target = self.group_dir(root) / f"checkpoints/epoch_{EPOCHS:03d}.pt"
        if checkpoint.is_file() and target.is_file():
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", 0)) >= EPOCHS:
                self.state["results"]["train"] = {"checkpoint": str(target),
                                                  "reused": True}
                self.save()
                return target
        command = [str(PY), "-u", str(TRAIN_TOOL), "--config", str(CONFIG),
                   "--target-root", str(TARGETS), "--rank-root", str(RANKS),
                   "--groups-root", str(root), "--base-lambda", "0.25",
                   "--rank-weight", "0", "--pair-seed", "20261122",
                   "--grad-every", "10"]
        if checkpoint.is_file():
            command.append("--resume")
        self.run("train_formal", command)
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != EPOCHS or not target.is_file():
            raise RuntimeError("formal training did not finish at epoch 200")
        config = payload.get("model_config") or {}
        if config.get("architecture_variant") != "query_film_bilinear" or \
                int(config.get("appearance_proj_dim", 0)) != 256:
            raise RuntimeError(f"unexpected model config: {config}")
        self.state["results"]["train"] = {
            "checkpoint": str(target), "sha256": sha256(target),
            "epoch": payload.get("epoch"), "global_step": payload.get("global_step")}
        self.save()
        return target

    def test(self, checkpoint):
        tracking = EXP / "test/tracking"
        if not tracking.is_dir():
            tracking.mkdir(parents=True)
        posts = [p for p in tracking.iterdir() if p.is_dir() and p.name.endswith("_post")]
        raws = [p for p in tracking.iterdir() if p.is_dir() and not p.name.endswith("_post")]
        expected_names = set(SM_SEQS)
        if len(posts) == 1 and len(raws) == 1 \
                and len(list(posts[0].glob("*.txt"))) == len(SM_SEQS) \
                and len(list(raws[0].glob("*.txt"))) == len(SM_SEQS):
            self.state["results"]["test"] = {"post_dir": str(posts[0]),
                                             "raw_dir": str(raws[0]), "reused": True}
            self.save()
        else:
            gates_log = EXP / "test/gates.jsonl"
            if gates_log.is_file():
                gates_log.replace(gates_log.with_name(
                    f"gates.jsonl.prev.{time.strftime('%Y%m%dT%H%M%S')}"))
            command = [str(PY), "-u", "run.py", "--dataset", "SportsMOT", "--mode", "test",
                       "--sequences", *SM_SEQS, "--seed", "10000", "--kf-type", "nsa",
                       "--agentguard-mode", "shutrack-v2",
                       "--shutrack-checkpoint", str(checkpoint),
                       "--agentguard-device", "cpu",
                       "--detection-cache-root", str(CACHE),
                       "--output_dir", str(tracking), "--tracker-suffix", "costtrainval",
                       "--agentguard-stats-output", str(EXP / "test/gates.jsonl"),
                       "--use_post", "--skip-eval",
                       "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
            self.run("test_inference", command, cwd=TRACKER)
            posts = [p for p in tracking.iterdir() if p.is_dir() and p.name.endswith("_post")]
            raws = [p for p in tracking.iterdir() if p.is_dir() and not p.name.endswith("_post")]
            if len(posts) != 1 or len(raws) != 1:
                raise RuntimeError(f"unexpected test outputs: raw={raws} post={posts}")
            self.state["results"]["test"] = {"post_dir": str(posts[0]),
                                             "raw_dir": str(raws[0])}
            self.save()
        post, raw = posts[0], raws[0]
        post_names = {p.stem for p in post.glob("*.txt")}
        raw_names = {p.stem for p in raw.glob("*.txt")}
        if post_names != expected_names or raw_names != expected_names:
            raise RuntimeError(f"sequence set mismatch: post_missing={sorted(expected_names - post_names)[:5]} "
                               f"raw_missing={sorted(expected_names - raw_names)[:5]}")
        stats = [json.loads(line) for line in (EXP / "test/gates.jsonl").read_text().splitlines()
                 if line.strip()]
        if len(stats) != len(SM_SEQS):
            raise RuntimeError(f"gate stats rows {len(stats)} != {len(SM_SEQS)}")
        motion = [row["shutrack_avg_motion_gate"] for row in stats]
        appearance = [row["shutrack_avg_appearance_gate"] for row in stats]
        gate_summary = {
            "rows": len(stats),
            "motion_min": min(motion), "motion_mean": sum(motion) / len(motion),
            "motion_max": max(motion),
            "appearance_min": min(appearance), "appearance_mean": sum(appearance) / len(appearance),
            "appearance_max": max(appearance),
            "motion_varied": min(motion) < max(motion) and max(motion) < 1.0,
        }
        if not gate_summary["motion_varied"]:
            raise RuntimeError(f"motion gate does not look enabled: {gate_summary}")
        self.state["results"]["gates"] = gate_summary
        self.save()
        return {"post_dir": post, "raw_dir": raw, "names": len(post_names),
                "gates": gate_summary}

    def package(self, post_dir):
        zip_path = SUBDIR / "SportsMOT.zip"
        package_log = SUBDIR / "SportsMOT_package.log"
        command = [str(PY), str(MAKE_PACKAGE), "--source", str(post_dir),
                   "--zip", str(zip_path), "--expected-count", str(len(SM_SEQS)),
                   "--expected-names-file", str(SM_LIST)]
        with package_log.open("wb") as handle:
            code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError("packaging failed")
        validate_log = SUBDIR / "SportsMOT_validate.log"
        validate = [str(PY), str(SUBMIT_TOOL), "validate", "--zip", str(zip_path),
                    "--dataset", "sportsmot",
                    "--seqinfo-root", "/home/shang/datasets/SportsMOT/dataset/test",
                    "--sequences-file", str(SM_LIST)]
        with validate_log.open("wb") as handle:
            code = subprocess.call(validate, stdout=handle, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError("local zip validation failed")
        self.state["results"]["package"] = {
            "zip": str(zip_path), "sha256": sha256(zip_path),
            "package_log": str(package_log), "validate_log": str(validate_log)}
        self.save()
        return zip_path

    def _api_get(self, path):
        token = (Path.home() / ".config/codabench/token").read_text().strip()
        for _ in range(5):
            result = subprocess.run(
                ["curl", "-sS", "--max-time", "30", "-H", f"Authorization: Token {token}",
                 f"https://www.codabench.org{path}"],
                capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass
            time.sleep(3)
        raise RuntimeError(f"Codabench API unavailable: {path}")

    def _phase_submissions(self):
        payload = self._api_get("/api/submissions/?phase=21402")
        return payload if isinstance(payload, list) else payload.get("results", [])

    def _latest_submission_id(self):
        ids = [int(item["id"]) for item in self._phase_submissions() if item.get("id")]
        return max(ids) if ids else 0

    def _poll_submission(self, submission_id, record_path, timeout=2400):
        deadline = time.time() + timeout
        non_terminal = ("Submitting", "Running", "Preparing", "Queued", "Pending")
        while time.time() < deadline:
            payload = self._api_get(f"/api/submissions/{submission_id}/")
            status = payload.get("status")
            if status not in non_terminal:
                record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return payload
            print(f"{now()} submission {submission_id} status={status}", flush=True)
            time.sleep(30)
        raise RuntimeError(f"submission {submission_id} still running after {timeout}s")

    def submit(self, zip_path):
        record = SUBDIR / "SportsMOT_submission.json"
        if record.is_file():
            payload = json.loads(record.read_text())
            if payload.get("id") is not None:
                status = payload.get("status")
                if status in ("Submitting", "Running", "Preparing", "Queued", "Pending"):
                    payload = self._poll_submission(payload["id"], record)
                self.state["results"]["submission"] = {
                    "submission_id": payload["id"], "status": payload.get("status"),
                    "scores": payload.get("scores"), "record": str(record), "reused": True}
                self.save()
                return payload
        baseline = self.state.get("submit_baseline_id")
        if baseline is None:
            baseline = self._latest_submission_id()
            self.state["submit_baseline_id"] = baseline
            self.save()
        else:
            fresh = [item for item in self._phase_submissions()
                     if int(item.get("id", 0)) > int(baseline)]
            if fresh:
                submission_id = max(int(item["id"]) for item in fresh)
                payload = self._poll_submission(submission_id, record)
                self.state["results"]["submission"] = {
                    "submission_id": submission_id, "status": payload.get("status"),
                    "scores": payload.get("scores"), "record": str(record),
                    "recovered_after_error": True}
                self.save()
                return payload
        last_error = None
        for attempt in range(3):
            name = f"shutrack_sportsmot_cost_trainval_e200_{time.strftime('%Y%m%dT%H%M%S')}_{attempt + 1}"
            command = [str(PY), str(SUBMIT_TOOL), "submit", "--zip", str(zip_path),
                       "--competition", "13077", "--phase", "21402", "--task", "26486",
                       "--name", name, "--dataset", "sportsmot",
                       "--seqinfo-root", "/home/shang/datasets/SportsMOT/dataset/test",
                       "--sequences-file", str(SM_LIST),
                       "--poll", "--poll-interval", "30", "--poll-timeout", "2400",
                       "--save-record", str(record)]
            try:
                self.run(f"codabench_submit_attempt{attempt + 1}", command)
            except RuntimeError as exc:
                last_error = exc
            if record.is_file():
                payload = json.loads(record.read_text())
                if payload.get("id") is not None:
                    if payload.get("status") in ("Submitting", "Running", "Preparing",
                                                 "Queued", "Pending"):
                        payload = self._poll_submission(payload["id"], record)
                    self.state["results"]["submission"] = {
                        "submission_id": payload["id"], "status": payload.get("status"),
                        "scores": payload.get("scores"), "record": str(record)}
                    self.save()
                    return payload
            fresh = [item for item in self._phase_submissions()
                     if int(item.get("id", 0)) > baseline]
            if fresh:
                submission_id = max(int(item["id"]) for item in fresh)
                payload = self._poll_submission(submission_id, record)
                self.state["results"]["submission"] = {
                    "submission_id": submission_id, "status": payload.get("status"),
                    "scores": payload.get("scores"), "record": str(record),
                    "recovered_after_error": True}
                self.save()
                return payload
            time.sleep(15)
        raise RuntimeError(f"submission failed after retries: {last_error}")


def report(driver):
    prior = json.loads(PRIOR_RECORD.read_text()) if PRIOR_RECORD.is_file() else {}
    record_path = SUBDIR / "SportsMOT_submission.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}

    def score_dict(scores):
        if isinstance(scores, dict):
            return scores
        if isinstance(scores, list):
            return {item.get("column_key"): item.get("score") for item in scores
                    if isinstance(item, dict)}
        return {}

    prior_scores = score_dict(prior.get("scores"))
    new_scores = score_dict(record.get("scores"))
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")
    training_metrics = []
    metrics_path = driver.group_dir(EXP / "runs/groups") / "metrics.jsonl"
    if metrics_path.is_file():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        training_metrics = [{"epoch": r["epoch"], "epoch_seconds": r.get("epoch_seconds"),
                             "loss": r.get("loss"), "motion_loss": r.get("motion_loss"),
                             "appearance_loss": r.get("appearance_loss"),
                             "motion_gate_mean": r.get("motion_gate_mean"),
                             "appearance_gate_mean": r.get("appearance_gate_mean")}
                            for r in rows]
    payload = {
        "prior_submission": {"id": prior.get("id"), "scores": prior_scores},
        "new_submission": {"id": record.get("id"), "status": record.get("status"),
                           "scores": new_scores},
        "deltas": {key: ((float(new_scores[key]) - float(prior_scores[key]))
                         if new_scores.get(key) is not None and prior_scores.get(key) is not None
                         else None) for key in keys},
        "test": driver.state["results"].get("test"),
        "gates": driver.state["results"].get("gates"),
        "package": driver.state["results"].get("package"),
        "train": driver.state["results"].get("train"),
        "smoke_schedule": driver.state.get("smoke_schedule"),
        "training_metrics": training_metrics,
    }
    (EXP / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# SportsMOT 正式 trainval：代价监督 200 轮 + test 提交", "",
             "协议：query_film_bilinear，90 条官方 trainval，from_scratch，seed 42，200 分片轮；"
             "运动损失 = GT DIoU + 0.25×错误代价加权 BCE；外观 rank+column 权重不变。", "",
             "## 训练调度核验", "",
             f"- 冒烟日志：sequences={driver.state.get('smoke_schedule', {}).get('sequences')}，"
             f"sequence_counts={driver.state.get('smoke_schedule', {}).get('sequence_counts')}，"
             f"total_steps={driver.state.get('smoke_schedule', {}).get('total_steps')}，"
             f"warmup_steps={driver.state.get('smoke_schedule', {}).get('warmup_steps')}。",
             "- 未沿用 45 条实验的 18800/94，训练器在 90 条数据上重新计算为 37000/185。", "",
             "## 官方 test 分数", "",
             "| 指标 | 上一版新架构（75.727） | 本轮 | 变化 |",
             "|---|---:|---:|---:|"]
    for key in keys:
        old = prior_scores.get(key)
        new = new_scores.get(key)
        delta = payload["deltas"][key]
        integer = key in ("IDSW", "Frag")
        def f(v):
            if v is None:
                return "-"
            return f"{float(v):.0f}" if integer else f"{float(v):.3f}"
        lines.append(f"| {key} | {f(old)} | {f(new)} | "
                     f"{'-' if delta is None else (f'{delta:+.0f}' if integer else f'{delta:+.3f}')} |")
    test = driver.state["results"].get("test") or {}
    gates = driver.state["results"].get("gates") or {}
    package = driver.state["results"].get("package") or {}
    train = driver.state["results"].get("train") or {}
    lines += ["", "## 产物", "",
              f"- checkpoint: `{train.get('checkpoint')}`",
              f"- raw: `{test.get('raw_dir')}`",
              f"- post: `{test.get('post_dir')}`",
              f"- ZIP: `{package.get('zip')}` sha256 `{str(package.get('sha256'))[:16]}…`",
              f"- 提交记录: `{SUBDIR / 'SportsMOT_submission.json'}`", "",
              "## 门控与序列核验", "",
              f"- test 序列数：{gates.get('rows', len(SM_SEQS))}（预期 {len(SM_SEQS)}）。",
              f"- 运动 gate 统计：min {gates.get('motion_min'):.4f} / mean "
              f"{gates.get('motion_mean'):.4f} / max {gates.get('motion_max'):.4f}；"
              f"外观 gate：min {gates.get('appearance_min'):.4f} / mean "
              f"{gates.get('appearance_mean'):.4f} / max {gates.get('appearance_max'):.4f}。",
              "", "## 训练成本", ""]
    if training_metrics:
        total = sum(r["epoch_seconds"] or 0 for r in training_metrics)
        lines.append(f"- 200 轮总训练时间（各轮墙钟求和）：{total:.1f}s；"
                     f"平均 {total/len(training_metrics):.2f}s/轮。")
        last = training_metrics[-1]
        lines.append(f"- 末轮：loss {last['loss']:.6f}，motion {last['motion_loss']:.6f}，"
                     f"appearance {last['appearance_loss']:.6f}。")
    (EXP / "report.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "preflight", "smoke", "train",
                                            "test", "package", "submit", "report"),
                        default="all")
    args = parser.parse_args()
    driver = Driver()
    if not CONFIG.is_file():
        raise SystemExit(f"missing config {CONFIG}")
    SM_LIST.parent.mkdir(parents=True, exist_ok=True)
    SM_LIST.write_text("\n".join(SM_SEQS) + "\n")
    if args.stage in ("all", "preflight"):
        payload = driver.preflight()
        print(json.dumps({"init": payload.get("init"),
                          "loss": payload.get("loss_on_real_batch")}, indent=2), flush=True)
    if args.stage in ("all", "smoke"):
        print(json.dumps({"smoke_schedule": driver.smoke()}, indent=2), flush=True)
    if args.stage in ("all", "train"):
        checkpoint = driver.train()
        print(json.dumps({"checkpoint": str(checkpoint)}, indent=2), flush=True)
    if args.stage in ("all", "test"):
        test = driver.test(driver.state["results"]["train"]["checkpoint"])
        print(json.dumps({"test": {"names": test["names"],
                                   "post_dir": str(test["post_dir"]),
                                   "gates": test["gates"]}}, indent=2), flush=True)
    zip_path = SUBDIR / "SportsMOT.zip"
    if args.stage == "package" or (args.stage == "all"
                                   and "package" not in driver.state["results"]):
        driver.package(driver.state["results"]["test"]["post_dir"])
    if args.stage in ("all", "submit"):
        record = driver.submit(zip_path)
        print(json.dumps({"submission": {"id": record.get("id"),
                                         "status": record.get("status"),
                                         "scores": record.get("scores")}}, indent=2), flush=True)
    if args.stage in ("all", "report"):
        report(driver)


if __name__ == "__main__":
    main()
