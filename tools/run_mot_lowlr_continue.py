#!/usr/bin/env python3
"""MOT20 e80 test and MOT17 retrained from MOT20 e80 to 50 epochs, then test+submit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
CACHE = ROOT / "outputs/agentguard/detection_cache"
NIGHT = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
FINAL = NIGHT / "final"
EXP = ROOT / "outputs/agentguard/experiments/mot_lowlr_continue_20261123"
TRAIN_EXP = EXP / "mot17_from_mot20_e80"
TRAIN_GROUP = "appearance_row_column"
MOT20_E80 = FINAL / "MOT20/groups/appearance_row_column/checkpoints/epoch_080.pt"
MOT17_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT17/data"
SUBDIR = EXP / "submissions"
LOGS = EXP / "logs"
MAKE_PACKAGE = ROOT / "tools/package_tracking_results.py"
SUBMIT_TOOL = ROOT / "tools/codabench_submit.py"
TRAIN_TOOL = ROOT / "tools/train_supervision_groups.py"
SM_SEQMAPP = TRACKER / "trackeval/seqmap/sportsmot/test.txt"
MOT17_SEQS = ("MOT17-01-FRCNN", "MOT17-03-FRCNN", "MOT17-06-FRCNN", "MOT17-07-FRCNN",
              "MOT17-08-FRCNN", "MOT17-12-FRCNN", "MOT17-14-FRCNN")
MOT20_SEQS = ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08")
TOKENS = (("jojo0", Path.home() / ".config/codabench/token"),
          ("shangqingnie", Path.home() / ".config/codabench/token_edge"))
NON_TERMINAL = ("Submitting", "Running", "Preparing", "Queued", "Pending", "Submitted", "Scoring")
HEAVY = {"run.py", "train_supervision_groups.py", "train_motion_rank.py", "train_stop_after.py"}
STATE_PATH = EXP / "state.json"

DATASETS = {
    "MOT20": {
        "checkpoint": EXP / "mot20/groups/appearance_row_column/checkpoints/epoch_050.pt",
        "epoch": 50, "sequences": MOT20_SEQS, "expected": 4,
        "seqinfo": Path("/home/shang/datasets/MOT20/test"),
        "competition": "10050", "phase": "16383", "task": "20804",
        "dataset_flag": "mot20", "previous_hota": 66.637,
    },
    "MOT17": {
        "checkpoint": EXP / "mot17/groups/appearance_row_column/checkpoints/epoch_050.pt",
        "epoch": 50, "sequences": MOT17_SEQS, "expected": 21,
        "seqinfo": Path("/home/shang/datasets/MOT17/test"),
        "competition": "10049", "phase": "16382", "task": "20803",
        "dataset_flag": "mot17", "previous_hota": 67.827,
    },
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        for path in (EXP, SUBDIR, LOGS):
            path.mkdir(parents=True, exist_ok=True)
        self.state = (json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file()
                      else {"created_at": now(), "stages": {}, "results": {}})

    def save(self):
        tmp = STATE_PATH.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        tmp.replace(STATE_PATH)

    def wait_jobs(self, label):
        while True:
            jobs = active_jobs()
            if not jobs:
                return
            print(f"{now()} waiting before {label}: jobs={jobs}", flush=True)
            time.sleep(30)

    def run(self, label, command, cwd=ROOT):
        self.wait_jobs(label)
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

    def checkpoint_info(self, name, spec):
        import torch
        path = spec["checkpoint"]
        if not path.is_file():
            raise RuntimeError(f"{name}: missing checkpoint {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("model_config") or {}
        if config.get("architecture_variant") != "query_film_bilinear":
            raise RuntimeError(f"{name}: wrong architecture {config.get('architecture_variant')}")
        if int(config.get("appearance_proj_dim", 0)) != 256:
            raise RuntimeError(f"{name}: wrong projection {config.get('appearance_proj_dim')}")
        if int(payload.get("epoch", -1)) != spec["epoch"]:
            raise RuntimeError(f"{name}: epoch {payload.get('epoch')} != {spec['epoch']}")
        return {"path": str(path), "sha256": sha256(path), "epoch": payload["epoch"],
                "global_step": payload.get("global_step"), "model_config": config}

    def train_mot17(self):
        group_dir = TRAIN_EXP / "groups" / TRAIN_GROUP
        checkpoint = group_dir / "checkpoints/last.pt"
        target = group_dir / "checkpoints/epoch_050.pt"
        if checkpoint.is_file() and target.is_file():
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", 0)) >= 50:
                self.state["results"].setdefault("MOT17", {})["train"] = {
                    "checkpoint": str(target), "reused": True}
                self.save()
                return target
        command = [str(PY), "-u", str(TRAIN_TOOL),
                   "--group", TRAIN_GROUP, "--dataset", "MOT17",
                   "--init", str(MOT20_E80), "--split", "train",
                   "--data-root", str(MOT17_DATA),
                   "--groups-root", str(TRAIN_EXP / "groups"),
                   "--batch-size", "1024", "--lr", "1e-5", "--weight-decay", "1e-4",
                   "--seed", "42", "--shards", "1", "--max-epochs", "50",
                   "--snapshot-every", "20", "--rank-weight", "18",
                   "--rank-alpha", "0.95", "--rank-tau", "0.1",
                   "--column-weight", "50", "--device", "cuda"]
        if checkpoint.is_file():
            command.append("--resume")
        self.run("train_mot17_from_mot20e80", command)
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != 50 or not target.is_file():
            raise RuntimeError("MOT17 retrain did not finish at epoch 50")
        self.state["results"].setdefault("MOT17", {})["train"] = {
            "checkpoint": str(target), "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step")}
        self.save()
        return target

    def tracking_and_post(self, name, spec):
        out_dir = EXP / name / "test/tracking"
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [str(PY), "-u", "run.py", "--dataset", name, "--mode", "test",
                   "--sequences", *spec["sequences"], "--seed", "10000", "--kf-type", "nsa",
                   "--agentguard-mode", "shutrack-v2",
                   "--shutrack-checkpoint", str(spec["checkpoint"]),
                   "--agentguard-device", "cpu", "--detection-cache-root", str(CACHE),
                   "--output_dir", str(out_dir), "--tracker-suffix", f"midpoint_e{spec['epoch']}",
                   "--agentguard-stats-output", str(EXP / name / "test/gates.jsonl"),
                   "--use_post", "--skip-eval",
                   "--torch-num-threads", "4", "--torch-num-interop-threads", "1"]
        self.run(f"test_{name}", command, cwd=TRACKER)
        posts = [p for p in out_dir.iterdir() if p.is_dir() and p.name.endswith("_post")]
        if len(posts) != 1:
            raise RuntimeError(f"{name}: expected one post dir, found {posts}")
        post = posts[0]
        if name == "MOT17":
            frcnn = sorted(post.glob("*-FRCNN.txt"))
            if len(frcnn) != 7:
                raise RuntimeError(f"MOT17: expected 7 FRCNN txt, found {len(frcnn)}")
            for source in frcnn:
                stem = source.name[: -len("-FRCNN.txt")]
                for detector in ("DPM", "SDP"):
                    shutil.copyfile(source, post / f"{stem}-{detector}.txt")
        txts = sorted(post.glob("*.txt"))
        if len(txts) != spec["expected"]:
            raise RuntimeError(f"{name}: expected {spec['expected']} txt, found {len(txts)}")
        gates = [json.loads(line) for line in (EXP / name / "test/gates.jsonl").read_text().splitlines()
                 if line.strip()]
        motion = [r["shutrack_avg_motion_gate"] for r in gates if r.get("shutrack_avg_motion_gate") is not None]
        appearance = [r["shutrack_avg_appearance_gate"] for r in gates if r.get("shutrack_avg_appearance_gate") is not None]
        if len(gates) != len(spec["sequences"]):
            raise RuntimeError(f"{name}: gate rows {len(gates)} != {len(spec['sequences'])}")
        result = {"post_dir": str(post), "files": len(txts),
                  "gates": {"rows": len(gates),
                            "motion_min": min(motion), "motion_mean": sum(motion) / len(motion),
                            "motion_max": max(motion),
                            "appearance_min": min(appearance),
                            "appearance_mean": sum(appearance) / len(appearance),
                            "appearance_max": max(appearance)}}
        self.state["results"].setdefault(name, {})["test"] = result
        self.save()
        return post

    def package(self, name, spec, post_dir):
        zip_path = SUBDIR / f"{name}.zip"
        package_log = SUBDIR / f"{name}_package.log"
        command = [str(PY), str(MAKE_PACKAGE), "--source", str(post_dir),
                   "--zip", str(zip_path), "--expected-count", str(spec["expected"])]
        if name == "SportsMOT":
            command += ["--expected-names-file", str(EXP / "logs/sportsmot_test_sequences.txt")]
        with package_log.open("wb") as handle:
            code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"{name}: packaging failed")
        validate_log = SUBDIR / f"{name}_validate.log"
        validate = [str(PY), str(SUBMIT_TOOL), "validate", "--zip", str(zip_path),
                    "--dataset", spec["dataset_flag"], "--seqinfo-root", str(spec["seqinfo"])]
        if name == "SportsMOT":
            validate += ["--sequences-file", str(EXP / "logs/sportsmot_test_sequences.txt")]
        with validate_log.open("wb") as handle:
            code = subprocess.call(validate, stdout=handle, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"{name}: local validation failed")
        result = {"zip": str(zip_path), "sha256": sha256(zip_path),
                  "package_log": str(package_log), "validate_log": str(validate_log)}
        self.state["results"].setdefault(name, {})["package"] = result
        self.save()
        return result

    @staticmethod
    def _api(token, path, timeout=60):
        last = None
        for _ in range(6):
            result = subprocess.run(
                ["curl", "-sS", "--max-time", str(timeout), "-H", f"Authorization: Token {token}",
                 f"https://www.codabench.org{path}"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    last = result.stdout[:200]
            else:
                last = result.stderr[:200]
            time.sleep(4)
        raise RuntimeError(f"Codabench API unavailable: {path} ({last})")

    def _submissions(self, token, phase):
        payload = self._api(token, f"/api/submissions/?phase={phase}")
        return payload if isinstance(payload, list) else payload.get("results", [])

    def _poll(self, token, submission_id, record, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            payload = self._api(token, f"/api/submissions/{submission_id}/")
            status = payload.get("status")
            if status is None and payload.get("id") is None:
                raise RuntimeError(f"cannot read submission {submission_id}: {str(payload)[:200]}")
            print(f"{now()} {submission_id} status={status}", flush=True)
            if status not in NON_TERMINAL:
                record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return payload
            time.sleep(30)
        raise RuntimeError(f"submission {submission_id} still running after {timeout}s")

    def _resolve(self, submission_id):
        for account, token_file in TOKENS:
            if not token_file.is_file():
                continue
            token = token_file.read_text().strip()
            payload = self._api(token, f"/api/submissions/{submission_id}/")
            if payload.get("id") is not None or payload.get("status") is not None:
                return token, payload
        raise RuntimeError(f"cannot resolve submission {submission_id} with either account")

    def submit(self, name, spec, package):
        record = SUBDIR / f"{name}_submission.json"
        state_path = SUBDIR / f"{name}_submission_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {"baseline": {}}
        if record.is_file():
            payload = json.loads(record.read_text())
            if payload.get("id") is not None:
                token = None
                if payload.get("status") in NON_TERMINAL:
                    token, payload = self._resolve(payload["id"])
                    payload = self._poll(token, payload["id"], record, 2700)
                self.state["results"].setdefault(name, {})["submission"] = {
                    "id": payload["id"], "status": payload.get("status"),
                    "scores": payload.get("scores"), "reused": True}
                self.save()
                return payload
        last_error = None
        for account, token_file in TOKENS:
            if not token_file.is_file():
                continue
            token = token_file.read_text().strip()
            can = self._api(token, f"/api/can_make_submission/{spec['phase']}/")
            if not can.get("can"):
                last_error = f"{account}: cannot submit ({can.get('reason')})"
                continue
            baseline = state["baseline"].get(account)
            if baseline is None:
                baseline = max([int(s["id"]) for s in self._submissions(token, spec["phase"])
                                if s.get("id")] or [0])
                state["baseline"][account] = baseline
                state_path.write_text(json.dumps(state, indent=2) + "\n")
            else:
                fresh = [s for s in self._submissions(token, spec["phase"])
                         if int(s.get("id", 0)) > int(baseline)]
                if fresh:
                    submission_id = max(int(s["id"]) for s in fresh)
                    payload = self._poll(token, submission_id, record, 2700)
                    self.state["results"].setdefault(name, {})["submission"] = {
                        "id": submission_id, "status": payload.get("status"),
                        "scores": payload.get("scores"), "account": account,
                        "recovered": True}
                    self.save()
                    return payload
            command = [str(PY), str(SUBMIT_TOOL), "submit", "--zip", package["zip"],
                       "--competition", spec["competition"], "--phase", spec["phase"],
                       "--task", spec["task"],
                       "--name", f"shutrack_midpoint_{name.lower()}_e{spec['epoch']}_"
                                 f"{time.strftime('%Y%m%dT%H%M%S')}",
                       "--dataset", spec["dataset_flag"], "--seqinfo-root", str(spec["seqinfo"]),
                       "--token-file", str(token_file),
                       "--poll", "--poll-interval", "30", "--poll-timeout", "2400",
                       "--save-record", str(record)]
            if name == "SportsMOT":
                command += ["--sequences-file", str(EXP / "logs/sportsmot_test_sequences.txt")]
            code = subprocess.call(command)
            if record.is_file():
                payload = json.loads(record.read_text())
                if payload.get("id") is not None:
                    if payload.get("status") in NON_TERMINAL:
                        payload = self._poll(token, payload["id"], record, 2700)
                    self.state["results"].setdefault(name, {})["submission"] = {
                        "id": payload["id"], "status": payload.get("status"),
                        "scores": payload.get("scores"), "account": account}
                    self.save()
                    return payload
            fresh = [s for s in self._submissions(token, spec["phase"])
                     if int(s.get("id", 0)) > int(baseline)]
            if fresh:
                submission_id = max(int(s["id"]) for s in fresh)
                payload = self._poll(token, submission_id, record, 2700)
                self.state["results"].setdefault(name, {})["submission"] = {
                    "id": submission_id, "status": payload.get("status"),
                    "scores": payload.get("scores"), "account": account, "recovered": True}
                self.save()
                return payload
            last_error = f"{account}: submit exited {code}"
        raise RuntimeError(f"{name}: submission failed ({last_error})")


def report(driver):
    lines = ["# MOT20 / MOT17 低学习率续训 50 轮 test 提交", "",
             "| 数据集 | 权重 | 旧官方 HOTA | 新官方 HOTA | 提交 ID | 状态 |",
             "|---|---|---:|---:|---|---|"]
    for name, spec in DATASETS.items():
        entry = driver.state["results"].get(name, {})
        submission = entry.get("submission") or {}
        scores = submission.get("scores") or {}
        if isinstance(scores, list):
            scores = {item.get("column_key"): item.get("score") for item in scores if isinstance(item, dict)}
        new_hota = scores.get("HOTA")
        weight = "续训 e50（源 e100）" if name == "MOT20" else "续训 e50（源 e50）"
        lines.append(f"| {name} | {weight} | {spec['previous_hota']:.3f} | "
                     f"{'-' if new_hota is None else f'{float(new_hota):.3f}'} | "
                     f"{submission.get('id', '-')} | {submission.get('status', '-')} |")
    (EXP / "report.md").write_text("\n".join(lines) + "\n")
    (EXP / "report.json").write_text(json.dumps(driver.state, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "test", "package", "submit", "report"),
                        default="all")
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASETS), default=None)
    args = parser.parse_args()
    driver = Driver()
    selected = args.datasets or list(DATASETS)
    for name in selected:
        spec = DATASETS[name]
        driver.state["results"].setdefault(name, {})["checkpoint"] = driver.checkpoint_info(name, spec)
        driver.save()
    if args.stage in ("all", "test"):
        for name in selected:
            spec = DATASETS[name]
            post = driver.tracking_and_post(name, spec)
            print(json.dumps({"dataset": name, "post": str(post),
                              "gates": driver.state["results"][name]["test"]["gates"]}), flush=True)
    if args.stage in ("all", "package"):
        for name in selected:
            spec = DATASETS[name]
            post = Path(driver.state["results"][name]["test"]["post_dir"])
            package = driver.package(name, spec, post)
            print(json.dumps({"dataset": name, "package": package}), flush=True)
    if args.stage in ("all", "submit"):
        for name in selected:
            spec = DATASETS[name]
            package = driver.state["results"][name]["package"]
            payload = driver.submit(name, spec, package)
            print(json.dumps({"dataset": name, "id": payload.get("id"),
                              "status": payload.get("status"),
                              "scores": payload.get("scores")}), flush=True)
    if args.stage in ("all", "report"):
        report(driver)


if __name__ == "__main__":
    main()
