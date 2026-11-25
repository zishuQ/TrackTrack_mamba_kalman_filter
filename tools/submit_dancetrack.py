#!/usr/bin/env python3
"""Submit one DanceTrack test package to Codabench with duplicate protection.

Flow: local validation -> baseline submission id -> reserve/upload/create/poll
via tools/codabench_submit.py. On network errors, look for a newly created
submission before retrying so a lost response never creates a duplicate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TOOL = ROOT / "tools/codabench_submit.py"
TOKEN = Path.home() / ".config" / "codabench" / "token"
BASE_URL = "https://www.codabench.org"
NON_TERMINAL = ("Submitting", "Running", "Preparing", "Queued", "Pending")


def run_json(argv, timeout=60):
    token = TOKEN.read_text().strip()
    last = None
    for _ in range(6):
        result = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout),
             "-H", f"Authorization: Token {token}", *argv],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                last = result.stdout[:200]
        else:
            last = result.stderr[:200]
        time.sleep(4)
    raise RuntimeError(f"Codabench API unavailable: {argv[-1]} ({last})")


def phase_submissions(phase):
    payload = run_json([f"{BASE_URL}/api/submissions/?phase={phase}"])
    return payload if isinstance(payload, list) else payload.get("results", [])


def latest_id(phase):
    ids = [int(item["id"]) for item in phase_submissions(phase) if item.get("id")]
    return max(ids) if ids else 0


def poll(submission_id, record, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = run_json([f"{BASE_URL}/api/submissions/{submission_id}/"])
        status = payload.get("status")
        print(f"{time.strftime('%H:%M:%S')} submission {submission_id} status={status}", flush=True)
        if status not in NON_TERMINAL:
            record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return payload
        time.sleep(30)
    raise RuntimeError(f"submission {submission_id} still running after {timeout}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--sequences-file", required=True)
    parser.add_argument("--seqinfo-root", required=True)
    parser.add_argument("--competition", default="14885")
    parser.add_argument("--phase", default="24635")
    parser.add_argument("--task", default="30364")
    parser.add_argument("--name", default="")
    parser.add_argument("--poll-timeout", type=float, default=2700)
    args = parser.parse_args()
    record = Path(args.record)
    state_path = record.with_name("submission_state.json")
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    zip_path = Path(args.zip)

    validate = [str(PY), str(TOOL), "validate", "--zip", str(zip_path),
                "--dataset", "dancetrack", "--seqinfo-root", args.seqinfo_root,
                "--sequences-file", args.sequences_file]
    if subprocess.call(validate) != 0:
        raise SystemExit("local zip validation failed")

    if record.is_file():
        payload = json.loads(record.read_text())
        if payload.get("id") is not None:
            if payload.get("status") in NON_TERMINAL:
                payload = poll(payload["id"], record, args.poll_timeout)
            print(json.dumps({"submission_id": payload["id"],
                              "status": payload.get("status"),
                              "scores": payload.get("scores")}, indent=2), flush=True)
            return

    baseline = state.get("baseline_id")
    if baseline is None:
        baseline = latest_id(int(args.phase))
        state["baseline_id"] = baseline
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    else:
        fresh = [item for item in phase_submissions(int(args.phase))
                 if int(item.get("id", 0)) > int(baseline)]
        if fresh:
            submission_id = max(int(item["id"]) for item in fresh)
            payload = poll(submission_id, record, args.poll_timeout)
            print(json.dumps({"submission_id": submission_id,
                              "status": payload.get("status"),
                              "scores": payload.get("scores")}, indent=2), flush=True)
            return

    last_error = None
    for attempt in range(3):
        name = args.name or f"shutrack_dance_trainval_e80_{time.strftime('%Y%m%dT%H%M%S')}_{attempt + 1}"
        command = [str(PY), str(TOOL), "submit", "--zip", str(zip_path),
                   "--competition", args.competition, "--phase", args.phase,
                   "--task", args.task, "--name", name, "--dataset", "dancetrack",
                   "--seqinfo-root", args.seqinfo_root,
                   "--sequences-file", args.sequences_file,
                   "--poll", "--poll-interval", "30",
                   "--poll-timeout", str(int(args.poll_timeout)),
                   "--save-record", str(record)]
        code = subprocess.call(command)
        if record.is_file():
            payload = json.loads(record.read_text())
            if payload.get("id") is not None:
                if payload.get("status") in NON_TERMINAL:
                    payload = poll(payload["id"], record, args.poll_timeout)
                print(json.dumps({"submission_id": payload["id"],
                                  "status": payload.get("status"),
                                  "scores": payload.get("scores")}, indent=2), flush=True)
                return
        if code:
            last_error = f"submit exited {code}"
        fresh = [item for item in phase_submissions(int(args.phase))
                 if int(item.get("id", 0)) > int(baseline)]
        if fresh:
            submission_id = max(int(item["id"]) for item in fresh)
            payload = poll(submission_id, record, args.poll_timeout)
            print(json.dumps({"submission_id": submission_id,
                              "status": payload.get("status"),
                              "scores": payload.get("scores")}, indent=2), flush=True)
            return
        time.sleep(15)
    raise SystemExit(f"submission failed after retries: {last_error}")


if __name__ == "__main__":
    main()
