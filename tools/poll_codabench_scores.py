#!/usr/bin/env python3
"""Poll one Codabench submission until it reaches a terminal status.

Writes the final poll payload into the per-dataset submission record and a
separate SCORES_<dataset>.json so the shared SUBMISSION_RESULTS.json is never
edited while the main-line driver is still writing it.
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
CLIENT = ROOT / "tools/codabench_submit.py"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
SUBDIR = EXP / "submissions"

PENDING = {"Preparing", "Submitting", "Running", "Queued", "Waiting", "Pending"}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def poll(submission_id: int) -> dict:
    result = subprocess.run(
        [str(PY), str(CLIENT), "poll", "--submission-id", str(submission_id)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"poll failed rc={result.returncode}: {result.stderr[:300]}")
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    SUBDIR.mkdir(parents=True, exist_ok=True)
    record_path = Path(args.record) if args.record else SUBDIR / f"{args.dataset}_submission.json"
    scores_path = SUBDIR / f"SCORES_{args.dataset}.json"
    log_path = SUBDIR / f"{args.dataset}_poll.log"
    started = time.time()
    errors = 0
    last_status = None
    while time.time() - started < args.timeout:
        try:
            payload = poll(args.submission_id)
            errors = 0
        except Exception as exc:  # noqa: BLE001
            errors += 1
            with log_path.open("a") as handle:
                handle.write(f"{now()} error {errors}: {exc}\n")
            if errors >= 10:
                raise SystemExit(f"poll failed 10 times: {exc}")
            time.sleep(args.interval)
            continue
        status = str(payload.get("status") or "")
        scores = payload.get("scores") or {}
        if status != last_status or scores:
            with log_path.open("a") as handle:
                handle.write(f"{now()} status={status} scores={scores}\n")
            last_status = status
        if status not in PENDING and (status or scores):
            summary = {
                "dataset": args.dataset, "submission_id": args.submission_id,
                "status": status, "scores": scores,
                "phase_name": payload.get("phase_name"),
                "is_public": payload.get("is_public"),
                "finished_at": now(),
            }
            scores_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            if record_path.is_file():
                try:
                    record = json.loads(record_path.read_text())
                except json.JSONDecodeError:
                    record = {}
            else:
                record = {}
            record.update({"status": status, "scores": scores,
                           "phase_name": payload.get("phase_name"),
                           "polled_at": now()})
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            return
        time.sleep(args.interval)
    raise SystemExit(f"submission {args.submission_id} still {last_status} after {args.timeout:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
