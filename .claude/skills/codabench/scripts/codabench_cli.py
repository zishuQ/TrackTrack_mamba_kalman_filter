#!/usr/bin/env python3
"""Small Codabench helper: quota, today's submissions, poll one record.

Tokens are read from ~/.config/codabench/{token,token_edge} and never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.codabench.org"
TOKENS = (("jojo0", Path.home() / ".config/codabench/token"),
          ("shangqingnie", Path.home() / ".config/codabench/token_edge"))
NON_TERMINAL = {"Submitting", "Running", "Preparing", "Queued", "Pending",
                "Submitted", "Scoring", "Waiting", "Waiting for scoring"}


def accounts():
    for name, path in TOKENS:
        if path.is_file():
            yield name, path.read_text().strip()


def api(token, path, timeout=60, retries=6):
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                BASE + path, headers={"Authorization": f"Token {token}"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Codabench API unavailable: {path} ({last})")


def cmd_quota(args):
    for phase in args.phase:
        try:
            detail = api(next(accounts())[1], f"/api/phases/{phase}/")
            print(json.dumps({"phase": phase, "limit": {
                key: detail.get(key) for key in
                ("has_max_submissions", "max_submissions_per_day", "max_submissions_scope")}}))
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"phase": phase, "limit_error": str(error)[:160]}))
        for name, token in accounts():
            try:
                can = api(token, f"/api/can_make_submission/{phase}/")
            except Exception as error:  # noqa: BLE001
                can = {"can": None, "reason": str(error)[:160]}
            print(json.dumps({"phase": phase, "account": name, **can}))
    return 0


def cmd_today(args):
    today = time.strftime("%Y-%m-%d", time.gmtime())
    for name, token in accounts():
        for phase in args.phase:
            try:
                payload = api(token, f"/api/submissions/?phase={phase}")
            except Exception as error:  # noqa: BLE001
                print(json.dumps({"account": name, "phase": phase,
                                  "error": str(error)[:160]}))
                continue
            rows = payload if isinstance(payload, list) else payload.get("results", [])
            todays = [row for row in rows
                      if str(row.get("created_when", "")).startswith(today)]
            print(json.dumps({"account": name, "phase": phase, "utc_date": today,
                              "count": len(todays),
                              "submissions": [{key: row.get(key) for key in
                                               ("id", "status", "created_when")}
                                              for row in todays]}))
    return 0


def cmd_poll(args):
    record = Path(args.record)
    token = Path(args.token_file).read_text().strip()
    payload = json.loads(record.read_text())
    submission_id = int(payload["id"])
    deadline = time.time() + args.timeout_min * 60
    while time.time() < deadline:
        try:
            current = api(token, f"/api/submissions/{submission_id}/")
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"id": submission_id, "error": str(error)[:160]}),
                  flush=True)
            time.sleep(60)
            continue
        status = current.get("status")
        if status not in NON_TERMINAL:
            record.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"id": submission_id, "status": status,
                              "scores": current.get("scores")}))
            return 0
        print(json.dumps({"id": submission_id, "status": status,
                          "at": time.strftime("%H:%M:%S")}), flush=True)
        time.sleep(args.interval)
    print(json.dumps({"id": submission_id, "status": "timeout"}))
    return 2


def main():
    parser = argparse.ArgumentParser(description="Codabench quota/today/poll helper.")
    sub = parser.add_subparsers(dest="command", required=True)
    quota = sub.add_parser("quota", help="can_make_submission per account + phase limits")
    quota.add_argument("--phase", action="append", required=True)
    today = sub.add_parser("today", help="today's (UTC) submissions per account")
    today.add_argument("--phase", action="append", required=True)
    poll = sub.add_parser("poll", help="poll a saved submission record until terminal")
    poll.add_argument("--record", required=True)
    poll.add_argument("--token-file", required=True)
    poll.add_argument("--interval", type=int, default=60)
    poll.add_argument("--timeout-min", type=int, default=360)
    args = parser.parse_args()
    if args.command == "quota":
        return cmd_quota(args)
    if args.command == "today":
        return cmd_today(args)
    return cmd_poll(args)


if __name__ == "__main__":
    sys.exit(main())
