#!/usr/bin/env python3
"""CPU vs GPU inference benchmark for the formal MOT17 NEW checkpoint.

Uses two MOT17 train sequences with GT, raw tracking (no post), identical
tracker settings. Waits for any running MOT17 test inference to exit first so
the timings are not contaminated. Runs each device twice, alternating.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
CKPT = EXP / "final/MOT17/groups/appearance_row_column/checkpoints/last.pt"
CACHE = ROOT / "outputs/agentguard/detection_cache"
OUT = EXP / "bench_device"
SEQS = ("MOT17-02-FRCNN", "MOT17-04-FRCNN")
DEVICES = ("cpu", "cuda", "cpu", "cuda")


def mot17_test_running() -> bool:
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{entry}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "run.py" in cmd and "--dataset MOT17" in cmd and "--mode test" in cmd:
            return True
    return False


def run_once(device: str, run_index: int) -> dict:
    tag = f"{device}_{run_index}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    command = [
        str(PY), "-u", "run.py", "--dataset", "MOT17", "--mode", "all",
        "--sequences", *SEQS, "--seed", "10000", "--kf-type", "nsa",
        "--agentguard-mode", "shutrack-v2", "--shutrack-checkpoint", str(CKPT),
        "--agentguard-device", device,
        "--detection-cache-root", str(CACHE),
        "--output_dir", str(out), "--tracker-suffix", f"devbench_{tag}",
        "--agentguard-stats-output", str(out / "gates.jsonl"),
        "--print-per-sequence-metrics", "--eval-json", str(summary),
        "--torch-num-threads", "4", "--torch-num-interop-threads", "1",
    ]
    log = OUT / f"{tag}.log"
    started = time.monotonic()
    with log.open("wb") as handle:
        code = subprocess.call(command, cwd=str(TRACKER), stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    record = {"device": device, "run": run_index, "tag": tag,
              "seconds": round(elapsed, 2), "exit_code": code, "log": str(log)}
    if code == 0 and summary.is_file():
        payload = json.loads(summary.read_text())
        record["combined"] = payload.get("combined")
        record["per_sequence"] = payload.get("per_sequence")
        gates = out / "gates.jsonl"
        if gates.is_file():
            record["tracker_elapsed"] = [
                {"sequence": row.get("sequence"), "elapsed_sec": row.get("elapsed_sec")}
                for row in (json.loads(line) for line in gates.read_text().splitlines() if line.strip())
            ]
    else:
        record["error"] = f"exit {code}"
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    while mot17_test_running():
        print(f"{time.strftime('%F %T')} waiting for MOT17 test inference to finish", flush=True)
        time.sleep(10)
    results = []
    for index, device in enumerate(DEVICES, start=1):
        print(f"{time.strftime('%F %T')} run {index}/{len(DEVICES)} device={device}", flush=True)
        results.append(run_once(device, index))
        (OUT / "benchmark_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
