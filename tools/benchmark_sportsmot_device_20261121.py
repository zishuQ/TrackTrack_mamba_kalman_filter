#!/usr/bin/env python3
"""CPU vs GPU inference benchmark on the SportsMOT val split (45 sequences).

Formal SportsMOT NEW checkpoint, raw tracking (no post), identical settings.
Runs CPU then GPU once each. Wall time includes the CPU-side TrackEval
evaluation; per-sequence tracker elapsed from gates.jsonl isolates tracking.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
CKPT = EXP / "final/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"
CACHE = ROOT / "outputs/agentguard/detection_cache"
OUT = EXP / "bench_device"
SEQS = tuple(
    line.strip() for line in (TRACKER / "trackeval/seqmap/sportsmot/val.txt").read_text().splitlines()
    if line.strip() and line.strip() != "name"
)
DEVICES = ("cpu", "cuda")


def run_once(device: str) -> dict:
    out = OUT / f"sportsmot_{device}"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    command = [
        str(PY), "-u", "run.py", "--dataset", "SportsMOT", "--mode", "val",
        "--sequences", *SEQS, "--seed", "10000", "--kf-type", "nsa",
        "--agentguard-mode", "shutrack-v2", "--shutrack-checkpoint", str(CKPT),
        "--agentguard-device", device,
        "--detection-cache-root", str(CACHE),
        "--output_dir", str(out), "--tracker-suffix", f"devbench_sm_{device}",
        "--agentguard-stats-output", str(out / "gates.jsonl"),
        "--print-per-sequence-metrics", "--eval-json", str(summary),
        "--torch-num-threads", "4", "--torch-num-interop-threads", "1",
    ]
    log = OUT / f"sportsmot_{device}.log"
    started = time.monotonic()
    with log.open("wb") as handle:
        code = subprocess.call(command, cwd=str(TRACKER), stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    record = {"device": device, "seconds": round(elapsed, 2), "exit_code": code, "log": str(log)}
    if code == 0 and summary.is_file():
        payload = json.loads(summary.read_text())
        record["combined"] = payload.get("combined")
        record["per_sequence"] = payload.get("per_sequence")
        gates = out / "gates.jsonl"
        if gates.is_file():
            rows = [json.loads(line) for line in gates.read_text().splitlines() if line.strip()]
            record["tracking_elapsed_total"] = round(sum(r.get("elapsed_sec", 0.0) for r in rows), 2)
            record["tracking_elapsed"] = [
                {"sequence": r.get("sequence"), "elapsed_sec": r.get("elapsed_sec")} for r in rows
            ]
    else:
        record["error"] = f"exit {code}"
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for device in DEVICES:
        print(f"{time.strftime('%F %T')} SportsMOT val device={device}", flush=True)
        record = run_once(device)
        results.append(record)
        (OUT / "sportsmot_benchmark_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )
        combined = record.get("combined") or {}
        print(f"{device}: wall {record['seconds']}s tracking_sum {record.get('tracking_elapsed_total')}s "
              f"HOTA {combined.get('HOTA')} AssA {combined.get('AssA')} IDF1 {combined.get('IDF1')}",
              flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
