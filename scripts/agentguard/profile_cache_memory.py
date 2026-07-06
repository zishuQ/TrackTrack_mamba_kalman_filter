#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _process_rss_mb(pid: int) -> float:
    status = Path(f"/proc/{pid}/status")
    try:
        with status.open("r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except FileNotFoundError:
        return 0.0
    return 0.0


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _read_manifest(path: Path) -> dict:
    with (path / "manifest.json").open("r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile compact AgentGuard cache memory using mmap detection cache."
    )
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--sequence", default="MOT17-04-FRCNN")
    parser.add_argument("--frames", default="5,20,50,200")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
    )
    parser.add_argument(
        "--detection-cache-root",
        default=str(REPO_ROOT / "outputs" / "agentguard" / "detection_cache"),
    )
    parser.add_argument(
        "--event-cache-root",
        default=str(REPO_ROOT / "outputs" / "agentguard" / "event_cache_profile"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "agentguard" / "cache_memory_profile.json"),
    )
    parser.add_argument(
        "--keep-caches",
        action="store_true",
        help="Keep per-frame-count output caches instead of rebuilding each run.",
    )
    args = parser.parse_args()

    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    report = {
        "dataset": args.dataset,
        "mode": args.mode,
        "sequence": args.sequence,
        "runs": [],
    }

    env = os.environ.copy()
    py_path_parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "3. Tracker"),
        str(REPO_ROOT / "2. FastReID"),
        str(REPO_ROOT / "1. YOLOX"),
        str(REPO_ROOT / "agentguard" / "src"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts + [env.get("PYTHONPATH", "")])

    for n in frames:
        root = Path(args.event_cache_root) / f"{n:04d}"
        seq_dir = root / args.dataset / args.mode / args.sequence
        if root.exists() and not args.keep_caches:
            shutil.rmtree(root)

        before = _rss_mb()
        cmd = [
            args.python_bin,
            "-m",
            "agentguard.cli",
            "cache_events",
            "--dataset",
            args.dataset,
            "--mode",
            args.mode,
            "--sequence",
            args.sequence,
            "--max-frames",
            str(n),
            "--detection-cache-root",
            args.detection_cache_root,
            "--event-cache-root",
            str(root),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        peak_child = 0.0
        while proc.poll() is None:
            peak_child = max(peak_child, _process_rss_mb(proc.pid))
            time.sleep(0.2)
        stdout, stderr = proc.communicate()
        after = _rss_mb()
        peak_child = max(peak_child, _process_rss_mb(proc.pid))
        if proc.returncode != 0:
            raise SystemExit(
                json.dumps(
                    {
                        "failed_frames": n,
                        "returncode": proc.returncode,
                        "stdout": stdout[-4000:],
                        "stderr": stderr[-4000:],
                    },
                    indent=2,
                )
            )

        manifest = _read_manifest(seq_dir)
        cache_bytes = _dir_size(seq_dir)
        events = int(manifest.get("num_events", 0))
        report["runs"].append(
            {
                "frames": n,
                "initial_rss_mb": before,
                "peak_rss_mb": peak_child,
                "end_rss_mb": after,
                "detections": int(manifest.get("num_detections", 0)),
                "events": events,
                "association_records": int(manifest.get("num_association_records", 0)),
                "cache_bytes": cache_bytes,
                "bytes_per_event": float(cache_bytes) / max(events, 1),
                "stdout_tail": stdout[-2000:],
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
