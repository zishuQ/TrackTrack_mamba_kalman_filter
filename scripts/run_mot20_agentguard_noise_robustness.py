#!/usr/bin/env python3
"""Run MOT20 detection-noise robustness jobs with an AgentGuard checkpoint.

The default job list contains 19 unique settings: position noise 0..9 px
(including the clean baseline) and size noise 1..9 px.  A clean size=0 run is
identical to the clean position=0 run, so it is intentionally not duplicated.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKER_DIR = ROOT / "3. Tracker"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_CSV = ROOT / "outputs" / "mot20_noise_robustness_summary.csv"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/agentguard/experiments/"
    / "iwg_rg_cma_sqrt_size_mot20_bound005_context6_seed42_bs1024_shard4x1_100e"
    / "checkpoints/iwg_rg_cma_epoch100.pt"
)
DEFAULT_CACHE = ROOT / "outputs/agentguard/detection_cache"
DEFAULT_LOG_DIR = ROOT / "outputs/mot20_agentguard_noise_logs"
# Match the existing robustness CSV, which evaluates the MOT20 validation pair.
DEFAULT_SEQUENCES = ("MOT20-01", "MOT20-03")
MODEL_NAME = "agentguard_iwg_rg_cma_epoch100_final"

HEADER = ["model", "HOTA", "MOTA", "IDF1", "DetA", "AssA", "px", "noise_type"]
FLOAT_LINE = re.compile(
    r"^\s*([+-]?(?:\d+\.\d+|\d+\.\d*[eE][+-]?\d+))\s+"
    r"([+-]?(?:\d+\.\d+|\d+\.\d*[eE][+-]?\d+))\s+"
    r"([+-]?(?:\d+\.\d+|\d+\.\d*[eE][+-]?\d+))\s+"
    r"([+-]?(?:\d+\.\d+|\d+\.\d*[eE][+-]?\d+))\s+"
    r"([+-]?(?:\d+\.\d+|\d+\.\d*[eE][+-]?\d+))\s*$"
)


def completed_jobs(csv_path: Path) -> set[tuple[str, str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(newline="") as handle:
        return {
            (row.get("model", ""), row.get("noise_type", ""), row.get("px", ""))
            for row in csv.DictReader(handle)
            if row.get("model") and row.get("noise_type") and row.get("px")
        }


def ensure_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(HEADER)
        return
    with csv_path.open(newline="") as handle:
        actual = next(csv.reader(handle), [])
    missing = [name for name in HEADER if name not in actual]
    if missing:
        raise ValueError(
            f"{csv_path} has incompatible columns; missing {missing}. "
            "Use --csv with a new output file."
        )


def parse_aggregate(output: str) -> tuple[float, float, float, float, float]:
    """Parse the first five-column metric row printed by TrackEval."""
    for line in output.splitlines():
        match = FLOAT_LINE.match(line)
        if match:
            return tuple(float(value) for value in match.groups())  # type: ignore[return-value]
    raise RuntimeError("Could not parse aggregate HOTA/MOTA/IDF1/DetA/AssA metrics")


def append_result(
    csv_path: Path,
    noise_type: str,
    px: int,
    metrics: tuple[float, float, float, float, float],
) -> None:
    row = [MODEL_NAME, *[f"{value:.6f}" for value in metrics], f"{px}px", noise_type]
    with csv_path.open("a", newline="") as handle:
        csv.writer(handle).writerow(row)
    print(",".join(row), flush=True)


def command_for_job(args: argparse.Namespace, noise_type: str, px: int) -> list[str]:
    command = [
        str(args.python),
        "run.py",
        "--dataset",
        "MOT20",
        "--mode",
        "all",
        "--sequences",
        *args.sequences,
        "--seed",
        str(args.seed),
        "--agentguard-mode",
        "iwg-rg-cma",
        "--agentguard-checkpoint",
        str(args.checkpoint),
        "--iwg-rg-cma-output",
        "final",
        "--agentguard-device",
        args.agentguard_device,
        "--detection-cache-root",
        str(args.detection_cache_root),
        "--tracker-suffix",
        f"mot20_agentguard_{noise_type}_{px}px",
        "--print-per-sequence-metrics",
    ]
    if noise_type == "pos" and px:
        command += ["--noise-pos-std", str(px)]
    elif noise_type == "size" and px:
        command += ["--noise-size-std", str(px)]
    return command


def run_one(args: argparse.Namespace, noise_type: str, px: int) -> None:
    command = command_for_job(args, noise_type, px)
    label = f"{noise_type}_{px}px"
    log_path = args.log_dir / f"{label}.log"
    print(f"START {label}", flush=True)
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=TRACKER_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = process.stdout
    log_path.write_text(output)
    if process.returncode:
        print(output[-8000:], flush=True)
        raise RuntimeError(f"Command failed for {label}; see {log_path}")
    metrics = parse_aggregate(output)
    append_result(args.csv, noise_type, px, metrics)
    print(f"DONE {label} ({time.monotonic() - started:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--detection-cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--agentguard-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--max-px",
        type=int,
        default=9,
        help="Run position 0..max-px and size 1..max-px (default: 9 => 19 jobs).",
    )
    parser.add_argument("--force", action="store_true", help="Ignore rows already present in --csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print all commands without running them.")
    args = parser.parse_args()

    if args.max_px < 1:
        parser.error("--max-px must be at least 1")
    for sequence in args.sequences:
        if sequence not in DEFAULT_SEQUENCES:
            parser.error(f"unsupported MOT20 sequence: {sequence}")
    if not (TRACKER_DIR / "run.py").is_file():
        raise FileNotFoundError(TRACKER_DIR / "run.py")
    for path in (args.python, args.checkpoint, args.detection_cache_root):
        if not path.exists():
            raise FileNotFoundError(path)
    for sequence in args.sequences:
        manifest = (
            args.detection_cache_root / "MOT20" / "all" / sequence / "manifest.json"
        )
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

    jobs = [("pos", px) for px in range(args.max_px + 1)]
    jobs += [("size", px) for px in range(1, args.max_px + 1)]
    print(f"Planned jobs: {len(jobs)} ({args.max_px + 1} position + {args.max_px} size)")
    ensure_csv(args.csv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_jobs(args.csv)

    for noise_type, px in jobs:
        key = (MODEL_NAME, noise_type, f"{px}px")
        command = command_for_job(args, noise_type, px)
        if args.dry_run:
            print(" ".join(command))
            continue
        if key in completed and not args.force:
            print(f"SKIP existing {noise_type}_{px}px", flush=True)
            continue
        run_one(args, noise_type, px)
        completed.add(key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
