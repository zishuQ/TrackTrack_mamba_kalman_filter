#!/usr/bin/env python3
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
DEFAULT_MAMBA_MODEL = Path("/home/shang/workspace/mamba_kalman_filter/checkpoints/MOT20_ablation_baseline.pth")

HEADER = [
    "model",
    "noise_type",
    "px",
    "HOTA",
    "MOTA",
    "IDF1",
    "DetA",
    "AssA",
    "seconds",
]

METRIC_RE = re.compile(
    r"^\s*([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s*$"
)

def read_completed(csv_path):
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {
            (row["model"], row["noise_type"], row["px"])
            for row in csv.DictReader(f)
            if row.get("model") and row.get("noise_type") and row.get("px")
        }


def ensure_header(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(HEADER)


def parse_metrics(output):
    metrics = []
    for line in output.splitlines():
        metric_match = METRIC_RE.match(line)
        if metric_match:
            metrics.append(tuple(float(x) for x in metric_match.groups()))
    if not metrics:
        raise RuntimeError("Could not parse aggregate metrics from command output")
    return metrics[0]


def append_row(csv_path, model, noise_type, px, aggregate, seconds):
    row = [
        model,
        noise_type,
        px,
        *[f"{value:.6f}" for value in aggregate],
        f"{seconds:.1f}",
    ]
    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    print(",".join(str(x) for x in row), flush=True)


def command_for_job(args, model, noise_type, px):
    python = str(args.python)
    common = [
        python,
        "run.py" if model == "kf_all" else "run_mamba.py",
        "--dataset",
        "MOT20",
        "--mode",
        "all",
        "--sequences",
        "MOT20-01",
        "MOT20-03",
        "--print-per-sequence-metrics",
    ]

    if model == "kf_all":
        common += ["--kf-type", "kf"]
    elif model == "mamba_ablation_baseline":
        common += ["--mamba_model_path", str(args.mamba_model)]
    else:
        raise ValueError(model)

    if noise_type == "pos" and px > 0:
        common += ["--noise-pos-std", str(px)]
    if noise_type == "size" and px > 0:
        common += ["--noise-size-std", str(px)]

    common += ["--tracker-suffix", f"{model}_{noise_type}_{px}"]
    return common


def run_one(args, model, noise_type, px):
    cmd = command_for_job(args, model, noise_type, px)
    print(f"START {model} {noise_type} {px}px", flush=True)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=TRACKER_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    seconds = time.time() - start
    if proc.returncode != 0:
        print(proc.stdout[-6000:], flush=True)
        raise RuntimeError(f"Command failed: {model} {noise_type} {px}px")
    aggregate = parse_metrics(proc.stdout)
    append_row(args.csv, model, noise_type, px, aggregate, seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--mamba-model", type=Path, default=DEFAULT_MAMBA_MODEL)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["kf_all", "mamba_ablation_baseline"],
        choices=["kf_all", "mamba_ablation_baseline"],
    )
    parser.add_argument("--max-px", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Run even if a row already exists in the CSV")
    args = parser.parse_args()

    ensure_header(args.csv)
    completed = read_completed(args.csv)

    print(f"CSV: {args.csv}", flush=True)
    for model in args.models:
        if model == "mamba_ablation_baseline" and not args.mamba_model.exists():
            raise FileNotFoundError(args.mamba_model)
        for noise_type in ("pos", "size"):
            for px in range(0, args.max_px + 1):
                key = (model, noise_type, str(px))
                if key in completed and not args.force:
                    print(f"SKIP existing {model} {noise_type} {px}px", flush=True)
                    continue
                run_one(args, model, noise_type, px)
                completed.add(key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
