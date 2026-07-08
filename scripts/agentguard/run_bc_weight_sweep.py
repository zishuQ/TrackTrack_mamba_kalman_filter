#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASELINE_METRICS_BY_DATASET = {
    ("MOT17", "all"): {
        "HOTA": 0.770209,
        "AssA": 0.770913,
        "IDF1": 0.873123,
        "MOTA": 0.885963,
    },
}

METRIC_HEADER_RE = re.compile(r"^\s*HOTA\s+MOTA\s+IDF1\s+DetA\s+AssA\s*$")
METRIC_ROW_RE = re.compile(
    r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)
SEQ_ROW_RE = re.compile(r"^\s*(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


@dataclass(frozen=True)
class Experiment:
    name: str
    candidate_types: str
    candidate_weights: str
    max_per_candidate_type: int = 0


@dataclass(frozen=True)
class RunSpec:
    experiment: Experiment
    tgr_stride: int

    @property
    def name(self) -> str:
        return f"{self.experiment.name}_tgrs{self.tgr_stride}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python(default: str) -> str:
    return os.environ.get("PYTHON_BIN", default)


def _env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(repo_root),
        str(repo_root / "3. Tracker"),
        str(repo_root / "agentguard" / "src"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(shlex.quote(part) for part in cmd)
    with log_path.open("w") as log:
        log.write(f"\n$ {line}\n")
        log.flush()
        print(f"$ {line}", flush=True)
        if dry_run:
            return
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}: {line}")


def _ensure_default_a_only_labels(
    *,
    repo: Path,
    py: str,
    env: dict[str, str],
    dataset: str,
    mode: str,
    label_dir: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    if label_dir.is_dir() and any(label_dir.glob("*_labels.json")):
        return
    source_dir = repo / "outputs" / "agentguard" / "labels" / dataset / mode
    if not source_dir.is_dir() or not any(source_dir.glob("*_labels.json")):
        raise FileNotFoundError(
            f"A-only label directory not found: {label_dir}; "
            f"full label source also missing: {source_dir}. "
            "Run build_rollout_labels first."
        )
    cmd = [
        py,
        "scripts/agentguard/derive_a_only_labels.py",
        "--dataset",
        dataset,
        "--mode",
        mode,
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(label_dir),
        "--overwrite",
    ]
    _run(cmd, cwd=repo, env=env, log_path=log_path, dry_run=dry_run)


def _parse_tracking_log(path: Path) -> dict[str, Any]:
    combined: dict[str, float] | None = None
    per_sequence: dict[str, dict[str, float]] = {}
    lines = path.read_text(errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if METRIC_HEADER_RE.match(line):
            for row in lines[idx + 1: idx + 4]:
                match = METRIC_ROW_RE.match(row)
                if match:
                    hota, mota, idf1, deta, assa = map(float, match.groups())
                    combined = {
                        "HOTA": hota,
                        "MOTA": mota,
                        "IDF1": idf1,
                        "DetA": deta,
                        "AssA": assa,
                    }
                    break
        seq_match = SEQ_ROW_RE.match(line)
        if seq_match:
            seq, hota, mota, idf1, deta, assa = seq_match.groups()
            per_sequence[seq] = {
                "HOTA": float(hota),
                "MOTA": float(mota),
                "IDF1": float(idf1),
                "DetA": float(deta),
                "AssA": float(assa),
            }
    if combined is None:
        raise RuntimeError(f"Could not parse combined TrackEval metrics from {path}")
    return {"combined": combined, "per_sequence": per_sequence}


def _default_experiments() -> list[Experiment]:
    return [
        Experiment("a_only", "A", "A:1,B:0,C:0"),
    ]


def _bc_ablation_experiments() -> list[Experiment]:
    return [
        Experiment("a1_b005_c01", "A,B,C", "A:1,B:0.05,C:0.1"),
        Experiment("a1_b01_c02", "A,B,C", "A:1,B:0.1,C:0.2"),
        Experiment("a1_b025_c025", "A,B,C", "A:1,B:0.25,C:0.25"),
        Experiment("a1_b05_c05", "A,B,C", "A:1,B:0.5,C:0.5"),
        Experiment("a1_b1_c05", "A,B,C", "A:1,B:1,C:0.5"),
        Experiment("a1_b05_c1", "A,B,C", "A:1,B:0.5,C:1"),
        Experiment("a1_b1_c1", "A,B,C", "A:1,B:1,C:1"),
        Experiment("a1_b02_c08", "A,B,C", "A:1,B:0.2,C:0.8"),
    ]


def _selected_experiments(names: str, include_bc_ablations: bool) -> list[Experiment]:
    experiments = _default_experiments()
    if include_bc_ablations:
        experiments += _bc_ablation_experiments()
    if not names:
        return experiments
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    chosen = [exp for exp in experiments if exp.name in wanted]
    missing = wanted.difference({exp.name for exp in chosen})
    if missing:
        raise ValueError(f"Unknown experiment name(s): {sorted(missing)}")
    return chosen


def _parse_custom_experiments(values: list[str]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for value in values:
        parts = value.split("|")
        if len(parts) != 3:
            raise ValueError(
                "--custom-experiment must be 'name|candidate_types|candidate_weights', "
                "for example 'low_bc|A,B,C|A:1,B:0.1,C:0.2'"
            )
        name, candidate_types, candidate_weights = (part.strip() for part in parts)
        if not name:
            raise ValueError("custom experiment name cannot be empty")
        experiments.append(Experiment(name, candidate_types, candidate_weights))
    return experiments


def _parse_strides(value: str) -> list[int]:
    strides = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        stride = int(part)
        if stride <= 0:
            raise ValueError(f"Invalid TGR stride {stride}; expected positive integer")
        strides.append(stride)
    return strides or [1]


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = summary_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    fieldnames = [
        "experiment",
        "candidate_types",
        "candidate_weights",
        "tgr_window_stride",
        "tgr_frame_stride",
        "seed",
        "batch_size",
        "HOTA",
        "AssA",
        "IDF1",
        "MOTA",
        "DetA",
        "delta_HOTA",
        "delta_AssA",
        "delta_IDF1",
        "delta_MOTA",
        "tracking_log",
        "checkpoint_dir",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _infer_sequences(
    detection_cache_root: Path,
    dataset: str,
    mode: str,
    detector: str,
) -> list[str]:
    manifest_path = detection_cache_root / dataset / mode / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Detection cache manifest not found: {manifest_path}. "
            "Run scripts/agentguard/00_split_detection_cache.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    sequences = sorted(str(seq) for seq in manifest.get("sequences", {}).keys())
    if detector:
        suffix = f"-{detector.upper()}"
        filtered = [seq for seq in sequences if seq.upper().endswith(suffix)]
        if filtered:
            sequences = filtered
    if not sequences:
        raise RuntimeError(f"No sequences found in {manifest_path}")
    return sequences


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate AgentGuard Student-V0 A-only by default, with optional B/C ablations."
    )
    parser.add_argument("--dataset", default="MOT17")
    parser.add_argument("--mode", default="all", help="Use all for MOT17 train-all cache.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tgr-batch-size", type=int, default=0, help="TGR batch size. Defaults to --batch-size when 0.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tgr-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-max-samples", type=int, default=20000)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-policy", default="train_all", choices=["train_all", "sequence_holdout"])
    parser.add_argument("--experiments", default="", help="Comma-separated subset of experiment names.")
    parser.add_argument(
        "--include-bc-ablations",
        action="store_true",
        help="Include TrackTrack-specific B/C hard-candidate weight ablations.",
    )
    parser.add_argument(
        "--custom-experiment",
        action="append",
        default=[],
        help="Add one experiment as 'name|candidate_types|candidate_weights'. Can be repeated.",
    )
    parser.add_argument("--max-per-candidate-type", type=int, default=0)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--detector", default="FRCNN", help="Filter inferred MOT17 detector sequences; empty disables.")
    parser.add_argument("--sweep-root", default="outputs/agentguard/sweeps")
    parser.add_argument("--detection-cache-root", default="outputs/agentguard/detection_cache")
    parser.add_argument("--event-cache-root", default="outputs/agentguard/event_cache")
    parser.add_argument("--label-dir", default="", help="Default: outputs/agentguard/labels/<dataset>/<mode>_a_only")
    parser.add_argument("--agentguard-device", default="cpu", help="Online inference device; CPU is usually faster here.")
    parser.add_argument(
        "--tgr-strides",
        default="1,4",
        help="Comma-separated TGR train/inference strides. Default runs both 1 and 4.",
    )
    parser.add_argument("--tgr-frame-stride", type=int, default=4, help="Deprecated single-stride fallback; use --tgr-strides.")
    parser.add_argument("--replay-diff-threshold", type=float, default=0.15)
    parser.add_argument("--profile-every", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = _repo_root()
    py = _python(str(repo / ".venv" / "bin" / "python"))
    env = _env(repo)
    sweep_root = (repo / args.sweep_root).resolve()
    label_dir = args.label_dir or str(repo / "outputs" / "agentguard" / "labels" / args.dataset / f"{args.mode}_a_only")
    if not args.label_dir:
        _ensure_default_a_only_labels(
            repo=repo,
            py=py,
            env=env,
            dataset=args.dataset,
            mode=args.mode,
            label_dir=Path(label_dir),
            log_path=sweep_root / "derive_a_only_labels.log",
            dry_run=args.dry_run,
        )
    detection_cache_root_path = (repo / args.detection_cache_root).resolve()
    detection_cache_root = str(detection_cache_root_path)
    event_cache_root = str((repo / args.event_cache_root).resolve())
    sequences = args.sequences or _infer_sequences(
        detection_cache_root_path,
        args.dataset,
        args.mode,
        args.detector,
    )
    baseline_metrics = BASELINE_METRICS_BY_DATASET.get((args.dataset, args.mode), {})
    rows: list[dict[str, Any]] = []

    experiments = _selected_experiments(args.experiments, args.include_bc_ablations) + _parse_custom_experiments(args.custom_experiment)
    run_specs = [
        RunSpec(exp, stride)
        for exp in experiments
        for stride in _parse_strides(args.tgr_strides or str(args.tgr_frame_stride))
    ]
    seen_names: set[str] = set()
    for spec in run_specs:
        exp = spec.experiment
        if spec.name in seen_names:
            raise ValueError(f"Duplicate experiment name: {spec.name}")
        seen_names.add(spec.name)
        exp_dir = sweep_root / spec.name
        dataset_dir = exp_dir / "dataset"
        checkpoint_dir = exp_dir / "checkpoints"
        tracking_dir = exp_dir / "tracking_results"
        logs_dir = exp_dir / "logs"
        tracking_log = logs_dir / "tracktrack_full.log"
        metrics_json = exp_dir / "metrics.json"
        if args.skip_existing and metrics_json.is_file():
            rows.append(json.loads(metrics_json.read_text()))
            continue

        build_cmd = [
            py,
            "-m",
            "agentguard.cli",
            "build_student_v0_data",
            "--dataset",
            args.dataset,
            "--mode",
            args.mode,
            "--split-policy",
            args.split_policy,
            "--label-dir",
            label_dir,
            "--output-dir",
            str(dataset_dir),
            "--event-cache-root",
            event_cache_root,
            "--detection-cache-root",
            detection_cache_root,
            "--candidate-types",
            exp.candidate_types,
            "--candidate-weights",
            exp.candidate_weights,
            "--max-per-candidate-type",
            str(args.max_per_candidate_type or exp.max_per_candidate_type),
        ]
        _run(build_cmd, cwd=repo, env=env, log_path=logs_dir / "build_data.log", dry_run=args.dry_run)

        train_cmd = [
            py,
            "-m",
            "agentguard.cli",
            "train_student_v0",
            "--dataset",
            args.dataset,
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--tgr-batch-size",
            str(args.tgr_batch_size),
            "--lr",
            str(args.lr),
            "--tgr-lr",
            str(args.tgr_lr),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--tgr-window-stride",
            str(spec.tgr_stride),
            "--val-max-samples",
            str(args.val_max_samples),
            "--full-val-every",
            str(args.full_val_every),
            "--dataset-dir",
            str(dataset_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
        _run(train_cmd, cwd=repo, env=env, log_path=logs_dir / "train.log", dry_run=args.dry_run)

        track_cmd = [
            py,
            "run.py",
            "--dataset",
            args.dataset,
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--sequences",
            *sequences,
            "--output_dir",
            str(tracking_dir),
            "--detection-cache-root",
            detection_cache_root,
            "--resource-log",
            str(logs_dir / "resource.jsonl"),
            "--profile-every",
            str(args.profile_every),
            "--agentguard-mode",
            "full",
            "--iwg-checkpoint",
            str(checkpoint_dir / "iwg" / "iwg_best.pt"),
            "--tgr-checkpoint",
            str(checkpoint_dir / "tgr" / "tgr_best.pt"),
            "--agentguard-device",
            args.agentguard_device,
            "--agentguard-tgr-frame-stride",
            str(spec.tgr_stride),
            "--agentguard-replay-diff-threshold",
            str(args.replay_diff_threshold),
            "--tracker-suffix",
            spec.name,
            "--print-per-sequence-metrics",
        ]
        _run(track_cmd, cwd=repo / "3. Tracker", env=env, log_path=tracking_log, dry_run=args.dry_run)
        if args.dry_run:
            continue

        parsed = _parse_tracking_log(tracking_log)
        combined = parsed["combined"]
        row = {
            "experiment": spec.name,
            "candidate_types": exp.candidate_types,
            "candidate_weights": exp.candidate_weights,
            "tgr_window_stride": spec.tgr_stride,
            "tgr_frame_stride": spec.tgr_stride,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "tracking_log": str(tracking_log),
            "checkpoint_dir": str(checkpoint_dir),
            "per_sequence": parsed["per_sequence"],
            **combined,
        }
        if baseline_metrics:
            row.update(
                {
                    "delta_HOTA": combined["HOTA"] - baseline_metrics["HOTA"],
                    "delta_AssA": combined["AssA"] - baseline_metrics["AssA"],
                    "delta_IDF1": combined["IDF1"] - baseline_metrics["IDF1"],
                    "delta_MOTA": combined["MOTA"] - baseline_metrics["MOTA"],
                }
            )
        metrics_json.write_text(json.dumps(row, indent=2, sort_keys=True))
        rows.append(row)
        _write_summary(sweep_root / "summary.csv", rows)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    if rows:
        _write_summary(sweep_root / "summary.csv", rows)
        print(f"Wrote {sweep_root / 'summary.csv'}")
        print(f"Wrote {sweep_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
