#!/usr/bin/env python3
"""Evaluate every saved cross-dataset checkpoint serially on the target split.

Training has already been completed by ``run_cross_dataset_generalization.py``.
This runner only performs CPU tracker inference and TrackEval evaluation for
epochs 5, 10, ..., 100 in both directions.  The previously completed epoch 65
results are reused and copied into the all-checkpoints result tree.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
TRACKER_DIR = ROOT / "3. Tracker"
GENERALIZATION_ROOT = ROOT / "outputs" / "agentguard" / "generalization"
ALL_ROOT = GENERALIZATION_ROOT / "all_checkpoints"
DATASET_ROOT = ROOT / "outputs" / "agentguard" / "datasets" / "iwg_rg_cma"
EVENT_CACHE_ROOT = ROOT / "outputs" / "agentguard" / "event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT = ROOT / "outputs" / "agentguard" / "detection_cache"
DATA_ROOT = Path("/home/shang/datasets")

CONTEXT_SIZE = 6
TRAINING_EPOCHS = 100
CHECKPOINT_EPOCHS = tuple(range(5, TRAINING_EPOCHS + 1, 5))
SEED = 42
SELECTED_REUSE_EPOCH = 65

METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")
SUMMARY_FIELDS = (
    "experiment_id",
    "train_dataset",
    "train_split",
    "eval_dataset",
    "eval_split",
    "context_size",
    "epoch",
    "checkpoint",
    "scope",
    "sequence",
    *METRICS,
    "training_device",
    "inference_device",
    "git_commit",
    "dataset_dir",
    "config_path",
    "train_log_path",
    "track_log_path",
    "evaluation_log_path",
    "tracker_folder",
    "tracker_source",
)


EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "experiment_id": "mot17_to_sportsmot_val_all_checkpoints",
        "legacy_id": "mot17_to_sportsmot_val_context6_100e_epoch065",
        "train_dataset": "MOT17",
        "train_split": "all",
        "dataset_dir": DATASET_ROOT / "MOT17" / "nsa_all_v3_compact",
        "eval_dataset": "SportsMOT",
        "eval_split": "val",
        "eval_data_dir": DATA_ROOT / "SportsMOT" / "dataset" / "val",
        "seqmap": TRACKER_DIR / "trackeval" / "seqmap" / "sportsmot" / "val.txt",
        "detection_split": "val",
        "tracker_prefix": "sportsmot_val_0.80",
    },
    {
        "experiment_id": "sportsmot_to_mot17_all_all_checkpoints",
        "legacy_id": "sportsmot_to_mot17_all_context6_100e_epoch065",
        "train_dataset": "SportsMOT",
        "train_split": "train",
        "dataset_dir": DATASET_ROOT / "SportsMOT" / "nsa_train_v3_compact",
        "eval_dataset": "MOT17",
        "eval_split": "all",
        "eval_data_dir": DATA_ROOT / "MOT17" / "train",
        "seqmap": TRACKER_DIR / "trackeval" / "seqmap" / "mot17" / "all.txt",
        "detection_split": "all",
        "tracker_prefix": "mot17_all_0.80",
    },
)


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(ROOT), str(TRACKER_DIR), str(ROOT / "agentguard" / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_seqmap(path: Path) -> list[str]:
    sequences = []
    for line in path.read_text().splitlines():
        name = line.strip()
        if not name or name.lower() == "name" or name.startswith("#"):
            continue
        sequences.append(name)
    if not sequences:
        raise RuntimeError(f"sequence map is empty: {path}")
    return sequences


def _check_detection_caches(
    *, dataset: str, split: str, sequences: list[str]
) -> None:
    for sequence in sequences:
        detection_dir = DETECTION_CACHE_ROOT / dataset / split / sequence
        manifest = detection_dir / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        metadata = json.loads(manifest.read_text())
        if not metadata.get("complete", False):
            raise RuntimeError(f"incomplete detection cache: {detection_dir}")


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    """Run a tracker command without flooding the parent terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    print(f"START {log_path.parent.parent.name}: {rendered}", flush=True)
    with log_path.open("w") as log:
        log.write(f"$ (cd {cwd} && {rendered})\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=_runtime_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"tracker command failed with exit code {return_code}")
    print(f"DONE  {log_path.parent.parent.name}", flush=True)


def _tracker_command(
    *,
    experiment: dict[str, Any],
    checkpoint: Path,
    tracker_root: Path,
    tracker_suffix: str,
    sequences: list[str],
) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        "run.py",
        "--dataset",
        experiment["eval_dataset"],
        "--mode",
        experiment["eval_split"],
        "--data_dir",
        str(DATA_ROOT) + "/",
        "--output_dir",
        str(tracker_root),
        "--agentguard-mode",
        "iwg-rg-cma",
        "--agentguard-checkpoint",
        str(checkpoint),
        "--agentguard-device",
        "cpu",
        "--detection-cache-root",
        str(DETECTION_CACHE_ROOT),
        "--tracker-suffix",
        tracker_suffix,
        "--sequences",
        *sequences,
        "--skip-eval",
    ]


def _tracker_folder(experiment: dict[str, Any], suffix: str) -> str:
    return f"{experiment['tracker_prefix']}_{suffix}_iwg_rg_cma_final"


def _evaluate_once(
    *,
    experiment: dict[str, Any],
    tracker_root: Path,
    tracker_folder: str,
    sequences: list[str],
    log_path: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(TRACKER_DIR))
    from utils.etc import evaluate_sequences

    args = SimpleNamespace(
        mode=experiment["eval_split"],
        data_path=str(experiment["eval_data_dir"]) + "/",
        output_dir=str(tracker_root),
        print_per_sequence_metrics=False,
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        summary = evaluate_sequences(
            args, tracker_folder, experiment["eval_dataset"], sequences
        )
    output = buffer.getvalue()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    print(output, end="", flush=True)
    return summary


def _rows(
    *,
    experiment: dict[str, Any],
    epoch: int,
    checkpoint: Path,
    tracker_root: Path,
    tracker_folder: str,
    summary: dict[str, Any],
    paths: dict[str, Path],
    git_commit: str,
    tracker_source: str,
) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, dict[str, Any]]] = [
        ("combined", "COMBINED_SEQ", summary["combined"])
    ]
    scopes.extend(
        ("sequence", sequence, summary["per_sequence"][sequence])
        for sequence in sorted(summary["per_sequence"])
    )
    result = []
    for scope, sequence, metrics in scopes:
        row: dict[str, Any] = {
            "experiment_id": experiment["experiment_id"],
            "train_dataset": experiment["train_dataset"],
            "train_split": experiment["train_split"],
            "eval_dataset": experiment["eval_dataset"],
            "eval_split": experiment["eval_split"],
            "context_size": CONTEXT_SIZE,
            "epoch": epoch,
            "checkpoint": str(checkpoint.resolve()),
            "scope": scope,
            "sequence": sequence,
            "training_device": "cuda",
            "inference_device": "cpu",
            "git_commit": git_commit,
            "dataset_dir": str(experiment["dataset_dir"].resolve()),
            "config_path": str(paths["protocol"].resolve()),
            "train_log_path": str(
                (GENERALIZATION_ROOT / experiment["legacy_id"] / "logs" / "train.log").resolve()
            ),
            "track_log_path": str(paths["track"].resolve()),
            "evaluation_log_path": str(paths["evaluation"].resolve()),
            "tracker_folder": str((tracker_root / tracker_folder).resolve()),
            "tracker_source": tracker_source,
        }
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        result.append(row)
    return result


def _write_summary(rows: list[dict[str, Any]]) -> None:
    path = ALL_ROOT / "summary.csv"
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(SUMMARY_FIELDS):
                raise ValueError(f"unexpected all-checkpoints schema: {reader.fieldnames}")
            existing = list(reader)

    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, str, str, str], int] = {}
    for row in existing + rows:
        key = (
            str(row["experiment_id"]),
            str(row["epoch"]),
            str(row["scope"]),
            str(row["sequence"]),
        )
        if key in positions:
            merged[positions[key]] = row
        else:
            positions[key] = len(merged)
            merged.append(row)

    ALL_ROOT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(merged)


def _checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"iwg_rg_cma_epoch(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"unexpected checkpoint name: {path}")
    return int(match.group(1))


def _reuse_epoch65(
    *,
    experiment: dict[str, Any],
    epoch: int,
    checkpoint: Path,
    run_root: Path,
    sequences: list[str],
) -> tuple[dict[str, Any], dict[str, Path], Path, str, str] | None:
    if epoch != SELECTED_REUSE_EPOCH:
        return None
    legacy_root = GENERALIZATION_ROOT / experiment["legacy_id"]
    legacy_metrics = legacy_root / "metrics.json"
    legacy_tracker_root = legacy_root / "tracker_outputs"
    suffix = f"generalization_{experiment['legacy_id']}_e{epoch:03d}"
    tracker_folder = _tracker_folder(experiment, suffix)
    tracker_path = legacy_tracker_root / tracker_folder
    if not legacy_metrics.is_file() or not tracker_path.is_dir():
        return None
    missing = [
        sequence
        for sequence in sequences
        if not (tracker_path / f"{sequence}.txt").is_file()
    ]
    if missing:
        return None

    run_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol": run_root / "protocol.json",
        "track": run_root / "logs" / "track.log",
        "evaluation": run_root / "logs" / "evaluation.log",
    }
    summary = json.loads(legacy_metrics.read_text())
    paths["track"].parent.mkdir(parents=True, exist_ok=True)
    paths["track"].write_text(f"Reused tracker output from {legacy_root}\n")
    paths["evaluation"].write_text(
        f"Reused evaluated metrics from {legacy_metrics}\n"
    )
    _write_json(
        run_root / "metrics.json",
        {**summary, "reused_from": str(legacy_root.resolve())},
    )
    return summary, paths, legacy_tracker_root, tracker_folder, "reused_epoch065"


def _run_epoch(
    *,
    experiment: dict[str, Any],
    epoch: int,
    checkpoint: Path,
    git_commit: str,
) -> list[dict[str, Any]]:
    run_root = ALL_ROOT / experiment["experiment_id"] / f"epoch{epoch:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    sequences = _read_seqmap(experiment["seqmap"])
    tracker_root = run_root / "tracker_outputs"
    tracker_suffix = f"all_checkpoints_{experiment['experiment_id']}_e{epoch:03d}"
    tracker_folder = _tracker_folder(experiment, tracker_suffix)
    tracker_path = tracker_root / tracker_folder
    paths = {
        "protocol": run_root / "protocol.json",
        "track": run_root / "logs" / "track.log",
        "evaluation": run_root / "logs" / "evaluation.log",
    }

    reused = _reuse_epoch65(
        experiment=experiment,
        epoch=epoch,
        checkpoint=checkpoint,
        run_root=run_root,
        sequences=sequences,
    )
    if reused is not None:
        summary, paths, tracker_root, tracker_folder, source = reused
    else:
        if tracker_path.exists():
            missing = [
                sequence
                for sequence in sequences
                if not (tracker_path / f"{sequence}.txt").is_file()
            ]
            if missing:
                print(
                    f"Existing partial tracker output will be completed: {tracker_path}",
                    flush=True,
                )
        command = _tracker_command(
            experiment=experiment,
            checkpoint=checkpoint,
            tracker_root=tracker_root,
            tracker_suffix=tracker_suffix,
            sequences=sequences,
        )
        _run_logged(command, cwd=TRACKER_DIR, log_path=paths["track"])
        missing = [
            sequence
            for sequence in sequences
            if not (tracker_path / f"{sequence}.txt").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing tracker outputs: {missing}")
        summary = _evaluate_once(
            experiment=experiment,
            tracker_root=tracker_root,
            tracker_folder=tracker_folder,
            sequences=sequences,
            log_path=paths["evaluation"],
        )
        _write_json(run_root / "metrics.json", summary)
        source = "new_cpu_inference"

    protocol = {
        "experiment_id": experiment["experiment_id"],
        "epoch": epoch,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "train_dataset": experiment["train_dataset"],
        "train_split": experiment["train_split"],
        "eval_dataset": experiment["eval_dataset"],
        "eval_split": experiment["eval_split"],
        "context_size": CONTEXT_SIZE,
        "seed": SEED,
        "training_device": "cuda",
        "inference_device": "cpu",
        "post_processing": "disabled",
        "sequences": sequences,
        "tracker_folder": str((tracker_root / tracker_folder).resolve()),
        "tracker_source": source,
        "git_commit": git_commit,
    }
    _write_json(paths["protocol"], protocol)
    rows = _rows(
        experiment=experiment,
        epoch=epoch,
        checkpoint=checkpoint,
        tracker_root=tracker_root,
        tracker_folder=tracker_folder,
        summary=summary,
        paths=paths,
        git_commit=git_commit,
        tracker_source=source,
    )
    _write_summary(rows)
    (run_root / "completed").touch()
    print(
        f"METRICS {experiment['experiment_id']} epoch={epoch:03d} "
        f"HOTA={summary['combined']['HOTA']:.6f} "
        f"MOTA={summary['combined']['MOTA']:.6f} "
        f"IDF1={summary['combined']['IDF1']:.6f}",
        flush=True,
    )
    return rows


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    ALL_ROOT.mkdir(parents=True, exist_ok=True)
    if (ALL_ROOT / "running").exists():
        raise SystemExit(f"all-checkpoints runner is already running: {ALL_ROOT}")
    if (ALL_ROOT / "completed").exists():
        raise SystemExit(f"all-checkpoints runner is already complete: {ALL_ROOT}")

    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    protocol = {
        "runner": str(Path(__file__).resolve()),
        "git_commit": git_commit,
        "epochs": list(CHECKPOINT_EPOCHS),
        "checkpoint_every": 5,
        "training_device": "cuda",
        "inference_device": "cpu",
        "post_processing": False,
        "reuse_epoch065": True,
        "experiments": [
            {
                "experiment_id": item["experiment_id"],
                "legacy_training_run": item["legacy_id"],
                "train_dataset": item["train_dataset"],
                "eval_dataset": item["eval_dataset"],
                "eval_split": item["eval_split"],
            }
            for item in EXPERIMENTS
        ],
    }
    _write_json(ALL_ROOT / "protocol.json", protocol)
    (ALL_ROOT / "running").touch()
    complete = False
    try:
        for experiment in EXPERIMENTS:
            checkpoint_dir = (
                GENERALIZATION_ROOT / experiment["legacy_id"] / "checkpoints"
            )
            for epoch in CHECKPOINT_EPOCHS:
                checkpoint = checkpoint_dir / f"iwg_rg_cma_epoch{epoch:03d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                _check_detection_caches(
                    dataset=experiment["eval_dataset"],
                    split=experiment["detection_split"],
                    sequences=_read_seqmap(experiment["seqmap"]),
                )
                run_root = (
                    ALL_ROOT / experiment["experiment_id"] / f"epoch{epoch:03d}"
                )
                if (run_root / "completed").is_file():
                    print(
                        f"SKIP  {experiment['experiment_id']} epoch={epoch:03d}",
                        flush=True,
                    )
                    continue
                _run_epoch(
                    experiment=experiment,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    git_commit=git_commit,
                )
        complete = True
        (ALL_ROOT / "completed").touch()
    finally:
        (ALL_ROOT / "running").unlink(missing_ok=True)
        (ALL_ROOT / "pipeline.exit_code").write_text("0\n" if complete else "1\n")
        if not complete:
            (ALL_ROOT / "failed").touch()


if __name__ == "__main__":
    main()
