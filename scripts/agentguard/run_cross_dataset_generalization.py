#!/usr/bin/env python3
"""Run the two AgentGuard cross-dataset generalization experiments serially.

The runner deliberately keeps these experiments outside the ablation tree. It
builds the official SportsMOT-train compact index when needed, trains each
direction with the current context=6 baseline protocol, and evaluates only the
epoch-65 checkpoint on the target split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agentguard" / "src"))
from agentguard.cli import validate_train_iwg_rg_cma_command
from agentguard.datasets.iwg_rg_cma_dataset import inspect_packed_train_data
PYTHON = ROOT / ".venv" / "bin" / "python"
TRACKER_DIR = ROOT / "3. Tracker"
GENERALIZATION_ROOT = ROOT / "outputs" / "agentguard" / "generalization"
DATASET_ROOT = ROOT / "outputs" / "agentguard" / "datasets" / "iwg_rg_cma"
LABEL_ROOT = ROOT / "outputs" / "agentguard" / "labels" / "iwg_rg_cma"
EVENT_CACHE_ROOT = ROOT / "outputs" / "agentguard" / "event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT = ROOT / "outputs" / "agentguard" / "detection_cache"
DATA_ROOT = Path("/home/shang/datasets")

MOT17_DATASET_DIR = Path(
    os.environ.get(
        "MOT17_DATASET_DIR",
        str(DATASET_ROOT / "MOT17" / "nsa_all_v3_compact"),
    )
)
SPORTSMOT_TRAIN_DATASET_DIR = Path(
    os.environ.get(
        "SPORTSMOT_TRAIN_DATASET_DIR",
        str(DATASET_ROOT / "SportsMOT" / "nsa_train_v3_compact"),
    )
)
SPORTSMOT_TRAINVAL_DATASET_DIR = Path(
    os.environ.get(
        "SPORTSMOT_TRAINVAL_DATASET_DIR",
        str(DATASET_ROOT / "SportsMOT" / "nsa_trainval_v3_compact"),
    )
)
SPORTSMOT_TRAIN_LABEL_DIR = LABEL_ROOT / "SportsMOT" / "nsa_train_v3_compact"

CONTEXT_SIZE = 6
SELECTED_EPOCH = 65
EPOCHS = 100
CHECKPOINT_EVERY = 5
SEED = 42

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
)


EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "experiment_id": "mot17_to_sportsmot_val_context6_100e_epoch065",
        "train_dataset": "MOT17",
        "train_split": "all",
        "dataset_dir": MOT17_DATASET_DIR,
        "eval_dataset": "SportsMOT",
        "eval_split": "val",
        "eval_data_dir": DATA_ROOT / "SportsMOT" / "dataset" / "val",
        "seqmap": TRACKER_DIR / "trackeval" / "seqmap" / "sportsmot" / "val.txt",
        "detection_split": "val",
        "tracker_prefix": "sportsmot_val_0.80",
    },
    {
        "experiment_id": "sportsmot_to_mot17_all_context6_100e_epoch065",
        "train_dataset": "SportsMOT",
        "train_split": "train",
        "dataset_dir": SPORTSMOT_TRAIN_DATASET_DIR,
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


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    """Run one command and stream its output into a durable log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    print(f"\n$ (cd {cwd} && {rendered})", flush=True)
    with log_path.open("w") as log:
        log.write(f"$ (cd {cwd} && {rendered})\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=_runtime_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {rendered}")


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


def _check_detection_and_event_caches(
    *, dataset: str, split: str, sequences: list[str]
) -> None:
    for sequence in sequences:
        detection_dir = DETECTION_CACHE_ROOT / dataset / split / sequence
        event_dir = EVENT_CACHE_ROOT / dataset / split / sequence
        for required in (detection_dir / "manifest.json", event_dir / "manifest.json"):
            if not required.is_file():
                raise FileNotFoundError(required)
        detection_manifest = json.loads(
            (detection_dir / "manifest.json").read_text()
        )
        event_manifest = json.loads((event_dir / "manifest.json").read_text())
        if not detection_manifest.get("complete", False):
            raise RuntimeError(f"incomplete detection cache: {detection_dir}")
        if not event_manifest.get("complete", False):
            raise RuntimeError(f"incomplete event cache: {event_dir}")


def _ensure_sportsmot_train_dataset(*, log_path: Path) -> None:
    if SPORTSMOT_TRAIN_DATASET_DIR.is_file():
        raise RuntimeError(f"dataset path is a file: {SPORTSMOT_TRAIN_DATASET_DIR}")
    if (SPORTSMOT_TRAIN_DATASET_DIR / "metadata.json").is_file():
        report = inspect_packed_train_data(SPORTSMOT_TRAIN_DATASET_DIR)
        expected = {"dataset": "SportsMOT", "split": "train", "context_size": CONTEXT_SIZE}
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(
                    f"existing SportsMOT train dataset has {key}="
                    f"{report.get(key)!r}, expected {value!r}"
                )
        return

    if not (SPORTSMOT_TRAIN_LABEL_DIR / "summary.json").is_file():
        raise FileNotFoundError(
            f"official SportsMOT train compact labels are missing: "
            f"{SPORTSMOT_TRAIN_LABEL_DIR}"
        )

    sequences = _read_seqmap(
        TRACKER_DIR / "trackeval" / "seqmap" / "sportsmot" / "train.txt"
    )
    _check_detection_and_event_caches(
        dataset="SportsMOT", split="train", sequences=sequences
    )
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "build_iwg_rg_cma_data",
        "--dataset",
        "SportsMOT",
        "--mode",
        "train",
        "--event-cache-root",
        str(EVENT_CACHE_ROOT),
        "--detection-cache-root",
        str(DETECTION_CACHE_ROOT),
        "--label-dir",
        str(SPORTSMOT_TRAIN_LABEL_DIR),
        "--output-dir",
        str(SPORTSMOT_TRAIN_DATASET_DIR),
        "--max-frame-gap",
        "30",
        "--context-size",
        str(CONTEXT_SIZE),
    ]
    _run_logged(command, cwd=ROOT, log_path=log_path)


def _training_command(experiment: dict[str, Any], checkpoint_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "train_iwg_rg_cma",
        "--dataset-dir",
        str(experiment["dataset_dir"]),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
        "--device",
        "cuda",
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        "1024",
        "--num-workers",
        "4",
        "--lr",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--warmup-epochs",
        "1",
        "--grad-clip",
        "1.0",
        "--seed",
        str(SEED),
        "--correction-bound",
        "0.05",
        "--context-size",
        str(CONTEXT_SIZE),
        "--architecture-variant",
        "legacy",
        "--memory-shards",
        "1",
        "--epochs-per-shard",
        str(EPOCHS),
        "--shard-cycles",
        "1",
        "--sequence-sampling",
        "sample-proportional",
    ]


def _validate_checkpoint(
    *, checkpoint: Path, dataset_dir: Path, output_path: Path, log_path: Path
) -> None:
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "validate_iwg_rg_cma_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--dataset-dir",
        str(dataset_dir),
        "--device",
        "cuda",
        "--max-batches",
        "8",
        "--output",
        str(output_path),
    ]
    _run_logged(command, cwd=ROOT, log_path=log_path)


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


def _expected_tracker_folder(experiment: dict[str, Any], tracker_suffix: str) -> str:
    # run.py appends the user suffix before the AgentGuard output suffix.
    return f"{experiment['tracker_prefix']}_{tracker_suffix}_iwg_rg_cma_final"


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
        summary = evaluate_sequences(args, tracker_folder, experiment["eval_dataset"], sequences)
    output = buffer.getvalue()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    print(output, end="", flush=True)
    return summary


def _summary_rows(
    *,
    experiment: dict[str, Any],
    checkpoint: Path,
    tracker_folder: str,
    summary: dict[str, Any],
    paths: dict[str, Path],
    git_commit: str,
) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, dict[str, Any]]] = [
        ("combined", "COMBINED_SEQ", summary["combined"])
    ]
    scopes.extend(
        ("sequence", sequence, summary["per_sequence"][sequence])
        for sequence in sorted(summary["per_sequence"])
    )
    rows = []
    for scope, sequence, metrics in scopes:
        row: dict[str, Any] = {
            "experiment_id": experiment["experiment_id"],
            "train_dataset": experiment["train_dataset"],
            "train_split": experiment["train_split"],
            "eval_dataset": experiment["eval_dataset"],
            "eval_split": experiment["eval_split"],
            "context_size": CONTEXT_SIZE,
            "epoch": SELECTED_EPOCH,
            "checkpoint": str(checkpoint.resolve()),
            "scope": scope,
            "sequence": sequence,
            "training_device": "cuda",
            "inference_device": "cpu",
            "git_commit": git_commit,
            "dataset_dir": str(experiment["dataset_dir"].resolve()),
            "config_path": str(paths["protocol"].resolve()),
            "train_log_path": str(paths["train_log"].resolve()),
            "track_log_path": str(paths["track_log"].resolve()),
            "evaluation_log_path": str(paths["evaluation_log"].resolve()),
            "tracker_folder": tracker_folder,
        }
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)
    return rows


def _write_summary(rows: list[dict[str, Any]]) -> None:
    SUMMARY_CSV = GENERALIZATION_ROOT / "summary.csv"
    existing: list[dict[str, Any]] = []
    if SUMMARY_CSV.is_file():
        with SUMMARY_CSV.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(SUMMARY_FIELDS):
                raise ValueError(
                    f"unexpected generalization summary schema: {reader.fieldnames}"
                )
            existing = list(reader)
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    for row in existing + rows:
        key = (str(row["experiment_id"]), str(row["sequence"]))
        if key in positions:
            merged[positions[key]] = row
        else:
            positions[key] = len(merged)
            merged.append(row)
    GENERALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(merged)


def _run_experiment(experiment: dict[str, Any], *, git_commit: str) -> list[dict[str, Any]]:
    run_root = GENERALIZATION_ROOT / experiment["experiment_id"]
    if (run_root / "running").exists():
        raise RuntimeError(f"run is already marked running: {run_root}")
    if (run_root / "completed").exists():
        raise RuntimeError(f"run is already completed: {run_root}")

    checkpoint_dir = run_root / "checkpoints"
    tracker_root = run_root / "tracker_outputs"
    logs = run_root / "logs"
    provenance = run_root / "provenance"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "running").touch()
    complete = False
    try:
        sequences = _read_seqmap(experiment["seqmap"])
        _check_detection_and_event_caches(
            dataset=experiment["eval_dataset"],
            split=experiment["detection_split"],
            sequences=sequences,
        )
        dataset_metadata = experiment["dataset_dir"] / "metadata.json"
        dataset_report = inspect_packed_train_data(experiment["dataset_dir"])
        for key, expected in (
            ("context_size", CONTEXT_SIZE),
            ("split", experiment["train_split"]),
        ):
            if dataset_report.get(key) != expected:
                raise ValueError(
                    f"{dataset_metadata}: {key}={dataset_report.get(key)!r}, "
                    f"expected {expected!r}"
                )

        paths = {
            "protocol": run_root / "protocol.json",
            "manifest": run_root / "manifest.json",
            "train_log": logs / "train.log",
            "checkpoint_validation_log": logs / "checkpoint_validation.log",
            "checkpoint_validation": run_root / "checkpoint_validation.json",
            "track_log": logs / "track.log",
            "evaluation_log": logs / "evaluation.log",
        }
        protocol = {
            "experiment_id": experiment["experiment_id"],
            "purpose": "cross-dataset generalization",
            "train_dataset": experiment["train_dataset"],
            "train_split": experiment["train_split"],
            "eval_dataset": experiment["eval_dataset"],
            "eval_split": experiment["eval_split"],
            "dataset_dir": str(experiment["dataset_dir"].resolve()),
            "dataset_metadata_sha256": _sha256(dataset_metadata),
            "context_size": CONTEXT_SIZE,
            "epochs": EPOCHS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "selected_epoch": SELECTED_EPOCH,
            "seed": SEED,
            "architecture_variant": "legacy",
            "sequence_sampling": "sample-proportional",
            "memory_shards": 1,
            "epochs_per_shard": EPOCHS,
            "shard_cycles": 1,
            "training_device": "cuda",
            "inference_device": "cpu",
            "post_processing": "disabled",
            "sequence_map": str(experiment["seqmap"].resolve()),
            "sequences": sequences,
            "git_commit": git_commit,
            "git_status_porcelain": _git_value(["git", "status", "--porcelain"], ""),
        }
        _write_json(paths["protocol"], protocol)
        _write_json(
            paths["manifest"],
            {
                **protocol,
                "checkpoint": str(
                    checkpoint_dir / f"iwg_rg_cma_epoch{SELECTED_EPOCH:03d}.pt"
                ),
                "tracker_root": str(tracker_root),
            },
        )
        provenance.mkdir(parents=True, exist_ok=True)
        (provenance / "git_commit.txt").write_text(git_commit + "\n")
        (provenance / "git_status_porcelain.txt").write_text(
            protocol["git_status_porcelain"] + "\n"
        )
        (provenance / "dataset_metadata.sha256").write_text(
            _sha256(dataset_metadata) + "  metadata.json\n"
        )

        checkpoint = checkpoint_dir / f"iwg_rg_cma_epoch{SELECTED_EPOCH:03d}.pt"
        training_summary_path = checkpoint_dir / "training_summary.json"
        training_complete = False
        if checkpoint.is_file() and training_summary_path.is_file():
            try:
                training_summary = json.loads(training_summary_path.read_text())
                training_complete = (
                    training_summary.get("status") == "completed"
                    and float(training_summary.get("effective_full_epochs", 0.0))
                    >= EPOCHS
                )
            except (OSError, ValueError, TypeError):
                training_complete = False
        train_command = _training_command(experiment, checkpoint_dir)
        train_config = validate_train_iwg_rg_cma_command(
            train_command, selected_epoch=SELECTED_EPOCH
        )
        protocol["checkpoint_epochs"] = list(train_config["checkpoint_epochs"])
        protocol["checkpoint_every"] = train_config["checkpoint_every"]
        _write_json(paths["protocol"], protocol)
        (provenance / "train.command.txt").write_text(shlex.join(train_command) + "\n")
        if training_complete:
            print(f"Reusing completed training: {training_summary_path}", flush=True)
        else:
            _run_logged(train_command, cwd=ROOT, log_path=paths["train_log"])

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"selected checkpoint was not produced: {checkpoint}"
            )
        (provenance / "selected_checkpoint.sha256").write_text(
            _sha256(checkpoint) + f"  {checkpoint.name}\n"
        )
        if paths["checkpoint_validation"].is_file():
            print(
                f"Reusing checkpoint validation: {paths['checkpoint_validation']}",
                flush=True,
            )
        else:
            _validate_checkpoint(
                checkpoint=checkpoint,
                dataset_dir=experiment["dataset_dir"],
                output_path=paths["checkpoint_validation"],
                log_path=paths["checkpoint_validation_log"],
            )

        tracker_suffix = f"generalization_{experiment['experiment_id']}_e{SELECTED_EPOCH:03d}"
        tracker_folder = _expected_tracker_folder(experiment, tracker_suffix)
        tracker_path = tracker_root / tracker_folder
        if tracker_path.exists():
            missing = [
                sequence
                for sequence in sequences
                if not (tracker_path / f"{sequence}.txt").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"tracker output is incomplete: {tracker_path}; missing {missing}"
                )
            print(f"Reusing tracker output: {tracker_path}", flush=True)
        else:
            track_command = _tracker_command(
                experiment=experiment,
                checkpoint=checkpoint,
                tracker_root=tracker_root,
                tracker_suffix=tracker_suffix,
                sequences=sequences,
            )
            (provenance / "track.command.txt").write_text(
                shlex.join(track_command) + "\n"
            )
            _run_logged(track_command, cwd=TRACKER_DIR, log_path=paths["track_log"])
            missing = [
                sequence
                for sequence in sequences
                if not (tracker_path / f"{sequence}.txt").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"tracker output is incomplete: {tracker_path}; missing {missing}"
                )

        summary = _evaluate_once(
            experiment=experiment,
            tracker_root=tracker_root,
            tracker_folder=tracker_folder,
            sequences=sequences,
            log_path=paths["evaluation_log"],
        )
        _write_json(run_root / "metrics.json", summary)
        rows = _summary_rows(
            experiment=experiment,
            checkpoint=checkpoint,
            tracker_folder=tracker_folder,
            summary=summary,
            paths=paths,
            git_commit=git_commit,
        )
        _write_summary(rows)
        complete = True
        (run_root / "completed").touch()
        return rows
    finally:
        (run_root / "running").unlink(missing_ok=True)
        (run_root / "pipeline.exit_code").write_text(
            ("0" if complete else "1") + "\n"
        )
        if not complete:
            (run_root / "failed").touch()


def main() -> None:
    GENERALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    if (GENERALIZATION_ROOT / "running").exists():
        raise SystemExit(
            f"generalization runner is already marked running: {GENERALIZATION_ROOT}"
        )
    if (GENERALIZATION_ROOT / "completed").exists():
        raise SystemExit(
            f"generalization runner is already completed: {GENERALIZATION_ROOT}"
        )

    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    inspect_packed_train_data(MOT17_DATASET_DIR)
    if not SPORTSMOT_TRAINVAL_DATASET_DIR.joinpath("metadata.json").is_file():
        raise FileNotFoundError(SPORTSMOT_TRAINVAL_DATASET_DIR / "metadata.json")

    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    protocol = {
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "git_commit": git_commit,
        "experiments": [
            {
                "experiment_id": item["experiment_id"],
                "train_dataset": item["train_dataset"],
                "train_split": item["train_split"],
                "eval_dataset": item["eval_dataset"],
                "eval_split": item["eval_split"],
            }
            for item in EXPERIMENTS
        ],
        "shared_training_config": {
            "epochs": EPOCHS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "batch_size": 1024,
            "num_workers": 4,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "warmup_epochs": 1,
            "grad_clip": 1.0,
            "seed": SEED,
            "correction_bound": 0.05,
            "context_size": CONTEXT_SIZE,
            "architecture_variant": "legacy",
            "memory_shards": 1,
            "epochs_per_shard": EPOCHS,
            "shard_cycles": 1,
            "sequence_sampling": "sample-proportional",
        },
        "training_device": "cuda",
        "inference_device": "cpu",
        "selected_epoch": SELECTED_EPOCH,
        "post_processing": False,
        "sportsmot_train_dataset_note": (
            "Built from official SportsMOT train compact labels and train caches; "
            "SportsMOT val is not included in this training dataset."
        ),
    }
    _write_json(GENERALIZATION_ROOT / "protocol.json", protocol)
    (GENERALIZATION_ROOT / "running").touch()
    complete = False
    rows: list[dict[str, Any]] = []
    try:
        _ensure_sportsmot_train_dataset(
            log_path=GENERALIZATION_ROOT / "preparation" / "build_sportsmot_train_dataset.log"
        )
        for experiment in EXPERIMENTS:
            rows.extend(_run_experiment(experiment, git_commit=git_commit))
        _write_summary(rows)
        complete = True
        (GENERALIZATION_ROOT / "completed").touch()
    finally:
        (GENERALIZATION_ROOT / "running").unlink(missing_ok=True)
        (GENERALIZATION_ROOT / "pipeline.exit_code").write_text(
            ("0" if complete else "1") + "\n"
        )
        if not complete:
            (GENERALIZATION_ROOT / "failed").touch()


if __name__ == "__main__":
    main()
