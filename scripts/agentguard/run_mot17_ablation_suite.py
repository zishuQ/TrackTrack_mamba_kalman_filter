#!/usr/bin/env python3
"""Build and run the MOT17 AgentGuard ablation suite serially."""

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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
ABLATION_ROOT = ROOT / "outputs/agentguard/ablation"
DATASET_ROOT = ABLATION_ROOT / "datasets"
TRACKER_ROOT = ABLATION_ROOT / "tracker_outputs"
SUMMARY_CSV = ABLATION_ROOT / "summary.csv"
DATASET_DIR_CONTEXT6 = (
    ROOT / "outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_compact"
)
EVENT_CACHE_ROOT = ROOT / "outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT = ROOT / "outputs/agentguard/detection_cache"
LABEL_DIR = ROOT / "outputs/agentguard/labels/iwg_rg_cma/MOT17/nsa_candidate_a_compact"
DATA_ROOT = Path("/home/shang/datasets")
BASELINE_CHECKPOINT = (
    ABLATION_ROOT
    / "mot17_all_tracktrack_gain_02_09_10"
    / "checkpoints"
    / "iwg_rg_cma_epoch065.pt"
)
REFERENCE_TRACKER_ROOT = ROOT / "outputs/3. track"
REFERENCE_TRACKTRACK_FOLDER = "mot17_all_0.80"
REFERENCE_BASELINE_FOLDER = (
    "mot17_all_0.80_mot17_baseline_e065_final_all_raw_iwg_rg_cma_final"
)
SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
)
SEQUENCE_SET = "+".join(SEQUENCES)
TRACKER_PREFIX = "mot17_all_0.80"
METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")
RATE_METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")
COUNT_METRICS = ("IDSW", "Frag")
CSV_FIELDS = (
    "experiment_id",
    "method",
    "architecture_variant",
    "dataset",
    "split",
    "evaluation",
    "scope",
    "sequence",
    "sequence_set",
    "seq1",
    "seq2",
    "seq3",
    "epoch",
    "checkpoint",
    "tracker_folder",
    *METRICS,
    *(f"delta_{metric}_pp" for metric in RATE_METRICS),
    *(f"delta_{metric}" for metric in COUNT_METRICS),
    "source_artifact",
    "git_commit",
)
SUMMARY_KEY_FIELDS = ("experiment_id", "method", "scope", "sequence")


RUNS: tuple[dict[str, Any], ...] = (
    {
        "experiment_id": "mot17_iwg_policy_only_100e_epoch065",
        "method": "AgentGuard IWG policy-only",
        "architecture_variant": "iwg-policy-only",
        "context_size": 6,
        "dataset_dir": DATASET_DIR_CONTEXT6,
        "init_checkpoint": "",
        "freeze_iwg": False,
    },
    {
        "experiment_id": "mot17_iwg_direct_2gate_100e_epoch065",
        "method": "AgentGuard IWG direct-2gate",
        "architecture_variant": "iwg-direct-2gate",
        "context_size": 6,
        "dataset_dir": DATASET_DIR_CONTEXT6,
        "init_checkpoint": "",
        "freeze_iwg": False,
    },
    {
        "experiment_id": "mot17_cma_without_reliability_token_100e_epoch065",
        "method": "AgentGuard CMA without reliability token",
        "architecture_variant": "cma-without-reliability-token",
        "context_size": 6,
        "dataset_dir": DATASET_DIR_CONTEXT6,
        "init_checkpoint": BASELINE_CHECKPOINT,
        "freeze_iwg": True,
    },
    {
        "experiment_id": "mot17_cma_without_cross_evidence_100e_epoch065",
        "method": "AgentGuard CMA without cross-evidence",
        "architecture_variant": "cma-without-cross-evidence",
        "context_size": 6,
        "dataset_dir": DATASET_DIR_CONTEXT6,
        "init_checkpoint": BASELINE_CHECKPOINT,
        "freeze_iwg": True,
    },
)


for _context_size in (1, 2, 4, 8, 10):
    RUNS += (
        {
            "experiment_id": f"mot17_context{_context_size}_100e_epoch065",
            "method": f"AgentGuard context={_context_size}",
            "architecture_variant": "legacy",
            "context_size": _context_size,
            "dataset_dir": DATASET_ROOT / f"mot17_context{_context_size}_v3_compact",
            "init_checkpoint": "",
            "freeze_iwg": False,
        },
    )


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(ROOT), str(ROOT / "3. Tracker"), str(ROOT / "agentguard" / "src")]
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


def _evaluate_tracker(
    tracker_root: Path,
    tracker_folder: str,
    *,
    log_path: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "3. Tracker"))
    from utils.etc import evaluate_sequences

    args = type(
        "EvaluationArgs",
        (),
        {
            "mode": "all",
            "data_path": str(DATA_ROOT / "MOT17" / "train") + "/",
            "output_dir": str(tracker_root),
            "print_per_sequence_metrics": False,
        },
    )()
    buffer = StringIO()
    with redirect_stdout(buffer):
        summary = evaluate_sequences(args, tracker_folder, "MOT17", list(SEQUENCES))
    output = buffer.getvalue()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    print(output, end="", flush=True)
    return summary


def _tracker_ready(path: Path) -> bool:
    return path.is_dir() and all(
        (path / f"{sequence}.txt").is_file() for sequence in SEQUENCES
    )


def _summary_rows(
    *,
    experiment_id: str,
    method: str,
    architecture_variant: str,
    checkpoint: Path,
    tracker_folder: str,
    source_artifact: Path,
    summary: dict[str, Any],
    git_commit: str,
) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, dict[str, Any]]] = [
        ("combined", "COMBINED_SEQ", summary["combined"])
    ]
    scopes.extend(
        ("sequence", sequence, summary["per_sequence"][sequence])
        for sequence in SEQUENCES
    )
    rows = []
    for scope, sequence, metrics in scopes:
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "method": method,
            "architecture_variant": architecture_variant,
            "dataset": "MOT17",
            "split": "all",
            "evaluation": "raw_final_gate",
            "scope": scope,
            "sequence": sequence,
            "sequence_set": SEQUENCE_SET,
            "seq1": SEQUENCES[0],
            "seq2": SEQUENCES[1],
            "seq3": SEQUENCES[2],
            "epoch": 65,
            "checkpoint": str(checkpoint.resolve()),
            "tracker_folder": tracker_folder,
            "source_artifact": str(source_artifact.resolve()),
            "git_commit": git_commit,
        }
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)
    return rows


def _upsert_summary(rows: list[dict[str, Any]]) -> None:
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(
            f"shared ablation summary is missing: {SUMMARY_CSV}; "
            "preserve the existing reference rows before running the suite"
        )
    with SUMMARY_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                f"unexpected ablation summary schema: {reader.fieldnames}"
            )
        existing = list(reader)
    references = {
        (row["scope"], row["sequence"]): row
        for row in existing
        if row["method"] == "TrackTrack"
    }
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, ...], int] = {}
    for row in existing + rows:
        key = tuple(str(row.get(field, "")) for field in SUMMARY_KEY_FIELDS)
        if key in positions:
            merged[positions[key]] = row
        else:
            positions[key] = len(merged)
            merged.append(row)
    for row in merged:
        reference = references.get((row["scope"], row["sequence"]))
        if reference is None:
            raise ValueError(
                f"TrackTrack reference missing for {row['scope']}/{row['sequence']}"
            )
        for metric in RATE_METRICS:
            row[f"delta_{metric}_pp"] = (
                float(row[metric]) - float(reference[metric])
            ) * 100.0
        for metric in COUNT_METRICS:
            row[f"delta_{metric}"] = float(row[metric]) - float(
                reference[metric]
            )
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(merged)


def _build_temporal_datasets() -> None:
    for context_size in (1, 2, 4, 8, 10):
        output_dir = DATASET_ROOT / f"mot17_context{context_size}_v3_compact"
        metadata_path = output_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if int(metadata.get("context_size", -1)) != context_size:
                raise ValueError(f"dataset context mismatch: {output_dir}")
            print(f"Using existing temporal dataset: {output_dir}", flush=True)
            continue
        command = [
            str(PYTHON),
            "-u",
            "-m",
            "agentguard.cli",
            "build_iwg_rg_cma_data",
            "--dataset",
            "MOT17",
            "--mode",
            "all",
            "--event-cache-root",
            str(EVENT_CACHE_ROOT),
            "--detection-cache-root",
            str(DETECTION_CACHE_ROOT),
            "--label-dir",
            str(LABEL_DIR),
            "--output-dir",
            str(output_dir),
            "--max-frame-gap",
            "30",
            "--context-size",
            str(context_size),
        ]
        log_path = ABLATION_ROOT / "logs" / f"build_context{context_size}.log"
        _run_logged(command, cwd=ROOT, log_path=log_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)


def _protocol(spec: dict[str, Any], run_root: Path) -> dict[str, Any]:
    return {
        "experiment_id": spec["experiment_id"],
        "dataset": "MOT17",
        "split": "all",
        "sequence_set": list(SEQUENCES),
        "training": {
            "dataset_dir": str(Path(spec["dataset_dir"]).resolve()),
            "device": "cuda",
            "architecture_variant": spec["architecture_variant"],
            "context_size": spec["context_size"],
            "epochs": 100,
            "selected_epoch": 65,
            "checkpoint_every": 5,
            "batch_size": 1024,
            "num_workers": 4,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "grad_clip": 1.0,
            "seed": 42,
            "correction_bound": 0.05,
            "memory_shards": 1,
            "epochs_per_shard": 100,
            "shard_cycles": 1,
            "sequence_sampling": "sample-proportional",
            "init_checkpoint": (
                str(Path(spec["init_checkpoint"]).resolve())
                if spec["init_checkpoint"]
                else ""
            ),
            "freeze_iwg": bool(spec["freeze_iwg"]),
        },
        "inference": {
            "device": "cpu",
            "evaluation": "raw_final_gate",
            "tracker_seed": 10000,
            "detection_cache_root": str(DETECTION_CACHE_ROOT.resolve()),
            "tracker_root": str(TRACKER_ROOT.resolve()),
        },
        "selection": "fixed epoch 65; no best-of-epoch selection",
        "execution": "serial build, training, tracking, and evaluation",
        "run_root": str(run_root.resolve()),
    }


def _run_one(spec: dict[str, Any], git_commit: str) -> None:
    run_root = ABLATION_ROOT / str(spec["experiment_id"])
    checkpoint_dir = run_root / "checkpoints"
    selected_checkpoint = checkpoint_dir / "iwg_rg_cma_epoch065.pt"
    tracker_suffix = f"{spec['experiment_id']}_raw"
    tracker_folder = f"{TRACKER_PREFIX}_{tracker_suffix}_iwg_rg_cma_final"
    tracker_path = TRACKER_ROOT / tracker_folder
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "provenance").mkdir(exist_ok=True)
    _write_json(run_root / "protocol.json", _protocol(spec, run_root))
    (run_root / "provenance" / "git_commit.txt").write_text(git_commit + "\n")
    dataset_dir = Path(spec["dataset_dir"])
    if not (dataset_dir / "metadata.json").is_file():
        raise FileNotFoundError(dataset_dir / "metadata.json")
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASELINE_CHECKPOINT)
    (run_root / "provenance" / "dataset_metadata_sha256.txt").write_text(
        _sha256(dataset_dir / "metadata.json") + "  metadata.json\n"
    )
    if spec["init_checkpoint"]:
        init_checkpoint = Path(spec["init_checkpoint"])
        (run_root / "provenance" / "init_checkpoint_sha256.txt").write_text(
            _sha256(init_checkpoint) + f"  {init_checkpoint.name}\n"
        )
    if (run_root / "completed").is_file():
        print(f"Skipping completed run: {spec['experiment_id']}", flush=True)
        return
    if not selected_checkpoint.is_file() and (
        checkpoint_dir / "iwg_rg_cma_last.pt"
    ).is_file():
        raise FileExistsError(
            f"incomplete run has checkpoints but no epoch65 checkpoint: {run_root}"
        )
    if tracker_path.exists() and not _tracker_ready(tracker_path):
        raise FileExistsError(f"incomplete tracker output exists: {tracker_path}")

    train_command = [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "train_iwg_rg_cma",
        "--dataset-dir",
        str(dataset_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "5",
        "--device",
        "cuda",
        "--epochs",
        "100",
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
        "42",
        "--correction-bound",
        "0.05",
        "--context-size",
        str(spec["context_size"]),
        "--architecture-variant",
        str(spec["architecture_variant"]),
        "--memory-shards",
        "1",
        "--epochs-per-shard",
        "100",
        "--shard-cycles",
        "1",
        "--sequence-sampling",
        "sample-proportional",
    ]
    if spec["init_checkpoint"]:
        train_command.extend(
            ["--init-checkpoint", str(Path(spec["init_checkpoint"]).resolve())]
        )
    if spec["freeze_iwg"]:
        train_command.append("--freeze-iwg")
    (run_root / "provenance" / "train.command.txt").write_text(
        shlex.join(train_command) + "\n"
    )
    if not selected_checkpoint.is_file():
        _run_logged(
            train_command,
            cwd=ROOT,
            log_path=run_root / "logs" / "train.log",
        )
    else:
        print(f"Using existing checkpoint: {selected_checkpoint}", flush=True)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(selected_checkpoint)

    validate_command = [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "validate_iwg_rg_cma_checkpoint",
        "--checkpoint",
        str(selected_checkpoint),
        "--dataset-dir",
        str(dataset_dir),
        "--device",
        "cpu",
        "--max-batches",
        "8",
        "--output",
        str(run_root / "checkpoint_validation.json"),
    ]
    (run_root / "provenance" / "checkpoint_validation.command.txt").write_text(
        shlex.join(validate_command) + "\n"
    )
    if not (run_root / "checkpoint_validation.json").is_file():
        _run_logged(
            validate_command,
            cwd=ROOT,
            log_path=run_root / "logs" / "checkpoint_validation.log",
        )

    track_command = [
        str(PYTHON),
        "-u",
        "run.py",
        "--dataset",
        "MOT17",
        "--mode",
        "all",
        "--sequences",
        *SEQUENCES,
        "--agentguard-mode",
        "iwg-rg-cma",
        "--agentguard-checkpoint",
        str(selected_checkpoint),
        "--output_dir",
        str(TRACKER_ROOT),
        "--detection-cache-root",
        str(DETECTION_CACHE_ROOT),
        "--tracker-suffix",
        tracker_suffix,
        "--skip-eval",
    ]
    (run_root / "provenance" / "track.command.txt").write_text(
        shlex.join(track_command) + "\n"
    )
    if not _tracker_ready(tracker_path):
        _run_logged(
            track_command,
            cwd=ROOT / "3. Tracker",
            log_path=run_root / "logs" / "track.log",
        )
    if not _tracker_ready(tracker_path):
        raise FileNotFoundError(tracker_path)

    result_summary = _evaluate_tracker(
        TRACKER_ROOT,
        tracker_folder,
        log_path=run_root / "logs" / "evaluation.log",
    )
    rows = _summary_rows(
        experiment_id=str(spec["experiment_id"]),
        method=str(spec["method"]),
        architecture_variant=str(spec["architecture_variant"]),
        checkpoint=selected_checkpoint,
        tracker_folder=tracker_folder,
        source_artifact=run_root,
        summary=result_summary,
        git_commit=git_commit,
    )
    _upsert_summary(rows)
    _write_json(
        run_root / "manifest.json",
        {
            "experiment_id": spec["experiment_id"],
            "status": "completed",
            "selected_epoch": 65,
            "checkpoint": str(selected_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(selected_checkpoint),
            "tracker_folder": tracker_folder,
            "tracker_root": str(TRACKER_ROOT.resolve()),
            "sequence_set": list(SEQUENCES),
            "summary_csv": str(SUMMARY_CSV.resolve()),
            "architecture_variant": spec["architecture_variant"],
            "context_size": spec["context_size"],
            "training_device": "cuda",
            "inference_device": "cpu",
            "freeze_iwg": bool(spec["freeze_iwg"]),
            "git_commit": git_commit,
        },
    )
    (run_root / "completed").write_text("completed\n")
    (run_root / "failed").unlink(missing_ok=True)
    print(f"Completed {spec['experiment_id']}", flush=True)


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not (DATASET_DIR_CONTEXT6 / "metadata.json").is_file():
        raise FileNotFoundError(DATASET_DIR_CONTEXT6 / "metadata.json")
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKER_ROOT.mkdir(parents=True, exist_ok=True)
    _build_temporal_datasets()
    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    for spec in RUNS:
        try:
            _run_one(spec, git_commit)
        except Exception as exc:
            run_root = ABLATION_ROOT / str(spec["experiment_id"])
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "failed").write_text(str(exc) + "\n")
            print(
                f"ABLATION FAILED: {spec['experiment_id']}: {exc}",
                file=sys.stderr,
            )
            raise
    print(f"All ablations complete; summary: {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
