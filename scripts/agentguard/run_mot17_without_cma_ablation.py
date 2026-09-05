#!/usr/bin/env python3
"""Run the MOT17 without-CMA ablation and write one comparison table.

The run is deliberately serial:

1. evaluate the retained TrackTrack and AgentGuard baseline outputs;
2. train the without-CMA architecture for 100 epochs;
3. validate and select epoch 65;
4. track MOT17-02/09/10 with that checkpoint;
5. evaluate the same three-sequence subset and join all results.

No tracker result directory is overwritten. If the process fails before the
first checkpoint is written, the same run directory can be resumed.
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
PYTHON = ROOT / ".venv" / "bin" / "python"
DATASET_DIR = ROOT / "outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_compact"
DETECTION_CACHE_ROOT = ROOT / "outputs/agentguard/detection_cache"
TRACKER_ROOT = ROOT / "outputs/3. track"
DATA_ROOT = Path("/home/shang/datasets")
SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
)
SEQUENCE_SET = "+".join(SEQUENCES)

REFERENCE_ROOT = ROOT / "outputs/agentguard/ablation/mot17_all_tracktrack_gain_02_09_10"
BASELINE_CHECKPOINT = REFERENCE_ROOT / "checkpoints/iwg_rg_cma_epoch065.pt"
BASELINE_TRACKER_FOLDER = (
    "mot17_all_0.80_mot17_baseline_e065_final_all_raw_iwg_rg_cma_final"
)
TRACKTRACK_TRACKER_FOLDER = "mot17_all_0.80"

RUN_ID = "mot17_all_without_cma_100e_epoch065"
RUN_ROOT = ROOT / "outputs/agentguard/ablation" / RUN_ID
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
SELECTED_CHECKPOINT = CHECKPOINT_DIR / "iwg_rg_cma_epoch065.pt"
SUMMARY_CSV = ROOT / "outputs/agentguard/ablation/summary.csv"
TRACKER_SUFFIX = "mot17_without_cma_e065_raw"
WITHOUT_CMA_TRACKER_FOLDER = (
    f"mot17_all_0.80_{TRACKER_SUFFIX}_iwg_rg_cma_final"
)

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


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [
        str(ROOT),
        str(ROOT / "3. Tracker"),
        str(ROOT / "agentguard" / "src"),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
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
        raise RuntimeError(
            f"command failed with exit code {return_code}: {rendered}"
        )


def _evaluate_tracker(
    tracker_folder: str,
    *,
    log_path: Path,
) -> dict[str, Any]:
    """Evaluate one existing raw tracker folder on the fixed subset."""
    sys.path.insert(0, str(ROOT / "3. Tracker"))
    from utils.etc import evaluate_sequences

    args = SimpleNamespace(
        mode="all",
        data_path=str(DATA_ROOT / "MOT17" / "train") + "/",
        output_dir=str(TRACKER_ROOT),
        print_per_sequence_metrics=False,
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        summary = evaluate_sequences(
            args, tracker_folder, "MOT17", list(SEQUENCES)
        )
    text = buffer.getvalue()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text)
    print(text, end="", flush=True)
    return summary


def _summary_rows(
    *,
    experiment_id: str,
    method: str,
    architecture_variant: str,
    epoch: int | str,
    checkpoint: Path | None,
    tracker_folder: str,
    source_artifact: Path,
    summary: dict[str, Any],
    git_commit: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, dict[str, Any]]] = [
        ("combined", "COMBINED_SEQ", summary["combined"])
    ]
    scopes.extend(
        ("sequence", sequence, summary["per_sequence"][sequence])
        for sequence in SEQUENCES
    )
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
            "epoch": epoch,
            "checkpoint": str(checkpoint.resolve()) if checkpoint else "",
            "tracker_folder": tracker_folder,
            "source_artifact": str(source_artifact.resolve()),
            "git_commit": git_commit,
        }
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)
    return rows


def _join_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = {
        row["sequence"]: row
        for row in rows
        if row["method"] == "TrackTrack"
    }
    for row in rows:
        reference = references[row["sequence"]]
        for metric in RATE_METRICS:
            row[f"delta_{metric}_pp"] = (
                float(row[metric]) - float(reference[metric])
            ) * 100.0
        for metric in COUNT_METRICS:
            row[f"delta_{metric}"] = float(row[metric]) - float(
                reference[metric]
            )
    return rows


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Upsert experiment rows into the shared ablation summary.

    The key keeps rerunning one experiment idempotent while allowing later
    ablations to append their own rows to the same table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(CSV_FIELDS):
                raise ValueError(
                    f"unexpected ablation summary schema in {path}: "
                    f"{reader.fieldnames}"
                )
            existing = list(reader)

    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, ...], int] = {}
    for row in existing + rows:
        key = tuple(str(row.get(field, "")) for field in SUMMARY_KEY_FIELDS)
        if key in positions:
            merged[positions[key]] = row
        else:
            positions[key] = len(merged)
            merged.append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(merged)


def _protocol() -> dict[str, Any]:
    return {
        "experiment_id": RUN_ID,
        "dataset": "MOT17",
        "split": "all",
        "sequence_set": list(SEQUENCES),
        "training": {
            "dataset_dir": str(DATASET_DIR.resolve()),
            "architecture_variant": "without-cma",
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
            "context_size": 6,
            "memory_shards": 1,
            "epochs_per_shard": 100,
            "shard_cycles": 1,
            "sequence_sampling": "sample-proportional",
        },
        "inference": {
            "device": "cpu",
            "evaluation": "raw_final_gate",
            "tracker_seed": 10000,
            "detection_cache_root": str(DETECTION_CACHE_ROOT.resolve()),
        },
        "references": {
            "tracktrack_tracker_folder": TRACKTRACK_TRACKER_FOLDER,
            "baseline_checkpoint": str(BASELINE_CHECKPOINT.resolve()),
            "baseline_tracker_folder": BASELINE_TRACKER_FOLDER,
            "baseline_source_csv": str(
                (REFERENCE_ROOT / "agentguard_vs_tracktrack_epoch065.csv").resolve()
            ),
        },
        "selection_rule": "fixed epoch 65; no best-of-epoch selection for this ablation",
        "execution": "serial training, tracking, and evaluation",
    }


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    for path in (DATASET_DIR / "metadata.json", BASELINE_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    for folder in (TRACKTRACK_TRACKER_FOLDER, BASELINE_TRACKER_FOLDER):
        path = TRACKER_ROOT / folder
        if not path.is_dir():
            raise FileNotFoundError(path)
        missing = [
            sequence
            for sequence in SEQUENCES
            if not (path / f"{sequence}.txt").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{path}: missing {missing}")
    if RUN_ROOT.exists():
        if (RUN_ROOT / "completed").is_file():
            header: list[str] = []
            if SUMMARY_CSV.is_file():
                with SUMMARY_CSV.open(newline="") as handle:
                    header = next(csv.reader(handle), [])
            expected_delta_fields = {
                "delta_IDSW",
                "delta_Frag",
            }
            if expected_delta_fields.issubset(header):
                (RUN_ROOT / "failed").unlink(missing_ok=True)
                print(f"Ablation is already complete: {RUN_ROOT}", flush=True)
                return
            print(
                f"Refreshing completed run with corrected summary schema: "
                f"{RUN_ROOT}",
                flush=True,
            )
            (RUN_ROOT / "completed").unlink()
        if SELECTED_CHECKPOINT.is_file():
            print(
                f"Resuming from existing selected checkpoint: "
                f"{SELECTED_CHECKPOINT}",
                flush=True,
            )
        elif (CHECKPOINT_DIR / "iwg_rg_cma_last.pt").exists():
            raise FileExistsError(
                "an incomplete run contains checkpoints but not the selected "
                f"epoch 65 checkpoint; refusing to overwrite it: {RUN_ROOT}"
            )
        else:
            print(
                f"Resuming initialization of failed run: {RUN_ROOT}",
                flush=True,
            )
        (RUN_ROOT / "failed").unlink(missing_ok=True)
    produced_tracker = TRACKER_ROOT / WITHOUT_CMA_TRACKER_FOLDER
    tracker_ready = produced_tracker.is_dir() and all(
        (produced_tracker / f"{sequence}.txt").is_file()
        for sequence in SEQUENCES
    )
    if produced_tracker.exists() and not tracker_ready:
        raise FileExistsError(
            f"refusing to overwrite existing tracker directory: "
            f"{produced_tracker}"
        )

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "logs").mkdir(exist_ok=True)
    (RUN_ROOT / "provenance").mkdir(exist_ok=True)
    _write_json(RUN_ROOT / "protocol.json", _protocol())
    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    (RUN_ROOT / "provenance" / "git_commit.txt").write_text(git_commit + "\n")
    (RUN_ROOT / "provenance" / "git_status_porcelain.txt").write_text(
        _git_value(["git", "status", "--porcelain"], "unavailable") + "\n"
    )
    (RUN_ROOT / "provenance" / "dataset_sha256.txt").write_text(
        _sha256(DATASET_DIR / "metadata.json") + "  metadata.json\n"
    )
    (RUN_ROOT / "provenance" / "baseline_checkpoint_sha256.txt").write_text(
        _sha256(BASELINE_CHECKPOINT)
        + f"  {BASELINE_CHECKPOINT.name}\n"
    )

    print(f"Starting serial ablation run: {RUN_ID}", flush=True)
    print(f"Reference subset: {SEQUENCE_SET}", flush=True)
    tracktrack_summary = _evaluate_tracker(
        TRACKTRACK_TRACKER_FOLDER,
        log_path=RUN_ROOT / "logs" / "tracktrack_eval.log",
    )
    baseline_summary = _evaluate_tracker(
        BASELINE_TRACKER_FOLDER,
        log_path=RUN_ROOT / "logs" / "baseline_eval.log",
    )

    train_command = [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "train_iwg_rg_cma",
        "--dataset-dir",
        str(DATASET_DIR),
        "--checkpoint-dir",
        str(CHECKPOINT_DIR),
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
        "6",
        "--architecture-variant",
        "without-cma",
        "--memory-shards",
        "1",
        "--epochs-per-shard",
        "100",
        "--shard-cycles",
        "1",
        "--sequence-sampling",
        "sample-proportional",
    ]
    (RUN_ROOT / "provenance" / "train.command.txt").write_text(
        shlex.join(train_command) + "\n"
    )
    if SELECTED_CHECKPOINT.is_file():
        print(
            f"Using existing epoch 65 checkpoint; skipping training: "
            f"{SELECTED_CHECKPOINT}",
            flush=True,
        )
    else:
        _run_logged(
            train_command,
            cwd=ROOT,
            log_path=RUN_ROOT / "logs" / "train.log",
        )
    if not SELECTED_CHECKPOINT.is_file():
        raise FileNotFoundError(SELECTED_CHECKPOINT)

    validate_command = [
        str(PYTHON),
        "-u",
        "-m",
        "agentguard.cli",
        "validate_iwg_rg_cma_checkpoint",
        "--checkpoint",
        str(SELECTED_CHECKPOINT),
        "--dataset-dir",
        str(DATASET_DIR),
        "--device",
        "cpu",
        "--max-batches",
        "8",
        "--output",
        str(RUN_ROOT / "checkpoint_validation.json"),
    ]
    (RUN_ROOT / "provenance" / "checkpoint_validation.command.txt").write_text(
        shlex.join(validate_command) + "\n"
    )
    _run_logged(
        validate_command,
        cwd=ROOT,
        log_path=RUN_ROOT / "logs" / "checkpoint_validation.log",
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
        str(SELECTED_CHECKPOINT),
        "--output_dir",
        str(TRACKER_ROOT),
        "--detection-cache-root",
        str(DETECTION_CACHE_ROOT),
        "--tracker-suffix",
        TRACKER_SUFFIX,
        "--skip-eval",
    ]
    (RUN_ROOT / "provenance" / "track.command.txt").write_text(
        shlex.join(track_command) + "\n"
    )
    if tracker_ready:
        print(f"Using existing tracker output: {produced_tracker}", flush=True)
    else:
        _run_logged(
            track_command,
            cwd=ROOT / "3. Tracker",
            log_path=RUN_ROOT / "logs" / "track.log",
        )
    if not produced_tracker.is_dir():
        raise FileNotFoundError(produced_tracker)
    for sequence in SEQUENCES:
        if not (produced_tracker / f"{sequence}.txt").is_file():
            raise FileNotFoundError(produced_tracker / f"{sequence}.txt")

    without_cma_summary = _evaluate_tracker(
        WITHOUT_CMA_TRACKER_FOLDER,
        log_path=RUN_ROOT / "logs" / "without_cma_eval.log",
    )

    rows: list[dict[str, Any]] = []
    rows.extend(
        _summary_rows(
            experiment_id="tracktrack_reference",
            method="TrackTrack",
            architecture_variant="tracktrack-reference",
            epoch="",
            checkpoint=None,
            tracker_folder=TRACKTRACK_TRACKER_FOLDER,
            source_artifact=TRACKER_ROOT / TRACKTRACK_TRACKER_FOLDER,
            summary=tracktrack_summary,
            git_commit=git_commit,
        )
    )
    rows.extend(
        _summary_rows(
            experiment_id="mot17_all_baseline_epoch065",
            method="AgentGuard baseline",
            architecture_variant="legacy",
            epoch=65,
            checkpoint=BASELINE_CHECKPOINT,
            tracker_folder=BASELINE_TRACKER_FOLDER,
            source_artifact=REFERENCE_ROOT
            / "agentguard_vs_tracktrack_epoch065.csv",
            summary=baseline_summary,
            git_commit=git_commit,
        )
    )
    rows.extend(
        _summary_rows(
            experiment_id=RUN_ID,
            method="AgentGuard without-CMA",
            architecture_variant="without-cma",
            epoch=65,
            checkpoint=SELECTED_CHECKPOINT,
            tracker_folder=WITHOUT_CMA_TRACKER_FOLDER,
            source_artifact=RUN_ROOT,
            summary=without_cma_summary,
            git_commit=git_commit,
        )
    )
    rows = _join_deltas(rows)
    _write_summary_csv(rows, SUMMARY_CSV)
    _write_json(
        RUN_ROOT / "manifest.json",
        {
            "experiment_id": RUN_ID,
            "status": "completed",
            "selected_epoch": 65,
            "checkpoint": str(SELECTED_CHECKPOINT.resolve()),
            "tracker_folder": WITHOUT_CMA_TRACKER_FOLDER,
            "sequence_set": list(SEQUENCES),
            "summary_csv": str(SUMMARY_CSV.resolve()),
            "checkpoint_sha256": _sha256(SELECTED_CHECKPOINT),
            "git_commit": git_commit,
        },
    )
    (RUN_ROOT / "completed").write_text("completed\n")
    print(f"Wrote {len(rows)} rows to {SUMMARY_CSV}", flush=True)
    print(f"Completed ablation run: {RUN_ROOT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if RUN_ROOT.is_dir():
            (RUN_ROOT / "failed").write_text(str(exc) + "\n")
        print(f"ABLATION FAILED: {exc}", file=sys.stderr, flush=True)
        raise
