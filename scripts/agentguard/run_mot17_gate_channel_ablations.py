#!/usr/bin/env python3
"""Run inference-only ablations for the KF and EMA gate channels.

Both variants reuse the same epoch-65 baseline checkpoint and the same raw
MOT17-02/09/10 protocol.  A disabled channel is forced to gate=1.0, which
restores the original full update for that channel while leaving the other
learned gate untouched.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from run_mot17_without_cma_ablation import (
    BASELINE_CHECKPOINT,
    BASELINE_TRACKER_FOLDER,
    DATASET_DIR,
    DETECTION_CACHE_ROOT,
    REFERENCE_ROOT,
    ROOT,
    SEQUENCES,
    SEQUENCE_SET,
    SUMMARY_CSV,
    TRACKER_ROOT,
    TRACKTRACK_TRACKER_FOLDER,
    _evaluate_tracker,
    _git_value,
    _join_deltas,
    _run_logged,
    _sha256,
    _summary_rows,
    _write_json,
    _write_summary_csv,
)


EXPERIMENTS = (
    {
        "experiment_id": "mot17_all_disable_gate_ema_epoch065",
        "method": "AgentGuard baseline (Gate_EMA disabled)",
        "architecture_variant": "legacy-disable-ema-gate",
        "tracker_suffix": "mot17_gate_ema_disabled_e065_raw",
        "tracker_folder": (
            "mot17_all_0.80_mot17_gate_ema_disabled_e065_raw"
            "_iwg_rg_cma_final"
        ),
        "flag": "--agentguard-disable-ema-gate",
        "disabled_channel": "Gate_EMA",
    },
    {
        "experiment_id": "mot17_all_disable_gate_kf_epoch065",
        "method": "AgentGuard baseline (Gate_KF disabled)",
        "architecture_variant": "legacy-disable-kf-gate",
        "tracker_suffix": "mot17_gate_kf_disabled_e065_raw",
        "tracker_folder": (
            "mot17_all_0.80_mot17_gate_kf_disabled_e065_raw"
            "_iwg_rg_cma_final"
        ),
        "flag": "--agentguard-disable-kf-gate",
        "disabled_channel": "Gate_KF",
    },
)


def _run_root(experiment_id: str) -> Path:
    return ROOT / "outputs/agentguard/ablation" / experiment_id


def _tracker_ready(path: Path) -> bool:
    return path.is_dir() and all(
        (path / f"{sequence}.txt").is_file() for sequence in SEQUENCES
    )


def _protocol(experiment: dict[str, str]) -> dict:
    return {
        "experiment_id": experiment["experiment_id"],
        "dataset": "MOT17",
        "split": "all",
        "sequence_set": list(SEQUENCES),
        "checkpoint": str(BASELINE_CHECKPOINT.resolve()),
        "checkpoint_epoch": 65,
        "architecture_variant": experiment["architecture_variant"],
        "disabled_channel": experiment["disabled_channel"],
        "disabled_gate_value": 1.0,
        "remaining_channel": (
            "Gate_KF"
            if experiment["disabled_channel"] == "Gate_EMA"
            else "Gate_EMA"
        ),
        "training": "none; inference-only channel ablation",
        "inference": {
            "agentguard_device": "cpu",
            "evaluation": "raw_final_gate",
            "tracker_seed": 10000,
        },
        "references": {
            "tracktrack_tracker_folder": TRACKTRACK_TRACKER_FOLDER,
            "baseline_tracker_folder": BASELINE_TRACKER_FOLDER,
            "baseline_checkpoint": str(BASELINE_CHECKPOINT.resolve()),
        },
        "execution": "serial tracking and evaluation",
    }


def main() -> None:
    if not (DATASET_DIR / "metadata.json").is_file():
        raise FileNotFoundError(DATASET_DIR / "metadata.json")
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASELINE_CHECKPOINT)
    for folder in (TRACKTRACK_TRACKER_FOLDER, BASELINE_TRACKER_FOLDER):
        path = TRACKER_ROOT / folder
        if not _tracker_ready(path):
            raise FileNotFoundError(
                f"reference tracker output is incomplete: {path}"
            )

    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    reference_root = _run_root(EXPERIMENTS[0]["experiment_id"])
    (reference_root / "logs").mkdir(parents=True, exist_ok=True)
    tracktrack_summary = _evaluate_tracker(
        TRACKTRACK_TRACKER_FOLDER,
        log_path=reference_root / "logs" / "tracktrack_eval.log",
    )
    baseline_summary = _evaluate_tracker(
        BASELINE_TRACKER_FOLDER,
        log_path=reference_root / "logs" / "baseline_eval.log",
    )

    for experiment in EXPERIMENTS:
        experiment_id = experiment["experiment_id"]
        run_root = _run_root(experiment_id)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "logs").mkdir(exist_ok=True)
        (run_root / "provenance").mkdir(exist_ok=True)
        _write_json(run_root / "protocol.json", _protocol(experiment))
        (run_root / "provenance" / "git_commit.txt").write_text(
            git_commit + "\n"
        )
        (run_root / "provenance" / "baseline_checkpoint_sha256.txt").write_text(
            _sha256(BASELINE_CHECKPOINT)
            + f"  {BASELINE_CHECKPOINT.name}\n"
        )

        tracker_path = TRACKER_ROOT / experiment["tracker_folder"]
        if tracker_path.exists() and not _tracker_ready(tracker_path):
            raise FileExistsError(
                f"refusing to overwrite incomplete tracker output: {tracker_path}"
            )

        track_command = [
            str(ROOT / ".venv" / "bin" / "python"),
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
            str(BASELINE_CHECKPOINT),
            experiment["flag"],
            "--output_dir",
            str(TRACKER_ROOT),
            "--detection-cache-root",
            str(DETECTION_CACHE_ROOT),
            "--tracker-suffix",
            experiment["tracker_suffix"],
            "--skip-eval",
        ]
        (run_root / "provenance" / "track.command.txt").write_text(
            shlex.join(track_command) + "\n"
        )

        if _tracker_ready(tracker_path):
            print(f"Using existing tracker output: {tracker_path}", flush=True)
        else:
            print(
                f"Running {experiment['disabled_channel']} ablation: "
                f"{experiment_id}",
                flush=True,
            )
            _run_logged(
                track_command,
                cwd=ROOT / "3. Tracker",
                log_path=run_root / "logs" / "track.log",
            )
        if not _tracker_ready(tracker_path):
            raise FileNotFoundError(tracker_path)

        result_summary = _evaluate_tracker(
            experiment["tracker_folder"],
            log_path=run_root / "logs" / "evaluation.log",
        )
        rows = []
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
                experiment_id=experiment_id,
                method=experiment["method"],
                architecture_variant=experiment["architecture_variant"],
                epoch=65,
                checkpoint=BASELINE_CHECKPOINT,
                tracker_folder=experiment["tracker_folder"],
                source_artifact=run_root,
                summary=result_summary,
                git_commit=git_commit,
            )
        )
        _write_summary_csv(_join_deltas(rows), SUMMARY_CSV)
        _write_json(
            run_root / "manifest.json",
            {
                "experiment_id": experiment_id,
                "status": "completed",
                "checkpoint": str(BASELINE_CHECKPOINT.resolve()),
                "checkpoint_epoch": 65,
                "disabled_channel": experiment["disabled_channel"],
                "tracker_folder": experiment["tracker_folder"],
                "sequence_set": list(SEQUENCES),
                "summary_csv": str(SUMMARY_CSV.resolve()),
                "git_commit": git_commit,
            },
        )
        (run_root / "completed").write_text("completed\n")
        print(f"Completed {experiment_id}", flush=True)


if __name__ == "__main__":
    main()
