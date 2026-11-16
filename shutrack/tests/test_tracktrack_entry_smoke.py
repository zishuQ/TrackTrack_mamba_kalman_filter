"""Short real TrackTrack-entry smoke for SHUTrack v2.

Run manually:

    SHUTRACK_RUN_TRACKER_SMOKE=1 \
    SHUTRACK_SMOKE_CHECKPOINT=outputs/.../best.pt \
    ./.venv/bin/python -m pytest shutrack/tests/test_tracktrack_entry_smoke.py -q -s

It uses the real `run.py` entry, real detection cache, real Track objects and
the existing `update_with_gates`.  It is skipped by default because it needs
the local datasets and a trained checkpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_DIR = REPO_ROOT / "3. Tracker"


def run_entry_smoke(
    checkpoint: str | Path,
    *,
    sequence: str = "MOT17-02-FRCNN",
    max_frames: int = 300,
    output_dir: str | Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"smoke checkpoint not found: {checkpoint}")
    roots = output_dir or (REPO_ROOT / "outputs/agentguard/experiments/shutrack_v2_entry_smoke")
    roots = Path(roots)
    roots.mkdir(parents=True, exist_ok=True)
    stats_path = roots / f"stats_{sequence}_{max_frames}.jsonl"
    if stats_path.exists():
        stats_path.unlink()
    suffix = f"shutrack_v2_entry_smoke_{sequence}_{max_frames}"
    early = bool(os.environ.get("SHUTRACK_SMOKE_EARLY"))
    if early:
        suffix += "_early"
    command = [
        sys.executable,
        "-u",
        "run.py",
        "--dataset",
        "MOT17",
        "--mode",
        "all",
        "--sequences",
        sequence,
        "--agentguard-mode",
        "shutrack-v2",
        "--shutrack-checkpoint",
        str(checkpoint),
        "--agentguard-device",
        "cpu",
        "--max-frames",
        str(max_frames),
        "--skip-eval",
        "--tracker-suffix",
        suffix,
        "--agentguard-stats-output",
        str(stats_path),
    ]
    if early:
        command.append("--shutrack-early-enable")
    result = subprocess.run(
        command,
        cwd=TRACKER_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TrackTrack entry smoke failed:\n"
            + result.stdout[-4000:]
            + "\n"
            + result.stderr[-4000:]
        )
    if not stats_path.is_file():
        raise RuntimeError(
            "TrackTrack entry smoke produced no statistics file:\n"
            + result.stdout[-2000:]
            + "\n"
            + result.stderr[-2000:]
        )
    records = [
        json.loads(line)
        for line in stats_path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("TrackTrack entry smoke statistics file is empty")
    return {"command": command, "stats": records[-1], "stdout_tail": result.stdout[-1000:]}


def _checkpoint_from_env() -> str:
    checkpoint = os.environ.get("SHUTRACK_SMOKE_CHECKPOINT", "").strip()
    if not checkpoint:
        raise RuntimeError("SHUTRACK_SMOKE_CHECKPOINT is required")
    return checkpoint


@pytest.mark.skipif(
    not os.environ.get("SHUTRACK_RUN_TRACKER_SMOKE"),
    reason="set SHUTRACK_RUN_TRACKER_SMOKE=1 to run the real entry smoke",
)
def test_real_tracktrack_entry_smoke():
    result = run_entry_smoke(_checkpoint_from_env())
    stats = result["stats"]
    assert stats["shutrack_matched_events"] > 0
    assert stats["shutrack_unmatched_events"] > 0
    assert stats["shutrack_model_inference_count"] > 0
    assert stats["shutrack_gate_update_count"] > 0
    assert 0.0 <= stats["shutrack_min_motion_gate"] <= stats["shutrack_max_motion_gate"] <= 1.0
    assert 0.0 <= stats["shutrack_min_appearance_gate"] <= stats["shutrack_max_appearance_gate"] <= 1.0


if __name__ == "__main__":
    checkpoint = _checkpoint_from_env()
    frames = int(os.environ.get("SHUTRACK_SMOKE_FRAMES", "300"))
    sequence = os.environ.get("SHUTRACK_SMOKE_SEQUENCE", "MOT17-02-FRCNN")
    result = run_entry_smoke(checkpoint, sequence=sequence, max_frames=frames)
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print(result["stdout_tail"])
