#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def _timestamp(value: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if value is None else value).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _info(handle, message: str, *, timestamp: float | None = None) -> None:
    handle.write(f"{_timestamp(timestamp)} | INFO     | {message}\n")
    handle.flush()


def _render_epoch(handle, metrics: dict, total_epochs: int, timestamp: float) -> None:
    epoch = int(metrics["epoch"])
    _info(handle, f"--- Epoch {epoch}/{total_epochs} ---", timestamp=timestamp)
    _info(
        handle,
        (
            f"Epoch {epoch:3d} - train_loss={metrics['loss']:.4f}  "
            f"base={metrics['base_loss']:.4f}  final={metrics['final_loss']:.4f}  "
            f"policy={metrics['policy_loss']:.4f}  cue={metrics['cue_loss']:.4f}  "
            f"risk={metrics['risk_loss']:.4f}  residual={metrics['residual_loss']:.6f}  "
            f"revision={metrics['revision_loss']:.6f}  "
            f"lr={metrics['learning_rate']:.2e}  "
            f"time={metrics['epoch_time_s']:.1f}s"
        ),
        timestamp=timestamp,
    )
    motion = metrics.get("motion_history_attention")
    appearance = metrics.get("appearance_history_attention")
    motion_entropy = metrics.get("motion_attention_entropy")
    appearance_entropy = metrics.get("appearance_attention_entropy")
    if (
        motion is None
        or appearance is None
        or motion_entropy is None
        or appearance_entropy is None
        or not metrics.get("diagnostics_available", True)
    ):
        interval = metrics.get("diagnostic_interval", "n/a")
        sampled = metrics.get("diagnostic_batches", 0)
        attention_line = (
            "Attention  motion_H=n/a  appearance_H=n/a  "
            f"diagnostics=unsampled interval={interval} batches={sampled}"
        )
    else:
        attention_line = (
            "Attention  "
            f"motion_H={motion_entropy:.4f}  "
            f"appearance_H={appearance_entropy:.4f}  "
            f"motion_pos=[{', '.join(f'{value:.3f}' for value in motion)}]  "
            f"appearance_pos=[{', '.join(f'{value:.3f}' for value in appearance)}]"
        )
    _info(handle, attention_line, timestamp=timestamp)


def _read_complete_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open() as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # The trainer may be in the middle of appending the final line.
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render live RG-CMA JSON metrics as a readable training.log."
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    metrics_path = Path(args.metrics).resolve()
    output_path = Path(args.output).resolve()
    run_root = Path(args.run_root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_marker = run_root / "pipeline.pid"
    start_time = start_marker.stat().st_mtime if start_marker.exists() else time.time()
    rendered = 0
    elapsed = 0.0
    with output_path.open("w") as output:
        _info(output, "=" * 60, timestamp=start_time)
        _info(output, "Safe-Direct IWG + RG-CMA Training - start", timestamp=start_time)
        _info(output, f"Output directory: {output_path.parent}", timestamp=start_time)
        _info(
            output,
            "Config: epochs=100 batch_size=2048 workers=4 lr=1e-4 "
            "weight_decay=1e-4 warmup=1 amp=false seed=42 validation=none",
            timestamp=start_time,
        )
        while True:
            records = _read_complete_lines(metrics_path)
            for metrics in records[rendered:]:
                elapsed += float(metrics.get("epoch_time_s", 0.0))
                _render_epoch(output, metrics, args.epochs, start_time + elapsed)
                rendered += 1
            if rendered >= args.epochs:
                _info(output, "IWG RG-CMA training complete.")
                _info(output, "=" * 60)
                return
            if args.once:
                return
            if (run_root / "failed").exists() and not (run_root / "running").exists():
                _info(output, f"Training stopped after epoch {rendered}; pipeline failed.")
                return
            time.sleep(max(float(args.interval), 0.25))


if __name__ == "__main__":
    main()
