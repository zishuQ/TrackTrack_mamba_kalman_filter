from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure a logger that writes both to stdout and a log file.

    Parameters
    ----------
    output_dir : str
        Directory where the log file ``training.log`` will be written.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    logger = logging.getLogger("agentguard.training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — detailed level.
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler — info level.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    return logger


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    epoch: int,
    writer: Optional[Any] = None,
) -> None:
    """Log a metrics dict to the console and optionally to a TensorBoard writer.

    Parameters
    ----------
    metrics : dict
        Metric name → scalar value.
    step : int
        Global step (used for TensorBoard x-axis).
    epoch : int
        Current epoch number (for display).
    writer : torch.utils.tensorboard.SummaryWriter or None
        Optional TensorBoard writer.
    """
    logger = logging.getLogger("agentguard.training")

    # Build a compact, informative message.
    parts = [f"Epoch {epoch:2d}  step {step:6d}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))

    # TensorBoard logging.
    if writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)


def save_training_summary(
    metrics_history: List[Dict[str, Any]],
    path: str,
) -> str:
    """Persist a full training metrics history to a JSON file.

    Parameters
    ----------
    metrics_history : list of dict
        Per-epoch (or per-step) metrics dictionaries.
    path : str
        Destination file path.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Convert any non-serialisable types.
    clean: List[Dict[str, Any]] = []
    for entry in metrics_history:
        clean_entry: Dict[str, Any] = {}
        for key, value in entry.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                clean_entry[key] = value
            elif isinstance(value, bytes):
                clean_entry[key] = value.decode("utf-8", errors="replace")
            else:
                clean_entry[key] = str(value)
        clean.append(clean_entry)

    with open(path, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    return os.path.abspath(path)
