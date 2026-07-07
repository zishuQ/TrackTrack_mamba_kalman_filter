"""Evaluation module for AgentGuard tracking quality assessment.

Provides utilities to evaluate tracking results against ground truth,
compare tracking baselines against AgentGuard outputs, and detect
Oracle-supervision failures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agentguard.contracts.events import TrackEvent

# ---------------------------------------------------------------------------
#  TrackEval integration
# ---------------------------------------------------------------------------

try:
    import trackeval

    _TRACKEVAL_AVAILABLE = True
except ImportError:
    _TRACKEVAL_AVAILABLE = False


def evaluate_tracking_results(
    gt_path: str,
    tracker_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a tracker's output against ground truth using TrackEval.

    Computes HOTA, AssA, DetA, IDF1, IDs, and Fragmentation metrics.

    Parameters
    ----------
    gt_path : str
        Path to the directory containing ground-truth data (MOTChallenge
        format).  Expects ``<gt_path>/<sequence>/gt/gt.txt``.
    tracker_path : str
        Path to the tracker's output directory.  Expects
        ``<tracker_path>/<sequence>.txt`` or
        ``<tracker_path>/<sequence>/data/`` depending on TrackEval
        configuration.
    output_path : str, optional
        If provided, the raw evaluation output is saved here as JSON.

    Returns
    -------
    dict
        Dictionary of metric names to values.  Includes keys::

            HOTA, AssA, DetA, IDF1, IDs, Frag, MOTA, MOTP,
            Recall, Precision, MT, ML, and per-class variants where
            available.

    Raises
    ------
    RuntimeError
        If TrackEval is not installed.
    """
    if not _TRACKEVAL_AVAILABLE:
        raise RuntimeError(
            "TrackEval is required but not installed. "
            "Install it with: pip install trackeval"
        )

    # Build a minimal TrackEval configuration
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_EMPTY_CLASSES"] = False
    eval_config["OUTPUT_DETAILED"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_path
    dataset_config["TRACKERS_FOLDER"] = os.path.dirname(tracker_path)
    dataset_config["TRACKERS_TO_EVAL"] = [os.path.basename(tracker_path)]
    dataset_config["BENCHMARK"] = "MOT17"
    dataset_config["SPLIT_TO_EVAL"] = ""
    dataset_config["PRINT_CONFIG"] = False
    dataset_config["DO_PREPROC"] = False

    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity", "VACE"],
        "PRINT_CONFIG": False,
    }

    # Run evaluation
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity(), trackeval.metrics.VACE()

    output_res, output_msg = evaluator.evaluate([dataset], [metrics])

    # Flatten the nested output into a single metric dict
    results: Dict[str, float] = _flatten_trackeval_output(output_res)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def _flatten_trackeval_output(
    output_res: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Flatten the nested TrackEval output into a single metric dict.

    TrackEval returns deeply nested dicts.  This helper extracts the
    COMBINED_SEQ-level metrics into a flat dict.
    """
    results: Dict[str, float] = {}

    try:
        # output_res is a list of lists of dicts:
        # [[{benchmark: {seq: {metric: value}}}], [...]]
        for outer in output_res:
            if not isinstance(outer, list):
                continue
            for inner in outer:
                if not isinstance(inner, dict):
                    continue
                for benchmark_data in inner.values():
                    if not isinstance(benchmark_data, dict):
                        continue
                    for seq_data in benchmark_data.values():
                        if not isinstance(seq_data, dict):
                            continue
                        for metric_name, metric_value in seq_data.items():
                            if isinstance(metric_value, (int, float)):
                                results[metric_name] = float(metric_value)
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
#  Comparison
# ---------------------------------------------------------------------------


def compare_results(
    baseline_path: str,
    agentguard_path: str,
) -> Dict[str, float]:
    """Compare two tracking result folders and return delta metrics.

    Both folders should contain MOTChallenge-format results.

    Parameters
    ----------
    baseline_path : str
        Path to the baseline tracker results (e.g. SORT/OCSORT).
    agentguard_path : str
        Path to the AgentGuard-enhanced results.

    Returns
    -------
    dict
        Delta metrics::

            {
                "HOTA_delta": float,
                "AssA_delta": float,
                "DetA_delta": float,
                "IDF1_delta": float,
                "IDs_delta": float,
                "Frag_delta": float,
                "MOTA_delta": float,
                ...
            }

        Positive values mean AgentGuard improved over the baseline.
    """
    # Evaluate both
    baseline_results = evaluate_tracking_results(
        gt_path=_find_gt_path(baseline_path),
        tracker_path=baseline_path,
        output_path=None,
    )

    agentguard_results = evaluate_tracking_results(
        gt_path=_find_gt_path(agentguard_path),
        tracker_path=agentguard_path,
        output_path=None,
    )

    # Compute deltas: AgentGuard - Baseline
    all_keys = set(baseline_results.keys()) | set(agentguard_results.keys())
    deltas: Dict[str, float] = {}
    for key in all_keys:
        base_val = baseline_results.get(key, 0.0)
        ag_val = agentguard_results.get(key, 0.0)
        deltas[f"{key}_delta"] = ag_val - base_val

    return deltas


def _find_gt_path(result_path: str) -> str:
    """Try to locate the GT directory relative to a results path.

    Scans common parent directory conventions.
    """
    result_dir = os.path.dirname(os.path.abspath(result_path))

    candidates = [
        os.path.join(result_dir, "gt"),
        os.path.join(result_dir, "..", "gt"),
        os.path.join(result_dir, "..", "..", "gt"),
        os.path.join(os.path.dirname(result_dir), "gt"),
    ]

    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if os.path.isdir(norm):
            # Check if it has a gt subfolder for any sequence
            for sub in os.listdir(norm):
                gt_file = os.path.join(norm, sub, "gt", "gt.txt")
                if os.path.isfile(gt_file):
                    return norm

    # Fallback: return a plausible path (evaluation will fail gracefully)
    return os.path.join(os.path.dirname(result_dir), "gt")


# ---------------------------------------------------------------------------
#  Oracle blocking detection
# ---------------------------------------------------------------------------


def check_oracle_blocked(
    oracle_results: Dict[str, Any],
) -> bool:
    """Check if the Oracle supervision (IWG / Full) failed to improve
    over the baseline.

    The Oracle is considered "blocked" when its metrics are not
    significantly better than the baseline across all key metrics.

    Parameters
    ----------
    oracle_results : dict
        A dict with at least ``"oracle"`` and ``"baseline"`` sub-dicts,
        each containing metric keys like ``HOTA``, ``AssA``, ``DetA``,
        ``IDF1``, ``IDs``, ``MOTA``.

    Returns
    -------
    bool
        ``True`` if the Oracle is blocked (failed to improve).
    """
    oracle = oracle_results.get("oracle", {})
    baseline = oracle_results.get("baseline", {})

    if not oracle or not baseline:
        return True  # insufficient data

    # Metrics where higher is better
    higher_is_better = ["HOTA", "AssA", "DetA", "IDF1", "MOTA", "Recall", "Precision"]

    # Metrics where lower is better (e.g. ID switches, fragmentations)
    lower_is_better = ["IDs", "Frag"]

    improved_count = 0
    total_count = 0

    for metric in higher_is_better:
        if metric in oracle and metric in baseline:
            total_count += 1
            if oracle[metric] > baseline[metric]:
                improved_count += 1

    for metric in lower_is_better:
        if metric in oracle and metric in baseline:
            total_count += 1
            if oracle[metric] < baseline[metric]:
                improved_count += 1

    if total_count == 0:
        return True  # no comparable metrics

    # Blocked if fewer than 60% of metrics improved
    improvement_rate = improved_count / total_count
    return improvement_rate < 0.6


def save_oracle_blocked(
    path: str,
    oracle_results: Dict[str, Any],
) -> None:
    """Save an ``oracle_blocked.json`` file with failure details.

    Parameters
    ----------
    path : str
        Output file path (should end in ``oracle_blocked.json``).
    oracle_results : dict
        Results dict containing ``oracle``, ``baseline``, and optional
        ``details`` keys.
    """
    blocked = check_oracle_blocked(oracle_results)

    output = {
        "oracle_blocked": blocked,
        "oracle": oracle_results.get("oracle", {}),
        "baseline": oracle_results.get("baseline", {}),
        "details": oracle_results.get("details", {}),
        "analysis": _analyze_blocking(oracle_results) if blocked else None,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def _analyze_blocking(oracle_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze why the Oracle is blocked and provide insights."""
    oracle = oracle_results.get("oracle", {})
    baseline = oracle_results.get("baseline", {})
    analysis: Dict[str, Any] = {}

    for metric in ["HOTA", "AssA", "DetA", "IDF1", "MOTA"]:
        o_val = oracle.get(metric)
        b_val = baseline.get(metric)
        if o_val is not None and b_val is not None:
            analysis[metric] = {
                "baseline": b_val,
                "oracle": o_val,
                "delta": o_val - b_val,
            }

    # Check ID switches specifically
    ids_oracle = oracle.get("IDs")
    ids_baseline = baseline.get("IDs")
    if ids_oracle is not None and ids_baseline is not None:
        analysis["IDs"] = {
            "baseline": ids_baseline,
            "oracle": ids_oracle,
            "delta": ids_oracle - ids_baseline,
            "worsened": ids_oracle > ids_baseline,
        }

    return analysis


# ---------------------------------------------------------------------------
#  Convenience: evaluate from cached events
# ---------------------------------------------------------------------------


def evaluate_from_events(
    events: List[TrackEvent],
    gt_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate tracking performance directly from a list of events.

    This is a convenience wrapper that first exports the events to
    MOTChallenge format and then runs TrackEval on the exported file.

    Parameters
    ----------
    events : list of TrackEvent
        Events from the data cache.
    gt_path : str
        Path to the ground-truth data.
    output_path : str, optional
        If provided, per-sequence results are saved here.

    Returns
    -------
    dict
        Metric dictionary (same keys as :func:`evaluate_tracking_results`).
    """
    # Export events to a temporary MOTChallenge-format file
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="agentguard_eval_")

    try:
        # Group events by sequence and export
        seq_events: Dict[str, List[TrackEvent]] = {}
        for ev in events:
            seq_events.setdefault(ev.sequence, []).append(ev)

        for seq_name, seq_evts in seq_events.items():
            seq_dir = os.path.join(tmp_dir, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            out_path = os.path.join(seq_dir, f"{seq_name}.txt")

            # Write in MOTChallenge format: frame, id, x, y, w, h, score, -1, -1, -1
            lines: List[str] = []
            for ev in seq_evts:
                if ev.has_detection and ev.detection is not None:
                    box = ev.detection.box
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    lines.append(
                        f"{ev.frame_id},{ev.track_id},{box[0]:.1f},{box[1]:.1f},"
                        f"{w:.1f},{h:.1f},{ev.detection.score:.3f},-1,-1,-1"
                    )
            lines.sort(key=lambda x: (int(x.split(",")[0]), int(x.split(",")[1])))

            with open(out_path, "w") as f:
                f.write("\n".join(lines))

        # Evaluate the exported results
        results = evaluate_tracking_results(
            gt_path=gt_path,
            tracker_path=tmp_dir,
            output_path=output_path,
        )

    finally:
        # Clean up
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


__all__ = [
    "evaluate_tracking_results",
    "compare_results",
    "check_oracle_blocked",
    "save_oracle_blocked",
    "evaluate_from_events",
]
