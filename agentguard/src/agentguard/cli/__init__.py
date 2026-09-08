"""Command-line entry points for the current IWG + RG-CMA pipeline.

Usage
-----
    python -m agentguard.cli --help
    python -m agentguard.cli cache_events --dataset MOT17 --mode train_custom
    python -m agentguard.cli validate_cache --dataset MOT17 --mode train_custom
    python -m agentguard.cli build_rollout_labels --dataset MOT17 --mode train_custom
    python -m agentguard.cli build_iwg_rg_cma_data --help
    python -m agentguard.cli validate_iwg_rg_cma_checkpoint --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_DESCRIPTOR,
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Path handling – always reach the TrackTrack project root from this file.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_AG_PYPROJECT = _PROJECT_ROOT / "agentguard" / "pyproject.toml"
_TRACKER_DIR = _PROJECT_ROOT / "3. Tracker"
assert _AG_PYPROJECT.is_file(), (
    f"agentguard/pyproject.toml not found at {_PROJECT_ROOT}"
)
assert _TRACKER_DIR.is_dir(), (
    f"'3. Tracker' directory not found at {_PROJECT_ROOT}"
)
PROJECT_ROOT = str(_PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mode → split routing
# ---------------------------------------------------------------------------
MODE_TO_SPLIT: Dict[str, str] = {
    "train_custom": "train",
    "val_custom": "val",
    "train": "train",
    "val": "val",
    "all": "all",
    "test": "test",
}


def _resolve_split(mode: str) -> str:
    """Map user-facing *mode* to a split YAML name.

    Must never index a split YAML file with ``train_custom`` or
    ``val_custom`` — those map to the standard ``train`` / ``val`` splits.
    """
    split = MODE_TO_SPLIT.get(mode)
    if split is None:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of {list(MODE_TO_SPLIT)}"
        )
    return split

# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------


def _outputs_dir() -> str:
    return os.path.join(PROJECT_ROOT, "outputs")


def _default_event_cache_root() -> str:
    return os.path.join(_outputs_dir(), "agentguard", "event_cache_v3_iwg_v2")


def _event_cache_dir(dataset: str, mode: str, root: Optional[str] = None) -> str:
    split = _resolve_split(mode)
    base = root or _default_event_cache_root()
    return os.path.join(base, dataset, split)


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _labels_dir(dataset: str) -> str:
    return os.path.join(_outputs_dir(), "agentguard", "labels", dataset)



def _gt_root_for_dataset_mode(data_dir: str, dataset: str, mode: str) -> str:
    split = _resolve_split(mode)
    lowered = dataset.lower()
    if "dance" in lowered:
        if split in {"val", "test", "train"}:
            return os.path.join(data_dir, dataset, split)
        return os.path.join(data_dir, dataset, "train")
    if "mot" in lowered:
        return os.path.join(data_dir, dataset, "train")
    return os.path.join(data_dir, dataset, split)



def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_sequences(
    dataset: str, mode: str, explicit: Optional[List[str]] = None
) -> List[str]:
    """Return the list of sequence names for *dataset/mode*.

    When *explicit* is provided it is returned as-is.  Otherwise the
    :class:`~agentguard.data.split_manager.SplitManager` is consulted.
    """
    if explicit:
        return explicit
    from agentguard.data.split_manager import SplitManager

    manager = SplitManager(
        config_dir=os.path.join(PROJECT_ROOT, "agentguard", "configs", "splits")
    )
    return manager.get_sequences(dataset, _resolve_split(mode))


def _base_sequence_id(sequence: str) -> str:
    parts = sequence.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else sequence


def _resolve_detection_cache_sequences(
    dataset: str,
    mode: str,
    detection_cache_root: str,
    explicit: Optional[List[str]] = None,
    detector: Optional[str] = None,
) -> List[str]:
    if explicit:
        sequences = list(explicit)
    else:
        split = _resolve_split(mode)
        manifest_path = Path(detection_cache_root) / dataset / split / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Detection cache manifest not found: {manifest_path}")
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        sequences = sorted((manifest.get("sequences") or {}).keys())
        if split != "all":
            try:
                base_ids = set(_resolve_sequences(dataset, mode))
            except FileNotFoundError:
                base_ids = set()
            if base_ids:
                sequences = [s for s in sequences if _base_sequence_id(s) in base_ids]
    if detector:
        suffix = f"-{detector}"
        filtered = [s for s in sequences if s.endswith(suffix)]
        if filtered:
            sequences = filtered
    return sequences


# ===================================================================
# Subcommand: cache_events
# ===================================================================


def _add_cache_events_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cache_events",
        help="Run TrackTrack (AgentGuard off) and cache per-frame events.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name (e.g. MOT17).")
    p.add_argument(
        "--mode",
        default="train_custom",
        choices=["val", "val_custom", "train_custom", "all", "test"],
        help="TrackTrack evaluation mode.",
    )
    p.add_argument(
        "--sequence", type=str, default=None, help="Single sequence to process."
    )
    p.add_argument(
        "--detector",
        type=str,
        default="FRCNN",
        help="Detector suffix to select from detection-cache manifest (e.g. FRCNN).",
    )
    p.add_argument("--max-frames", type=int, default=0, help="Max frames per seq (0=unlimited).")
    p.add_argument(
        "--data-dir",
        default=os.environ.get("TRACKTRACK_DATA_DIR", "/home/shang/datasets/"),
        help="Root directory for dataset images / GT.",
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
        help="Root of per-sequence mmap detection caches.",
    )
    p.add_argument(
        "--event-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache_v3_iwg_v2"),
        help="Root for compact AgentGuard event caches.",
    )


def _cmd_cache_events(args: argparse.Namespace) -> None:
    import hashlib
    import gc
    import subprocess

    _ensure_dir(getattr(args, "event_cache_root", _default_event_cache_root()))

    from trackers.tracker import Tracker
    from utils.etc import set_parameters
    from agentguard.data.compact_event_cache import CompactEventCacheSink
    from agentguard.data.detection_cache import SequenceDetectionCache, sequence_cache_dir

    dataset = args.dataset
    mode = args.mode
    split = _resolve_split(mode)
    data_dir = args.data_dir
    max_frames = getattr(args, 'max_frames', 0)
    detection_cache_root = getattr(
        args,
        "detection_cache_root",
        os.path.join(_outputs_dir(), "agentguard", "detection_cache"),
    )
    event_cache_root = getattr(
        args,
        "event_cache_root",
        _default_event_cache_root(),
    )

    class _TrackerArgs:
        pass

    tracker_args = _TrackerArgs()
    # TrackTrack's shared parameter loader still populates unused pickle paths.
    tracker_args.pickle_dir = os.path.join(PROJECT_ROOT, "outputs", "2. det_feat")
    tracker_args.data_dir = data_dir
    tracker_args.min_len = 3
    tracker_args.min_box_area = 100
    tracker_args.max_time_lost = 30
    tracker_args.penalty_p = 0.20
    tracker_args.penalty_q = 0.40
    tracker_args.reduce_step = 0.05
    tracker_args.tai_thr = 0.55
    tracker_args.disable_gmc = False
    tracker_args.kf_type = "nsa"
    # Capture constructs the recorder while preserving pure TrackTrack updates.
    # 'off' no longer creates an adapter in the current Tracker implementation.
    tracker_args.agentguard_mode = "capture"
    tracker_args.capture_agentguard_events = True
    tracker_args.no_reid = False

    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
            source_commit = ""

    explicit = [args.sequence] if args.sequence else None
    sequences = _resolve_detection_cache_sequences(
        dataset=dataset,
        mode=mode,
        detection_cache_root=detection_cache_root,
        explicit=explicit,
        detector=getattr(args, "detector", None),
    )
    if not sequences:
        print(f"No sequences found for {dataset}/{mode} in {detection_cache_root}")
        return

    for seq_name in sequences:
        print(f"\n{'='*60}")
        print(f"Processing {seq_name} ({dataset}/{mode})")
        print(f"{'='*60}")

        set_parameters(tracker_args, seq_name, mode)

        seqinfo_path = os.path.join(tracker_args.data_path, seq_name, "seqinfo.ini")
        img_w, img_h = 1920, 1080
        if os.path.isfile(seqinfo_path):
            with open(seqinfo_path, "r") as f:
                for line in f:
                    if "imWidth" in line:
                        img_w = int(line.split("=")[-1])
                    if "imHeight" in line:
                        img_h = int(line.split("=")[-1])
                    if "frameRate" in line:
                        tracker_args.max_time_lost = int(line.split("=")[-1]) * 2
        tracker_args.img_w = img_w
        tracker_args.img_h = img_h
        tracker_args.dataset = dataset

        seq_cache_path = sequence_cache_dir(
            detection_cache_root,
            dataset,
            split,
            seq_name,
        )
        if not seq_cache_path.is_dir():
            raise FileNotFoundError(
                f"Per-sequence detection cache not found: {seq_cache_path}. "
                "Run scripts/agentguard/build_detection_mmap_cache.py first."
            )
        detection_cache = SequenceDetectionCache(seq_cache_path)
        total_frames = detection_cache.num_frames
        reid_dim = detection_cache.reid_dim
        print(f"  Using mmap detection cache: {seq_cache_path}")

        detection_manifest_path = Path(seq_cache_path) / "manifest.json"
        detection_manifest_sha = (
            _sha256_file(detection_manifest_path)
            if detection_manifest_path.is_file()
            else ""
        )
        tracker_config = {
            "agentguard_mode": "capture",
            "capture_agentguard_events": True,
            "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "dataset": dataset,
            "mode": mode,
            "sequence": seq_name,
            "max_frames": int(max_frames),
            "kf_type": tracker_args.kf_type,
            "disable_gmc": bool(tracker_args.disable_gmc),
            "det_thr": float(getattr(tracker_args, "det_thr", 0.0)),
            "init_thr": float(getattr(tracker_args, "init_thr", 0.0)),
            "match_thr": float(getattr(tracker_args, "match_thr", 0.0)),
            "max_time_lost": int(getattr(tracker_args, "max_time_lost", 0)),
        }
        tracker_config_hash = _sha256_json(tracker_config)
        config_hash = tracker_config_hash

        final_event_dir = Path(event_cache_root) / dataset / split / seq_name
        final_manifest = final_event_dir / "manifest.json"
        if final_manifest.is_file():
            with final_manifest.open("r") as f:
                existing = json.load(f)
            if (
                existing.get("complete")
                and int(existing.get("schema_version", 0)) == COMPACT_CACHE_SCHEMA_VERSION
                and existing.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256
                and existing.get("tracker_config_sha256") == tracker_config_hash
                and existing.get("detection_cache_manifest_sha256") == detection_manifest_sha
            ):
                print(f"  Existing complete compact cache matches config; skipping {seq_name}.")
                if detection_cache is not None:
                    detection_cache.close()
                continue

        event_sink = CompactEventCacheSink(
            cache_root=event_cache_root,
            dataset=dataset,
            split=split,
            sequence=seq_name,
            reid_dim=reid_dim,
            event_flush_size=256,
            frame_flush_size=32,
            association_flush_size=32,
            config_hash=config_hash,
            source_commit=source_commit,
            feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            detection_cache_manifest_sha256=detection_manifest_sha,
            tracker_config_sha256=tracker_config_hash,
            total_sequence_frames=total_frames,
            image_width=img_w,
            image_height=img_h,
            tracker_config=tracker_config,
        )
        tracker_args.event_sink = event_sink
        event_sink.on_sequence_start(
            sequence=seq_name,
            reid_dim=reid_dim,
            image_width=img_w,
            image_height=img_h,
        )
        event_sink._truncated = bool(max_frames > 0 and max_frames < total_frames)
        event_sink._num_detections = int(
            detection_cache.num_detections if detection_cache is not None else 0
        )

        tracker = Tracker(tracker_args, seq_name)
        try:
            frame_count = total_frames if max_frames <= 0 else min(max_frames, total_frames)
            for frame_idx in range(frame_count):
                target_frame = detection_cache.get_frame(frame_idx, view="target")
                source_frame = detection_cache.get_frame(frame_idx, view="source")
                tracker_args.agentguard_target_detection_indices = target_frame[
                    "detection_indices"
                ]
                tracker_args.agentguard_source_detection_indices = source_frame[
                    "detection_indices"
                ]
                det_frame = detection_cache.get_frame_array(frame_idx, view="target")
                det_frame_95 = detection_cache.get_frame_array(frame_idx, view="source")

                if det_frame is not None and len(det_frame) > 0:
                    if det_frame_95 is None:
                        det_frame_95 = det_frame
                    tracker.update(det_frame, det_frame_95)
                else:
                    tracker.update_without_detections()
        finally:
            detection_cache.close()
            tracker_args.agentguard_target_detection_indices = None
            tracker_args.agentguard_source_detection_indices = None

        manifest_info = event_sink.on_sequence_end()
        total_events = manifest_info.get("num_events", 0)
        print(
            f"  Wrote {total_events} events, {manifest_info.get('num_frames', 0)} frames "
            f"to {event_cache_root}/{dataset}/{split}/{seq_name}"
        )
        del tracker
        del detection_cache
        gc.collect()


# _extract_tracker_events removed — events now come from the adapter during processing


# ===================================================================
# Subcommand: validate_cache
# ===================================================================


def _add_validate_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "validate_cache",
        help="Validate cached event data for a dataset/mode.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        choices=["val", "val_custom", "train_custom", "all", "test"],
        help="Dataset split mode.",
    )
    p.add_argument(
        "--event-cache-root",
        default=_default_event_cache_root(),
        help="Root of compact AgentGuard event caches.",
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
        help="Root of per-sequence mmap detection caches.",
    )


def _cmd_validate_cache(args: argparse.Namespace) -> None:
    from agentguard.data.cache_reader import CompactEventCacheReader
    from agentguard.data.detection_cache import sequence_cache_dir
    from agentguard.features.scalar import compute_scalar_features

    dataset = args.dataset
    mode = args.mode
    split = _resolve_split(mode)
    cache_root = os.path.join(args.event_cache_root, dataset, split)

    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Cache directory not found: {cache_root}")

    sequences = sorted(
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
        and not d.endswith(".incomplete")
        and not d.startswith("_")
    )

    totals = {
        "total_events": 0,
        "matched_events": 0,
        "unmatched_events": 0,
        "association_record_count": 0,
        "invalid_detection_indices": 0,
        "duplicate_event_ids": 0,
        "warp_alignment_errors": 0,
        "history_length_mismatches": 0,
        "context_shape_errors": 0,
        "overlap_contract_errors": 0,
        "association_local_column_errors": 0,
        "scalar_nonfinite_events": 0,
    }
    scalar_abs_sum = 0.0
    scalar_abs_count = 0
    scalar_max_abs = 0.0
    scalar_index_abs = {25: [], 56: [], 57: []}
    feature_values = {25: [], 56: [], 57: []}
    feature_nonzero_count = feature_element_count = 0
    reid_dim = 0
    empty_caches: List[str] = []
    incomplete_sequences: List[str] = []
    schema_contracts: set[tuple[int, str]] = set()

    per_seq: List[Dict[str, Any]] = []

    for seq in sequences:
        seq_dir = os.path.join(cache_root, seq)
        det_dir = sequence_cache_dir(args.detection_cache_root, dataset, split, seq)
        try:
            reader = CompactEventCacheReader(seq_dir, det_dir)
            manifest = reader.manifest
        except Exception as e:
            print(f"  FAIL {seq}: {e}")
            incomplete_sequences.append(seq)
            continue

        schema_contracts.add(
            (
                int(manifest.get("schema_version", 0)),
                str(manifest.get("feature_schema_sha256", "")),
            )
        )
        seq_counts = {key: 0 for key in totals}
        n_assoc = int(manifest.get("num_association_records", 0))
        seq_counts["association_record_count"] = n_assoc
        seq_feature_nonzero = 0
        seq_feature_total = 0
        ids_seen: set[str] = set()
        try:
            for shard_id in range(len(reader.state_shards)):
                for state in reader._load_shard("states", shard_id):
                    lengths = (
                        len(np.asarray(state.get("recent_history_frames", []))),
                        len(np.asarray(state.get("recent_history_boxes", []))),
                        len(np.asarray(state.get("recent_history_scores", []))),
                    )
                    if len(set(lengths)) != 1:
                        seq_counts["history_length_mismatches"] += 1

            for ev in reader.iter_event_records():
                seq_counts["total_events"] += 1
                event_id = str(ev.get("event_id", ""))
                if event_id in ids_seen:
                    seq_counts["duplicate_event_ids"] += 1
                ids_seen.add(event_id)

                f = np.asarray(ev.get("track_feature", []), dtype=np.float32)
                seq_feature_nonzero += int(np.count_nonzero(f))
                seq_feature_total += int(f.size)
                cached_scalar = np.asarray(ev.get("scalar_features", []), dtype=np.float64).reshape(-1)
                if cached_scalar.shape != (63,) or not np.all(np.isfinite(cached_scalar)):
                    seq_counts["scalar_nonfinite_events"] += 1

                frame = reader.get_frame_record(ev.get("frame_index", int(ev["frame_id"]) - 1))
                wm = None if frame is None else frame.get("warp_matrix", frame.get("effective_warp"))
                if wm is not None and np.asarray(wm).shape != (2, 3):
                    seq_counts["warp_alignment_errors"] += 1

                if not bool(ev.get("matched", False)):
                    seq_counts["unmatched_events"] += 1
                    continue
                seq_counts["matched_events"] += 1
                global_det_idx = int(ev.get("accepted_detection_index", -1))
                assoc = reader.get_association(
                    ev.get("association_shard_id", -1),
                    ev.get("association_offset", -1),
                )
                if global_det_idx < 0 or assoc is None:
                    seq_counts["invalid_detection_indices"] += 1
                    continue
                detection_indices = np.asarray(
                    assoc.get("detection_indices", []),
                    dtype=np.int64,
                ).reshape(-1)
                local_matches = np.flatnonzero(detection_indices == global_det_idx)
                if local_matches.size != 1:
                    seq_counts["association_local_column_errors"] += 1
                    continue
                try:
                    reader.get_detection(global_det_idx)
                    event = reader.materialize_training_event(ev)
                except Exception:
                    seq_counts["context_shape_errors"] += 1
                    continue

                ctx = event.association_context
                local_col = int(local_matches[0])
                if ctx is None or ctx.accepted_detection_index != local_col:
                    seq_counts["association_local_column_errors"] += 1
                    continue
                overlap = np.asarray(ctx.detection_overlap_row, dtype=np.float64)
                if (
                    overlap.shape != (ctx.num_detections,)
                    or not np.all(np.isfinite(overlap))
                    or np.any(overlap < 0.0)
                    or np.any(overlap > 1.0 + 1e-12)
                    or abs(float(overlap[local_col])) > 1e-12
                ):
                    seq_counts["overlap_contract_errors"] += 1

                recomputed = compute_scalar_features(event)
                if cached_scalar.shape == (63,):
                    abs_error = np.abs(cached_scalar - recomputed)
                    scalar_abs_sum += float(abs_error.sum())
                    scalar_abs_count += int(abs_error.size)
                    scalar_max_abs = max(scalar_max_abs, float(abs_error.max(initial=0.0)))
                    for index in scalar_index_abs:
                        scalar_index_abs[index].append(float(abs_error[index]))
                        feature_values[index].append(float(recomputed[index]))
        finally:
            reader.close()

        for key, value in seq_counts.items():
            totals[key] += value
        feature_nonzero_count += seq_feature_nonzero
        feature_element_count += seq_feature_total
        reid_dim = int(manifest.get("reid_dim", reid_dim))

        if seq_counts["total_events"] == 0:
            empty_caches.append(seq)
        if not bool(manifest.get("complete", False)):
            incomplete_sequences.append(seq)

        seq_feature_ratio = (seq_feature_nonzero / max(seq_feature_total, 1))

        per_seq.append(
            {
                "sequence": seq,
                "processed_frames": int(manifest.get("processed_frames", manifest.get("num_frames", 0))),
                **seq_counts,
                "events_with_real_detection": seq_counts["matched_events"],
                "association_context_count": n_assoc,
                "feature_nonzero_ratio": seq_feature_ratio,
                "complete": bool(manifest.get("complete", False)),
                "truncated": bool(manifest.get("truncated", False)),
                "num_detection_records": int(manifest.get("num_detections", 0)),
                "reid_dim": int(manifest.get("reid_dim", 0)),
            }
        )

    feature_nonzero_ratio = (
        feature_nonzero_count / max(feature_element_count, 1)
    )

    contract_error_keys = (
        "invalid_detection_indices",
        "duplicate_event_ids",
        "warp_alignment_errors",
        "history_length_mismatches",
        "context_shape_errors",
        "overlap_contract_errors",
        "association_local_column_errors",
        "scalar_nonfinite_events",
    )
    schema_consistent = schema_contracts == {
        (COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256)
    }
    scalar_parity = {
        "max_abs_error": scalar_max_abs,
        "mean_abs_error": scalar_abs_sum / max(scalar_abs_count, 1),
        **{
            f"feature_{index}_max_abs_error": max(values, default=0.0)
            for index, values in scalar_index_abs.items()
        },
    }
    has_errors = (
        bool(empty_caches)
        or bool(incomplete_sequences)
        or not schema_consistent
        or any(totals[key] > 0 for key in contract_error_keys)
        or scalar_max_abs >= 1e-5
    )
    report = {
        "dataset": dataset,
        "mode": mode,
        "num_sequences": len(sequences),
        "processed_frames": sum(item["processed_frames"] for item in per_seq),
        **totals,
        "events_with_real_detection": totals["matched_events"],
        "association_context_count": totals["association_record_count"],
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "schema_consistent": schema_consistent,
        "scalar_parity": scalar_parity,
        "feature_statistics": {
            str(index): {
                "count": len(values),
                "min": min(values, default=0.0),
                "max": max(values, default=0.0),
                "mean": float(np.mean(values)) if values else 0.0,
            }
            for index, values in feature_values.items()
        },
        "feature_nonzero_ratio": feature_nonzero_ratio,
        "complete": not has_errors,
        "truncated": any(item.get("truncated", False) for item in per_seq),
        "reid_dim": reid_dim,
        "per_sequence": per_seq,
        "empty_caches": empty_caches,
        "incomplete_sequences": incomplete_sequences,
        "status": (
            "ok" if not has_errors else "issues_found"
        ),
    }

    out_dir = _ensure_dir(
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "agentguard",
            "reports",
            "cache_validation",
            dataset,
            split,
        )
    )
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Validation report saved to {out_path}")
    print(json.dumps(report, indent=2, default=str))
    if has_errors:
        raise RuntimeError(f"Cache contract validation failed; see {out_path}")


# ===================================================================
# Subcommand: rollout_smoke
# ===================================================================


def _add_rollout_smoke_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rollout_smoke",
        help="Materialize compact-cache events through the rollout input path.",
    )
    p.add_argument("--dataset", default="MOT17")
    p.add_argument(
        "--mode",
        default="all",
        choices=["val", "val_custom", "train_custom", "all", "test"],
    )
    p.add_argument("--sequence", default=None)
    p.add_argument("--max-events", type=int, default=100)
    p.add_argument(
        "--event-cache-root",
        default=_default_event_cache_root(),
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
    )


def _cmd_rollout_smoke(args: argparse.Namespace) -> None:
    from agentguard.data.cache_reader import CompactEventCacheReader
    from agentguard.data.detection_cache import sequence_cache_dir

    dataset = args.dataset
    split = _resolve_split(args.mode)
    cache_root = Path(args.event_cache_root) / dataset / split
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = sorted(
            p.name
            for p in cache_root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and not p.name.endswith(".incomplete")
        )

    checked = 0
    matched = 0
    unmatched = 0
    for seq in sequences:
        reader = CompactEventCacheReader(
            cache_root / seq,
            sequence_cache_dir(args.detection_cache_root, dataset, split, seq),
        )
        try:
            for record in reader.iter_event_records():
                event = reader.materialize_training_event(record)
                assert event.frame_start_state is not None
                assert event.pre_update_state is not None
                assert event.track_feature.ndim == 1
                assert event.warp_matrix.shape == (2, 3)
                if event.has_detection:
                    assert event.detection is not None
                    assert event.detection.feature.reshape(-1).shape[0] == reader.detection_cache.reid_dim
                    assert event.association_context is not None
                    matched += 1
                else:
                    assert event.detection is None
                    unmatched += 1
                checked += 1
                if checked >= args.max_events:
                    break
        finally:
            reader.close()
        if checked >= args.max_events:
            break

    report = {
        "dataset": dataset,
        "mode": args.mode,
        "checked_events": checked,
        "matched_events": matched,
        "unmatched_events": unmatched,
        "status": "ok" if checked == args.max_events else "incomplete",
    }
    print(json.dumps(report, indent=2))


# ===================================================================
# Subcommand: build_rollout_labels
# ===================================================================


def _add_build_rollout_labels_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build_rollout_labels",
        help="Build rollout labels (motion & appearance benefit, soft targets) "
        "from cached events.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        choices=["val", "val_custom", "train_custom", "all", "test"],
        help="Dataset split mode.",
    )
    p.add_argument("--sequence", type=str, default=None, help="Single sequence to process.")
    p.add_argument("--max-events", type=int, default=0, help="Limit labels per run (0=all).")
    p.add_argument("--future-frames", type=int, default=5, help="Future rollout frames.")
    p.add_argument(
        "--data-dir",
        default=os.environ.get("TRACKTRACK_DATA_DIR", "/home/shang/datasets/"),
        help="Root directory containing MOT datasets.",
    )
    p.add_argument(
        "--event-cache-root",
        default=_default_event_cache_root(),
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
    )
    p.add_argument("--label-dir", default=None, help="Override rollout label directory.")
def _cmd_build_rollout_labels(args: argparse.Namespace) -> None:
    from agentguard.data.rollout_label_builder import (
        build_compact_rollout_labels_for_sequence,
    )

    dataset = args.dataset
    mode = args.mode
    split = _resolve_split(mode)
    cache_root = _event_cache_dir(dataset, mode, args.event_cache_root)
    label_dir = _ensure_dir(_labels_dir(dataset))
    os.environ["AG_GUARD_DATASET"] = dataset
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Event cache split directory not found: {cache_root}")

    sequences = [args.sequence] if args.sequence else sorted(
        d for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d)) and not d.startswith("_")
    )
    label_mode = _label_mode_name(mode)
    labels_out = _ensure_dir(args.label_dir or os.path.join(label_dir, label_mode))
    gt_root = _gt_root_for_dataset_mode(args.data_dir, dataset, mode)
    total_labels = 0
    sequence_summaries: Dict[str, Dict[str, Any]] = {}
    remaining = int(args.max_events)

    for seq in sequences:
        seq_event_dir = os.path.join(cache_root, seq)
        seq_det_dir = os.path.join(args.detection_cache_root, dataset, split, seq)
        if not os.path.isdir(seq_det_dir):
            raise FileNotFoundError(f"Detection cache for {seq} not found: {seq_det_dir}")
        limit = remaining if remaining > 0 else 0
        labels, seq_summary = build_compact_rollout_labels_for_sequence(
            seq_event_dir,
            seq_det_dir,
            gt_root,
            max_events=limit,
            future_frames=args.future_frames,
        )
        if labels:
            seq_path = os.path.join(labels_out, f"{seq}_labels.json")
            with open(seq_path, "w") as f:
                json.dump(labels, f, indent=2, default=str)
        sequence_summaries[seq] = seq_summary
        total_labels += len(labels)
        if remaining > 0:
            remaining -= len(labels)
            if remaining <= 0:
                break

    summary = {
        "dataset": dataset,
        "mode": mode,
        "split": split,
        "label_mode": label_mode,
        "candidate_types": ["A"],
        "num_labels": total_labels,
        "num_sequences": len(sequence_summaries),
        "sequences": sequence_summaries,
    }
    summary.update(
        {
            "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
            "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
            "label_schema_descriptor": ROLLOUT_LABEL_SCHEMA_DESCRIPTOR,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        }
    )
    summary_path = os.path.join(labels_out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Labels saved to {labels_out}")
    print(json.dumps(summary, indent=2))





def _add_build_iwg_rg_cma_data_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "build_iwg_rg_cma_data",
        help="Build train-all causal context windows for safe-direct IWG+RG-CMA.",
    )
    parser.add_argument(
        "--dataset",
        default="MOT17",
        choices=["MOT17", "MOT20", "DanceTrack", "SportsMOT"],
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "train", "trainval"],
        help="Training source split; trainval is supported for SportsMOT only.",
    )
    parser.add_argument("--event-cache-root", required=True)
    parser.add_argument(
        "--detection-cache-root",
        required=True,
        help="Canonical detection/ReID cache used to build the separate ReID file.",
    )
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence subset, e.g. MOT17-09-FRCNN for a low-disk smoke build.",
    )
    parser.add_argument("--max-frame-gap", type=int, default=30)
    parser.add_argument(
        "--context-size",
        type=int,
        choices=[6, 8],
        default=6,
        help="Number of events passed to IWG/RG-CMA (6 preserves old datasets).",
    )


def _cmd_build_iwg_rg_cma_data(args: argparse.Namespace) -> None:
    from agentguard.datasets.iwg_rg_cma_dataset import build_iwg_rg_cma_dataset

    metadata = build_iwg_rg_cma_dataset(
        dataset=args.dataset,
        split=args.mode,
        event_cache_root=args.event_cache_root,
        detection_cache_root=args.detection_cache_root,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        sequence_subset=args.sequences,
        max_frame_gap=args.max_frame_gap,
        context_size=args.context_size,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {parsed}"
        )
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative integer, got {parsed}"
        )
    return parsed


class _RaisingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _add_train_iwg_rg_cma_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Additionally save a numbered checkpoint every N epochs. "
            "Omitted keeps the default 100e/200e/finetune cadence. "
            "Does not replace last.pt or the final numbered checkpoint."
        ),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--base-lr",
        type=float,
        default=None,
        help="Base IWG learning rate; defaults to --lr.",
    )
    parser.add_argument(
        "--cma-lr",
        type=float,
        default=None,
        help="RG-CMA learning rate; defaults to --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--diagnostic-interval",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Compute expensive diagnostics every N batches. "
            "0 disables them (default). 1 restores per-batch diagnostics. "
            "Does not change the training loss, clipping, or optimizer."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Write intra-epoch progress every N batches. "
            "0 keeps the previous auto cadence of about five logs per epoch. "
            "Independent of --diagnostic-interval."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--correction-bound",
        type=float,
        choices=[0.05, 0.10],
        default=0.05,
        help="RG-CMA correction bound; the official baseline uses 0.05.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        choices=[6, 8],
        default=6,
        help="Number of events passed to IWG/RG-CMA (official baseline: 6).",
    )
    parser.add_argument(
        "--reliability-mode",
        choices=["full", "no-scalar"],
        default="full",
        help="Reliability token input mode; no-scalar masks normalized scalar features.",
    )
    parser.add_argument(
        "--architecture-variant",
        choices=[
            "legacy",
            "legacy-clean-cross-modal",
            "legacy-clean-cross-modal-base-conditioned",
            "legacy-clean-bidirectional-cma",
            "legacy-clean-6x6-cma",
            "direct-base",
            "clean-cross-modal",
            "selective-correction",
        ],
        default="legacy",
        help="Model structure used for controlled RG-CMA ablations.",
    )
    parser.add_argument(
        "--residual-beta",
        type=float,
        default=1.0,
        help="Smooth-L1 beta for the RG-CMA correction residual loss.",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.5,
        help="Weight of the RG-CMA correction residual loss.",
    )
    parser.add_argument(
        "--revision-weight",
        type=float,
        default=0.01,
        help="L1 penalty on RG-CMA corrections; 0 disables this penalty.",
    )
    parser.add_argument(
        "--hard-example-gain",
        type=float,
        default=0.0,
        help="Extra residual-loss weight for targets near the correction bound.",
    )
    parser.add_argument(
        "--no-harm-weight",
        type=float,
        default=0.0,
        help="Penalty when a final gate has larger target error than its base gate.",
    )
    parser.add_argument(
        "--residual-target-mode",
        choices=["safe", "oracle-confidence", "safe-selective"],
        default="safe",
        help=(
            "CMA residual target. safe-selective learns only the additional "
            "safe-to-oracle correction and exactly abstains on uncertain or "
            "indecisive labels."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help=(
            "Strictly warm-start model weights from an IWG RG-CMA checkpoint. "
            "Optimizer, scheduler, epoch, and normalization are not restored."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--memory-shards",
        type=int,
        default=1,
        help="Split every sequence into this many contiguous training fractions.",
    )
    parser.add_argument(
        "--epochs-per-shard",
        type=int,
        default=0,
        help="Epochs per memory shard (0 infers from total epochs and cycles).",
    )
    parser.add_argument("--shard-cycles", type=int, default=1)
    parser.add_argument(
        "--randomize-shard-order",
        action="store_true",
        help="Shuffle shard order independently for each cycle using --seed.",
    )
    parser.add_argument(
        "--sequence-sampling",
        choices=["sample-proportional", "sqrt-size"],
        default="sample-proportional",
        help=(
            "Sequence sampling policy. sqrt-size keeps the phase sample/step "
            "budget and samples within sequences with replacement."
        ),
    )


def _add_train_iwg_rg_cma_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train_iwg_rg_cma",
        help="Train fixed safe-direct IWG+RG-CMA for 100 or 200 epochs.",
    )
    _add_train_iwg_rg_cma_arguments(parser)


def train_iwg_rg_cma_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "checkpoint_every": (
            None if args.checkpoint_every is None else int(args.checkpoint_every)
        ),
        "device": args.device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "base_lr": (
            float(args.base_lr) if args.base_lr is not None else float(args.lr)
        ),
        "cma_lr": (
            float(args.cma_lr) if args.cma_lr is not None else float(args.lr)
        ),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "grad_clip": float(args.grad_clip),
        "diagnostic_interval": int(args.diagnostic_interval),
        "log_interval": int(args.log_interval),
        "seed": int(args.seed),
        "correction_bound": float(args.correction_bound),
        "context_size": int(args.context_size),
        "reliability_mode": str(args.reliability_mode),
        "architecture_variant": str(args.architecture_variant),
        "residual_beta": float(args.residual_beta),
        "residual_weight": float(args.residual_weight),
        "revision_weight": float(args.revision_weight),
        "hard_example_gain": float(args.hard_example_gain),
        "no_harm_weight": float(args.no_harm_weight),
        "residual_target_mode": str(args.residual_target_mode),
        "init_checkpoint": (
            str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else ""
        ),
        "amp": False,
        "max_train_samples": int(args.max_train_samples),
        "memory_shards": int(args.memory_shards),
        "epochs_per_shard": int(args.epochs_per_shard),
        "shard_cycles": int(args.shard_cycles),
        "randomize_shard_order": bool(args.randomize_shard_order),
        "sequence_sampling": str(args.sequence_sampling),
    }


def parse_train_iwg_rg_cma_command(command: List[str]) -> dict[str, Any]:
    """Parse a process command list that invokes train_iwg_rg_cma."""
    argv = [str(item) for item in command]
    try:
        index = argv.index("train_iwg_rg_cma")
    except ValueError as exc:
        raise ValueError("command does not invoke train_iwg_rg_cma") from exc
    parser = _RaisingArgumentParser(prog="train_iwg_rg_cma")
    _add_train_iwg_rg_cma_arguments(parser)
    args = parser.parse_args(argv[index + 1 :])
    return train_iwg_rg_cma_config_from_args(args)


def validate_train_iwg_rg_cma_command(
    command: List[str],
    *,
    selected_epoch: int | None = None,
) -> dict[str, Any]:
    """Parse a train command and check it against the current training contract."""
    from agentguard.training.train_iwg_rg_cma import (
        _validate_formal_config,
        numbered_checkpoint_epochs,
    )

    config = parse_train_iwg_rg_cma_command(command)
    _validate_formal_config(config)
    planned = numbered_checkpoint_epochs(
        int(config["epochs"]),
        checkpoint_every=config.get("checkpoint_every"),
        warm_start=bool(str(config.get("init_checkpoint", "")).strip()),
    )
    if selected_epoch is not None and int(selected_epoch) not in planned:
        raise ValueError(
            f"selected epoch {int(selected_epoch):03d} would not be saved; "
            f"checkpoint plan is {[f'{epoch:03d}' for epoch in planned]}"
        )
    return config


def _cmd_train_iwg_rg_cma(args: argparse.Namespace) -> None:
    from agentguard.training.train_iwg_rg_cma import train_iwg_rg_cma

    config = train_iwg_rg_cma_config_from_args(args)
    summary = train_iwg_rg_cma(config)
    checkpoint_dir = Path(config["checkpoint_dir"])
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _add_validate_iwg_rg_cma_checkpoint_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "validate_iwg_rg_cma_checkpoint",
        help="Strictly load and evaluate an IWG RG-CMA checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--output", default="")


def _cmd_validate_iwg_rg_cma_checkpoint(args: argparse.Namespace) -> None:
    from agentguard.training.train_iwg_rg_cma import (
        validate_iwg_rg_cma_checkpoint,
    )

    report = validate_iwg_rg_cma_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        device=args.device,
        max_batches=args.max_batches,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


# ===================================================================
# Main entry point
# ===================================================================


def main(argv: Optional[List[str]] = None) -> None:
    """AgentGuard CLI — unified entry point for all subcommands.

    Usage::

        python -m agentguard.cli <subcommand> [options]
        python -m agentguard.cli --help
    """
    parser = argparse.ArgumentParser(
        prog="agentguard",
        description="AgentGuard CLI v%s — Tracking pipeline automation." % VERSION,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"AgentGuard CLI {VERSION}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Valid subcommands",
        required=True,
    )

    # Current IWG-RG-CMA pipeline.
    _add_cache_events_parser(subparsers)
    _add_validate_cache_parser(subparsers)
    _add_rollout_smoke_parser(subparsers)
    _add_build_rollout_labels_parser(subparsers)
    _add_build_iwg_rg_cma_data_parser(subparsers)
    _add_train_iwg_rg_cma_parser(subparsers)
    _add_validate_iwg_rg_cma_checkpoint_parser(subparsers)

    parsed = parser.parse_args(argv)

    # Dispatch to command handler
    dispatch = {
        "cache_events": _cmd_cache_events,
        "validate_cache": _cmd_validate_cache,
        "rollout_smoke": _cmd_rollout_smoke,
        "build_rollout_labels": _cmd_build_rollout_labels,
        "build_iwg_rg_cma_data": _cmd_build_iwg_rg_cma_data,
        "train_iwg_rg_cma": _cmd_train_iwg_rg_cma,
        "validate_iwg_rg_cma_checkpoint": _cmd_validate_iwg_rg_cma_checkpoint,
    }

    handler = dispatch.get(parsed.command)
    if handler is not None:
        handler(parsed)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
