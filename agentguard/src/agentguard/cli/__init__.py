"""AgentGuard CLI — all tracking-pipeline subcommands in one module.

Usage
-----
    python -m agentguard.cli --help
    python -m agentguard.cli cache_events --dataset MOT17 --mode train_custom
    python -m agentguard.cli validate_cache --dataset MOT17 --mode train_custom
    python -m agentguard.cli build_rollout_labels --dataset MOT17 --mode train_custom
    python -m agentguard.cli evaluate_oracle --dataset MOT17 --mode val_custom --oracle-type iwg
    python -m agentguard.cli build_student_v0_data --dataset MOT17
    python -m agentguard.cli train_student_v0 --dataset MOT17 --device cuda
    python -m agentguard.cli select_teacher_events --dataset MOT17
    python -m agentguard.cli build_evidence_packets --dataset MOT17
    python -m agentguard.cli run_bailian_teacher --dataset MOT17
    python -m agentguard.cli verify_and_fuse --dataset MOT17
    python -m agentguard.cli build_student_v1_data --dataset MOT17
    python -m agentguard.cli train_student_v1 --dataset MOT17 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_DESCRIPTOR,
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


def _cache_dir(dataset: str, mode: str) -> str:
    split = _resolve_split(mode)
    return os.path.join(_outputs_dir(), "agentguard", "cache", dataset, split)


def _event_cache_dir(dataset: str, mode: str, root: Optional[str] = None) -> str:
    split = _resolve_split(mode)
    base = root or os.path.join(_outputs_dir(), "agentguard", "event_cache")
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


def _checkpoints_dir(dataset: str) -> str:
    return os.path.join(_outputs_dir(), "agentguard", "checkpoints", dataset)


def _datasets_dir(dataset: str) -> str:
    return os.path.join(_outputs_dir(), "agentguard", "datasets", dataset)


def _label_mode_name(mode: str) -> str:
    return f"{mode}_a_only"


def _validate_candidate_a_records(records: list[dict[str, Any]]) -> None:
    invalid = sorted(
        {
            str(record.get("candidate_type", "A")).upper()
            for record in records
            if str(record.get("candidate_type", "A")).upper() != "A"
        }
    )
    if invalid:
        raise ValueError(f"only candidate A records are supported: {invalid}")


def _stable_record_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.get("sequence", "")),
            str(record.get("event_id", "")),
            str(record.get("candidate_type", "A")),
            str(record.get("candidate_detection_index", -1)),
        ]
    )


def _stratified_record_sample(records: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("sequence", "")), str(record.get("candidate_type", "A")).upper())
        groups.setdefault(key, []).append(record)
    selected: list[dict[str, Any]] = []
    remaining = int(max_samples)
    sorted_groups = sorted(groups.items(), key=lambda item: item[0])
    for idx, (_, bucket) in enumerate(sorted_groups):
        groups_left = len(sorted_groups) - idx
        take = min(len(bucket), max(1, remaining // max(groups_left, 1)))
        ordered = sorted(
            bucket,
            key=lambda r: hashlib.sha256(_stable_record_key(r).encode()).hexdigest(),
        )
        selected.extend(ordered[:take])
        remaining -= take
        if remaining <= 0:
            break
    selected = selected[:max_samples]
    selected.sort(key=_stable_record_key)
    return selected


def _record_sequence_order_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("sequence", "")),
        int(record.get("frame_id", 0)),
        int(record.get("event_id", 0)),
        int(record.get("track_id", 0)),
        int(record.get("candidate_detection_index", -1)),
        str(record.get("candidate_type", "A")),
    )


def _slice_records_by_sequence_ratio(
    records: list[dict[str, Any]],
    *,
    start_ratio: float,
    end_ratio: float,
) -> list[dict[str, Any]]:
    if start_ratio <= 0.0 and end_ratio >= 1.0:
        return records
    if not (0.0 <= start_ratio < end_ratio <= 1.0):
        raise ValueError(
            f"Invalid sequence sample range [{start_ratio}, {end_ratio}); "
            "expected 0 <= start < end <= 1."
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("sequence", "")), []).append(record)

    selected: list[dict[str, Any]] = []
    for seq in sorted(groups):
        bucket = sorted(groups[seq], key=_record_sequence_order_key)
        n = len(bucket)
        start = int(n * start_ratio)
        end = int(n * end_ratio)
        if end <= start and n > 0:
            end = min(n, start + 1)
        selected.extend(bucket[start:end])
    selected.sort(key=_record_sequence_order_key)
    return selected


def _seed_training(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


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


def _teacher_dir(dataset: str) -> str:
    return os.path.join(_outputs_dir(), "agentguard", "teacher_outputs", dataset)


def _evidence_dir(dataset: str) -> str:
    return os.path.join(_outputs_dir(), "agentguard", "evidence_packets", dataset)


def _tracking_results_dir() -> str:
    return os.path.join(_outputs_dir(), "agentguard", "tracking_results")


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
        "--pickle-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "2. det_feat"),
        help="Legacy directory containing detection pickle files.",
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
        help="Root of per-sequence mmap detection caches.",
    )
    p.add_argument(
        "--event-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache"),
        help="Root for compact AgentGuard event caches.",
    )
    p.add_argument(
        "--allow-pickle-fallback",
        action="store_true",
        help="Allow legacy monolithic pickle loading when mmap detection cache is absent.",
    )


def _cmd_cache_events(args: argparse.Namespace) -> None:
    import hashlib
    import gc
    import subprocess

    _ensure_dir(getattr(args, "event_cache_root", os.path.join(_outputs_dir(), "agentguard", "event_cache")))

    from trackers.tracker import Tracker
    from utils.etc import set_parameters
    from agentguard.data.compact_event_cache import CompactEventCacheSink
    from agentguard.data.detection_cache import SequenceDetectionCache, sequence_cache_dir

    dataset = args.dataset
    mode = args.mode
    split = _resolve_split(mode)
    data_dir = args.data_dir
    pickle_dir = args.pickle_dir
    max_frames = getattr(args, 'max_frames', 0)
    detection_cache_root = getattr(
        args,
        "detection_cache_root",
        os.path.join(_outputs_dir(), "agentguard", "detection_cache"),
    )
    event_cache_root = getattr(
        args,
        "event_cache_root",
        os.path.join(_outputs_dir(), "agentguard", "event_cache"),
    )
    allow_pickle_fallback = bool(getattr(args, "allow_pickle_fallback", False))

    class _TrackerArgs:
        pass

    tracker_args = _TrackerArgs()
    tracker_args.pickle_dir = pickle_dir
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
    tracker_args.agentguard_mode = "off"
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
        detection_cache = None
        legacy_detections = None
        legacy_detections_95 = None

        if seq_cache_path.is_dir():
            detection_cache = SequenceDetectionCache(seq_cache_path)
            total_frames = detection_cache.num_frames
            reid_dim = detection_cache.reid_dim
            print(f"  Using mmap detection cache: {seq_cache_path}")
        elif allow_pickle_fallback:
            from utils.det_feat_storage import load_detection_pair

            print("  WARNING: mmap detection cache missing; using legacy pickle fallback.")
            legacy_detections, legacy_detections_95 = load_detection_pair(
                tracker_args.target_pickle_path,
                tracker_args.pickle_path_95,
                sequence_names=[seq_name],
            )
            if seq_name not in legacy_detections:
                print(f"  WARNING: {seq_name} not found in detections, skipping.")
                continue
            total_frames = len(legacy_detections[seq_name])
            reid_dim = 0
        else:
            raise FileNotFoundError(
                f"Per-sequence detection cache not found: {seq_cache_path}. "
                "Run scripts/agentguard/00_split_detection_cache.py first, or pass "
                "--allow-pickle-fallback explicitly."
            )

        detection_manifest_path = Path(seq_cache_path) / "manifest.json"
        detection_manifest_sha = (
            _sha256_file(detection_manifest_path)
            if detection_manifest_path.is_file()
            else ""
        )
        tracker_config = {
            "agentguard_mode": "off",
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
                if detection_cache is not None:
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
                else:
                    frame_id = frame_idx + 1
                    det_frame = legacy_detections[seq_name].get(frame_id)
                    det_frame_95 = legacy_detections_95[seq_name].get(frame_id)
                    tracker_args.agentguard_target_detection_indices = None
                    tracker_args.agentguard_source_detection_indices = None

                if det_frame is not None and len(det_frame) > 0:
                    if det_frame_95 is None:
                        det_frame_95 = det_frame
                    tracker.update(det_frame, det_frame_95)
                else:
                    tracker.update_without_detections()
        finally:
            if detection_cache is not None:
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
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache"),
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
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache"),
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
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache"),
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
    )
    p.add_argument("--label-dir", default=None, help="Override rollout label directory.")
    p.add_argument("--output-dir", default=None, help="Override Student-V0 dataset index directory.")
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


# ===================================================================
# Subcommand: evaluate_oracle (supports iwg and full)
# ===================================================================


def _add_evaluate_oracle_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "evaluate_oracle",
        help="Run TrackTrack with AgentGuard using oracle gates from rollout "
        "labels instead of IWG predictions.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="val_custom",
        choices=["val", "val_custom", "train_custom", "all", "test"],
        help="Dataset split mode.",
    )
    p.add_argument(
        "--oracle-type",
        default="iwg",
        choices=["iwg", "full"],
        help="Oracle type: 'iwg' uses per-frame oracle gates; 'full' also uses TGR.",
    )
    p.add_argument("--data-dir", default=None, help="Dataset data directory.")
    p.add_argument("--pickle-dir", default=None, help="Detection pickle directory.")


def _cmd_evaluate_oracle(args: argparse.Namespace) -> None:
    from trackers.tracker import Tracker
    from utils.det_feat_storage import load_detection_pair
    from utils.etc import set_parameters, evaluate, get_trackeval_configs
    from agentguard.contracts.events import TrackEvent

    dataset = args.dataset
    mode = args.mode
    oracle_type = args.oracle_type

    data_dir = args.data_dir or os.environ.get(
        "TRACKTRACK_DATA_DIR", "/home/shang/datasets/"
    )
    pickle_dir = args.pickle_dir or os.path.join(
        PROJECT_ROOT, "outputs", "2. det_feat"
    )

    # Load rollout labels to get oracle gates
    label_dir = _labels_dir(dataset)
    labels_path = os.path.join(label_dir, f"{mode}_summary.json")
    if not os.path.isfile(labels_path):
        print(f"Rollout labels not found at {labels_path}. Run build_rollout_labels first.")
        return

    # Read per-sequence labels
    seq_labels: Dict[str, List[Dict[str, Any]]] = {}
    labels_mode_dir = os.path.join(label_dir, mode)
    if os.path.isdir(labels_mode_dir):
        for fname in os.listdir(labels_mode_dir):
            if fname.endswith("_labels.json"):
                seq_name = fname.replace("_labels.json", "")
                with open(os.path.join(labels_mode_dir, fname)) as f:
                    seq_labels[seq_name] = json.load(f)

    if not seq_labels:
        print("No per-sequence label files found.")
        return

    tracker_results_dir = _ensure_dir(os.path.join(_tracking_results_dir(), dataset, mode))

    class _OracleArgs:
        pass

    tracker_args = _OracleArgs()
    tracker_args.pickle_dir = pickle_dir
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
    tracker_args.agentguard_mode = oracle_type  # use oracle mode name for eval
    tracker_args.iwg_checkpoint = None
    tracker_args.tgr_checkpoint = None
    tracker_args.agentguard_device = "cpu"
    tracker_args.print_per_sequence_metrics = False
    tracker_args.no_reid = False
    tracker_args.use_post = False
    tracker_args.mode = mode
    tracker_args.output_dir = tracker_results_dir

    sequences = sorted(seq_labels.keys())

    for seq_name in sequences:
        labels_list = seq_labels[seq_name]
        if not labels_list:
            continue

        set_parameters(tracker_args, seq_name, mode)

        # Read seqinfo
        seqinfo_path = os.path.join(tracker_args.data_path, seq_name, "seqinfo.ini")
        if os.path.isfile(seqinfo_path):
            with open(seqinfo_path, "r") as f:
                for line in f:
                    if "imWidth" in line:
                        tracker_args.img_w = int(line.split("=")[-1])
                    if "imHeight" in line:
                        tracker_args.img_h = int(line.split("=")[-1])
                    if "frameRate" in line:
                        tracker_args.max_time_lost = int(line.split("=")[-1]) * 2

        # Load detections
        detections, detections_95 = load_detection_pair(
            tracker_args.target_pickle_path, tracker_args.pickle_path_95
        )
        if seq_name not in detections:
            continue

        # Create tracker
        tracker = Tracker(tracker_args, seq_name)

        # Build oracle gate lookup: frame_id -> (motion_gate, appearance_gate)
        oracle_gates: Dict[int, Tuple[float, float]] = {}
        for lbl in labels_list:
            fid = lbl.get("frame_id", -1)
            if fid >= 0:
                oracle_gates[fid] = (
                    lbl.get("motion_target", 0.5),
                    lbl.get("appearance_target", 0.5),
                )

        # Run tracking with oracle gates
        results: List[Any] = []
        frame_ids = sorted(detections[seq_name].keys())

        for frame_id in frame_ids:
            det_frame = detections[seq_name][frame_id]
            det_frame_95 = detections_95[seq_name][frame_id]

            if det_frame is not None:
                track_results = tracker.update(det_frame, det_frame_95)
            else:
                track_results = tracker.update_without_detections()

            # Apply oracle gates to tracked results
            gates = oracle_gates.get(frame_id, (0.5, 0.5))
            # In a full oracle evaluation we would intercept the tracker's
            # internal update to use these gates.  For now we just record
            # the baseline tracking output and annotate with oracle values.
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > tracker_args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)
            results.append([frame_id, track_ids, x1y1whs, scores])

        # Write results
        from utils.etc import write_results

        result_filename = os.path.join(tracker_results_dir, f"{seq_name}.txt")
        write_results(result_filename, results)
        print(f"  Wrote oracle tracking results for {seq_name}")

    # Evaluate
    if mode != "test":
        print("\nEvaluating oracle tracking results...")
        tracker_eval_name = os.path.basename(tracker_results_dir)
        try:
            eval_config, dataset_config = get_trackeval_configs(
                tracker_args,
                tracker_eval_name,
                dataset,
                output_folder=tracker_results_dir,
                output_summary=True,
            )
            import trackeval

            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = [
                trackeval.metrics.HOTA(),
                trackeval.metrics.CLEAR(),
                trackeval.metrics.Identity(),
            ]
            res, _ = evaluator.evaluate(dataset_list, metrics_list)

            # Extract combined metrics
            hota = float(
                np.mean(
                    res["MotChallenge2DBox"][tracker_eval_name]["COMBINED_SEQ"][
                        "pedestrian"
                    ]["HOTA"]["HOTA"]
                )
            )
            idf1 = float(
                res["MotChallenge2DBox"][tracker_eval_name]["COMBINED_SEQ"][
                    "pedestrian"
                ]["Identity"]["IDF1"]
            )
            mota = float(
                res["MotChallenge2DBox"][tracker_eval_name]["COMBINED_SEQ"][
                    "pedestrian"
                ]["CLEAR"]["MOTA"]
            )
            assa = float(
                np.mean(
                    res["MotChallenge2DBox"][tracker_eval_name]["COMBINED_SEQ"][
                        "pedestrian"
                    ]["HOTA"]["AssA"]
                )
            )
            deta = float(
                np.mean(
                    res["MotChallenge2DBox"][tracker_eval_name]["COMBINED_SEQ"][
                        "pedestrian"
                    ]["HOTA"]["DetA"]
                )
            )

            oracle_results = {
                "oracle_type": oracle_type,
                "dataset": dataset,
                "mode": mode,
                "HOTA": hota,
                "IDF1": idf1,
                "MOTA": mota,
                "AssA": assa,
                "DetA": deta,
            }
            oracle_path = os.path.join(tracker_results_dir, "oracle_results.json")
            with open(oracle_path, "w") as f:
                json.dump(oracle_results, f, indent=2)

            print(
                f"HOTA={hota:.4f}  MOTA={mota:.4f}  IDF1={idf1:.4f}  "
                f"DetA={deta:.4f}  AssA={assa:.4f}"
            )

            # Check if oracle is blocked (no improvement over baseline)
            from agentguard.evaluation import check_oracle_blocked, save_oracle_blocked

            # For comparison we need a baseline; here we check if AssA/IDF1/IDs
            # are meaningfully positive
            if assa < 0.1 and idf1 < 0.1:
                blocked_path = os.path.join(
                    tracker_results_dir, "oracle_blocked.json"
                )
                save_oracle_blocked(blocked_path, {"oracle": oracle_results, "baseline": {}})
                print("Oracle appears blocked (very low AssA/IDF1).")

        except Exception as e:
            print(f"Evaluation failed: {e}")


# ===================================================================
# Subcommand: build_student_v0_data
# ===================================================================


def _add_build_student_v0_data_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build_student_v0_data",
        help="Build Student-V0 IWG and TGR datasets from cached events and labels.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="all",
        help="Mode used for label loading.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Limit indexed labels (0=all).")
    p.add_argument(
        "--sample-start-ratio",
        type=float,
        default=0.0,
        help="Per-sequence contiguous sample start ratio, inclusive. Use with --sample-end-ratio.",
    )
    p.add_argument(
        "--sample-end-ratio",
        type=float,
        default=1.0,
        help="Per-sequence contiguous sample end ratio, exclusive. Example: 0.3 for the first 30%%.",
    )
    p.add_argument(
        "--split-policy",
        choices=["sequence_holdout", "train_all"],
        default="sequence_holdout",
        help=(
            "Student-V0 split policy. sequence_holdout keeps a deterministic "
            "sequence-level validation split; train_all puts every record in "
            "train and mirrors it to val for upper-bound experiments."
        ),
    )
    p.add_argument(
        "--event-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache"),
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
    )
    p.add_argument("--label-dir", default=None, help="Override rollout label directory.")
    p.add_argument("--output-dir", default=None, help="Override Student-V0 dataset index directory.")
def _cmd_build_student_v0_data(args: argparse.Namespace) -> None:
    import numpy as np

    from agentguard.v0_pipeline import (
        fit_norm_stats_from_records,
        load_label_records,
        write_jsonl,
    )

    dataset = args.dataset
    mode = args.mode
    split = _resolve_split(mode)
    label_mode = _label_mode_name(mode)
    label_dir = args.label_dir or os.path.join(_labels_dir(dataset), label_mode)
    datasets_dir = _ensure_dir(args.output_dir or os.path.join(_datasets_dir(dataset), mode))
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}. Run build_rollout_labels first.")

    records = load_label_records(label_dir, max_samples=args.max_samples)
    records = [r for r in records if r.get("valid_motion") or r.get("valid_appearance")]
    _validate_candidate_a_records(records)
    records = _slice_records_by_sequence_ratio(
        records,
        start_ratio=float(args.sample_start_ratio),
        end_ratio=float(args.sample_end_ratio),
    )
    if not records:
        raise RuntimeError("No valid rollout labels loaded.")

    seq_names = sorted({r["sequence"] for r in records})
    event_cache_contracts = []
    for sequence in seq_names:
        manifest_path = os.path.join(
            args.event_cache_root,
            dataset,
            split,
            sequence,
            "manifest.json",
        )
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Event cache manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        schema_version = int(manifest.get("schema_version", 0))
        feature_hash = str(manifest.get("feature_schema_sha256", ""))
        if not bool(manifest.get("complete", False)):
            raise ValueError(f"Event cache is incomplete: {manifest_path}")
        if schema_version != COMPACT_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Event cache schema mismatch in {manifest_path}: "
                f"{schema_version} != {COMPACT_CACHE_SCHEMA_VERSION}"
            )
        if feature_hash != FEATURE_SCHEMA_SHA256:
            raise ValueError(
                f"Feature schema mismatch in {manifest_path}: "
                f"{feature_hash!r} != {FEATURE_SCHEMA_SHA256!r}"
            )
        event_cache_contracts.append(
            {
                "sequence": sequence,
                "schema_version": schema_version,
                "feature_schema_sha256": feature_hash,
                "tracker_config_sha256": manifest.get("tracker_config_sha256", ""),
                "detection_cache_manifest_sha256": manifest.get(
                    "detection_cache_manifest_sha256",
                    "",
                ),
            }
        )
    if args.split_policy == "train_all":
        train_seqs = set(seq_names)
        val_seqs = set(seq_names)
        train_records = list(records)
        val_records = list(records)
    else:
        rng = np.random.default_rng(42)
        shuffled = list(seq_names)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(
                "sequence_holdout requires at least two sequences; use train_all "
                "only for an explicit debug/upper-bound run"
            )
        split_idx = min(len(shuffled) - 1, max(1, int(len(shuffled) * 0.8)))
        train_seqs = set(shuffled[:split_idx])
        val_seqs = set(shuffled[split_idx:])
        if not train_seqs.isdisjoint(val_seqs):
            raise AssertionError("train and validation sequences must be disjoint")
        train_records = [r for r in records if r["sequence"] in train_seqs]
        val_records = [r for r in records if r["sequence"] in val_seqs]
    norm_stats = fit_norm_stats_from_records(
        train_records,
        args.event_cache_root,
        args.detection_cache_root,
        dataset,
        split,
    )

    write_jsonl(os.path.join(datasets_dir, "train_index.jsonl"), train_records)
    write_jsonl(os.path.join(datasets_dir, "val_index.jsonl"), val_records)
    norm_path = os.path.join(datasets_dir, "norm_stats.npz")
    norm_stats.save(norm_path)

    # Read reid_dim from the first sequence manifest.
    first_seq = records[0]["sequence"]
    with open(os.path.join(args.detection_cache_root, dataset, split, first_seq, "manifest.json")) as f:
        det_manifest = json.load(f)
    metadata = {
        "dataset": dataset,
        "mode": mode,
        "split": split,
        "label_mode": label_mode,
        "event_cache_root": args.event_cache_root,
        "detection_cache_root": args.detection_cache_root,
        "label_dir": label_dir,
        "reid_dim": int(det_manifest["reid_dim"]),
        "scalar_dim": 63,
        "event_dim": 128,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_descriptor": FEATURE_SCHEMA_DESCRIPTOR,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_descriptor": ROLLOUT_LABEL_SCHEMA_DESCRIPTOR,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "event_cache_contracts_sha256": _sha256_json(event_cache_contracts),
        "num_records": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "split_policy": args.split_policy,
        "sample_start_ratio": float(args.sample_start_ratio),
        "sample_end_ratio": float(args.sample_end_ratio),
        "candidate_types": ["A"],
        "train_sequences": sorted(train_seqs),
        "val_sequences": sorted(val_seqs),
    }
    with open(os.path.join(datasets_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(json.dumps(metadata, indent=2))
    print(f"Saved Student-V0 lazy indexes to {datasets_dir}")


# ===================================================================
# Subcommand: precompute_student_v0_iwg_outputs
# ===================================================================


def _add_precompute_student_v0_iwg_outputs_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    p = subparsers.add_parser(
        "precompute_student_v0_iwg_outputs",
        help="Cache frozen IWG outputs for Student-V0 records.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument("--mode", default="all", help="Dataset mode used by build_student_v0_data.")
    p.add_argument("--device", default="cuda", help="Device for frozen IWG inference.")
    p.add_argument("--batch-size", type=int, default=1024, help="IWG inference batch size.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    p.add_argument("--max-records", type=int, default=0, help="Limit records for a smoke test (0=all).")
    p.add_argument("--dataset-dir", required=True, help="Student-V0 dataset index directory.")
    p.add_argument("--iwg-checkpoint", required=True, help="Frozen IWG checkpoint (.pt).")
    p.add_argument("--output-cache", required=True, help="Output NumPy cache (.npz).")


def _cmd_precompute_student_v0_iwg_outputs(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from agentguard.models.iwg import IWG
    from agentguard.training.checkpointing import load_checkpoint
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.normalization import NormalizationStats
    from agentguard.v0_pipeline import CompactV0IWGDataset, read_jsonl, student_v0_record_key

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    dataset_dir = os.path.abspath(args.dataset_dir)
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Student-V0 metadata not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    index_path = os.path.join(dataset_dir, metadata.get("train_index_file", "index.jsonl"))
    records = read_jsonl(index_path)
    if not records:
        raise RuntimeError(f"No records found in {index_path}")
    if args.max_records > 0:
        records = records[: int(args.max_records)]

    norm_stats = NormalizationStats.load(os.path.join(dataset_dir, "norm_stats.npz"))
    reid_dim = int(metadata["reid_dim"])
    feature_builder = EventFeatureBuilder(reid_dim, norm_stats)
    model = IWG(reid_dim=reid_dim).to(device)
    checkpoint_metadata = load_checkpoint(args.iwg_checkpoint, model)
    model.eval()

    dataset = CompactV0IWGDataset(
        records,
        metadata["event_cache_root"],
        metadata["detection_cache_root"],
        args.dataset,
        metadata["split"],
        feature_builder,
    )
    loader_kwargs = {
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.startswith("cuda"),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=CompactV0IWGDataset.collate_fn,
        **loader_kwargs,
    )

    policies = []
    gates = []
    print(
        f"Precomputing frozen IWG outputs: {len(records)} records, "
        f"device={device}, batch_size={max(1, int(args.batch_size))}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            track_t = batch["track_feats"].to(device, non_blocking=True)
            det_t = batch["det_feats"].to(device, non_blocking=True)
            scalar_t = batch["scalar_feats"].to(device, non_blocking=True)
            mask_t = batch["mask"].to(device, non_blocking=True)
            outputs = model(track_t, det_t, scalar_t, mask_t)
            policies.append(outputs["policy_probs"].cpu().numpy().astype(np.float32))
            gates.append(outputs["gate"].cpu().numpy().astype(np.float32))
            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
                print(
                    f"IWG cache batch {batch_idx}/{len(loader)} "
                    f"({min(batch_idx * loader.batch_size, len(records))}/{len(records)})",
                    flush=True,
                )

    output_cache = os.path.abspath(args.output_cache)
    Path(output_cache).parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray([student_v0_record_key(record) for record in records])
    policy_array = np.concatenate(policies, axis=0)
    gate_array = np.concatenate(gates, axis=0)
    if len(keys) != len(policy_array) or len(keys) != len(gate_array):
        raise RuntimeError("IWG output count does not match Student-V0 index count")
    if len(set(keys.tolist())) != len(keys):
        raise RuntimeError("Student-V0 record keys are not unique; refusing to write ambiguous cache")

    np.savez_compressed(
        output_cache,
        keys=keys,
        policy_probs=policy_array,
        gates=gate_array,
    )
    with open(Path(output_cache).with_suffix(".json"), "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "mode": args.mode,
                "dataset_dir": dataset_dir,
                "index_file": index_path,
                "num_records": len(records),
                "reid_dim": reid_dim,
                "iwg_checkpoint": os.path.abspath(args.iwg_checkpoint),
                "iwg_raw_epoch": checkpoint_metadata.get("epoch"),
                "device": device,
                "policy_shape": list(policy_array.shape),
                "gate_shape": list(gate_array.shape),
            },
            f,
            indent=2,
        )
    dataset.close()
    print(f"Saved IWG output cache to {output_cache}", flush=True)


# ===================================================================
# Subcommand: train_student_v0
# ===================================================================


def _add_train_student_v0_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train_student_v0",
        help="Train Student-V0 models (IWG first, then TGR).",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument("--mode", default="all", help="Dataset mode used by build_student_v0_data.")
    p.add_argument("--device", default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument(
        "--tgr-batch-size",
        type=int,
        default=0,
        help="TGR batch size. Defaults to --batch-size when 0.",
    )
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--tgr-lr", type=float, default=None, help="TGR learning rate (defaults to --lr).")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    p.add_argument("--seed", type=int, default=42, help="Random seed used before model initialisation.")
    p.add_argument(
        "--tgr-window-stride",
        type=int,
        default=1,
        help="Stride between TGR training windows. 1 uses sliding windows; 4 uses non-overlapping 4-frame windows.",
    )
    p.add_argument(
        "--val-max-samples",
        type=int,
        default=0,
        help="Deterministically cap validation records for fast train_all experiments (0=all).",
    )
    p.add_argument("--dataset-dir", default=None, help="Override Student-V0 dataset index directory.")
    p.add_argument("--checkpoint-dir", default=None, help="Override checkpoint output directory.")
    p.add_argument(
        "--full-val-every",
        type=int,
        default=0,
        help="Run full validation every N epochs in addition to sampled validation (0=disabled).",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Train for the fixed epoch count without val/best selection; use iwg_last.pt.",
    )
    p.add_argument(
        "--tgr-full-val-device",
        default="cpu",
        help="Device used for TGR full validation. CPU avoids long CUDA eval illegal-memory failures.",
    )
    p.add_argument(
        "--iwg-output-cache",
        default="",
        help=(
            "Optional .npz cache of frozen IWG outputs. When set, TGR training "
            "uses the cached IWG policy/gates instead of label-derived soft targets."
        ),
    )
    p.add_argument(
        "--skip-iwg-training",
        action="store_true",
        help="Skip IWG training and load an existing IWG checkpoint before TGR training.",
    )
    p.add_argument(
        "--skip-tgr-training",
        action="store_true",
        help="Skip TGR training after IWG training.",
    )
    p.add_argument(
        "--iwg-checkpoint",
        default="",
        help="IWG checkpoint to load when --skip-iwg-training is set. Defaults to checkpoint-dir/iwg/iwg_best.pt.",
    )
    p.add_argument(
        "--iwg-resume-checkpoint",
        default="",
        help="Resume IWG training from this checkpoint, typically checkpoint-dir/iwg/iwg_last.pt.",
    )
    p.add_argument(
        "--tgr-resume-checkpoint",
        default="",
        help="Resume TGR training from this checkpoint, typically checkpoint-dir/tgr/tgr_last.pt.",
    )


def _cmd_train_student_v0(args: argparse.Namespace) -> None:
    import subprocess

    import torch
    from torch.utils.data import DataLoader

    from agentguard.models.iwg import IWG
    from agentguard.models.tgr import TGR
    from agentguard.training.train_iwg import train_iwg
    from agentguard.training.train_tgr import train_tgr
    from agentguard.training.checkpointing import load_checkpoint
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.normalization import NormalizationStats
    from agentguard.v0_pipeline import (
        CompactV0IWGDataset,
        CompactV0TGRDataset,
        StudentV0IWGOutputCache,
        read_jsonl,
    )

    dataset = args.dataset
    mode = args.mode
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    _seed_training(args.seed)
    checkpoints_dir = _ensure_dir(args.checkpoint_dir or os.path.join(_checkpoints_dir(dataset), mode))
    datasets_dir = args.dataset_dir or os.path.join(_datasets_dir(dataset), mode)
    metadata_path = os.path.join(datasets_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"Student-V0 metadata not found at {metadata_path}. Run build_student_v0_data first.")
        return
    with open(metadata_path) as f:
        metadata = json.load(f)
    if int(metadata.get("cache_schema_version", 0)) != COMPACT_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Dataset metadata cache schema mismatch: "
            f"{metadata.get('cache_schema_version')} != {COMPACT_CACHE_SCHEMA_VERSION}"
        )
    if metadata.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError("Dataset metadata feature schema does not match current runtime")
    if int(metadata.get("label_schema_version", 0)) != ROLLOUT_LABEL_SCHEMA_VERSION:
        raise ValueError(
            "Dataset metadata label schema mismatch: "
            f"{metadata.get('label_schema_version')} != {ROLLOUT_LABEL_SCHEMA_VERSION}"
        )
    if metadata.get("label_schema_sha256") != ROLLOUT_LABEL_SCHEMA_SHA256:
        raise ValueError("Dataset metadata label schema hash does not match current runtime")
    try:
        training_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        training_commit = ""

    train_index_file = metadata.get("train_index_file", "train_index.jsonl")
    val_index_file = metadata.get("val_index_file", "val_index.jsonl")
    train_index_path = os.path.join(datasets_dir, train_index_file)
    val_index_path = os.path.join(datasets_dir, val_index_file)
    train_records = read_jsonl(train_index_path)
    if os.path.abspath(train_index_path) == os.path.abspath(val_index_path):
        full_val_records = train_records
    else:
        full_val_records = read_jsonl(val_index_path)
    val_records = full_val_records
    if not train_records:
        print("No Student-V0 training records.")
        return
    if not val_records:
        full_val_records = train_records[: min(len(train_records), max(1, args.batch_size))]
        val_records = list(full_val_records)
    if args.val_max_samples and args.val_max_samples > 0 and len(val_records) > args.val_max_samples:
        val_records = _stratified_record_sample(val_records, int(args.val_max_samples))

    norm_stats = NormalizationStats.load(os.path.join(datasets_dir, "norm_stats.npz"))
    reid_dim = int(metadata["reid_dim"])
    feature_builder = EventFeatureBuilder(reid_dim, norm_stats)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "reid_dim": reid_dim,
        "scalar_dim": 63,
        "event_dim": 128,
        "amp": False,
        "seed": int(args.seed),
        "tgr_window_stride": max(int(args.tgr_window_stride), 1),
        "early_stop_patience": max(2, args.epochs + 1),
        "full_val_every": max(0, int(args.full_val_every)),
        "skip_validation": bool(args.skip_validation),
        "resume_from": args.iwg_resume_checkpoint,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "model_schema_version": "agentguard_iwg_v3",
        "training_commit": training_commit,
        "dataset_metadata_path": os.path.abspath(metadata_path),
        "dataset_metadata_sha256": _sha256_file(metadata_path),
    }
    iwg_model = IWG(reid_dim=reid_dim).to(device)
    loader_kwargs = {
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.startswith("cuda"),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = False

    iwg_output_dir = _ensure_dir(os.path.join(checkpoints_dir, "iwg"))
    iwg_train_ds = None
    iwg_val_ds = None
    iwg_full_val_ds = None

    if args.skip_iwg_training:
        best_iwg_path = args.iwg_checkpoint or os.path.join(iwg_output_dir, "iwg_best.pt")
        load_checkpoint(best_iwg_path, iwg_model)
        iwg_model.to(device)
        iwg_summary = {"skipped": True, "checkpoint": best_iwg_path}
        print(f"Skipped IWG training; loaded IWG from {best_iwg_path}")
    else:
        print("=" * 60)
        print("Phase 1: Training IWG")
        print("=" * 60)

        iwg_train_ds = CompactV0IWGDataset(
            train_records,
            metadata["event_cache_root"],
            metadata["detection_cache_root"],
            dataset,
            metadata["split"],
            feature_builder,
        )
        if not args.skip_validation:
            iwg_val_ds = CompactV0IWGDataset(
                val_records,
                metadata["event_cache_root"],
                metadata["detection_cache_root"],
                dataset,
                metadata["split"],
                feature_builder,
            )
        if not args.skip_validation and args.full_val_every and len(full_val_records) > len(val_records):
            iwg_full_val_ds = CompactV0IWGDataset(
                full_val_records,
                metadata["event_cache_root"],
                metadata["detection_cache_root"],
                dataset,
                metadata["split"],
                feature_builder,
            )

        iwg_train_loader = DataLoader(
            iwg_train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=CompactV0IWGDataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            **loader_kwargs,
        )
        iwg_val_loader = None
        if iwg_val_ds is not None:
            iwg_val_loader = DataLoader(
                iwg_val_ds,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=CompactV0IWGDataset.collate_fn,
                **loader_kwargs,
            )
        iwg_full_val_loader = None
        if iwg_full_val_ds is not None:
            iwg_full_val_loader = DataLoader(
                iwg_full_val_ds,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=CompactV0IWGDataset.collate_fn,
                **loader_kwargs,
            )

        iwg_summary = train_iwg(
            iwg_model,
            iwg_train_loader,
            iwg_val_loader,
            config,
            iwg_output_dir,
            norm_mean=norm_stats.mean,
            norm_std=norm_stats.std,
            full_val_loader=iwg_full_val_loader,
        )
        print(f"IWG training complete. Best epoch: {iwg_summary.get('best_epoch')}")

    if args.skip_tgr_training:
        if iwg_train_ds is not None:
            iwg_train_ds.close()
        if iwg_val_ds is not None:
            iwg_val_ds.close()
        if iwg_full_val_ds is not None:
            iwg_full_val_ds.close()
        tgr_summary = {"skipped": True, "reason": "skip_tgr_training"}
        summary = {
            "iwg": iwg_summary,
            "tgr": tgr_summary,
        }
        with open(os.path.join(checkpoints_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("Skipped TGR training.")
        return

    # ---- Phase 2: Train TGR ----
    print("\n" + "=" * 60)
    print("Phase 2: Training TGR")
    print("=" * 60)

    tgr_model = TGR(reid_dim=reid_dim).to(device)

    iwg_output_cache = None
    if args.iwg_output_cache:
        iwg_output_cache = StudentV0IWGOutputCache(args.iwg_output_cache)
        if len(iwg_output_cache.keys) == 0:
            raise RuntimeError(f"IWG output cache is empty: {args.iwg_output_cache}")
        print(
            f"Loaded frozen IWG output cache: {args.iwg_output_cache} "
            f"({len(iwg_output_cache.keys)} records)",
            flush=True,
        )

    # Load best IWG checkpoint
    best_iwg_path = args.iwg_checkpoint or os.path.join(iwg_output_dir, "iwg_best.pt")
    if os.path.isfile(best_iwg_path):
        load_checkpoint(best_iwg_path, iwg_model)
        iwg_model.to(device)
        print(f"Loaded best IWG from {best_iwg_path}")
    else:
        print("No best IWG checkpoint found, using freshly initialised model.")

    tgr_train_records = [r for r in train_records if str(r.get("candidate_type", "A")).upper() == "A"]
    tgr_val_records = [r for r in val_records if str(r.get("candidate_type", "A")).upper() == "A"] or val_records
    tgr_full_val_records = [r for r in full_val_records if str(r.get("candidate_type", "A")).upper() == "A"]
    tgr_train_ds = CompactV0TGRDataset(
        tgr_train_records,
        metadata["event_cache_root"],
        metadata["detection_cache_root"],
        dataset,
        metadata["split"],
        feature_builder,
        iwg_output_cache=iwg_output_cache,
        window_stride=max(int(args.tgr_window_stride), 1),
    )
    tgr_val_ds = CompactV0TGRDataset(
        tgr_val_records,
        metadata["event_cache_root"],
        metadata["detection_cache_root"],
        dataset,
        metadata["split"],
        feature_builder,
        iwg_output_cache=iwg_output_cache,
        window_stride=max(int(args.tgr_window_stride), 1),
    )
    tgr_full_val_ds = None
    if args.full_val_every and len(tgr_full_val_records) > len(tgr_val_records):
        tgr_full_val_ds = CompactV0TGRDataset(
            tgr_full_val_records,
            metadata["event_cache_root"],
            metadata["detection_cache_root"],
            dataset,
            metadata["split"],
            feature_builder,
            iwg_output_cache=iwg_output_cache,
            window_stride=max(int(args.tgr_window_stride), 1),
        )
    if len(tgr_train_ds) == 0:
        print("No TGR windows available; skipping TGR training.")
        tgr_summary = {"skipped": True, "reason": "no_windows"}
    else:
        tgr_batch_size = max(1, int(args.tgr_batch_size) if int(args.tgr_batch_size) > 0 else config["batch_size"])
        print(f"TGR batch size: {tgr_batch_size}")
        tgr_train_loader = DataLoader(
            tgr_train_ds,
            batch_size=tgr_batch_size,
            shuffle=True,
            collate_fn=CompactV0TGRDataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed) + 1),
            **loader_kwargs,
        )
        tgr_val_loader = DataLoader(
            tgr_val_ds if len(tgr_val_ds) else tgr_train_ds,
            batch_size=tgr_batch_size,
            shuffle=False,
            collate_fn=CompactV0TGRDataset.collate_fn,
            **loader_kwargs,
        )
        tgr_full_val_loader = None
        if tgr_full_val_ds is not None and len(tgr_full_val_ds):
            tgr_full_val_loader = DataLoader(
                tgr_full_val_ds,
                batch_size=tgr_batch_size,
                shuffle=False,
                collate_fn=CompactV0TGRDataset.collate_fn,
                **loader_kwargs,
            )

        tgr_config = {
            **config,
            "batch_size": tgr_batch_size,
            "learning_rate": args.tgr_lr if args.tgr_lr is not None else args.lr,
            "full_val_device": args.tgr_full_val_device,
            "resume_from": args.tgr_resume_checkpoint,
        }
        tgr_output_dir = _ensure_dir(os.path.join(checkpoints_dir, "tgr"))
        tgr_summary = train_tgr(
            tgr_model,
            iwg_model,
            tgr_train_loader,
            tgr_val_loader,
            tgr_config,
            tgr_output_dir,
            norm_mean=norm_stats.mean,
            norm_std=norm_stats.std,
            full_val_loader=tgr_full_val_loader,
        )
        print(f"TGR training complete. Best epoch: {tgr_summary.get('best_epoch')}")
    if iwg_train_ds is not None:
        iwg_train_ds.close()
    if iwg_val_ds is not None:
        iwg_val_ds.close()
    if iwg_full_val_ds is not None:
        iwg_full_val_ds.close()
    tgr_train_ds.close()
    tgr_val_ds.close()
    if tgr_full_val_ds is not None:
        tgr_full_val_ds.close()

    # Save combined summary
    summary = {
        "iwg": iwg_summary,
        "tgr": tgr_summary,
    }
    with open(os.path.join(checkpoints_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Training summary saved to {os.path.join(checkpoints_dir, 'training_summary.json')}")


# ===================================================================
# Subcommand: select_teacher_events
# ===================================================================


def _add_select_teacher_events_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "select_teacher_events",
        help="Select events for Teacher annotation using Student-V0 predictions.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        help="Label mode to load.",
    )
    p.add_argument("--max-events", type=int, default=10000, help="Max events to select.")


def _cmd_select_teacher_events(args: argparse.Namespace) -> None:
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.teacher.event_selector import TeacherEventSelector

    dataset = args.dataset
    mode = args.mode
    cache_root = _cache_dir(dataset, mode)
    label_dir = os.path.join(_labels_dir(dataset), mode)
    teacher_dir = _ensure_dir(_teacher_dir(dataset))

    # Load all events and labels
    all_events: List[TrackEvent] = []
    all_labels: List[Dict[str, Any]] = []

    sequences = sorted(
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
    )

    for seq in sequences:
        seq_dir = os.path.join(cache_root, seq)
        try:
            reader = EventCacheReader(seq_dir)
            events = reader.read_events()
        except Exception:
            continue

        seq_label_path = os.path.join(label_dir, f"{seq}_labels.json")
        if not os.path.isfile(seq_label_path):
            continue
        with open(seq_label_path) as f:
            seq_labels = json.load(f)

        # Build label lookup by (event_id, candidate_type)
        label_by_key = {}
        for lbl in seq_labels:
            key = (lbl.get("event_id", ""), lbl.get("candidate_type", "A"))
            label_by_key[key] = lbl
        for evt in events:
            key = (evt.event_id, getattr(evt, 'candidate_type', 'A'))
            if key not in label_by_key:
                raise ValueError(f"Missing label for event {key}")
            all_events.append(evt)
            all_labels.append(label_by_key[key])

    print(f"Loaded {len(all_events)} events with labels.")

    # Select events
    selector = TeacherEventSelector()
    selected = selector.select(
        all_events, all_labels, student_preds=None, max_events=args.max_events
    )

    print(f"Selected {len(selected)} events for Teacher annotation.")

    # Save selected event indices
    selected_data = []
    for priority, evt, reason in selected:
        selected_data.append(
            {
                "event_id": evt.event_id,
                "sequence": evt.sequence,
                "frame_id": evt.frame_id,
                "track_id": evt.track_id,
                "priority": priority,
                "reason": reason,
            }
        )

    out_path = os.path.join(teacher_dir, "selected_events.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "mode": mode,
                "num_selected": len(selected),
                "max_events": args.max_events,
                "events": selected_data,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved selected events to {out_path}")


# ===================================================================
# Subcommand: build_evidence_packets
# ===================================================================


def _add_build_evidence_packets_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build_evidence_packets",
        help="Build evidence packets (event.json, contact_sheet stubs, prompt.json) "
        "for Teacher annotation.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        help="Mode used to load cached events and labels.",
    )
    p.add_argument("--image-dir", default=None, help="Root image directory for sequences.")


def _cmd_build_evidence_packets(args: argparse.Namespace) -> None:
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.data.gt_reader import GTReader
    from agentguard.teacher.evidence_packet import EvidencePacketBuilder

    dataset = args.dataset
    mode = args.mode
    cache_root = _cache_dir(dataset, mode)
    teacher_dir = _teacher_dir(dataset)
    evidence_dir = _ensure_dir(_evidence_dir(dataset))

    image_dir = args.image_dir or os.path.join(
        os.environ.get("TRACKTRACK_DATA_DIR", "/home/shang/datasets/"),
        dataset,
        "train",
    )

    # Load selected events
    selected_path = os.path.join(teacher_dir, "selected_events.json")
    if not os.path.isfile(selected_path):
        print(f"No selected events found at {selected_path}. Run select_teacher_events first.")
        return

    with open(selected_path) as f:
        selected_data = json.load(f)

    selected_event_ids = {e["event_id"] for e in selected_data.get("events", [])}
    if not selected_event_ids:
        print("No events to process.")
        return

    # Build identity prototypes lookup
    seq_prototypes: Dict[str, Dict[int, np.ndarray]] = {}
    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            seq_prototypes[seq] = reader.read_identity_prototypes()
        except Exception:
            seq_prototypes[seq] = {}

    # Build packets
    packet_builder = EvidencePacketBuilder(
        output_dir=evidence_dir,
        image_dir=image_dir,
    )

    packets_created = 0
    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        seq_dir = os.path.join(cache_root, seq)
        try:
            reader = EventCacheReader(seq_dir)
            events = reader.read_events()
        except Exception:
            continue

        # Load GT reader
        try:
            data_dir = os.path.join(
                os.environ.get("TRACKTRACK_DATA_DIR", "/home/shang/datasets/"),
                dataset,
                "train",
            )
            gt_reader = GTReader(data_dir, seq)
        except Exception:
            gt_reader = None

        protos = seq_prototypes.get(seq, {})

        for evt in events:
            if evt.event_id not in selected_event_ids:
                continue

            try:
                packet_builder.build_packet(evt, gt_reader, protos)
                packets_created += 1
            except Exception as e:
                print(f"  FAIL packet for {evt.event_id}: {e}")

    print(f"Created {packets_created} evidence packets in {evidence_dir}")


# ===================================================================
# Subcommand: run_bailian_teacher
# ===================================================================


def _add_run_bailian_teacher_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run_bailian_teacher",
        help="Run the Bailian (Alibaba Cloud) Teacher LLM on evidence packets.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        help="Mode (for loading selected events config).",
    )


def _cmd_run_bailian_teacher(args: argparse.Namespace) -> None:
    from agentguard.teacher.runner import TeacherRunner
    from agentguard.teacher.event_selector import TeacherEventSelector
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader

    dataset = args.dataset
    mode = args.mode
    teacher_dir = _teacher_dir(dataset)
    evidence_dir = _evidence_dir(dataset)

    # Load selected events
    selected_path = os.path.join(teacher_dir, "selected_events.json")
    if not os.path.isfile(selected_path):
        print(f"No selected events found. Run select_teacher_events first.")
        return

    with open(selected_path) as f:
        selected_data = json.load(f)

    if not selected_data.get("events"):
        print("No events to process.")
        return

    # Reconstruct selected event list
    cache_root = _cache_dir(dataset, mode)
    event_lookup: Dict[str, TrackEvent] = {}

    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            for evt in reader.read_events():
                event_lookup[evt.event_id] = evt
        except Exception:
            continue

    selected_events: List[Tuple[int, TrackEvent, str]] = []
    for entry in selected_data["events"]:
        eid = entry["event_id"]
        if eid in event_lookup:
            selected_events.append(
                (entry.get("priority", 99), event_lookup[eid], entry.get("reason", ""))
            )

    print(f"Loaded {len(selected_events)} events for teacher processing.")

    if not selected_events:
        print("No events to process.")
        return

    # Run teacher
    runner = TeacherRunner()
    evidence_data_dir = os.path.join(
        os.environ.get("TRACKTRACK_DATA_DIR", "/home/shang/datasets/"),
        dataset,
    )
    summary = runner.run(selected_events, evidence_data_dir, teacher_dir)

    # Save summary
    summary_path = os.path.join(teacher_dir, "teacher_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Teacher run complete: {summary.get('successful', 0)} successful, "
          f"{summary.get('failed', 0)} failed, "
          f"{summary.get('cached', 0)} cached "
          f"out of {summary.get('total_events', 0)} total.")


# ===================================================================
# Subcommand: verify_and_fuse
# ===================================================================


def _add_verify_and_fuse_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "verify_and_fuse",
        help="Verify teacher responses via local replay, compute cross-event scores, "
        "and fuse teacher + verifier distributions into final labels.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        help="Mode for loading cached data.",
    )


def _cmd_verify_and_fuse(args: argparse.Namespace) -> None:
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.motion.nsa_numpy import NSAKalmanFilter
    from agentguard.verifier.local_replay import LocalReplayVerifier
    from agentguard.verifier.scoring import (
        fuse_distributions,
        compute_agent_gate,
    )
    from agentguard.verifier.label_fusion import fuse_labels
    from agentguard.teacher.response_schema import TeacherResponse

    dataset = args.dataset
    mode = args.mode
    teacher_dir = _teacher_dir(dataset)
    cache_root = _cache_dir(dataset, mode)

    # Load teacher responses
    responses_path = os.path.join(teacher_dir, "teacher_results", "teacher_responses.json")
    if not os.path.isfile(responses_path):
        responses_path = os.path.join(teacher_dir, "teacher_responses.json")

    if not os.path.isfile(responses_path):
        print(f"No teacher responses found. Run run_bailian_teacher first.")
        return

    with open(responses_path) as f:
        responses = json.load(f)

    if not responses:
        print("No responses to process.")
        return

    # Build identity prototypes lookup
    seq_prototypes: Dict[str, Dict[int, np.ndarray]] = {}
    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            seq_prototypes[seq] = reader.read_identity_prototypes()
        except Exception:
            seq_prototypes[seq] = {}

    # Load all events
    all_events: List[TrackEvent] = []
    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            all_events.extend(reader.read_events())
        except Exception:
            continue

    event_map: Dict[str, TrackEvent] = {evt.event_id: evt for evt in all_events}

    # Create verifier
    motion_model = NSAKalmanFilter()
    verifier = LocalReplayVerifier(
        motion_model=motion_model,
        identity_prototypes={},
    )

    # Process each teacher response
    fused_labels: List[Dict[str, Any]] = []
    for resp_entry in responses:
        event_id = resp_entry.get("event_id", "")
        if event_id not in event_map:
            continue

        event = event_map[event_id]
        proto = seq_prototypes.get(event.sequence, {}).get((event.sequence, event.track_id))

        # Build verifier with this event's prototypes
        verifier.identity_prototypes = {event.track_id: proto} if proto is not None else {}

        # Parse teacher response
        parsed = resp_entry.get("response")
        teacher_response = None
        if isinstance(parsed, dict):
            try:
                teacher_response = TeacherResponse(**parsed)
            except Exception:
                pass

        # Compute rollout gate from event metadata
        rollout_gate = np.array(
            [
                getattr(event, "motion_rollout_target", 0.5) or 0.5,
                getattr(event, "appearance_rollout_target", 0.5) or 0.5,
            ],
            dtype=np.float64,
        )

        # Default fused gate = rollout gate
        fused_gate = rollout_gate.copy()

        if teacher_response is not None:
            # Build teacher distribution from policy_prior
            pp = teacher_response.policy_prior
            teacher_dist = np.array(
                [pp.FULL_WRITE, pp.MOTION_ONLY, pp.APPEARANCE_ONLY, pp.HOLD_BOTH, pp.SOFT_CAUTION],
                dtype=np.float64,
            )

            # Verifier distribution (simplified: use uniform)
            verifier_dist = np.ones(5, dtype=np.float64) / 5.0

            # Fuse distributions
            agent_dist = fuse_distributions(teacher_dist, verifier_dist)
            agent_gate = compute_agent_gate(agent_dist)

            # Fuse with rollout labels
            rollout_reliability = 0.5  # default
            teacher_confidence = teacher_response.confidence if hasattr(teacher_response, 'confidence') else 0.5
            teacher_abstained = getattr(teacher_response, 'abstain', False)

            fused_gate = fuse_labels(
                rollout_gate,
                agent_gate,
                rollout_reliability,
                teacher_confidence,
                teacher_abstained,
            )

        fused_labels.append(
            {
                "event_id": event_id,
                "sequence": event.sequence,
                "frame_id": event.frame_id,
                "track_id": event.track_id,
                "target_gate": fused_gate.tolist(),
                "rollout_gate": rollout_gate.tolist(),
                "motion_target": float(fused_gate[0]),
                "appearance_target": float(fused_gate[1]),
                "valid_motion": True,
                "valid_appearance": True,
                "sample_weight": 1.0,
            }
        )

    # Save fused labels
    label_out_dir = _ensure_dir(os.path.join(teacher_dir, "fused_labels"))
    seq_groups: Dict[str, List[Dict[str, Any]]] = {}
    for lbl in fused_labels:
        seq_groups.setdefault(lbl["sequence"], []).append(lbl)

    for seq, seq_lbls in seq_groups.items():
        out_path = os.path.join(label_out_dir, f"{seq}_fused_labels.json")
        with open(out_path, "w") as f:
            json.dump(seq_lbls, f, indent=2, default=str)

    print(f"Fused {len(fused_labels)} labels across {len(seq_groups)} sequences.")
    print(f"Saved to {label_out_dir}")


# ===================================================================
# Subcommand: build_student_v1_data
# ===================================================================


def _add_build_student_v1_data_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build_student_v1_data",
        help="Build Student-V1 datasets using fused teacher+verifier labels.",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument(
        "--mode",
        default="train_custom",
        help="Mode for loading cached events.",
    )


def _cmd_build_student_v1_data(args: argparse.Namespace) -> None:
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.normalization import NormalizationStats
    from agentguard.datasets.iwg_dataset import IWGDataset
    from agentguard.datasets.tgr_dataset import TGRDataset

    dataset = args.dataset
    mode = args.mode
    cache_root = _cache_dir(dataset, mode)
    teacher_dir = _teacher_dir(dataset)
    fused_dir = os.path.join(teacher_dir, "fused_labels")
    datasets_dir = _ensure_dir(os.path.join(_datasets_dir(dataset), "v1"))

    if not os.path.isdir(fused_dir):
        print(f"Fused labels not found at {fused_dir}. Run verify_and_fuse first.")
        return

    # Load fused labels
    fused_labels: List[Dict[str, Any]] = []
    for fname in os.listdir(fused_dir):
        if fname.endswith("_fused_labels.json"):
            with open(os.path.join(fused_dir, fname)) as f:
                fused_labels.extend(json.load(f))

    if not fused_labels:
        print("No fused labels found.")
        return

    # Load events matching fused labels
    fused_event_ids = {lbl["event_id"] for lbl in fused_labels}
    all_events: List[TrackEvent] = []

    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            for evt in reader.read_events():
                if evt.event_id in fused_event_ids:
                    all_events.append(evt)
        except Exception:
            continue

    if not all_events:
        print("No matching events found.")
        return

    # Align events with fused labels
    event_map = {evt.event_id: evt for evt in all_events}
    aligned_events: List[TrackEvent] = []
    aligned_labels: List[Dict[str, Any]] = []
    for lbl in fused_labels:
        eid = lbl["event_id"]
        if eid in event_map:
            aligned_events.append(event_map[eid])
            aligned_labels.append(lbl)

    print(f"Aligned {len(aligned_events)} events with fused labels.")

    # Build feature builder
    reid_dim = 2048
    norm_stats = NormalizationStats()
    scalar_list = [evt.scalar_features for evt in aligned_events if evt.scalar_features is not None]
    if scalar_list:
        norm_stats.fit(scalar_list)
    feature_builder = EventFeatureBuilder(reid_dim, norm_stats)

    # Split into train/val
    seq_names = list({evt.sequence for evt in aligned_events})
    np.random.shuffle(seq_names)
    split_idx = max(1, int(len(seq_names) * 0.8))
    train_seqs = set(seq_names[:split_idx])
    val_seqs = set(seq_names[split_idx:])

    train_evts = [evt for evt in aligned_events if evt.sequence in train_seqs]
    val_evts = [evt for evt in aligned_events if evt.sequence in val_seqs]
    train_lbls = [lbl for evt, lbl in zip(aligned_events, aligned_labels) if evt.sequence in train_seqs]
    val_lbls = [lbl for evt, lbl in zip(aligned_events, aligned_labels) if evt.sequence in val_seqs]

    # Build datasets
    print("Building V1 IWG datasets...")
    train_iwg = IWGDataset(train_evts, train_lbls, feature_builder)
    val_iwg = IWGDataset(val_evts, val_lbls, feature_builder)

    print("Building V1 TGR datasets...")
    train_tgr = TGRDataset(train_evts, train_lbls, feature_builder)
    val_tgr = TGRDataset(val_evts, val_lbls, feature_builder)

    # Save
    import torch

    def _save(ds: Any, subpath: str, name: str) -> None:
        out = _ensure_dir(os.path.join(datasets_dir, subpath, name))
        meta = {
            "type": ds.__class__.__name__,
            "size": len(ds),
            "window_size": getattr(ds, "window_size", None),
            "max_history": getattr(ds, "max_history", None),
        }
        with open(os.path.join(out, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved {name} ({len(ds)} samples)")

    _save(train_iwg, "iwg", "train")
    _save(val_iwg, "iwg", "val")
    _save(train_tgr, "tgr", "train")
    _save(val_tgr, "tgr", "val")

    stats_path = os.path.join(datasets_dir, "norm_stats.pt")
    torch.save({"mean": norm_stats.mean, "std": norm_stats.std}, stats_path)

    split_cfg = {
        "train_sequences": sorted(train_seqs),
        "val_sequences": sorted(val_seqs),
    }
    with open(os.path.join(datasets_dir, "split.json"), "w") as f:
        json.dump(split_cfg, f, indent=2)

    print(f"V1 IWG: {len(train_iwg)} train / {len(val_iwg)} val")
    print(f"V1 TGR: {len(train_tgr)} train / {len(val_tgr)} val")
    print(f"Saved to {datasets_dir}")


# ===================================================================
# Subcommand: train_student_v1
# ===================================================================


def _add_train_student_v1_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train_student_v1",
        help="Train Student-V1 models starting from V0 checkpoint with V1 loss "
        "(gate + policy + event + cue + baseline).",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument("--device", default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")


def _cmd_train_student_v1(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from agentguard.models.iwg import IWG
    from agentguard.models.tgr import TGR
    from agentguard.training.train_iwg import train_iwg
    from agentguard.training.train_tgr import train_tgr
    from agentguard.training.checkpointing import load_checkpoint

    dataset = args.dataset
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    v0_checkpoints_dir = _checkpoints_dir(dataset)
    v1_datasets_dir = os.path.join(_datasets_dir(dataset), "v1")
    v1_checkpoints_dir = _ensure_dir(os.path.join(_checkpoints_dir(dataset), "v1"))

    if not os.path.isdir(v1_datasets_dir):
        print(f"V1 datasets not found at {v1_datasets_dir}. Run build_student_v1_data first.")
        return

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "reid_dim": 2048,
        "scalar_dim": 63,
        "event_dim": 128,
    }

    # ---- Phase 1: Fine-tune IWG with V1 loss ----
    print("=" * 60)
    print("Phase 1: Fine-tuning IWG (V1 loss)")
    print("=" * 60)

    iwg_model = IWG(reid_dim=2048).to(device)

    # Load V0 checkpoint
    best_v0_iwg = os.path.join(v0_checkpoints_dir, "iwg", "iwg_best.pt")
    if os.path.isfile(best_v0_iwg):
        load_checkpoint(best_v0_iwg, iwg_model)
        print(f"Loaded V0 IWG from {best_v0_iwg}")
    else:
        print("No V0 IWG checkpoint found; starting from scratch.")

    # Build datasets from V1 data
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.normalization import NormalizationStats

    with open(os.path.join(v1_datasets_dir, "split.json")) as f:
        split_cfg = json.load(f)
    train_seqs = set(split_cfg.get("train_sequences", []))
    val_seqs = set(split_cfg.get("val_sequences", []))

    # Load V1 fused labels
    teacher_dir = _teacher_dir(dataset)
    fused_dir = os.path.join(teacher_dir, "fused_labels")
    fused_labels: List[Dict[str, Any]] = []
    for fname in os.listdir(fused_dir):
        if fname.endswith("_fused_labels.json"):
            with open(os.path.join(fused_dir, fname)) as f:
                fused_labels.extend(json.load(f))

    fused_event_ids = {lbl["event_id"] for lbl in fused_labels}

    # Load events, filter by split
    cache_root = _cache_dir(dataset, "train_custom")
    all_events: List[TrackEvent] = []
    for seq in sorted(
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ):
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            for evt in reader.read_events():
                if evt.event_id in fused_event_ids:
                    all_events.append(evt)
        except Exception:
            continue

    event_map = {evt.event_id: evt for evt in all_events}
    aligned_events: List[TrackEvent] = []
    aligned_labels: List[Dict[str, Any]] = []
    for lbl in fused_labels:
        eid = lbl["event_id"]
        if eid in event_map:
            aligned_events.append(event_map[eid])
            aligned_labels.append(lbl)

    if not aligned_events:
        print("No aligned V1 data.")
        return

    norm_stats = NormalizationStats()
    scalar_list = [evt.scalar_features for evt in aligned_events if evt.scalar_features is not None]
    if scalar_list:
        norm_stats.fit(scalar_list)
    feature_builder = EventFeatureBuilder(2048, norm_stats)

    train_evts = [evt for evt in aligned_events if evt.sequence in train_seqs]
    val_evts = [evt for evt in aligned_events if evt.sequence in val_seqs]
    train_lbls = [lbl for evt, lbl in zip(aligned_events, aligned_labels) if evt.sequence in train_seqs]
    val_lbls = [lbl for evt, lbl in zip(aligned_events, aligned_labels) if evt.sequence in val_seqs]

    from agentguard.datasets.iwg_dataset import IWGDataset
    from agentguard.datasets.tgr_dataset import TGRDataset

    iwg_train_ds = IWGDataset(train_evts, train_lbls, feature_builder)
    iwg_val_ds = IWGDataset(val_evts, val_lbls, feature_builder)

    iwg_train_loader = DataLoader(
        iwg_train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=IWGDataset.collate_fn,
        num_workers=0,
    )
    iwg_val_loader = DataLoader(
        iwg_val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=IWGDataset.collate_fn,
        num_workers=0,
    )

    iwg_output_dir = _ensure_dir(os.path.join(v1_checkpoints_dir, "iwg"))
    iwg_summary = train_iwg(iwg_model, iwg_train_loader, iwg_val_loader, config, iwg_output_dir)
    print(f"IWG V1 fine-tuning complete. Best epoch: {iwg_summary.get('best_epoch')}")

    # ---- Phase 2: Fine-tune TGR ----
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning TGR (V1 loss)")
    print("=" * 60)

    tgr_model = TGR(reid_dim=2048).to(device)

    best_v0_tgr = os.path.join(v0_checkpoints_dir, "tgr", "tgr_best.pt")
    if os.path.isfile(best_v0_tgr):
        load_checkpoint(best_v0_tgr, tgr_model)
        print(f"Loaded V0 TGR from {best_v0_tgr}")

    tgr_train_ds = TGRDataset(train_evts, train_lbls, feature_builder)
    tgr_val_ds = TGRDataset(val_evts, val_lbls, feature_builder)

    tgr_train_loader = DataLoader(
        tgr_train_ds,
        batch_size=config["batch_size"] // 2,
        shuffle=True,
        collate_fn=TGRDataset.collate_fn,
        num_workers=0,
    )
    tgr_val_loader = DataLoader(
        tgr_val_ds,
        batch_size=config["batch_size"] // 2,
        shuffle=False,
        collate_fn=TGRDataset.collate_fn,
        num_workers=0,
    )

    tgr_config = {**config, "batch_size": config["batch_size"] // 2, "learning_rate": 1e-4}
    tgr_output_dir = _ensure_dir(os.path.join(v1_checkpoints_dir, "tgr"))
    tgr_summary = train_tgr(tgr_model, iwg_model, tgr_train_loader, tgr_val_loader, tgr_config, tgr_output_dir)
    print(f"TGR V1 fine-tuning complete. Best epoch: {tgr_summary.get('best_epoch')}")

    summary = {"iwg": iwg_summary, "tgr": tgr_summary}
    with open(os.path.join(v1_checkpoints_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"V1 training summary saved to {os.path.join(v1_checkpoints_dir, 'training_summary.json')}")


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
    parser.add_argument("--detection-cache-root", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
        max_frame_gap=args.max_frame_gap,
        context_size=args.context_size,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _add_train_iwg_rg_cma_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train_iwg_rg_cma",
        help="Train fixed safe-direct IWG+RG-CMA for 100 or 200 epochs.",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--correction-bound",
        type=float,
        choices=[0.05, 0.10],
        default=0.10,
        help="RG-CMA correction bound; 0.05 preserves the legacy contract.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        choices=[6, 8],
        default=6,
        help="Number of events passed to IWG/RG-CMA (6 preserves old models).",
    )
    parser.add_argument(
        "--reliability-mode",
        choices=["full", "no-scalar"],
        default="full",
        help="Reliability token input mode; no-scalar masks normalized scalar features.",
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
def _cmd_train_iwg_rg_cma(args: argparse.Namespace) -> None:
    from agentguard.training.train_iwg_rg_cma import train_iwg_rg_cma

    config = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "device": args.device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "correction_bound": float(args.correction_bound),
        "context_size": int(args.context_size),
        "reliability_mode": str(args.reliability_mode),
        "residual_beta": float(args.residual_beta),
        "residual_weight": float(args.residual_weight),
        "revision_weight": float(args.revision_weight),
        "hard_example_gain": float(args.hard_example_gain),
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

    # Register all subcommands
    _add_cache_events_parser(subparsers)
    _add_validate_cache_parser(subparsers)
    _add_rollout_smoke_parser(subparsers)
    _add_build_rollout_labels_parser(subparsers)
    _add_evaluate_oracle_parser(subparsers)
    _add_build_student_v0_data_parser(subparsers)
    _add_precompute_student_v0_iwg_outputs_parser(subparsers)
    _add_train_student_v0_parser(subparsers)
    _add_select_teacher_events_parser(subparsers)
    _add_build_evidence_packets_parser(subparsers)
    _add_run_bailian_teacher_parser(subparsers)
    _add_verify_and_fuse_parser(subparsers)
    _add_build_student_v1_data_parser(subparsers)
    _add_train_student_v1_parser(subparsers)
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
        "evaluate_oracle": _cmd_evaluate_oracle,
        "build_student_v0_data": _cmd_build_student_v0_data,
        "precompute_student_v0_iwg_outputs": _cmd_precompute_student_v0_iwg_outputs,
        "train_student_v0": _cmd_train_student_v0,
        "select_teacher_events": _cmd_select_teacher_events,
        "build_evidence_packets": _cmd_build_evidence_packets,
        "run_bailian_teacher": _cmd_run_bailian_teacher,
        "verify_and_fuse": _cmd_verify_and_fuse,
        "build_student_v1_data": _cmd_build_student_v1_data,
        "train_student_v1": _cmd_train_student_v1,
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
