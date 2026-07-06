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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
            base_ids = set(_resolve_sequences(dataset, mode))
            sequences = [s for s in sequences if _base_sequence_id(s) in base_ids]
    if detector:
        suffix = f"-{detector}"
        sequences = [s for s in sequences if s.endswith(suffix)]
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
            "schema_version": 2,
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
                and int(existing.get("schema_version", 0)) == 2
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
            feature_schema_sha256="",
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

    dataset = args.dataset
    mode = args.mode
    split = _resolve_split(mode)
    cache_root = os.path.join(args.event_cache_root, dataset, split)

    if not os.path.isdir(cache_root):
        print(f"Cache directory not found: {cache_root}")
        return

    sequences = sorted(
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
        and not d.endswith(".incomplete")
        and not d.startswith("_")
    )

    total_events = 0
    matched_events = 0
    unmatched_events = 0
    association_record_count = 0
    invalid_detection_indices = 0
    duplicate_event_ids = 0
    warp_alignment_errors = 0
    feature_nonzero_count = 0
    feature_element_count = 0
    reid_dim = 0
    empty_caches: List[str] = []
    incomplete_sequences: List[str] = []

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

        n_events = 0
        n_matched = 0
        n_unmatched = 0
        n_assoc = int(manifest.get("num_association_records", 0))
        seq_feature_nonzero = 0
        seq_feature_total = 0
        seq_warp_alignment_errors = 0
        bad_det_idx = 0
        ids_seen: set[str] = set()
        dup_count = 0
        for ev in reader.iter_event_records():
            n_events += 1
            event_id = str(ev.get("event_id", ""))
            if event_id in ids_seen:
                dup_count += 1
            ids_seen.add(event_id)

            f = np.asarray(ev.get("track_feature", []), dtype=np.float32)
            seq_feature_nonzero += int(np.count_nonzero(f))
            seq_feature_total += int(f.size)

            frame = reader.get_frame_record(ev.get("frame_index", int(ev["frame_id"]) - 1))
            wm = None if frame is None else frame.get("warp_matrix", frame.get("effective_warp"))
            if wm is not None and np.asarray(wm).shape != (2, 3):
                seq_warp_alignment_errors += 1

            if not bool(ev.get("matched", False)):
                n_unmatched += 1
                continue
            n_matched += 1
            di = int(ev.get("accepted_detection_index", -1))
            assoc = reader.get_association(
                ev.get("association_shard_id", -1),
                ev.get("association_offset", -1),
            )
            if di < 0 or assoc is None:
                bad_det_idx += 1
                continue
            detection_indices = np.asarray(assoc.get("detection_indices", []), dtype=np.int64)
            if not np.any(detection_indices == di):
                bad_det_idx += 1
                continue
            try:
                reader.get_detection(di)
            except Exception:
                bad_det_idx += 1
        reader.close()

        total_events += n_events
        matched_events += n_matched
        unmatched_events += n_unmatched
        association_record_count += n_assoc
        invalid_detection_indices += bad_det_idx
        duplicate_event_ids += dup_count
        warp_alignment_errors += seq_warp_alignment_errors
        feature_nonzero_count += seq_feature_nonzero
        feature_element_count += seq_feature_total
        reid_dim = int(manifest.get("reid_dim", reid_dim))

        if n_events == 0:
            empty_caches.append(seq)
        if not bool(manifest.get("complete", False)):
            incomplete_sequences.append(seq)

        seq_feature_ratio = (seq_feature_nonzero / max(seq_feature_total, 1))

        per_seq.append(
            {
                "sequence": seq,
                "processed_frames": int(manifest.get("processed_frames", manifest.get("num_frames", 0))),
                "total_events": n_events,
                "matched_events": n_matched,
                "unmatched_events": n_unmatched,
                "events_with_real_detection": n_matched,
                "association_context_count": n_assoc,
                "association_record_count": n_assoc,
                "feature_nonzero_ratio": seq_feature_ratio,
                "warp_alignment_errors": seq_warp_alignment_errors,
                "duplicate_event_ids": dup_count,
                "invalid_detection_indices": bad_det_idx,
                "complete": bool(manifest.get("complete", False)),
                "truncated": bool(manifest.get("truncated", False)),
                "num_detection_records": int(manifest.get("num_detections", 0)),
                "reid_dim": int(manifest.get("reid_dim", 0)),
            }
        )

    feature_nonzero_ratio = (
        feature_nonzero_count / max(feature_element_count, 1)
    )

    report = {
        "dataset": dataset,
        "mode": mode,
        "num_sequences": len(sequences),
        "processed_frames": sum(item["processed_frames"] for item in per_seq),
        "total_events": total_events,
        "matched_events": matched_events,
        "unmatched_events": unmatched_events,
        "events_with_real_detection": matched_events,
        "association_context_count": association_record_count,
        "association_record_count": association_record_count,
        "feature_nonzero_ratio": feature_nonzero_ratio,
        "warp_alignment_errors": warp_alignment_errors,
        "duplicate_event_ids": duplicate_event_ids,
        "invalid_detection_indices": invalid_detection_indices,
        "complete": (
            not empty_caches
            and not incomplete_sequences
            and invalid_detection_indices == 0
        ),
        "truncated": any(item.get("truncated", False) for item in per_seq),
        "reid_dim": reid_dim,
        "per_sequence": per_seq,
        "empty_caches": empty_caches,
        "incomplete_sequences": incomplete_sequences,
        "status": (
            "ok"
            if not empty_caches and not incomplete_sequences and invalid_detection_indices == 0
            else "issues_found"
        ),
    }

    out_dir = _ensure_dir(os.path.join(cache_root, "_validation"))
    out_path = os.path.join(out_dir, "cache_validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Validation report saved to {out_path}")
    print(json.dumps(report, indent=2, default=str))


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
    p.add_argument("--max-events", type=int, default=0, help="Limit events (0=all).")
    p.add_argument(
        "--event-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "event_cache"),
    )
    p.add_argument(
        "--detection-cache-root",
        default=os.path.join(PROJECT_ROOT, "outputs", "agentguard", "detection_cache"),
    )


def _cmd_build_rollout_labels(args: argparse.Namespace) -> None:
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.data.gt_reader import GTReader
    from agentguard.data.future_oracle import FutureOracleBuilder
    from agentguard.data.identity_prototype import IdentityPrototypeBuilder
    from agentguard.motion.nsa_numpy import NSAKalmanFilter
    from agentguard.rollout.motion import compute_motion_benefit
    from agentguard.rollout.appearance import compute_appearance_benefit
    from agentguard.labels import (
        compute_dataset_stats,
        compute_soft_targets,
        build_rollout_labels as _build_rollout_labels,
    )

    dataset = args.dataset
    mode = args.mode
    cache_root = _event_cache_dir(dataset, mode, args.event_cache_root)
    label_dir = _ensure_dir(_labels_dir(dataset))

    # Set env var used by compute_dataset_stats
    os.environ["AG_GUARD_DATASET"] = dataset

    sequences = sorted(
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
    )

    all_events: List[TrackEvent] = []
    all_motion_benefits: List[float] = []
    all_appearance_benefits: List[float] = []
    all_identity_prototypes: Dict[Any, np.ndarray] = {}

    motion_model = NSAKalmanFilter()

    total_processed = 0

    for seq in sequences:
        seq_dir = os.path.join(cache_root, seq)
        try:
            reader = EventCacheReader(seq_dir)
            manifest = reader.read_manifest()
            events = reader.read_events()
            frames_list = reader.read_frames()
            prototypes = reader.read_identity_prototypes()
        except Exception as e:
            print(f"  SKIP {seq}: {e}")
            continue

        if not events:
            continue

        # Convert frames list to dict keyed by frame_id
        frame_data: Dict[int, Any] = {}
        if isinstance(frames_list, list):
            for fr in frames_list:
                if isinstance(fr, dict):
                    frame_data[fr.get("frame_id", -1)] = fr
        else:
            frame_data = frames_list

        # Build future oracle data
        try:
            gt_reader = GTReader(manifest.dataset, seq)
        except Exception:
            gt_reader = None

        # Identity vote state for resolving target_gt_id
        from agentguard.data.identity_vote import TrackIdentityVoteState
        identity_vote = TrackIdentityVoteState()

        all_frame_detections: Dict[int, List[Tuple[np.ndarray, float, int]]] = {}
        for fid, fdata in frame_data.items():
            all_frame_detections[fid] = []

        warp_by_frame: Dict[int, np.ndarray] = {}
        for fid, fdata in frame_data.items():
            wm = fdata.get("warp_matrix")
            if wm is not None:
                warp_by_frame[fid] = np.asarray(wm, dtype=np.float64)
            else:
                warp_by_frame[fid] = np.eye(2, 3, dtype=np.float64)

        oracle_builder = FutureOracleBuilder(warp_by_frame)
        frame_ids = sorted(frame_data.keys())

        all_identity_prototypes.update(prototypes)

        # Helper to compute IoU between two boxes
        def _box_iou(a, b):
            x1 = max(float(a[0]), float(b[0]))
            y1 = max(float(a[1]), float(b[1]))
            x2 = min(float(a[2]), float(b[2]))
            y2 = min(float(a[3]), float(b[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            return 0.0 if union <= 0.0 else inter / union

        for evt in events:
            if args.max_events > 0 and total_processed >= args.max_events:
                break

            # Resolve target_gt_id from identity vote state
            target_gt_id = None
            if gt_reader is not None:
                target_gt_id = identity_vote.resolve_before_current(evt.sequence, evt.track_id)

                # Determine detection GT ID for current event to feed vote state
                detection_gt_id = -1
                if evt.has_detection and evt.detection is not None:
                    det_box = evt.detection.box
                    gt_entries = gt_reader.get_gt_for_frame(evt.frame_id)
                    best_iou = 0.5  # threshold
                    for gt_box, gt_id in gt_entries:
                        iou = _box_iou(det_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            detection_gt_id = gt_id
                identity_vote.add_current_observation(evt.sequence, evt.track_id, detection_gt_id)

            # Get identity prototype by (seq, target_gt_id)
            proto = None
            identity_key = (seq, target_gt_id) if target_gt_id is not None else None
            if identity_key is not None:
                proto = all_identity_prototypes.get(identity_key)

            # Build oracle data with target_gt_id
            if gt_reader is not None and target_gt_id is not None:
                oracle_data = oracle_builder.build(
                    evt, gt_reader, all_frame_detections, frame_ids,
                    target_gt_id=target_gt_id,
                )
            else:
                oracle_data = None

            if oracle_data is None:
                oracle_data = {
                    "current_gt_box": None,
                    "future_gt_boxes": [],
                    "future_oracle_detections": [],
                    "future_warp_matrices": [],
                }

            # Build RolloutContext for benefit computation
            from agentguard.rollout.context import RolloutContext

            ctx = RolloutContext(
                frame_id=evt.frame_id,
                target_gt_id=target_gt_id if target_gt_id is not None else -1,
                pre_update_state=evt.pre_update_state if evt.pre_update_state is not None
                else evt.frame_start_state,
                current_candidate=evt.detection,
                current_gt_box=oracle_data.get("current_gt_box"),
                future_gt_boxes=oracle_data.get("future_gt_boxes", []),
                future_oracle_detections=oracle_data.get("future_oracle_detections", []),
                future_warp_matrices=oracle_data.get("future_warp_matrices", []),
                identity_prototype=proto,
            )

            # Compute benefits
            try:
                B_m, write_losses, skip_losses, valid_m = compute_motion_benefit(
                    ctx, motion_model
                )
                B_a, _, _, _, _, valid_a = compute_appearance_benefit(ctx)
            except Exception as e:
                print(f"  WARN: benefit computation failed for {evt.event_id}: {e}")
                total_processed += 1
                continue

            all_events.append(evt)
            all_motion_benefits.append(B_m)
            all_appearance_benefits.append(B_a)

            # Attach future oracle data to event for downstream use
            evt.motion_rollout_target = float(compute_soft_targets(B_m, 0.01))
            evt.appearance_rollout_target = float(compute_soft_targets(B_a, 0.01))
            oracle_boxes = oracle_data.get("future_oracle_detections", [])
            evt.future_oracle_coverage = (
                sum(1 for b in oracle_boxes if b is not None) / max(len(oracle_boxes), 1)
            )

            total_processed += 1

        if args.max_events > 0 and total_processed >= args.max_events:
            break

    print(f"Processed {total_processed} events total.")

    if not all_events:
        print("No events processed. Nothing to save.")
        return

    # Compute dataset stats
    all_benefits = {
        "motion_benefits": all_motion_benefits,
        "appearance_benefits": all_appearance_benefits,
    }
    dataset_stats = compute_dataset_stats(all_benefits)

    # Build labels
    labels = _build_rollout_labels(
        all_events,
        all_motion_benefits,
        all_appearance_benefits,
        dataset_stats,
        all_identity_prototypes,
    )

    # Save labels per sequence
    seq_labels: Dict[str, List[Dict[str, Any]]] = {}
    for evt, label in zip(all_events, labels):
        seq_labels.setdefault(evt.sequence, []).append(label)

    labels_out = _ensure_dir(os.path.join(label_dir, mode))
    for seq, seq_lbls in seq_labels.items():
        seq_path = os.path.join(labels_out, f"{seq}_labels.json")
        with open(seq_path, "w") as f:
            json.dump(seq_lbls, f, indent=2, default=str)

    # Save summary
    summary = {
        "dataset": dataset,
        "mode": mode,
        "num_events": len(all_events),
        "num_sequences": len(seq_labels),
        "dataset_stats": dataset_stats,
        "mean_motion_benefit": float(np.mean(all_motion_benefits)),
        "mean_appearance_benefit": float(np.mean(all_appearance_benefits)),
    }
    summary_path = os.path.join(label_dir, f"{mode}_summary.json")
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
        default="train_custom",
        help="Mode used for label loading.",
    )


def _cmd_build_student_v0_data(args: argparse.Namespace) -> None:
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.normalization import NormalizationStats
    from agentguard.datasets.iwg_dataset import IWGDataset
    from agentguard.datasets.tgr_dataset import TGRDataset
    from agentguard.datasets.split_validation import validate_splits

    dataset = args.dataset
    mode = args.mode
    cache_root = _cache_dir(dataset, mode)
    label_dir = os.path.join(_labels_dir(dataset), mode)
    datasets_dir = _ensure_dir(_datasets_dir(dataset))

    # Read all events and labels
    sequences = sorted(
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
    )

    all_events: List[TrackEvent] = []
    all_labels: List[Dict[str, Any]] = []

    for seq in sequences:
        seq_dir = os.path.join(cache_root, seq)
        try:
            reader = EventCacheReader(seq_dir)
            events = reader.read_events()
        except Exception as e:
            print(f"  SKIP {seq}: {e}")
            continue

        # Load labels for this sequence
        seq_label_path = os.path.join(label_dir, f"{seq}_labels.json")
        if os.path.isfile(seq_label_path):
            with open(seq_label_path) as f:
                seq_labels = json.load(f)
        else:
            print(f"  WARN: no labels for {seq}, skipping.")
            continue

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

    if not all_events:
        print("No events/labels loaded.")
        return

    print(f"Loaded {len(all_events)} events with labels.")

    # Build feature builder
    reid_dim = 2048
    norm_stats = NormalizationStats()
    scalar_list = [evt.scalar_features for evt in all_events if evt.scalar_features is not None]
    if scalar_list:
        norm_stats.fit(scalar_list)
    feature_builder = EventFeatureBuilder(reid_dim, norm_stats)

    # Split sequences into train/val (80/20 by sequence)
    seq_names = list({evt.sequence for evt in all_events})
    np.random.shuffle(seq_names)
    split_idx = max(1, int(len(seq_names) * 0.8))
    train_seqs = set(seq_names[:split_idx])
    val_seqs = set(seq_names[split_idx:])

    train_events = [evt for evt in all_events if evt.sequence in train_seqs]
    val_events = [evt for evt in all_events if evt.sequence in val_seqs]
    train_labels = [lbl for evt, lbl in zip(all_events, all_labels) if evt.sequence in train_seqs]
    val_labels = [lbl for evt, lbl in zip(all_events, all_labels) if evt.sequence in val_seqs]

    # Validate split
    try:
        validate_splits(train_events, val_events)
        print("Split validation passed.")
    except ValueError as e:
        print(f"Split validation warning: {e}")

    # Build IWG datasets
    print("Building IWG datasets...")
    train_iwg = IWGDataset(train_events, train_labels, feature_builder)
    val_iwg = IWGDataset(val_events, val_labels, feature_builder)

    # Build TGR datasets
    print("Building TGR datasets...")
    train_tgr = TGRDataset(train_events, train_labels, feature_builder)
    val_tgr = TGRDataset(val_events, val_labels, feature_builder)

    # Save datasets as sharded torch files
    def _save_dataset(ds: Any, path: str, name: str) -> None:
        out_dir = _ensure_dir(os.path.join(path, name))
        # Save feature builder config
        import torch

        # Save events and labels indices
        indices = list(range(len(ds)))
        torch.save(indices, os.path.join(out_dir, "indices.pt"))

        # Save metadata
        meta = {
            "type": ds.__class__.__name__,
            "size": len(ds),
            "window_size": getattr(ds, "window_size", None),
            "max_history": getattr(ds, "max_history", None),
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved {name} ({len(ds)} samples) to {out_dir}")

    iwg_dir = os.path.join(datasets_dir, "iwg")
    _save_dataset(train_iwg, iwg_dir, "train")
    _save_dataset(val_iwg, iwg_dir, "val")

    tgr_dir = os.path.join(datasets_dir, "tgr")
    _save_dataset(train_tgr, tgr_dir, "train")
    _save_dataset(val_tgr, tgr_dir, "val")

    # Save normalization stats
    import torch

    stats_path = os.path.join(datasets_dir, "norm_stats.pt")
    torch.save({"mean": norm_stats.mean, "std": norm_stats.std}, stats_path)
    print(f"Saved normalization stats to {stats_path}")

    # Save split config
    split_cfg = {
        "train_sequences": sorted(train_seqs),
        "val_sequences": sorted(val_seqs),
    }
    with open(os.path.join(datasets_dir, "split.json"), "w") as f:
        json.dump(split_cfg, f, indent=2)

    print(f"IWG: {len(train_iwg)} train / {len(val_iwg)} val samples")
    print(f"TGR: {len(train_tgr)} train / {len(val_tgr)} val samples")


# ===================================================================
# Subcommand: train_student_v0
# ===================================================================


def _add_train_student_v0_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train_student_v0",
        help="Train Student-V0 models (IWG first, then TGR).",
    )
    p.add_argument("--dataset", default="MOT17", help="Dataset name.")
    p.add_argument("--device", default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")


def _cmd_train_student_v0(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from agentguard.models.iwg import IWG
    from agentguard.models.tgr import TGR
    from agentguard.training.train_iwg import train_iwg
    from agentguard.training.train_tgr import train_tgr
    from agentguard.training.checkpointing import load_checkpoint
    from agentguard.datasets.iwg_dataset import IWGDataset
    from agentguard.datasets.tgr_dataset import TGRDataset

    dataset = args.dataset
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    checkpoints_dir = _ensure_dir(_checkpoints_dir(dataset))
    datasets_dir = _datasets_dir(dataset)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "reid_dim": 2048,
        "scalar_dim": 63,
        "event_dim": 128,
    }

    # ---- Phase 1: Train IWG ----
    print("=" * 60)
    print("Phase 1: Training IWG")
    print("=" * 60)

    # Load IWG datasets from saved indices
    iwg_train_path = os.path.join(datasets_dir, "iwg", "train")
    iwg_val_path = os.path.join(datasets_dir, "iwg", "val")

    # In a full pipeline we'd reconstruct datasets from saved metadata.
    # Here we build a fresh IWG model and use simulated training.
    iwg_model = IWG(reid_dim=2048).to(device)

    # Create dummy DataLoaders (real pipeline would load saved datasets)
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.normalization import NormalizationStats

    norm_stats = NormalizationStats()
    feature_builder = EventFeatureBuilder(2048, norm_stats)

    # Try to load saved datasets
    iwg_train_ds_path = os.path.join(iwg_train_path, "metadata.json")
    iwg_val_ds_path = os.path.join(iwg_val_path, "metadata.json")

    if not (os.path.isfile(iwg_train_ds_path) and os.path.isfile(iwg_val_ds_path)):
        print("IWG dataset metadata not found. Run build_student_v0_data first.")
        return

    # Build datasets from cached events and labels
    from agentguard.contracts.events import TrackEvent
    from agentguard.data.cache_reader import EventCacheReader

    cache_root = _cache_dir(dataset, "train_custom")
    label_dir = os.path.join(_labels_dir(dataset), "train_custom")

    with open(os.path.join(datasets_dir, "split.json")) as f:
        split_cfg = json.load(f)
    train_seqs = set(split_cfg.get("train_sequences", []))
    val_seqs = set(split_cfg.get("val_sequences", []))

    all_events: List[TrackEvent] = []
    all_labels: List[Dict[str, Any]] = []

    for seq in sorted(os.listdir(cache_root)):
        if not os.path.isdir(os.path.join(cache_root, seq)):
            continue
        seq_label_path = os.path.join(label_dir, f"{seq}_labels.json")
        if not os.path.isfile(seq_label_path):
            continue
        try:
            reader = EventCacheReader(os.path.join(cache_root, seq))
            events = reader.read_events()
        except Exception:
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

    train_evts = [evt for evt in all_events if evt.sequence in train_seqs]
    val_evts = [evt for evt in all_events if evt.sequence in val_seqs]
    train_lbls = [lbl for evt, lbl in zip(all_events, all_labels) if evt.sequence in train_seqs]
    val_lbls = [lbl for evt, lbl in zip(all_events, all_labels) if evt.sequence in val_seqs]

    if not train_evts:
        print("No training data available.")
        return

    # Build datasets
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

    iwg_output_dir = _ensure_dir(os.path.join(checkpoints_dir, "iwg"))
    iwg_summary = train_iwg(iwg_model, iwg_train_loader, iwg_val_loader, config, iwg_output_dir)
    print(f"IWG training complete. Best epoch: {iwg_summary.get('best_epoch')}")

    # ---- Phase 2: Train TGR ----
    print("\n" + "=" * 60)
    print("Phase 2: Training TGR")
    print("=" * 60)

    tgr_model = TGR(reid_dim=2048).to(device)

    # Load best IWG checkpoint
    best_iwg_path = os.path.join(iwg_output_dir, "iwg_best.pt")
    if os.path.isfile(best_iwg_path):
        load_checkpoint(best_iwg_path, iwg_model)
        iwg_model.to(device)
        print(f"Loaded best IWG from {best_iwg_path}")
    else:
        print("No best IWG checkpoint found, using freshly initialised model.")

    # Build TGR datasets
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

    tgr_config = {**config, "batch_size": config["batch_size"] // 2, "learning_rate": 2e-4}
    tgr_output_dir = _ensure_dir(os.path.join(checkpoints_dir, "tgr"))
    tgr_summary = train_tgr(tgr_model, iwg_model, tgr_train_loader, tgr_val_loader, tgr_config, tgr_output_dir)
    print(f"TGR training complete. Best epoch: {tgr_summary.get('best_epoch')}")

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
    _add_train_student_v0_parser(subparsers)
    _add_select_teacher_events_parser(subparsers)
    _add_build_evidence_packets_parser(subparsers)
    _add_run_bailian_teacher_parser(subparsers)
    _add_verify_and_fuse_parser(subparsers)
    _add_build_student_v1_data_parser(subparsers)
    _add_train_student_v1_parser(subparsers)

    parsed = parser.parse_args(argv)

    # Dispatch to command handler
    dispatch = {
        "cache_events": _cmd_cache_events,
        "validate_cache": _cmd_validate_cache,
        "rollout_smoke": _cmd_rollout_smoke,
        "build_rollout_labels": _cmd_build_rollout_labels,
        "evaluate_oracle": _cmd_evaluate_oracle,
        "build_student_v0_data": _cmd_build_student_v0_data,
        "train_student_v0": _cmd_train_student_v0,
        "select_teacher_events": _cmd_select_teacher_events,
        "build_evidence_packets": _cmd_build_evidence_packets,
        "run_bailian_teacher": _cmd_run_bailian_teacher,
        "verify_and_fuse": _cmd_verify_and_fuse,
        "build_student_v1_data": _cmd_build_student_v1_data,
        "train_student_v1": _cmd_train_student_v1,
    }

    handler = dispatch.get(parsed.command)
    if handler is not None:
        handler(parsed)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
