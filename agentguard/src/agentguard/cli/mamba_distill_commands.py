from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRACKER_DIR = PROJECT_ROOT / "3. Tracker"
MOT20_SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
MAMBA_TEACHER_CHECKPOINT = Path(
    "/home/shang/workspace/mamba_kalman_filter/checkpoints/"
    "MOT20_exp31_1_epoch55.pth"
)
MAMBA_TEACHER_SHA256 = (
    "3576c957e8e8041bfd216da52a14b0bbfb675566d234d1d298c1cb4fc51d92b9"
)


class _NullCompactCaptureSink:
    compact = True

    def __init__(self) -> None:
        self._reid_dim = 0

    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        return None


def _default_output(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath("outputs", "agentguard", *parts))


def _validated_sequences(values: list[str]) -> list[str]:
    sequences = values or list(MOT20_SEQUENCES)
    invalid = sorted(set(sequences).difference(MOT20_SEQUENCES))
    if invalid:
        raise ValueError(f"unsupported MOT20 sequences: {invalid}")
    if len(set(sequences)) != len(sequences):
        raise ValueError("duplicate MOT20 sequence requested")
    return sequences


def _teacher_config(checkpoint: Path, checkpoint_sha256: str) -> dict[str, Any]:
    try:
        repository_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=checkpoint.parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_commit = "unknown"
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "profile": "MOT20/exp31_1",
        "mamba_repository_commit": repository_commit,
        "d_model": 64,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_trajectory_tokens": 4,
        "tta_buffer_size": 16,
        "use_tta": True,
        "use_mamba": True,
        "use_innovation_feature": True,
        "use_diou_feature": True,
        "iou_feature_type": "diou",
        "zero_mask_innovation": False,
        "zero_mask_diou": False,
        "driver": "NSA accepted associations; no Mamba association",
    }


def add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export_mamba_shadow",
        help="Export an NSA-driven offline Mamba teacher sidecar for MOT20.",
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--checkpoint", default=str(MAMBA_TEACHER_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--data-dir", default="/home/shang/datasets")
    parser.add_argument(
        "--event-cache-root",
        default=_default_output("event_cache_v3_iwg_v2"),
    )
    parser.add_argument(
        "--detection-cache-root",
        default=_default_output("detection_cache"),
    )
    parser.add_argument(
        "--output-root",
        default=_default_output("mamba_teacher_v1"),
    )


def add_export_native_events_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "export_mamba_events",
        help="Export compact IWG events from Mamba's native tracker trajectories.",
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--checkpoint", default=str(MAMBA_TEACHER_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--data-dir", default="/home/shang/datasets")
    parser.add_argument(
        "--detection-cache-root",
        default=_default_output("detection_cache"),
    )
    parser.add_argument(
        "--output-root",
        default=_default_output("event_cache_mamba_v1"),
    )


def export_mamba_events(args: argparse.Namespace) -> None:
    tracker_path = str(TRACKER_DIR)
    if tracker_path not in sys.path:
        sys.path.insert(0, tracker_path)

    import torch

    from agentguard.data.cache_schema import FEATURE_SCHEMA_SHA256
    from agentguard.data.compact_event_cache import CompactEventCacheSink
    from agentguard.data.detection_cache import (
        SequenceDetectionCache,
        sequence_cache_dir,
    )
    from agentguard.motion.mamba_shadow import canonical_sha256, sha256_file
    from mamba_kalman_filter.config import Config
    from trackers.mamba_kalman_filter_wrapper import MambaKalmanFilterWrapper
    from trackers.tracker_mamba import TrackerMamba
    from utils.etc import set_parameters

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mamba checkpoint not found: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != MAMBA_TEACHER_SHA256:
        raise ValueError(
            "Mamba checkpoint SHA256 mismatch: "
            f"{checkpoint_sha256} != {MAMBA_TEACHER_SHA256}"
        )

    Config.load_dataset_config("MOT20")
    Config.USE_TTA = True
    Config.USE_MAMBA = True
    Config.USE_INNOVATION_FEATURE = True
    Config.USE_DIOU_FEATURE = True
    Config.IOU_FEATURE_TYPE = "diou"
    Config.ZERO_MASK_INNOVATION = False
    Config.ZERO_MASK_DIOU = False

    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"

    teacher_config = _teacher_config(checkpoint, checkpoint_sha256)
    teacher_config["driver"] = "native Mamba association and track lifecycle"
    summaries: dict[str, Any] = {}
    for sequence in _validated_sequences(args.sequence):
        detection_dir = sequence_cache_dir(
            args.detection_cache_root, "MOT20", "all", sequence
        )
        detection_manifest_path = detection_dir / "manifest.json"
        if not detection_manifest_path.is_file():
            raise FileNotFoundError(
                f"detection cache not found: {detection_manifest_path}"
            )
        detection_manifest_sha256 = sha256_file(detection_manifest_path)

        class _TrackerArgs:
            pass

        tracker_args = _TrackerArgs()
        tracker_args.pickle_dir = str(PROJECT_ROOT / "outputs" / "2. det_feat")
        tracker_args.data_dir = str(Path(args.data_dir).resolve()) + os.sep
        tracker_args.min_len = 3
        tracker_args.min_box_area = 100
        tracker_args.max_time_lost = 30
        tracker_args.penalty_p = 0.20
        tracker_args.penalty_q = 0.40
        tracker_args.reduce_step = 0.05
        tracker_args.tai_thr = 0.55
        tracker_args.disable_gmc = False
        tracker_args.kf_type = "mamba_native"
        tracker_args.agentguard_mode = "off"
        tracker_args.capture_agentguard_events = True
        tracker_args.no_reid = False
        tracker_args.dataset = "MOT20"
        set_parameters(tracker_args, sequence, "all")

        seqinfo = Path(tracker_args.data_path) / sequence / "seqinfo.ini"
        image_width, image_height = 1920, 1080
        if seqinfo.is_file():
            for line in seqinfo.read_text().splitlines():
                if line.startswith("imWidth"):
                    image_width = int(line.split("=", 1)[1])
                elif line.startswith("imHeight"):
                    image_height = int(line.split("=", 1)[1])
                elif line.startswith("frameRate"):
                    tracker_args.max_time_lost = int(line.split("=", 1)[1]) * 2
        tracker_args.img_w = image_width
        tracker_args.img_h = image_height

        cache = SequenceDetectionCache(detection_dir)
        total_frames = int(cache.num_frames)
        frame_count = total_frames
        if int(args.max_frames) > 0:
            frame_count = min(frame_count, int(args.max_frames))
        tracker_config = {
            "agentguard_mode": "off",
            "capture_agentguard_events": True,
            "event_source": "mamba_native",
            "dataset": "MOT20",
            "mode": "all",
            "sequence": sequence,
            "max_frames": int(args.max_frames),
            "kf_type": "mamba_native",
            "disable_gmc": False,
            "det_thr": float(tracker_args.det_thr),
            "init_thr": float(tracker_args.init_thr),
            "match_thr": float(tracker_args.match_thr),
            "max_time_lost": int(tracker_args.max_time_lost),
            "teacher_config": teacher_config,
        }
        tracker_config_sha256 = canonical_sha256(tracker_config)
        output_dir = (
            Path(args.output_root).resolve()
            / "MOT20"
            / "all"
            / sequence
        )
        existing_manifest = output_dir / "manifest.json"
        if existing_manifest.is_file():
            existing = json.loads(existing_manifest.read_text())
            if (
                existing.get("complete")
                and existing.get("feature_schema_sha256")
                == FEATURE_SCHEMA_SHA256
                and existing.get("tracker_config_sha256")
                == tracker_config_sha256
                and existing.get("detection_cache_manifest_sha256")
                == detection_manifest_sha256
            ):
                summaries[sequence] = existing
                cache.close()
                print(f"{sequence}: matching native event cache exists; skipping")
                continue

        event_sink = CompactEventCacheSink(
            cache_root=Path(args.output_root).resolve(),
            dataset="MOT20",
            split="all",
            sequence=sequence,
            reid_dim=int(cache.reid_dim),
            event_flush_size=256,
            frame_flush_size=32,
            association_flush_size=32,
            config_hash=tracker_config_sha256,
            source_commit=source_commit,
            feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            detection_cache_manifest_sha256=detection_manifest_sha256,
            tracker_config_sha256=tracker_config_sha256,
            total_sequence_frames=total_frames,
            image_width=image_width,
            image_height=image_height,
            tracker_config=tracker_config,
        )
        tracker_args.event_sink = event_sink
        event_sink.on_sequence_start(
            sequence=sequence,
            reid_dim=int(cache.reid_dim),
            image_width=image_width,
            image_height=image_height,
        )
        event_sink._truncated = frame_count < total_frames
        event_sink._num_detections = int(cache.num_detections)

        tracker = None
        mamba_filter = None
        try:
            mamba_filter = MambaKalmanFilterWrapper(
                model_path=str(checkpoint), device=str(args.device)
            )
            mamba_filter.set_image_size(image_width, image_height)
            tracker = TrackerMamba(tracker_args, sequence)
            tracker.shared_kalman_filter = mamba_filter
            for frame_index in range(frame_count):
                target = cache.get_frame(frame_index, view="target")
                source = cache.get_frame(frame_index, view="source")
                tracker_args.agentguard_target_detection_indices = target[
                    "detection_indices"
                ]
                tracker_args.agentguard_source_detection_indices = source[
                    "detection_indices"
                ]
                target_array = cache.get_frame_array(frame_index, view="target")
                source_array = cache.get_frame_array(frame_index, view="source")
                if target_array is not None and len(target_array) > 0:
                    if source_array is None:
                        source_array = target_array
                    tracker.update(target_array, source_array)
                else:
                    tracker.update_without_detections()
                if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
                    print(
                        f"{sequence}: frame {frame_index + 1}/{frame_count}",
                        flush=True,
                    )
            for track in tracker.tracks:
                mamba_filter.delete_track(track.track_id)
            summary = event_sink.on_sequence_end()
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["event_source"] = "mamba_native"
            manifest["mamba_checkpoint_path"] = str(checkpoint)
            manifest["mamba_checkpoint_sha256"] = checkpoint_sha256
            manifest["mamba_teacher_config"] = teacher_config
            manifest["mamba_teacher_config_sha256"] = canonical_sha256(
                teacher_config
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            summaries[sequence] = manifest
            print(
                f"{sequence}: events={summary['num_events']} "
                f"matched={summary['num_matched_events']} "
                f"unmatched={summary['num_unmatched_events']}",
                flush=True,
            )
        finally:
            cache.close()
            tracker_args.agentguard_target_detection_indices = None
            tracker_args.agentguard_source_detection_indices = None
            tracker_args.event_sink = None
            del tracker
            del mamba_filter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps(summaries, indent=2, sort_keys=True))


def export_mamba_shadow(args: argparse.Namespace) -> None:
    tracker_path = str(TRACKER_DIR)
    if tracker_path not in sys.path:
        sys.path.insert(0, tracker_path)

    from agentguard.data.detection_cache import (
        SequenceDetectionCache,
        sequence_cache_dir,
    )
    from agentguard.motion.mamba_shadow import (
        MambaShadow,
        MambaShadowWriter,
        canonical_sha256,
        sha256_file,
    )
    from trackers.tracker import Tracker
    from utils.etc import set_parameters

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mamba teacher checkpoint not found: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != MAMBA_TEACHER_SHA256:
        raise ValueError(
            "Mamba teacher checkpoint SHA256 mismatch: "
            f"{checkpoint_sha256} != {MAMBA_TEACHER_SHA256}"
        )
    teacher_config = _teacher_config(checkpoint, checkpoint_sha256)
    teacher_config_sha256 = canonical_sha256(teacher_config)
    summaries: dict[str, Any] = {}

    for sequence in _validated_sequences(args.sequence):
        event_dir = (
            Path(args.event_cache_root).resolve() / "MOT20" / "all" / sequence
        )
        event_manifest_path = event_dir / "manifest.json"
        detection_dir = sequence_cache_dir(
            args.detection_cache_root, "MOT20", "all", sequence
        )
        detection_manifest_path = detection_dir / "manifest.json"
        if not event_manifest_path.is_file():
            raise FileNotFoundError(f"NSA event cache not found: {event_manifest_path}")
        if not detection_manifest_path.is_file():
            raise FileNotFoundError(f"detection cache not found: {detection_manifest_path}")
        event_manifest = json.loads(event_manifest_path.read_text())
        event_manifest_sha256 = sha256_file(event_manifest_path)
        detection_manifest_sha256 = sha256_file(detection_manifest_path)
        output_dir = (
            Path(args.output_root).resolve() / "MOT20" / "all" / sequence
        )
        existing_manifest = output_dir / "manifest.json"
        if existing_manifest.is_file():
            existing = json.loads(existing_manifest.read_text())
            if (
                existing.get("complete")
                and existing.get("teacher_config_sha256")
                == teacher_config_sha256
                and existing.get("event_cache_manifest_sha256")
                == event_manifest_sha256
                and existing.get("detection_cache_manifest_sha256")
                == detection_manifest_sha256
            ):
                summaries[sequence] = existing
                print(f"{sequence}: matching shadow exists; skipping", flush=True)
                continue
            raise FileExistsError(f"refusing stale shadow teacher: {output_dir}")

        class _TrackerArgs:
            pass

        tracker_args = _TrackerArgs()
        tracker_args.pickle_dir = str(PROJECT_ROOT / "outputs" / "2. det_feat")
        tracker_args.data_dir = str(Path(args.data_dir).resolve())
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
        tracker_args.dataset = "MOT20"
        tracker_args.event_sink = _NullCompactCaptureSink()
        set_parameters(tracker_args, sequence, "all")

        seqinfo = Path(tracker_args.data_path) / sequence / "seqinfo.ini"
        image_width, image_height = 1920, 1080
        if seqinfo.is_file():
            for line in seqinfo.read_text().splitlines():
                if line.startswith("imWidth"):
                    image_width = int(line.split("=", 1)[1])
                elif line.startswith("imHeight"):
                    image_height = int(line.split("=", 1)[1])
                elif line.startswith("frameRate"):
                    tracker_args.max_time_lost = int(line.split("=", 1)[1]) * 2
        tracker_args.img_w = image_width
        tracker_args.img_h = image_height

        cache = SequenceDetectionCache(detection_dir)
        frame_count = int(cache.num_frames)
        if int(args.max_frames) > 0:
            frame_count = min(frame_count, int(args.max_frames))
        if int(args.max_frames) <= 0 and frame_count != int(
            event_manifest["num_frames"]
        ):
            cache.close()
            raise ValueError(
                f"NSA cache frame count mismatch for {sequence}: "
                f"{event_manifest['num_frames']} != {frame_count}"
            )

        writer = MambaShadowWriter(
            output_dir,
            sequence=sequence,
            teacher_config=teacher_config,
            event_cache_manifest_sha256=event_manifest_sha256,
            detection_cache_manifest_sha256=detection_manifest_sha256,
            shard_size=int(args.shard_size),
        )
        shadow = None
        try:
            shadow = MambaShadow(
                sequence=sequence,
                image_width=image_width,
                image_height=image_height,
                checkpoint_path=checkpoint,
                writer=writer,
                device=args.device,
            )
            tracker_args.mamba_shadow = shadow
            tracker = Tracker(tracker_args, sequence)
            for frame_index in range(frame_count):
                target = cache.get_frame(frame_index, view="target")
                source = cache.get_frame(frame_index, view="source")
                tracker_args.agentguard_target_detection_indices = target[
                    "detection_indices"
                ]
                tracker_args.agentguard_source_detection_indices = source[
                    "detection_indices"
                ]
                target_array = cache.get_frame_array(frame_index, view="target")
                source_array = cache.get_frame_array(frame_index, view="source")
                if target_array is not None and len(target_array) > 0:
                    if source_array is None:
                        source_array = target_array
                    tracker.update(target_array, source_array)
                else:
                    tracker.update_without_detections()
                if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
                    print(
                        f"{sequence}: frame {frame_index + 1}/{frame_count}",
                        flush=True,
                    )
            if int(args.max_frames) <= 0 and writer.num_records != int(
                event_manifest["num_events"]
            ):
                raise RuntimeError(
                    f"shadow/event count mismatch for {sequence}: "
                    f"{writer.num_records} != {event_manifest['num_events']}"
                )
            manifest = shadow.close()
            summaries[sequence] = manifest
            print(
                f"{sequence}: records={manifest['num_records']} "
                f"matched={manifest['num_matched']} "
                f"unmatched={manifest['num_unmatched']}",
                flush=True,
            )
        except Exception:
            if shadow is not None:
                shadow.abort()
            else:
                writer.abort()
            raise
        finally:
            cache.close()
            tracker_args.agentguard_target_detection_indices = None
            tracker_args.agentguard_source_detection_indices = None
            tracker_args.mamba_shadow = None
            gc.collect()
    print(json.dumps(summaries, indent=2, sort_keys=True))


def _add_shared_distill_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--gt-root", default="/home/shang/datasets/MOT20/train")
    parser.add_argument(
        "--event-cache-root",
        default=_default_output("event_cache_v3_iwg_v2"),
    )
    parser.add_argument(
        "--detection-cache-root",
        default=_default_output("detection_cache"),
    )
    parser.add_argument(
        "--teacher-root",
        default=_default_output("mamba_teacher_v1"),
    )
    parser.add_argument(
        "--nsa-label-root",
        default=_default_output(
            "labels",
            "iwg_rg_cma",
            "MOT20",
            "nsa_v3_compact",
        ),
    )


def add_audit_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "audit_mamba_shadow",
        help="Strictly join and audit NSA-driven Mamba teacher sidecars.",
    )
    _add_shared_distill_sources(parser)
    parser.add_argument(
        "--output",
        default=_default_output(
            "mamba_teacher_v1", "MOT20", "all", "audit.json"
        ),
    )


def _compute_distill_sources(args: argparse.Namespace):
    from agentguard.data.mamba_distill import (
        compute_sequence_distill_statistics,
    )

    results = {}
    for sequence in _validated_sequences(args.sequence):
        results[sequence] = compute_sequence_distill_statistics(
            sequence=sequence,
            event_cache_dir=(
                Path(args.event_cache_root).resolve()
                / "MOT20"
                / "all"
                / sequence
            ),
            detection_cache_dir=(
                Path(args.detection_cache_root).resolve()
                / "MOT20"
                / "all"
                / sequence
            ),
            teacher_dir=(
                Path(args.teacher_root).resolve()
                / "MOT20"
                / "all"
                / sequence
            ),
            nsa_label_dir=Path(args.nsa_label_root).resolve() / sequence,
            gt_root=Path(args.gt_root).resolve(),
        )
        summary = results[sequence][1]
        print(
            f"{sequence}: join={summary['join_rate']:.1%} "
            f"records={summary['records']} labels={summary['labels']}",
            flush=True,
        )
    return results


def audit_mamba_shadow_command(args: argparse.Namespace) -> None:
    from agentguard.data.mamba_distill import audit_mamba_shadow

    report = audit_mamba_shadow(_compute_distill_sources(args))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["structural_pass"]:
        raise RuntimeError("Mamba shadow structural audit failed")
    if report["full_mot20"] and not report["admission_pass"]:
        raise RuntimeError("Mamba teacher failed the full MOT20 training admission gate")


def add_build_labels_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build_mamba_distill_labels",
        help="Build fixed C-class Mamba/NSA hybrid compact motion labels.",
    )
    _add_shared_distill_sources(parser)
    parser.add_argument(
        "--output-root",
        default=_default_output(
            "labels_mamba_distill_v1", "MOT20", "all_a_only"
        ),
    )
    parser.add_argument(
        "--audit-output",
        default=_default_output(
            "mamba_teacher_v1", "MOT20", "all", "audit.json"
        ),
    )


def build_mamba_distill_labels_command(args: argparse.Namespace) -> None:
    from agentguard.data.mamba_distill import (
        audit_mamba_shadow,
        build_mamba_distill_labels,
    )

    results = _compute_distill_sources(args)
    report = audit_mamba_shadow(results)
    audit_output = Path(args.audit_output).resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["full_mot20"]:
        raise RuntimeError("formal Mamba distill labels require all four MOT20 sequences")
    if not report["admission_pass"]:
        raise RuntimeError("Mamba teacher failed admission; refusing distill labels")
    positives = np.concatenate(
        [
            arrays["advantage"][arrays["advantage"] > 0.0]
            for arrays, _summary in results.values()
        ]
    )
    if len(positives) == 0:
        raise RuntimeError("Mamba teacher has no positive advantages")
    tau_adv = float(np.median(positives))
    summary = build_mamba_distill_labels(
        sequence_results=results,
        nsa_label_root=args.nsa_label_root,
        output_root=args.output_root,
        tau_adv=tau_adv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
