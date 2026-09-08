#!/usr/bin/env python3
"""Collect pure TrackTrack and build replay-complete MOT17 GT+LOO data.

Final sequences contain data.pt, track-only reid_features.npy and manifest.json.
Existing detection/GT assets are referenced, never copied or modified. Old
train_data is not an input. Resume is per sequence/stage, not per video frame.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "agentguard/src"))
from agentguard.data.cache_reader import CompactEventCacheReader
from agentguard.data.rollout_label_builder import build_compact_rollout_labels_for_sequence
from agentguard.data.train_data_v2 import (
    RULES, PackedReplayReader, check_detection_reference, compare_replay,
    fingerprint_detection, sha256, validate_replay, write_packed_sequence,
)
from agentguard.datasets.iwg_rg_cma_dataset import MOT17_FRCNN_ALL_SEQUENCES, StreamingIWGRGCMADataset


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def validate_final(directory, detection_dir, *, hashes=True):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 2 or not manifest.get("complete"):
        raise ValueError(f"not a completed v2 sequence: {directory}")
    for name, value in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != value["bytes"]:
            raise ValueError(f"missing/truncated output: {path}")
        if hashes and sha256(path) != value["sha256"]:
            raise ValueError(f"output checksum mismatch: {path}")
    check_detection_reference(detection_dir, manifest, full=hashes)
    reader = PackedReplayReader(directory, detection_dir)
    try:
        validate_replay(reader)
    finally:
        reader.close()
    return manifest


def refresh_metadata(root):
    manifests = {}
    for seq in MOT17_FRCNN_ALL_SEQUENCES:
        path = root / seq / "manifest.json"
        if path.is_file():
            manifest = read_json(path)
            if manifest.get("schema_version") != 2 or not manifest.get("complete"):
                raise ValueError(f"unexpected output: {path}")
            manifests[seq] = manifest
    if not manifests:
        return
    first = next(iter(manifests.values()))
    for m in manifests.values():
        for field in ("context_size", "max_frame_gap", "generation_rules", "future_frames", "reid_dim"):
            if m[field] != first[field]:
                raise ValueError(f"incompatible sequences in dataset: {field}")
    n = sum(m["scalar_statistics"]["count"] for m in manifests.values())
    mean = sum(np.asarray(m["scalar_statistics"]["sum"], dtype=np.float64) for m in manifests.values()) / n
    second = sum(np.asarray(m["scalar_statistics"]["sum_sq"], dtype=np.float64) for m in manifests.values()) / n
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    std[std < 1e-8] = 1.0
    metadata = {"format": "train_data", "schema_version": 2, "index_format": "train_data_v2",
                "dataset": "MOT17", "split": "all", "train_sequences": list(manifests),
                "context_size": first["context_size"], "max_frame_gap": first["max_frame_gap"],
                "future_frames": first["future_frames"], "generation_rules": first["generation_rules"],
                "reid_dim": first["reid_dim"], "scalar_dim": 63, "event_dim": 128,
                "reid_feature_file": "reid_features.npy", "reid_layout": "track_only", "scalar_storage": "raw",
                "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
                "timeline_counts": {s: m["timeline_counts"] for s, m in manifests.items()},
                "num_train_samples": sum(m["num_train_samples"] for m in manifests.values())}
    atomic_json(root / "metadata.json", metadata)


def capture(args, sequence, work):
    if args.event_cache_root:
        source = args.event_cache_root.resolve() / "MOT17/all" / sequence
        if source.is_dir():
            return source
    root = work / "capture"
    final = root / "MOT17/all" / sequence
    if (final / "manifest.json").is_file():
        manifest = read_json(final / "manifest.json")
        if manifest.get("complete") and not manifest.get("truncated", False):
            print(f"Reuse completed capture: {sequence}", flush=True)
            return final
        raise ValueError(f"unexpected incomplete final capture: {final}")
    command = [sys.executable, "-m", "agentguard.cli", "cache_events", "--dataset", "MOT17",
               "--mode", "all", "--sequence", sequence, "--detector", "FRCNN",
               "--data-dir", str(args.data_dir.resolve()) + os.sep,
               "--detection-cache-root", str(args.detection_cache_root.resolve()),
               "--event-cache-root", str(root)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "agentguard/src"), str(REPO / "3. Tracker"), env.get("PYTHONPATH", "")])
    log_path = work / "capture.log"
    print(f"Pure TrackTrack, capture only (no learned gates), NSA + GMC: {sequence}\nLog: {log_path}", flush=True)
    # The existing sink discards only its own .incomplete directory on retry.
    # Each sequence runs in a fresh process to reset tracker-global ID counters.
    with log_path.open("a") as log:
        subprocess.run(command, cwd=REPO / "3. Tracker", env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return final


def run(args):
    output_root = args.output_root.resolve()
    for protected in (REPO / "outputs/agentguard/train_data", args.detection_cache_root.resolve(), args.data_dir.resolve()):
        protected = protected.resolve()
        if output_root == protected or output_root in protected.parents or protected in output_root.parents:
            raise ValueError(f"output must be separate from existing inputs: {protected}")
    root = output_root / "MOT17"
    if root.is_symlink():
        raise ValueError("output dataset must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if (root / ".work").is_symlink():
        raise ValueError("work root must not be a symlink")
    # Prevent two builders publishing inconsistent root metadata concurrently.
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, root)


def run_locked(args, root):
    if (root / "metadata.json").is_file():
        existing = read_json(root / "metadata.json")
        if existing.get("schema_version") != 2:
            raise ValueError("refusing to reuse a non-v2 output directory")
        for key, value in {"context_size": args.context_size, "max_frame_gap": args.max_frame_gap,
                           "future_frames": args.future_frames, "generation_rules": RULES}.items():
            if existing.get(key) != value:
                raise ValueError(f"different {key}; use another output root and --relabel-from")
    sequences = args.sequence or list(MOT17_FRCNN_ALL_SEQUENCES)
    sources = [REPO / "agentguard/src/agentguard" / p for p in (
        "data/train_data_v2.py", "data/rollout_label_builder.py", "data/identity_prototype.py",
        "data/compact_event_cache.py", "data/detection_cache.py", "data/cache_schema.py",
        "data/cache_reader.py", "rollout/motion.py", "rollout/appearance.py", "rollout/losses.py",
        "rollout_labels.py", "motion/nsa_numpy.py", "data/identity_vote.py", "data/gt_reader.py",
        "data/label_schema.py", "cli/__init__.py")]
    sources += [Path(__file__).resolve()]
    for directory in (REPO / "3. Tracker/trackers", REPO / "3. Tracker/integrations/agentguard", REPO / "3. Tracker/utils"):
        sources.extend(sorted(directory.glob("*.py")))
    source_hashes = {str(p.relative_to(REPO)): sha256(p) for p in sources}
    for sequence in sequences:
        final = root / sequence
        if final.is_symlink() or not final.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unsafe output path: {final}")
        detection = args.detection_cache_root.resolve() / "MOT17/all" / sequence
        gt = args.data_dir.resolve() / "MOT17/train" / sequence / "gt/gt.txt"
        seqinfo = gt.parent.parent / "seqinfo.ini"
        for p in (gt, seqinfo, detection / "manifest.json"):
            if not p.is_file():
                raise FileNotFoundError(p)
        gmc = REPO / "3. Tracker/trackers/cmc" / f"GMC-{sequence.removesuffix('-FRCNN')}.txt"
        supplied_capture = args.event_cache_root and (args.event_cache_root.resolve() / "MOT17/all" / sequence).is_dir()
        fresh_capture = not args.relabel_from and not supplied_capture
        if fresh_capture and not gmc.is_file():
            raise FileNotFoundError(f"TrackTrack requires its existing GMC file: {gmc}")
        det_fingerprint = fingerprint_detection(detection)
        request = {"sequence": sequence, "context_size": args.context_size, "max_frame_gap": args.max_frame_gap,
                   "future_frames": args.future_frames, "generation_rules": RULES,
                   "gt_sha256": sha256(gt), "seqinfo_sha256": sha256(seqinfo),
                   "gmc_sha256": sha256(gmc) if fresh_capture else None,
                   "detection_fingerprint": det_fingerprint, "source_sha256": source_hashes,
                   "relabel_from": str(args.relabel_from.resolve()) if args.relabel_from else None,
                   "event_cache_root": str(args.event_cache_root.resolve()) if args.event_cache_root else None}
        work = root / ".work" / sequence
        if work.is_symlink():
            raise ValueError("work directory must not be a symlink")
        if final.exists():
            manifest = validate_final(final, detection)
            if manifest["request"] != request:
                raise ValueError(f"completed inputs/config changed; use new output root: {final}")
            refresh_metadata(root)
            if work.is_dir() and (work / "request.json").is_file() and read_json(work / "request.json") == request:
                shutil.rmtree(work)
            print(f"Already complete: {sequence}", flush=True)
            continue
        if work.exists() and any(work.iterdir()):
            if not (work / "request.json").is_file() or read_json(work / "request.json") != request:
                raise ValueError(f"unfinished work has different inputs; use a new output root: {work}")
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "request.json", request)
        if args.relabel_from:
            source = args.relabel_from.resolve() / "MOT17" / sequence
            validate_final(source, detection)
            def open_reader():
                return PackedReplayReader(source, detection)
        else:
            source = capture(args, sequence, work)
            def open_reader():
                return CompactEventCacheReader(source, detection, max_cached_shards=4)
        source_files = sorted(source.glob("*.pt")) + [source / "manifest.json"]
        if args.relabel_from:
            source_files.append(source / "reid_features.npy")
        replay_fingerprint = {p.name: sha256(p) for p in source_files}
        source_marker = work / "source.json"
        if source_marker.exists() and read_json(source_marker) != replay_fingerprint:
            raise ValueError("replay source changed during unfinished build; use a new output root")
        atomic_json(source_marker, replay_fingerprint)
        reader = open_reader()
        try:
            if reader.manifest.get("sequence") != sequence or reader.manifest.get("dataset") != "MOT17":
                raise ValueError("wrong source sequence")
            recorded = reader.manifest.get("detection_cache_manifest_sha256")
            if recorded and recorded != det_fingerprint["manifest.json"]["sha256"]:
                raise ValueError("capture used a different detection cache")
            validate_replay(reader)
        finally:
            reader.close()
        labels_path = work / "labels.pt"
        if labels_path.is_file():
            result = torch.load(labels_path, weights_only=True, map_location="cpu")
            labels, summary = result["labels"], result["summary"]
            print(f"Reuse completed GT+LOO labels: {sequence}", flush=True)
        else:
            print(f"Compute GT+LOO labels: {sequence}", flush=True)
            labels, summary = build_compact_rollout_labels_for_sequence(
                source, detection, gt.parent.parent.parent, future_frames=args.future_frames,
                reader=open_reader(), persist_stats=False, **RULES)
            temp = work / "labels.pt.tmp"
            torch.save({"labels": labels, "summary": summary}, temp)
            temp.replace(labels_path)
        metadata = {"dataset": "MOT17", "split": "all", "source_split": "all", "sequence": sequence,
                    "context_size": args.context_size, "max_frame_gap": args.max_frame_gap,
                    "future_frames": args.future_frames, "generation_rules": RULES, "request": request,
                    "replay_source_fingerprint": replay_fingerprint,
                    "references": {"detection_cache": os.path.relpath(detection, final),
                                   "detection_fingerprint": det_fingerprint,
                                   "gt": os.path.relpath(gt, final), "gt_sha256": sha256(gt),
                                   "seqinfo": os.path.relpath(seqinfo, final), "seqinfo_sha256": sha256(seqinfo)}}
        stage = work / "packed"
        if stage.exists():
            shutil.rmtree(stage)  # Only this script's uncommitted packing stage.
        reader = open_reader()
        try:
            write_packed_sequence(reader, stage, metadata, labels, summary)
            packed = PackedReplayReader(stage, detection)
            try:
                validate_replay(packed)
                compare_replay(reader, packed)
            finally:
                packed.close()
        finally:
            reader.close()
        validate_final(stage, detection)
        stage.rename(final)
        refresh_metadata(root)
        # Cleanup only after replay round-trip, checksums and atomic publication.
        shutil.rmtree(work)
        print(f"Complete: {final}", flush=True)
    dataset = StreamingIWGRGCMADataset(root, detection_cache_root=args.detection_cache_root)
    try:
        if len(dataset):
            dataset[0]
        print(f"Training reader OK: {len(dataset)} samples; metadata contains {len(dataset.metadata['train_sequences'])} sequences", flush=True)
    finally:
        dataset.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", action="append", choices=MOT17_FRCNN_ALL_SEQUENCES,
                   help="Repeat for a subset; omitted = all seven MOT17 FRCNN sequences")
    p.add_argument("--output-root", type=Path, default=REPO / "outputs/agentguard/train_data_v2")
    p.add_argument("--detection-cache-root", type=Path, default=REPO / "outputs/agentguard/detection_cache")
    p.add_argument("--data-dir", type=Path, default=Path("/home/shang/datasets"))
    p.add_argument("--event-cache-root", type=Path, help="Optional complete raw cache root for initial packing")
    p.add_argument("--relabel-from", type=Path, help="Prior train_data_v2 root; recompute solely from packed states")
    p.add_argument("--context-size", type=int, choices=[6, 8], default=6)
    p.add_argument("--max-frame-gap", type=int, default=30)
    p.add_argument("--future-frames", type=int, default=5)
    args = p.parse_args()
    if args.event_cache_root and args.relabel_from:
        p.error("choose only one replay source")
    if args.future_frames < 0 or args.max_frame_gap <= 0:
        p.error("future-frames must be nonnegative and max-frame-gap positive")
    run(args)


if __name__ == "__main__":
    main()
