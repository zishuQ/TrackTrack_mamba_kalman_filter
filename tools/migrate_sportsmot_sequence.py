#!/usr/bin/env python3
"""Migrate SportsMOT compact AgentGuard data to ``train_data/SportsMOT``.

By default the canonical ``trainval`` compact dataset is migrated.  Sequences
are processed one at a time; each source sequence is deleted only after its
packed output has been validated.  With no sequence argument all sequences in
the selected split are processed in metadata order.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _verify(output_root: Path, sequence: str) -> tuple[int, int, tuple[int, ...]]:
    seq_dir = output_root / sequence
    data_path = seq_dir / "data.pt"
    reid_path = seq_dir / "reid_features.npy"
    root_metadata_path = output_root / "metadata.json"
    for path in (data_path, reid_path, root_metadata_path):
        if not path.is_file():
            raise RuntimeError(f"迁移输出缺少文件: {path}")

    root_metadata = json.loads(root_metadata_path.read_text())
    if root_metadata.get("format") != "train_data":
        raise RuntimeError("根 metadata 格式不是 train_data")
    if sequence not in root_metadata.get("train_sequences", []):
        raise RuntimeError(f"根 metadata 未登记 {sequence}")

    packed = torch.load(data_path, weights_only=False)
    sequence_metadata = packed.get("metadata")
    arrays = packed.get("arrays")
    if not isinstance(sequence_metadata, dict) or not isinstance(arrays, dict):
        raise RuntimeError(f"无效的 packed data.pt: {data_path}")
    if sequence_metadata.get("format") != "train_data":
        raise RuntimeError(f"序列 metadata 格式不是 train_data: {data_path}")

    required = {
        "event_indices", "track_ids", "segment_ids", "safe_gate_target",
        "policy_safe_soft_target", "cue_target", "risk_target",
        "valid_channels", "sample_weight", "timeline_scalar_feats",
        "timeline_has_detection",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise RuntimeError(f"{sequence} 缺少训练字段: {missing}")

    sample_count = int(arrays["track_ids"].shape[0])
    timeline_count = int(arrays["timeline_scalar_feats"].shape[0])
    if int(arrays["event_indices"].shape[0]) != sample_count:
        raise RuntimeError(f"{sequence} 样本数组长度不一致")

    reid = np.load(reid_path, mmap_mode="r", allow_pickle=False)
    if reid.ndim != 3 or int(reid.shape[0]) != timeline_count or int(reid.shape[1]) != 2:
        raise RuntimeError(f"{sequence} ReID 数组形状异常: {tuple(reid.shape)}")
    reid_shape = tuple(int(x) for x in reid.shape)
    del reid
    return sample_count, timeline_count, reid_shape


def _cleanup_root(root: Path) -> None:
    index = root / "compact_index"
    remaining = [p for p in index.iterdir() if p.is_dir()] if index.is_dir() else []
    if remaining:
        return
    for name in ("metadata.json", "metadata.sha256", "norm_stats.npz"):
        path = root / name
        if path.exists():
            path.unlink()
            print(f"已删除旧根文件: {path}")
    if index.is_dir():
        index.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"已删除旧数据: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", nargs="?", help="序列名；省略时迁移所选 split 的全部序列")
    parser.add_argument("--split", choices=("trainval", "train"), default="trainval")
    parser.add_argument("--keep-source", action="store_true", help="验证后保留旧源数据")
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    source_root = repo / f"outputs/agentguard/datasets/iwg_rg_cma/SportsMOT/nsa_{args.split}_v3_compact"
    output_root = repo / "outputs/agentguard/train_data/SportsMOT"
    detection_root = repo / "outputs/agentguard/detection_cache/SportsMOT"
    event_root = repo / "outputs/agentguard/event_cache_v3_iwg_v2/SportsMOT"
    labels_root = repo / f"outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_{args.split}_v3_compact"

    if not source_root.is_dir():
        raise SystemExit(f"旧 compact 根目录不存在: {source_root}")
    existing_output_metadata = output_root / "metadata.json"
    if existing_output_metadata.is_file():
        existing = json.loads(existing_output_metadata.read_text())
        if existing.get("dataset") not in (None, "SportsMOT"):
            raise SystemExit(f"目标目录不是 SportsMOT: {output_root}")
        existing_split = existing.get("split")
        if existing_split not in (None, args.split):
            raise SystemExit(
                f"目标目录已有 split={existing_split}，不能混入 split={args.split}: {output_root}"
            )
    metadata = json.loads((source_root / "metadata.json").read_text())
    all_sequences = tuple(metadata.get("train_sequences", ()))
    if args.sequence:
        if args.sequence not in all_sequences:
            raise SystemExit(f"序列不在 {args.split} metadata 中: {args.sequence}")
        sequences = (args.sequence,)
    else:
        sequences = all_sequences
    if not sequences:
        raise SystemExit("metadata 中没有可迁移序列")

    sys.path.insert(0, str(repo / "agentguard/src"))
    from agentguard.datasets.iwg_rg_cma_dataset import migrate_compact_iwg_rg_cma_sequence
    from agentguard.datasets.iwg_rg_cma_dataset import (
        SPORTSMOT_TRAIN_SEQUENCES,
    )

    for sequence in sequences:
        target_dir = output_root / sequence
        compact_dir = source_root / "compact_index" / sequence
        if target_dir.exists():
            if not compact_dir.exists() and (target_dir / "data.pt").is_file() and (target_dir / "reid_features.npy").is_file():
                print(f"跳过已完成序列: {sequence}")
                continue
            raise SystemExit(f"目标序列已存在，拒绝覆盖: {target_dir}")

        cache_split = "train" if sequence in SPORTSMOT_TRAIN_SEQUENCES else "val"
        for path in (compact_dir, detection_root / cache_split / sequence):
            if not path.is_dir():
                raise SystemExit(f"迁移所需源目录不存在: {path}")

        print(f"开始迁移 {sequence} ({args.split})")
        migrate_compact_iwg_rg_cma_sequence(
            source_dataset_dir=source_root,
            output_dir=output_root,
            detection_cache_dir=detection_root / cache_split / sequence,
            sequence=sequence,
        )
        samples, timelines, reid_shape = _verify(output_root, sequence)
        print(f"验证成功: {sequence}, samples={samples}, timelines={timelines}, reid_shape={reid_shape}")

        if not args.keep_source:
            _remove_dir(compact_dir)
            # Detection cache is intentionally retained for inference/rebuilds.
            _remove_dir(event_root / cache_split / sequence)
            _remove_dir(labels_root / sequence)

            # trainval contains the train subset again.  Remove that duplicate
            # compact/label copy, but never remove shared detection/event data twice.
            if args.split == "trainval" and sequence in SPORTSMOT_TRAIN_SEQUENCES:
                train_root = repo / "outputs/agentguard/datasets/iwg_rg_cma/SportsMOT/nsa_train_v3_compact"
                train_labels = repo / "outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_train_v3_compact"
                _remove_dir(train_root / "compact_index" / sequence)
                _remove_dir(train_labels / sequence)
                _cleanup_root(train_root)
        else:
            print("保留旧源数据（--keep-source）")
        _cleanup_root(source_root)
        print(f"完成: {target_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
