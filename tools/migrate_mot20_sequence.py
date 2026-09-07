#!/usr/bin/env python3
"""Migrate one MOT20 compact sequence to the packed ``train_data`` layout.

The source sequence is removed only after the destination has been written and
validated.  With no positional argument, all MOT20 sequences are processed in
order; passing a sequence name processes only that sequence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


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
        "event_indices",
        "track_ids",
        "segment_ids",
        "safe_gate_target",
        "policy_safe_soft_target",
        "cue_target",
        "risk_target",
        "valid_channels",
        "sample_weight",
        "timeline_scalar_feats",
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
        raise RuntimeError(
            f"{sequence} ReID 数组形状异常: {tuple(reid.shape)}, "
            f"expected ({timeline_count}, 2, reid_dim)"
        )
    reid_shape = tuple(int(x) for x in reid.shape)
    del reid
    return sample_count, timeline_count, reid_shape


def _remove_source_dirs(
    *,
    source_root: Path,
    event_root: Path,
    labels_root: Path,
    sequence: str,
) -> None:
    paths = (
        source_root / "compact_index" / sequence,
        # Detection cache is an inference asset and is intentionally retained.
        event_root / sequence,
        labels_root / sequence,
    )
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            print(f"已删除旧数据: {path}")

    # Only remove legacy root metadata after every compact sequence is gone.
    compact_index = source_root / "compact_index"
    remaining = [p for p in compact_index.iterdir() if p.is_dir()] if compact_index.is_dir() else []
    if not remaining:
        for name in ("metadata.json", "metadata.sha256", "norm_stats.npz"):
            path = source_root / name
            if path.exists():
                path.unlink()
                print(f"已删除旧根文件: {path}")
        if compact_index.is_dir():
            compact_index.rmdir()
        if source_root.is_dir() and not any(source_root.iterdir()):
            source_root.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        nargs="?",
        choices=SEQUENCES,
        help="要迁移的 MOT20 序列；省略时按顺序迁移全部序列",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="只迁移并验证，不删除旧 compact/detection/event/labels 数据",
    )
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    source_root = repo / "outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact"
    output_root = repo / "outputs/agentguard/train_data/MOT20"
    detection_root = repo / "outputs/agentguard/detection_cache/MOT20/all"
    event_root = repo / "outputs/agentguard/event_cache_v3_iwg_v2/MOT20/all"
    labels_root = repo / "outputs/agentguard/labels/iwg_rg_cma/MOT20/nsa_v3_compact"

    if not source_root.is_dir():
        raise SystemExit(f"旧 compact 根目录不存在: {source_root}")
    import sys

    sys.path.insert(0, str(repo / "agentguard/src"))
    from agentguard.datasets.iwg_rg_cma_dataset import migrate_compact_iwg_rg_cma_sequence

    sequences = (args.sequence,) if args.sequence else SEQUENCES
    for sequence in sequences:
        target_dir = output_root / sequence
        compact_dir = source_root / "compact_index" / sequence
        if target_dir.exists():
            if not compact_dir.exists() and (target_dir / "data.pt").is_file() and (target_dir / "reid_features.npy").is_file():
                print(f"跳过已完成序列: {sequence}")
                continue
            raise SystemExit(f"目标序列已存在，拒绝覆盖: {target_dir}")
        for path in (compact_dir, detection_root / sequence):
            if not path.is_dir():
                raise SystemExit(f"迁移所需源目录不存在: {path}")

        print(f"开始迁移 {sequence}")
        migrate_compact_iwg_rg_cma_sequence(
            source_dataset_dir=source_root,
            output_dir=output_root,
            detection_cache_dir=detection_root / sequence,
            sequence=sequence,
        )
        samples, timelines, reid_shape = _verify(output_root, sequence)
        print(f"验证成功: {sequence}, samples={samples}, timelines={timelines}, reid_shape={reid_shape}")

        if not args.keep_source:
            _remove_source_dirs(
                source_root=source_root,
                event_root=event_root,
                labels_root=labels_root,
                sequence=sequence,
            )
        else:
            print("保留旧源数据（--keep-source）")
        print(f"完成: {output_root / sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
