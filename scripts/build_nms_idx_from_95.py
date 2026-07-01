#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3. Tracker"))

from utils.det_feat_storage import compact_index_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact target-NMS index by applying NMS directly on existing "
            "source-NMS FastReID pickle files. No target pickle is read or written."
        )
    )
    parser.add_argument("--det-feat-dir", default="outputs/2. det_feat")
    parser.add_argument("--prefix", default=None, help="Only process one prefix, e.g. mot17_test")
    parser.add_argument("--nms-thr", type=float, default=0.80)
    parser.add_argument("--source-nms", default="0.95")
    parser.add_argument("--target-nms", default="0.80", help="Logical target NMS used in idx/output names")
    parser.add_argument("--score-col", type=int, default=4)
    parser.add_argument("--class-col", type=int, default=5)
    parser.add_argument("--class-agnostic", action="store_true", help="Ignore class column during NMS")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing loader-compatible idx file")
    parser.add_argument("--backup-existing", action="store_true", help="Rename existing idx to *.bak before overwriting")
    parser.add_argument("--delete-80", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def bbox_iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area_a + area_b - inter, 1e-12)
    return inter / union


def nms_indices(boxes, scores, nms_thr):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    # Stable tie-breaking: preserve original row order among equal scores.
    order = np.argsort(-scores, kind="mergesort").astype(np.int64)
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        ious = bbox_iou_one_to_many(boxes[current], boxes[rest])
        order = rest[ious <= nms_thr]
    return np.asarray(keep, dtype=np.int32)


def frame_nms_indices(dets, nms_thr, score_col, class_col, class_agnostic):
    if dets is None:
        return None
    if len(dets) == 0:
        return np.empty((0,), dtype=np.int32)

    boxes = np.asarray(dets[:, :4], dtype=np.float32)
    scores = np.asarray(dets[:, score_col], dtype=np.float32)

    if class_agnostic:
        return nms_indices(boxes, scores, nms_thr)

    classes = np.asarray(dets[:, class_col])
    keep_parts = []
    for cls in np.unique(classes):
        cls_indices = np.flatnonzero(classes == cls)
        cls_keep = nms_indices(boxes[cls_indices], scores[cls_indices], nms_thr)
        keep_parts.append(cls_indices[cls_keep])

    if not keep_parts:
        return np.empty((0,), dtype=np.int32)

    keep = np.concatenate(keep_parts).astype(np.int32)
    # Match batched_nms behavior: final kept rows are globally sorted by score.
    order = np.argsort(-scores[keep], kind="mergesort")
    return keep[order].astype(np.int32)


def build_nms_index_from_95(detections_95, nms_thr, score_col, class_col, class_agnostic):
    index = {}
    total_95 = 0
    total_target = 0
    none_frames = 0
    for vid_name, frames in detections_95.items():
        index[vid_name] = {}
        for frame_id, dets in frames.items():
            frame_index = frame_nms_indices(
                dets,
                nms_thr=nms_thr,
                score_col=score_col,
                class_col=class_col,
                class_agnostic=class_agnostic,
            )
            index[vid_name][frame_id] = frame_index
            if dets is None:
                none_frames += 1
            else:
                total_95 += len(dets)
                total_target += len(frame_index)

    return {
        "version": 2,
        "source": "nms_from_95",
        "nms_thr": nms_thr,
        "score_col": score_col,
        "class_col": None if class_agnostic else class_col,
        "class_agnostic": class_agnostic,
        "total_95": total_95,
        "total": total_target,
        "none_frames": none_frames,
        "index": index,
    }


def expected_paths(det_feat_dir, prefix, source_nms, target_nms):
    path_95 = det_feat_dir / f"{prefix}_{source_nms}.pickle"
    path_target = det_feat_dir / f"{prefix}_{target_nms}.pickle"
    idx_path = Path(compact_index_path(str(path_target), str(path_95)))
    return path_95, path_target, idx_path


def main():
    args = parse_args()
    det_feat_dir = Path(args.det_feat_dir)

    paths_95 = sorted(det_feat_dir.glob(f"*_{args.source_nms}.pickle"))
    if args.prefix:
        paths_95 = [p for p in paths_95 if p.name == f"{args.prefix}_{args.source_nms}.pickle"]
    if not paths_95:
        raise SystemExit(f"No *_{args.source_nms}.pickle files matched.")

    for path_95 in paths_95:
        prefix = path_95.name[: -len(f"_{args.source_nms}.pickle")]
        _, path_target, idx_path = expected_paths(det_feat_dir, prefix, args.source_nms, args.target_nms)

        if idx_path.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"{idx_path} already exists. Use --overwrite to replace it, "
                    "or move it aside first."
                )
            if args.backup_existing:
                backup_path = idx_path.with_suffix(idx_path.suffix + ".bak")
                if backup_path.exists():
                    raise SystemExit(f"Backup path already exists: {backup_path}")
                os.rename(idx_path, backup_path)
                print(f"backed up existing idx to {backup_path}")

        print(f"loading {path_95}")
        with path_95.open("rb") as f:
            detections_95 = pickle.load(f)

        compact_index = build_nms_index_from_95(
            detections_95,
            nms_thr=args.nms_thr,
            score_col=args.score_col,
            class_col=args.class_col,
            class_agnostic=args.class_agnostic,
        )

        with idx_path.open("wb") as f:
            pickle.dump(compact_index, f, protocol=pickle.HIGHEST_PROTOCOL)

        ratio = compact_index["total"] / compact_index["total_95"] if compact_index["total_95"] else 0.0
        print(
            f"wrote {idx_path} "
            f"({compact_index['total']} kept / {compact_index['total_95']} source rows, "
            f"ratio={ratio:.4f}, target view={path_target.name})"
        )


if __name__ == "__main__":
    main()
