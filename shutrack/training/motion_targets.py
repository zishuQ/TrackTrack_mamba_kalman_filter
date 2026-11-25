"""GT-derived continuous motion-gate targets for an explicitly selected experiment."""
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

GRID = np.linspace(1.0, 0.0, 21, dtype=np.float64)
TIE_TOLERANCE = 1e-6
SOURCE_KEYS = ("frame", "current_event_index", "pred_box_g0", "pred_box_g1",
               "gt_box_t1", "motion_gt_valid")


def source_fingerprint(arrays):
    digest = hashlib.sha256()
    for key in SOURCE_KEYS:
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def numpy_diou(a, b):
    lo = np.maximum(a[..., :2], b[..., :2])
    hi = np.minimum(a[..., 2:], b[..., 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=-1)
    area_a = np.maximum(a[..., 2:] - a[..., :2], 0).prod(axis=-1)
    area_b = np.maximum(b[..., 2:] - b[..., :2], 0).prod(axis=-1)
    iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
    center_a = (a[..., :2] + a[..., 2:]) / 2
    center_b = (b[..., :2] + b[..., 2:]) / 2
    enclosing = np.maximum(a[..., 2:], b[..., 2:]) - np.minimum(a[..., :2], b[..., :2])
    return iou - ((center_a - center_b)**2).sum(axis=-1) / np.maximum(
        (enclosing**2).sum(axis=-1), 1e-9)


def generate_targets(box0, box1, gt, valid, *, block_size=16384):
    """Choose the largest grid gate within 1e-6 of each event's minimum loss."""
    target = np.ones(len(valid), dtype=np.float32)
    indices = np.flatnonzero(valid)
    for start in range(0, len(indices), block_size):
        rows = indices[start:start + block_size]
        a = np.asarray(box0[rows], dtype=np.float64)
        delta = np.asarray(box1[rows], dtype=np.float64) - a
        boxes = a[None] + GRID[:, None, None] * delta[None]
        losses = 1 - numpy_diou(boxes, np.asarray(gt[rows], dtype=np.float64)[None])
        best = losses.min(axis=0)
        chosen = (losses <= best[None] + TIE_TOLERANCE).argmax(axis=0)
        target[rows] = GRID[chosen].astype(np.float32)
    return target


def motion_target_bce(outputs, batch, *, weight=1.0, harm_weight=1.0):
    if harm_weight != 1.0:
        raise ValueError("Motion target BCE uses unit event weights")
    logits = outputs["gate_logits"].float()[:, 0]
    target = batch["motion_gate_target"].float()
    valid = batch["motion_gt_valid"].bool()
    error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (error * valid).sum() / valid.sum().clamp_min(1)
    return loss * weight, {"loss": loss * weight, "motion_target_bce": loss.detach()}
