from __future__ import annotations

import numpy as np


def iou_loss(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """1 - IoU between two ``(4,)`` boxes in ``x1y1x2y2`` format.

    Parameters
    ----------
    box_a : ndarray, shape ``(4,)``
    box_b : ndarray, shape ``(4,)``

    Returns
    -------
    float
        ``1.0 - intersection / union`` (0 when boxes perfectly overlap).
    """
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 1.0
    return 1.0 - inter / union


def l1_normalized_loss(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Normalised L1 loss.

    .. math::
        \\frac{1}{4} \\sum_{i \\in \\{x_1,y_1,x_2,y_2\\}}
            \\frac{|\\,a_i - b_i\\,|}{\\text{span}_i}

    where width = ``b[2] - b[0]`` and height = ``b[3] - b[1]``.

    Parameters
    ----------
    box_a : ndarray, shape ``(4,)``
    box_b : ndarray, shape ``(4,)``  (reference — its width/height are used
        for normalisation).

    Returns
    -------
    float
    """
    w = max(float(box_b[2] - box_b[0]), 1e-6)
    h = max(float(box_b[3] - box_b[1]), 1e-6)
    loss = (
        abs(box_a[0] - box_b[0]) / w
        + abs(box_a[1] - box_b[1]) / h
        + abs(box_a[2] - box_b[2]) / w
        + abs(box_a[3] - box_b[3]) / h
    )
    return float(loss / 4.0)


def motion_frame_loss(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Per-frame motion loss: ``1 - IoU + 0.25 * L1_normalised``.

    Parameters
    ----------
    box_a : ndarray, shape ``(4,)``
    box_b : ndarray, shape ``(4,)``

    Returns
    -------
    float
    """
    return iou_loss(box_a, box_b) + 0.25 * l1_normalized_loss(box_a, box_b)


def appearance_loss(feat: np.ndarray, prototype: np.ndarray) -> float:
    """Appearance loss: ``1 - cosine_similarity(feat, prototype)``.

    Parameters
    ----------
    feat : ndarray, shape ``(D,)``
        The appearance feature vector.
    prototype : ndarray, shape ``(D,)``
        The identity prototype vector.

    Returns
    -------
    float
        Loss in ``[0, 2]``.  Returns ``1.0`` when either vector is zero-norm.
    """
    feat_norm = np.linalg.norm(feat)
    proto_norm = np.linalg.norm(prototype)
    if feat_norm < 1e-12 or proto_norm < 1e-12:
        return 1.0
    cos_sim = float(np.dot(feat, prototype) / (feat_norm * proto_norm))
    return 1.0 - cos_sim


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    Parameters
    ----------
    x : float

    Returns
    -------
    float
        ``1 / (1 + exp(-x))``, clipped to ``[-50, 50]`` for stability.
    """
    x_clipped = np.clip(x, -50.0, 50.0)
    return float(1.0 / (1.0 + np.exp(-x_clipped)))
