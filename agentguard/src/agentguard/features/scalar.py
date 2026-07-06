from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import TrackStateSnapshot


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_NUM_FEATURES: int = 63

# Indices where no-detection events must zero out the association block.
_ASSOCIATION_START: int = 0
_ASSOCIATION_END: int = 27  # exclusive  (indices 0-26)

# Default second-minimum value when fewer than 2 elements exist.
_DEFAULT_SECOND_MIN: float = 1.0


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return a float or the default if *value* is ``None`` or NaN."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _safe_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return *value* as an ndarray of *shape*, or zeros if unavailable."""
    if value is None:
        return np.zeros(shape, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(shape, dtype=np.float64)
    # If shape mismatches, try to reshape; fall back to zeros.
    if arr.shape != shape:
        try:
            arr = arr.reshape(shape)
        except ValueError:
            return np.zeros(shape, dtype=np.float64)
    return arr


def _resolve_cost_vector(
    raw: Any,
    fallback: list[float],
) -> np.ndarray:
    """Return a 1-D cost vector.

    If *raw* is a non-empty sequence it is used directly;
    otherwise the *fallback* list (at least a single ``raw_cost`` value) is used.
    """
    if raw is not None and isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=np.float64).ravel()
        if arr.size > 0:
            return arr
    arr = np.asarray(fallback, dtype=np.float64).ravel()
    return arr if arr.size > 0 else np.zeros(1)


def _row_column_stats(
    costs: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Compute row/column statistics from a 1-D cost vector.

    Returns
    -------
    (minimum, second_minimum, diff, mean, std, entropy)
    """
    n = costs.shape[0]
    if n == 0:
        return (0.0, _DEFAULT_SECOND_MIN, _DEFAULT_SECOND_MIN, 0.0, 0.0, 0.0)

    sorted_ = np.sort(costs)
    minimum = float(sorted_[0])
    second_min = float(sorted_[1]) if n >= 2 else _DEFAULT_SECOND_MIN
    diff = second_min - minimum
    mean = float(np.mean(costs))
    std = float(np.std(costs))

    # --- Normalised entropy ---
    if n == 1:
        entropy = 0.0
    else:
        # p_i = exp(-c_i / 0.1) / sum_j(exp(-c_j / 0.1))
        p = np.exp(-costs / 0.1)
        p = p / np.sum(p)
        # H = -sum(p_i * log(p_i + 1e-12)) / log(n)
        entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(n))
        entropy = max(0.0, entropy)  # guard against tiny negatives

    return (minimum, second_min, diff, mean, std, entropy)


def _compute_association_block(
    params: dict[str, Any],
) -> np.ndarray:
    """Compute indices 0-26 (Association Features).

    Returns a (27,) array.
    """
    block = np.zeros(27, dtype=np.float64)

    has_det = params.get("has_detection", False)
    if not has_det:
        return block  # all zeros

    # -- Per-pair costs ------------------------------------------------
    block[0] = _safe_float(params.get("iou_similarity"))
    block[1] = _safe_float(params.get("iou_distance"))
    block[2] = _safe_float(params.get("cosine_distance"))
    block[3] = _safe_float(params.get("confidence_distance"))
    block[4] = _safe_float(params.get("angle_distance"))
    block[5] = _safe_float(params.get("raw_cost"))
    block[6] = _safe_float(params.get("final_cost"))
    block[7] = _safe_float(params.get("assignment_threshold", -1.0))
    # assignment_round: -1 means unmatched
    assign_round = int(params.get("assignment_round", -1))
    if assign_round < 0:
        assign_round = 0
    block[8] = min(assign_round, 10) / 10.0
    block[9] = _safe_float(params.get("det_score"))  # detection score

    # Detection source one-hot
    det_source = int(params.get("detection_source", 0))
    block[10] = 1.0 if det_source == 0 else 0.0
    block[11] = 1.0 if det_source == 1 else 0.0
    block[12] = 1.0 if det_source == 2 else 0.0

    # -- Row statistics (costs for this track against all detections) --
    raw_cost = _safe_float(params.get("raw_cost"), 0.0)
    cost_row = _resolve_cost_vector(params.get("track_cost_row"), [raw_cost])
    row_min, row_2nd, row_diff, row_mean, row_std, row_ent = _row_column_stats(
        cost_row
    )
    block[13] = row_min
    block[14] = row_2nd
    block[15] = row_diff
    block[16] = row_mean
    block[17] = row_std
    block[18] = row_ent

    # -- Column statistics (costs for this detection against all tracks)
    cost_col = _resolve_cost_vector(params.get("detection_cost_col"), [raw_cost])
    col_min, col_2nd, col_diff, col_mean, col_std, col_ent = _row_column_stats(
        cost_col
    )
    block[19] = col_min
    block[20] = col_2nd
    block[21] = col_diff
    block[22] = col_mean
    block[23] = col_std
    block[24] = col_ent

    # -- Per-detection extras -----------------------------------------
    block[25] = _safe_float(params.get("max_iou_with_other"), 0.0)
    block[26] = 1.0  # has_detection

    return block


def _compute_geometry_motion_block(
    params: dict[str, Any],
) -> np.ndarray:
    """Compute indices 27-54 (Geometry & Motion Features).

    Returns a (28,) array.
    """
    block = np.zeros(28, dtype=np.float64)

    has_det = params.get("has_detection", False)

    # -- Predicted track box from KF mean ------------------------------
    track_mean = _safe_array(params.get("track_mean"), (8,))
    pred_cx, pred_cy, pred_w, pred_h = (
        float(track_mean[0]),
        float(track_mean[1]),
        float(track_mean[2]),
        float(track_mean[3]),
    )
    pred_w_safe = max(pred_w, 1.0)
    pred_h_safe = max(pred_h, 1.0)

    im_w = max(float(params.get("image_width", 1)), 1.0)
    im_h = max(float(params.get("image_height", 1)), 1.0)

    # Normalised predicted box (indices 31-34) — always computed
    block[31 - 27] = pred_cx / im_w
    block[32 - 27] = pred_cy / im_h
    block[33 - 27] = pred_w / im_w
    block[34 - 27] = pred_h / im_h

    # -- Velocity components from KF mean (indices 39-42) — always computed
    block[39 - 27] = float(track_mean[4]) / pred_w_safe
    block[40 - 27] = float(track_mean[5]) / pred_h_safe
    block[41 - 27] = float(track_mean[6]) / pred_w_safe
    block[42 - 27] = float(track_mean[7]) / pred_h_safe

    # -- Covariance diagonal elements (indices 43-46) — always computed
    cov = _safe_array(params.get("track_covariance"), (8, 8))
    # P[0,0] .. P[3,3] normalised by (image dimension)²
    block[43 - 27] = math.log(max(cov[0, 0] / (im_w * im_w) + 1e-6, 1e-12))
    block[44 - 27] = math.log(max(cov[1, 1] / (im_h * im_h) + 1e-6, 1e-12))
    block[45 - 27] = math.log(max(cov[2, 2] / (im_w * im_w) + 1e-6, 1e-12))
    block[46 - 27] = math.log(max(cov[3, 3] / (im_h * im_h) + 1e-6, 1e-12))

    # -- Velocity column-major flatten (indices 47-54) — always computed
    velocity = _safe_array(params.get("track_velocity"), (4, 2))
    block[47 - 27 : 55 - 27] = velocity.ravel(order="F")

    # -- Detection-related features (indices 27-30, 35-38) ----
    if has_det:
        det_box = _safe_array(params.get("det_box"), (4,))
        # Detection box in cxcywh format
        det_cx = float((det_box[0] + det_box[2]) / 2.0)
        det_cy = float((det_box[1] + det_box[3]) / 2.0)
        det_w = float(det_box[2] - det_box[0])
        det_h = float(det_box[3] - det_box[1])
        det_w_safe = max(det_w, 1.0)
        det_h_safe = max(det_h, 1.0)

        # Residuals (indices 27-30)
        block[27 - 27] = (det_cx - pred_cx) / pred_w_safe
        block[28 - 27] = (det_cy - pred_cy) / pred_h_safe
        block[29 - 27] = math.log(max(det_w, 1.0) / pred_w_safe)
        block[30 - 27] = math.log(max(det_h, 1.0) / pred_h_safe)

        # Normalised detection box (indices 35-38)
        block[35 - 27] = det_cx / im_w
        block[36 - 27] = det_cy / im_h
        block[37 - 27] = det_w / im_w
        block[38 - 27] = det_h / im_h
    # else: indices 27-42 remain zero (already zero-initialised)

    return block


def _compute_track_context_block(
    params: dict[str, Any],
) -> np.ndarray:
    """Compute indices 55-62 (Track Context Features).

    Returns a (8,) array.
    """
    block = np.zeros(8, dtype=np.float64)

    # 55: current track score
    block[0] = _safe_float(params.get("track_score"))

    # 56-57: history observation scores
    history = params.get("track_history", {})
    if isinstance(history, dict) and len(history) > 0:
        frame_ids = sorted(history.keys())
        # Each history entry: [box, score, mean, cov, feat]
        # score is at index 1
        scores: list[float] = []
        for fid in frame_ids:
            entry = history[fid]
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                sc = _safe_float(entry[1])
                scores.append(sc)

        if scores:
            block[1] = scores[-1]  # most recent
            # Mean of last 3
            last3 = scores[-3:]
            block[2] = float(np.mean(last3))

    # 58: normalised history length
    hist_len = len(history) if isinstance(history, dict) else 0
    block[3] = min(hist_len, 50) / 50.0

    # 59: normalised gap between current frame and last observation
    frame_id = int(params.get("frame_id", 0))
    end_frame_id = int(params.get("track_end_frame_id", 0))
    gap = frame_id - end_frame_id
    block[4] = min(max(gap, 0), 30) / 30.0

    # 60-62: track state one-hot
    state_val = int(params.get("track_state", 0))
    block[5] = 1.0 if state_val == 0 else 0.0  # New
    block[6] = 1.0 if state_val == 1 else 0.0  # Tracked
    block[7] = 1.0 if state_val == 2 else 0.0  # Lost

    return block


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def compute_scalar_features(
    event_or_params: Union[TrackEvent, Dict[str, Any]],
) -> np.ndarray:
    """Compute the 63-dimensional scalar feature vector.

    Accepts either a ``TrackEvent`` instance or a plain dictionary with the
    required fields.

    Required dict fields
    --------------------
    - ``has_detection`` (``bool``)
    - ``image_width``, ``image_height`` (``int`` or ``float``)
    - ``frame_id`` (``int``)

    Optional dict fields (used when available)
    ------------------------------------------
    - ``association`` (``dict`` or flat keys ``iou_similarity``, …)
    - ``detection`` (``dict`` or flat keys ``det_box``, ``det_score``, ``det_source``)
    - ``pre_update_state`` (``dict`` or flat keys ``track_mean``, …)
    - ``track_cost_row`` (``list[float]``) – cost vector for the track row
    - ``detection_cost_col`` (``list[float]``) – cost vector for the detection column
    - ``max_iou_with_other`` (``float``)

    Parameters
    ----------
    event_or_params : TrackEvent or dict
        Input data.

    Returns
    -------
    np.ndarray
        Shape ``(63,)``, dtype ``np.float64``.
    """
    params = _resolve_params(event_or_params)

    assoc = _compute_association_block(params)
    geo_motion = _compute_geometry_motion_block(params)
    track_ctx = _compute_track_context_block(params)

    return np.concatenate([assoc, geo_motion, track_ctx]).astype(np.float64)


# ---------------------------------------------------------------------------
#  Input resolution
# ---------------------------------------------------------------------------


def _resolve_params(
    event_or_params: Union[TrackEvent, Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalise a ``TrackEvent`` or ``dict`` into a flat parameter dict."""
    if isinstance(event_or_params, TrackEvent):
        return _event_to_params(event_or_params)
    if isinstance(event_or_params, dict):
        return _normalise_dict(event_or_params)
    raise TypeError(
        f"Expected TrackEvent or dict, got {type(event_or_params).__name__}"
    )


def _event_to_params(ev: TrackEvent) -> Dict[str, Any]:
    """Flatten a ``TrackEvent`` into the parameter dict expected by the
    compute helpers."""
    params: Dict[str, Any] = {
        "has_detection": ev.has_detection,
        "image_width": ev.image_width,
        "image_height": ev.image_height,
        "frame_id": ev.frame_id,
    }

    # Association pair data
    assoc = ev.association
    if assoc is not None:
        params["iou_similarity"] = assoc.iou_similarity
        params["iou_distance"] = assoc.iou_distance
        params["cosine_distance"] = assoc.cosine_distance
        params["confidence_distance"] = assoc.confidence_distance
        params["angle_distance"] = assoc.angle_distance
        params["raw_cost"] = assoc.raw_cost
        params["final_cost"] = assoc.final_cost
        params["assignment_round"] = assoc.assignment_round
        params["assignment_threshold"] = assoc.assignment_threshold
        params["detection_source"] = assoc.detection_source

        # Prefer full AssociationContext row/col when available
        ctx = getattr(ev, 'association_context', None)
        if ctx is not None:
            params["track_cost_row"] = np.asarray(ctx.track_cost_row, dtype=np.float64)
            params["detection_cost_col"] = np.asarray(ctx.detection_cost_col, dtype=np.float64)
            params["max_iou_with_other"] = float(np.max(ctx.detection_overlap_row)) if ctx.detection_overlap_row.size > 0 else 0.0

    # Detection data
    det = ev.detection
    if det is not None:
        params["det_box"] = det.box
        params["det_score"] = det.score
        params["det_source"] = det.source

    # Track snapshot data
    state: Optional[TrackStateSnapshot] = ev.pre_update_state
    if state is not None:
        params["track_mean"] = state.mean
        params["track_covariance"] = state.covariance
        params["track_velocity"] = state.velocity
        params["track_score"] = state.score
        params["track_history"] = state.history
        params["track_end_frame_id"] = state.end_frame_id
        params["track_state"] = state.state

    # Extra fields that may be attached to the event at runtime
    for extra in ("track_cost_row", "detection_cost_col", "max_iou_with_other"):
        val = getattr(ev, extra, None)
        if val is not None and extra not in params:
            params[extra] = val

    if "max_iou_with_other" not in params:
        params["max_iou_with_other"] = 0.0

    return params


def _normalise_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a dictionary that may contain nested ``association``,
    ``detection``, or ``pre_update_state`` sub-dicts."""
    params: Dict[str, Any] = {
        "has_detection": bool(d.get("has_detection", False)),
        "image_width": int(d.get("image_width", 1)),
        "image_height": int(d.get("image_height", 1)),
        "frame_id": int(d.get("frame_id", 0)),
    }

    # Unpack nested association dict
    assoc = d.get("association")
    if isinstance(assoc, dict):
        params.update(
            {
                "iou_similarity": _safe_float(assoc.get("iou_similarity")),
                "iou_distance": _safe_float(assoc.get("iou_distance")),
                "cosine_distance": _safe_float(assoc.get("cosine_distance")),
                "confidence_distance": _safe_float(assoc.get("confidence_distance")),
                "angle_distance": _safe_float(assoc.get("angle_distance")),
                "raw_cost": _safe_float(assoc.get("raw_cost")),
                "final_cost": _safe_float(assoc.get("final_cost")),
                "assignment_round": int(assoc.get("assignment_round", -1)),
                "assignment_threshold": _safe_float(
                    assoc.get("assignment_threshold", -1.0)
                ),
                "detection_source": int(assoc.get("detection_source", 0)),
            }
        )
        raw = _safe_float(assoc.get("raw_cost"), 0.0)
        params.setdefault("track_cost_row", [raw])
        params.setdefault("detection_cost_col", [raw])
    else:
        # Flat keys
        for k in (
            "iou_similarity",
            "iou_distance",
            "cosine_distance",
            "confidence_distance",
            "angle_distance",
            "raw_cost",
            "final_cost",
            "assignment_threshold",
            "detection_source",
        ):
            if k in d:
                params[k] = _safe_float(d[k])
        if "assignment_round" in d:
            params["assignment_round"] = int(d["assignment_round"])
        raw = _safe_float(d.get("raw_cost", 0.0), 0.0)
        params.setdefault("track_cost_row", [raw])
        params.setdefault("detection_cost_col", [raw])

    # Unpack nested detection dict
    det = d.get("detection")
    if isinstance(det, dict):
        params["det_box"] = np.asarray(det.get("box", np.zeros(4)), dtype=np.float64)
        params["det_score"] = _safe_float(det.get("score"))
        params["det_source"] = int(det.get("source", 0))
    else:
        if "det_box" in d:
            params["det_box"] = np.asarray(d["det_box"], dtype=np.float64)
        params["det_score"] = _safe_float(d.get("det_score", 0.0))
        params["det_source"] = int(d.get("det_source", 0))

    # Unpack nested pre_update_state dict
    state = d.get("pre_update_state")
    if isinstance(state, dict):
        params["track_mean"] = (
            np.asarray(state["mean"], dtype=np.float64) if state.get("mean") is not None else None
        )
        params["track_covariance"] = (
            np.asarray(state["covariance"], dtype=np.float64)
            if state.get("covariance") is not None
            else None
        )
        params["track_velocity"] = np.asarray(
            state.get("velocity", np.zeros((4, 2))), dtype=np.float64
        )
        params["track_score"] = _safe_float(state.get("score"))
        params["track_history"] = state.get("history", {})
        params["track_end_frame_id"] = int(state.get("end_frame_id", 0))
        params["track_state"] = int(state.get("state", 0))
    else:
        if "track_mean" in d:
            params["track_mean"] = np.asarray(d["track_mean"], dtype=np.float64)
        if "track_covariance" in d:
            params["track_covariance"] = np.asarray(
                d["track_covariance"], dtype=np.float64
            )
        if "track_velocity" in d:
            params["track_velocity"] = np.asarray(
                d["track_velocity"], dtype=np.float64
            )
        params["track_score"] = _safe_float(d.get("track_score", 0.0))
        params["track_history"] = d.get("track_history", {})
        params["track_end_frame_id"] = int(d.get("track_end_frame_id", 0))
        params["track_state"] = int(d.get("track_state", 0))

    # Cost row/col and max_iou passed at top level
    for k in ("track_cost_row", "detection_cost_col", "max_iou_with_other"):
        if k in d:
            params[k] = d[k]

    params.setdefault("max_iou_with_other", 0.0)

    return params
