"""Feature and model schema for SHUTrack v3 (HCE + RSC).

Inputs are split into two explicit branches:

* **evidence (13 dims)** — fused motion and observation context:
  4 box residuals, 4 KF velocities, 2 center prediction variances,
  detection confidence, observation gap, has-detection flag.
* **appearance** — pre-update track ReID and the event's detection ReID.
* **competition (3 dims, optional branch)** — legacy 63-dim columns
  15/21/25: row top-2 cost gap, column top-2 cost gap, and the maximum IoU
  between the accepted detection and the other detections in the same pool.

The values here describe the *new* model input schema.  They are deliberately
independent from the legacy AgentGuard cache/label schema: nothing in this
module changes or re-derives the old ``FEATURE_SCHEMA_SHA256``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ARCHITECTURE_NAME = "shutrack_hce_rsc"
ARCHITECTURE_VERSION = "shutrack_v3"
FEATURE_SCHEMA_VERSION = "shutrack_v3_evidence13_competition3"

# Six events = five past events + the current event.
CONTEXT_SIZE = 6
HISTORY_SIZE = CONTEXT_SIZE - 1

# Evidence branch: 12 normalized legacy scalar columns + a raw 0/1 flag.
EVIDENCE_SCALAR_INDICES = (
    27, 28, 29, 30,   # box residual: center offset (2) + log w/h ratio (2)
    39, 40, 41, 42,   # normalized KF velocity (4)
    43, 44,           # center prediction variance: log var(cx), log var(cy)
    9,                # detection confidence
    59,               # normalized observation gap
)
EVIDENCE_FIELDS = (
    "box_residual_cx",
    "box_residual_cy",
    "box_log_ratio_w",
    "box_log_ratio_h",
    "kf_velocity_cx",
    "kf_velocity_cy",
    "kf_velocity_w",
    "kf_velocity_h",
    "center_variance_cx_log",
    "center_variance_cy_log",
    "detection_confidence",
    "observation_gap",
    "has_detection",
)
EVIDENCE_DIM = len(EVIDENCE_FIELDS)  # 13

# Competition branch: legacy 63-dim columns 15, 21, 25.
COMPETITION_SCALAR_INDICES = (15, 21, 25)
COMPETITION_FIELDS = (
    "row_top2_cost_gap",
    "col_top2_cost_gap",
    "max_iou_with_other",
)
COMPETITION_DIM = len(COMPETITION_FIELDS)  # 3

# The legacy has-detection column is kept as a separate explicit flag for
# appearance masking and for the final evidence dimension.
HAS_DETECTION_SCALAR_INDEX = 26

LEGACY_SCALAR_DIM = 63
DEFAULT_REID_DIM = 2048

MAX_FRAME_GAP = 30

FEATURE_SCHEMA: dict[str, Any] = {
    "name": "shutrack_v3_evidence_competition_features",
    "version": FEATURE_SCHEMA_VERSION,
    "context_size": CONTEXT_SIZE,
    "history_size": HISTORY_SIZE,
    "evidence_scalar_indices": list(EVIDENCE_SCALAR_INDICES),
    "evidence_fields": list(EVIDENCE_FIELDS),
    "evidence_dim": EVIDENCE_DIM,
    "competition_scalar_indices": list(COMPETITION_SCALAR_INDICES),
    "competition_fields": list(COMPETITION_FIELDS),
    "competition_dim": COMPETITION_DIM,
    "has_detection_flag": "raw 0/1; appended as the 13th evidence value",
    "appearance": "pre_update_track_reid_and_event_detection_reid",
    "scalar_normalization": (
        "legacy 63-dim train-split z-score; packed scalars are already "
        "normalized and must not be normalized again"
    ),
}

FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def feature_schema_descriptor() -> dict[str, Any]:
    """Return a JSON-serializable copy of the feature schema."""

    return json.loads(json.dumps(FEATURE_SCHEMA))


__all__ = [
    "ARCHITECTURE_NAME",
    "ARCHITECTURE_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_SHA256",
    "CONTEXT_SIZE",
    "HISTORY_SIZE",
    "EVIDENCE_SCALAR_INDICES",
    "EVIDENCE_FIELDS",
    "EVIDENCE_DIM",
    "COMPETITION_SCALAR_INDICES",
    "COMPETITION_FIELDS",
    "COMPETITION_DIM",
    "HAS_DETECTION_SCALAR_INDEX",
    "LEGACY_SCALAR_DIM",
    "DEFAULT_REID_DIM",
    "MAX_FRAME_GAP",
    "feature_schema_descriptor",
]
