"""Feature and model schema for SHUTrack v2 (HCE + RSC).

The values here describe the *new* model input schema.  They are deliberately
independent from the legacy AgentGuard cache/label schema: nothing in this
module changes or re-derives the old ``FEATURE_SCHEMA_SHA256``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ARCHITECTURE_NAME = "shutrack_hce_rsc"
ARCHITECTURE_VERSION = "shutrack_v2"
FEATURE_SCHEMA_VERSION = "shutrack_v2_event18"

# Six events = five past events + the current event.
CONTEXT_SIZE = 6
HISTORY_SIZE = CONTEXT_SIZE - 1

# Motion evidence: legacy 63-dim scalar indices 27:31, 39:43, 43:47.
MOTION_SCALAR_INDICES = (
    27, 28, 29, 30,          # detection center offset (2) + log w/h ratio (2)
    39, 40, 41, 42,          # normalized KF velocity (4)
    43, 44, 45, 46,          # normalized log covariance diagonal (4)
)
MOTION_DIM = len(MOTION_SCALAR_INDICES)

# Context evidence: four legacy scalars plus two selected-margin values and
# two margin-validity bits.  Order is part of the schema.
SCALAR_CONTEXT_INDICES = (
    9,    # detection score
    58,   # normalized history length
    59,   # normalized observation gap
    26,   # has_detection (legacy scalar value, in normalized space)
)
CONTEXT_ORDER = (
    "det_score",
    "row_margin",
    "col_margin",
    "history_length",
    "observation_gap",
    "has_detection",
    "row_margin_valid",
    "col_margin_valid",
)
CONTEXT_DIM = len(CONTEXT_ORDER)

LEGACY_SCALAR_DIM = 63
DEFAULT_REID_DIM = 2048

# Margin definition: the accepted pair is compared against the recorded
# candidate pool of the same raw-cost matrix.  Candidate columns whose IoU
# similarity is at or below this floor are infeasible by construction of the
# legacy association step and are excluded from the margin pool.
MARGIN_INFEASIBLE_IOU = 0.10

MAX_FRAME_GAP = 30

FEATURE_SCHEMA: dict[str, Any] = {
    "name": "shutrack_v2_structured_event_features",
    "version": FEATURE_SCHEMA_VERSION,
    "context_size": CONTEXT_SIZE,
    "history_size": HISTORY_SIZE,
    "motion_scalar_indices": list(MOTION_SCALAR_INDICES),
    "motion_dim": MOTION_DIM,
    "scalar_context_indices": list(SCALAR_CONTEXT_INDICES),
    "context_order": list(CONTEXT_ORDER),
    "context_dim": CONTEXT_DIM,
    "margin_definition": (
        "min(other feasible raw-cost candidates) - accepted raw cost"
    ),
    "margin_infeasible_iou": MARGIN_INFEASIBLE_IOU,
    "margin_validity_bits": ["row_margin_valid", "col_margin_valid"],
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
    "MOTION_SCALAR_INDICES",
    "MOTION_DIM",
    "SCALAR_CONTEXT_INDICES",
    "CONTEXT_ORDER",
    "CONTEXT_DIM",
    "LEGACY_SCALAR_DIM",
    "DEFAULT_REID_DIM",
    "MARGIN_INFEASIBLE_IOU",
    "MAX_FRAME_GAP",
    "feature_schema_descriptor",
]
