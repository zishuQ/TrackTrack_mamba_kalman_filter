from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.enums import POLICY_PROTOTYPE_MATRIX
from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import TrackStateSnapshot
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from agentguard.runtime.replay import ReplayEngine

# The five policies' gates
_POLICY_GATES: Dict[str, np.ndarray] = {
    "FULL_WRITE": POLICY_PROTOTYPE_MATRIX[0],
    "MOTION_ONLY": POLICY_PROTOTYPE_MATRIX[1],
    "APPEARANCE_ONLY": POLICY_PROTOTYPE_MATRIX[2],
    "HOLD_BOTH": POLICY_PROTOTYPE_MATRIX[3],
    "SOFT_CAUTION": POLICY_PROTOTYPE_MATRIX[4],
}


class LocalReplayVerifier:
    """Verifies Teacher decisions by locally replaying events with each policy.

    For each event, replays with the 5 policies (**FULL_WRITE**, **MOTION_ONLY**,
    **APPEARANCE_ONLY**, **HOLD_BOTH**, **SOFT_CAUTION**) and computes the
    relative benefit for each.

    The relative benefit delta R_m for policy *m* is::

        Delta R_m = 0.5 * (L_full - L_m) / max(|L_full|, 1e-6)
                  + 0.5 * (L_full_app - L_m_app) / max(|L_full_app|, 1e-6)

    where L is the localisation loss and L_app is the appearance loss.

    Parameters
    ----------
    motion_model : NSAKalmanFilter, optional
        Motion model for replay.  If ``None``, a default instance is created.
    identity_prototypes : dict
        Mapping ``track_id -> prototype vector`` (ReID prototype).
    """

    def __init__(
        self,
        motion_model: Optional[NSAKalmanFilter] = None,
        identity_prototypes: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        self.engine = ReplayEngine(motion_model=motion_model)
        self.identity_prototypes = identity_prototypes or {}

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def verify_event(
        self,
        event: TrackEvent,
        future_oracle_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replay an event with all 5 policies and compute deltas.

        Parameters
        ----------
        event : TrackEvent
            The event to verify.
        future_oracle_data : dict
            From :class:`~agentguard.data.future_oracle.FutureOracleBuilder`.
            Must contain keys ``current_gt_box``, ``future_gt_boxes``,
            ``future_oracle_detections``, ``future_warp_matrices``.

        Returns
        -------
        dict
            Keys::

                "policy_deltas": dict of {policy_name: delta_R}
                "policy_losses": dict of {policy_name: {"loc_loss": float,
                                                         "app_loss": float}}
                "full_loss": {"loc_loss": float, "app_loss": float}
        """
        if event.pre_update_state is None:
            raise ValueError(
                f"Event {event.event_id} has no pre_update_state; "
                f"cannot replay."
            )

        # Snapshot before the event
        snapshot_before = event.pre_update_state

        # Gather future GT and oracle info
        current_gt = future_oracle_data.get("current_gt_box")
        future_gt_boxes: List[np.ndarray] = future_oracle_data.get(
            "future_gt_boxes", []
        )
        future_oracle_detections: List[np.ndarray] = future_oracle_data.get(
            "future_oracle_detections", []
        )

        results: Dict[str, Any] = {
            "policy_deltas": {},
            "policy_losses": {},
            "full_loss": None,
        }

        # Replay with each policy
        full_loss: Optional[Dict[str, float]] = None

        for policy_name, gate in _POLICY_GATES.items():
            snapshot_after = self.engine.replay_event(
                track_snapshot=snapshot_before,
                event=event,
                revised_gate=gate,
            )

            loss = self._compute_loss(
                snapshot_after=snapshot_after,
                event=event,
                current_gt=current_gt,
                future_gt_boxes=future_gt_boxes,
                future_oracle_detections=future_oracle_detections,
            )

            results["policy_losses"][policy_name] = loss

            if policy_name == "FULL_WRITE":
                full_loss = loss
                results["full_loss"] = loss

        # Compute deltas relative to FULL_WRITE
        if full_loss is not None:
            for policy_name, loss in results["policy_losses"].items():
                if policy_name == "FULL_WRITE":
                    results["policy_deltas"][policy_name] = 0.0
                    continue
                delta = self._compute_delta(full_loss, loss)
                results["policy_deltas"][policy_name] = delta

        return results

    # ------------------------------------------------------------------
    #  Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        snapshot_after: TrackStateSnapshot,
        event: TrackEvent,
        current_gt: Optional[np.ndarray],
        future_gt_boxes: List[np.ndarray],
        future_oracle_detections: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute localisation and appearance loss for a replayed snapshot.

        Parameters
        ----------
        snapshot_after : TrackStateSnapshot
            Track state after replaying the event.
        event : TrackEvent
            The original event (used for feature extraction).
        current_gt : ndarray or None
            Ground-truth box at the event frame ``(4,)``.
        future_gt_boxes : list of ndarray
            Ground-truth boxes for future frames.
        future_oracle_detections : list of ndarray
            Oracle detection boxes for future frames.

        Returns
        -------
        dict with ``loc_loss`` (float) and ``app_loss`` (float).
        """
        # --- Localisation loss: IoU-based ---
        # Loss at current frame
        current_loc_loss = 0.0
        if current_gt is not None:
            pred_box = snapshot_after.box
            iou = self._iou(pred_box, current_gt)
            current_loc_loss = 1.0 - iou

        # Future frame localisation loss (mean over future frames)
        future_loc_losses: List[float] = []
        for gt_box, oracle_box in zip(future_gt_boxes, future_oracle_detections):
            if gt_box is not None and oracle_box is not None:
                pred_box = oracle_box  # use oracle as the "replayed" prediction
                iou = self._iou(pred_box, gt_box)
                future_loc_losses.append(1.0 - iou)

        mean_future_loc = float(np.mean(future_loc_losses)) if future_loc_losses else 0.0
        loc_loss = 0.5 * current_loc_loss + 0.5 * mean_future_loc

        # --- Appearance loss: cosine distance to prototype ---
        app_loss = 0.0
        track_id = snapshot_after.track_id
        prototype = self.identity_prototypes.get(track_id)
        if prototype is not None:
            feat = snapshot_after.feature.ravel()
            if np.linalg.norm(feat) > 1e-12 and np.linalg.norm(prototype) > 1e-12:
                cos_sim = float(np.dot(feat, prototype))
                app_loss = 1.0 - cos_sim

        return {"loc_loss": float(loc_loss), "app_loss": float(app_loss)}

    # ------------------------------------------------------------------
    #  Delta computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta(
        full_loss: Dict[str, float],
        policy_loss: Dict[str, float],
    ) -> float:
        """Compute the relative benefit delta for a policy vs FULL_WRITE.

        Formula::

            Delta R_m = 0.5 * (L_full - L_m) / max(|L_full|, 1e-6)
                      + 0.5 * (L_full_app - L_m_app) / max(|L_full_app|, 1e-6)

        Returns
        -------
        float
            Positive values indicate that *policy* improved over FULL_WRITE.
        """
        loc_diff = full_loss["loc_loss"] - policy_loss["loc_loss"]
        app_diff = full_loss["app_loss"] - policy_loss["app_loss"]

        loc_denom = max(abs(full_loss["loc_loss"]), 1e-6)
        app_denom = max(abs(full_loss["app_loss"]), 1e-6)

        delta = 0.5 * (loc_diff / loc_denom) + 0.5 * (app_diff / app_denom)
        return float(delta)

    # ------------------------------------------------------------------
    #  Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two ``(4,)`` boxes (x1y1x2y2)."""
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union
