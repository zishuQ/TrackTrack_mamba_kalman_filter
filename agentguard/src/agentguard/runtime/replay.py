from __future__ import annotations

from typing import List, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import TrackStateSnapshot
from agentguard.motion.nsa_numpy import NSAKalmanFilter


class ReplayEngine:
    """
    Replays events with revised gates to update track state.

    For each event in window (oldest first):
    1. Apply warp matrix to KF state.
    2. KF predict.
    3. If ``has_detection``: perform a gated KF update using the revised gate.
    4. If no detection: mark track as LOST (state=2).
    """

    def __init__(
        self,
        motion_model: Optional[NSAKalmanFilter] = None,
        feature_alpha: float = 0.9,
    ):
        self.motion = motion_model or NSAKalmanFilter()
        self.alpha = feature_alpha  # feature momentum used in appearance update

    def replay_event(
        self,
        track_snapshot: TrackStateSnapshot,
        event: TrackEvent,
        revised_gate: Optional[np.ndarray],
    ) -> TrackStateSnapshot:
        """Replay a single event against a track snapshot and return the
        resulting state.

        Parameters
        ----------
        track_snapshot : TrackStateSnapshot
            Pre-event track state.
        event : TrackEvent
            The event to replay (carries warp, detection, features, …).
        revised_gate : (2,) ndarray or None
            ``[motion_gate, appearance_gate]`` in ``[0, 1]``.
            When ``None``, both gates default to 1.0.

        Returns
        -------
        TrackStateSnapshot
            Updated track state after replaying the event.
        """
        # ------------------------------------------------------------------
        # 1. Apply warp matrix to KF state
        # ------------------------------------------------------------------
        mean = (
            track_snapshot.mean.copy()
            if track_snapshot.mean is not None
            else None
        )
        covariance = (
            track_snapshot.covariance.copy()
            if track_snapshot.covariance is not None
            else None
        )

        if (
            mean is not None
            and covariance is not None
            and event.warp_matrix is not None
        ):
            mean, covariance = self.motion.apply_warp(
                mean, covariance, event.warp_matrix
            )

        # ------------------------------------------------------------------
        # 2. KF predict
        # ------------------------------------------------------------------
        if mean is not None and covariance is not None:
            mean, covariance = self.motion.predict(mean, covariance)

        # Default to current snapshot values; update below as needed.
        new_state = track_snapshot.state
        new_feature = track_snapshot.feature.copy()
        new_box = track_snapshot.box.copy()

        # ------------------------------------------------------------------
        # 3a. Detection present — perform gated update
        # ------------------------------------------------------------------
        if event.has_detection and event.detection is not None and mean is not None:
            motion_gate = float(revised_gate[0]) if revised_gate is not None else 1.0
            appearance_gate = (
                float(revised_gate[1]) if revised_gate is not None else 1.0
            )

            # Full KF update (as if gate = 1.0)
            measurement = self.motion.bbox_to_measurement(event.detection.box)
            mean_full, cov_full = self.motion.update(
                mean, covariance, measurement, event.detection.score
            )

            # Motion gate: interpolate between prior (predict) and full update
            mean_prior = mean.copy()
            cov_prior = covariance.copy()
            mean_final = mean_prior + motion_gate * (mean_full - mean_prior)
            cov_final = (1.0 - motion_gate) * cov_prior + motion_gate * cov_full
            cov_final = 0.5 * (cov_final + cov_final.T)  # enforce symmetry
            mean = mean_final
            covariance = cov_final

            # Effective box (interpolated between predicted box and detection)
            predicted_box = self.motion.mean_to_bbox(mean_prior)
            detection_box = event.detection.box.copy()
            effective_box = (
                1.0 - motion_gate
            ) * predicted_box + motion_gate * detection_box
            new_box = effective_box

            # Appearance gate: interpolate feature update
            old_feat = track_snapshot.feature.copy()
            det_feat = event.detection.feature.copy()
            beta = self.alpha + (1.0 - self.alpha) * (1.0 - event.detection.score)
            feat_full = beta * old_feat + (1.0 - beta) * det_feat
            feat_final = (1.0 - appearance_gate) * old_feat + appearance_gate * feat_full
            feat_norm = np.linalg.norm(feat_final)
            if feat_norm > 1e-12:
                feat_final = feat_final / feat_norm
            new_feature = feat_final

            new_state = track_snapshot.state  # keep previous state (typically Tracked)

        # ------------------------------------------------------------------
        # 3b. No detection — track is lost
        # ------------------------------------------------------------------
        elif not event.has_detection:
            new_state = 2  # TrackLifecycle.LOST
            # feature is kept from the snapshot
            if mean is not None:
                new_box = self.motion.mean_to_bbox(mean)

        # ------------------------------------------------------------------
        # Build new snapshot
        # ------------------------------------------------------------------
        if mean is not None:
            velocity = np.column_stack([mean[:4], mean[4:]])  # (4, 2)
        else:
            velocity = track_snapshot.velocity.copy()

        # Update history: append the replayed frame's data
        new_history = dict(track_snapshot.history)
        new_history[event.frame_id] = [
            new_box.copy(),
            event.detection.score if event.has_detection else 0.0,
            mean.copy() if mean is not None else None,
            covariance.copy() if covariance is not None else None,
            new_feature.copy(),
        ]

        return TrackStateSnapshot(
            track_id=track_snapshot.track_id,
            box=new_box.copy(),
            score=track_snapshot.score,
            mean=mean.copy() if mean is not None else None,
            covariance=covariance.copy() if covariance is not None else None,
            velocity=velocity.copy(),
            feature=new_feature.copy(),
            history=new_history,
            end_frame_id=event.frame_id,
            state=new_state,
        )

    def replay_window(
        self,
        checkpoint_snapshot: TrackStateSnapshot,
        events: List[TrackEvent],
        revised_gates: np.ndarray,
    ) -> TrackStateSnapshot:
        """Replay a full window of events starting from a checkpoint snapshot.

        Parameters
        ----------
        checkpoint_snapshot : TrackStateSnapshot
            State before the oldest event in the window.
        events : list of TrackEvent
            Window events ordered from oldest to newest.
        revised_gates : ndarray, shape ``(len(events), 2)``
            Revised ``[motion_gate, appearance_gate]`` for each event.

        Returns
        -------
        TrackStateSnapshot
            Final track state after processing all events.
        """
        current = checkpoint_snapshot
        for event, gate in zip(events, revised_gates):
            current = self.replay_event(current, event, gate)
        return current
