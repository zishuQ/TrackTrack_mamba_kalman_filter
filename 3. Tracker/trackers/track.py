import numpy as np
from trackers.utils import (
    COMPACT_SNAPSHOT_HISTORY,
    ONLINE_HISTORY_KEEP,
    chronological_history_frame_ids,
    get_prev_box,
    recent_history_frame_ids,
)
from trackers.kalman_filter import create_kalman_filter


def get_vel(b_1, b_2):
    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[0]**2 + deltas[1]**2) + 1e-5
    norm_lb = np.sqrt(deltas[0]**2 + deltas[3]**2) + 1e-5
    norm_rt = np.sqrt(deltas[2]**2 + deltas[1]**2) + 1e-5
    norm_rb = np.sqrt(deltas[2]**2 + deltas[3]**2) + 1e-5

    # Get velocities
    vel_lt = np.array([b_2[0] - b_1[0], b_2[1] - b_1[1]]) / norm_lt
    vel_lb = np.array([b_2[0] - b_1[0], b_2[3] - b_1[3]]) / norm_lb
    vel_rt = np.array([b_2[2] - b_1[2], b_2[1] - b_1[1]]) / norm_rt
    vel_rb = np.array([b_2[2] - b_1[2], b_2[3] - b_1[3]]) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=0)


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackCounter(object):
    track_count = 0

    def get_track_id(self):
        self.track_count += 1
        return self.track_count


class BaseTrack(object):
    track_id = 0
    end_frame_id = 0
    state = TrackState.New

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class Track(BaseTrack):
    def __init__(self, args, detection, retain_full_history=False):
        # Initialize 1
        self.args = args
        self.box = detection[:4]  # x1y1x2y2
        self.score = detection[4]

        # Initialize 2
        self.delta_t = 3
        self.history = {}
        # Cumulative matched observations. Independent of the bounded history
        # window so New/Tracked and AgentGuard maturity keep the original
        # "number of observations" semantics.
        self.observation_count = 0
        # Opt-in debug/test path. Production Trackers leave this False so
        # runtime memory does not keep both a full and a compact history.
        self._retain_full_history = bool(retain_full_history)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))

        # Initialize 3
        self.alpha = 0.95
        self.feat = detection[6:][np.newaxis, :].copy()

    def _write_history(self, frame_id, box, score, mean=None, covariance=None, feat=None):
        """Record one matched observation and prune the online window.

        Unmatched frames never call this, matching the original write policy.
        Duplicate frame ids overwrite the entry and do not increase the
        observation count, matching ``len(history)`` growth of the unbounded
        dict.
        """
        is_new = frame_id not in self.history
        box = np.asarray(box).copy()
        score = float(score)
        if self._retain_full_history:
            self.history[frame_id] = [
                box,
                score,
                None if mean is None else np.asarray(mean).copy(),
                None if covariance is None else np.asarray(covariance).copy(),
                None if feat is None else np.asarray(feat).copy(),
            ]
        else:
            # Current KF mean/covariance/ReID live on the Track itself. Historical
            # copies of those arrays have no remaining online consumer.
            self.history[frame_id] = [box, score]
        if is_new:
            self.observation_count += 1
        self._prune_online_history()

    def _prune_online_history(self):
        if self._retain_full_history:
            return
        extra = len(self.history) - ONLINE_HISTORY_KEEP
        if extra <= 0:
            return
        for old_frame_id in chronological_history_frame_ids(self.history)[:extra]:
            del self.history[old_frame_id]

    def update_features(self, feat, score):
        # Update and normalize
        beta = self.alpha + (1 - self.alpha) * (1 - score)
        self.feat = beta * self.feat + (1 - beta) * feat
        self.feat /= np.linalg.norm(self.feat)

    def initiate(self, frame_id, counter):
        # Get new track id
        self.track_id = counter.get_track_id()

        # Initiate Kalman filter
        self.kalman_filter = create_kalman_filter(getattr(self.args, 'kf_type', 'nsa'))
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())

        # Initiate history
        self._write_history(
            frame_id, self.box, self.score, self.mean, self.covariance, self.feat
        )

        # Initiate parameters
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self):
        # Zero out the velocity of w and h when track is lost or new.
        if self.state != TrackState.Tracked and 'Dance' in self.args.data_path:
            self.mean[6] = 0
            self.mean[7] = 0

        # Predict
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, frame_id, detection):
        # Update Kalman filter & Feature
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               detection.cxcywh.copy(), detection.score)
        self.update_features(detection.feat.copy(), detection.score)

        # Update history
        self._write_history(
            frame_id,
            detection.box,
            detection.score,
            self.mean,
            self.covariance,
            self.feat,
        )

        # Update velocity
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t

        # Update parameters
        self.box = detection.box.copy()
        self.score = detection.score
        self.end_frame_id = frame_id
        self.state = (
            TrackState.Tracked if self.observation_count >= self.args.min_len else TrackState.New
        )

    def snapshot_state(self, compact_history=False):
        from agentguard.contracts.states import TrackStateSnapshot

        observation_count = int(getattr(self, "observation_count", len(self.history)))
        history_copy = {}
        if compact_history:
            frame_ids = recent_history_frame_ids(self.history, COMPACT_SNAPSHOT_HISTORY)
            for frame_id in frame_ids:
                item = self.history[frame_id]
                history_copy[frame_id] = [np.asarray(item[0]).copy(), float(item[1])]
        else:
            if not getattr(self, "_retain_full_history", False):
                raise ValueError(
                    "snapshot_state(compact_history=False) requires unbounded "
                    "per-frame KF mean/covariance/ReID history, which runtime "
                    "Track no longer retains. Use compact_history=True for the "
                    f"last {COMPACT_SNAPSHOT_HISTORY} observation boxes and "
                    "scores (plus observation_count), or construct "
                    "Track(..., retain_full_history=True)."
                )
            for frame_id, item in self.history.items():
                history_copy[frame_id] = [
                    item[0].copy(),
                    item[1],
                    item[2].copy() if item[2] is not None else None,
                    item[3].copy() if item[3] is not None else None,
                    item[4].copy(),
                ]

        return TrackStateSnapshot(
            track_id=self.track_id,
            box=self.box.copy(),
            score=self.score,
            mean=self.mean.copy() if self.mean is not None else None,
            covariance=self.covariance.copy() if self.covariance is not None else None,
            velocity=self.velocity.copy(),
            feature=self.feat.copy(),
            history=history_copy,
            end_frame_id=self.end_frame_id,
            state=self.state,
            observation_count=observation_count,
        )

    def update_with_gates(self, frame_id, detection, motion_gate, appearance_gate):
        # Validate
        motion_gate = float(np.clip(motion_gate, 0.0, 1.0))
        appearance_gate = float(np.clip(appearance_gate, 0.0, 1.0))
        if not np.isfinite(motion_gate) or not np.isfinite(appearance_gate):
            raise ValueError(f"Non-finite gate: m={motion_gate}, a={appearance_gate}")

        # Step 1: Save prior state (before KF update)
        mean_prior = self.mean.copy()
        covariance_prior = self.covariance.copy()
        predicted_box = self.x1y1x2y2.copy()

        # Step 2: Run full KF update
        mean_full, cov_full = self.kalman_filter.update(
            self.mean, self.covariance, detection.cxcywh.copy(), detection.score
        )

        # Step 3: Motion gate with EXACT endpoint handling
        if motion_gate == 1.0:
            self.mean = mean_full
            self.covariance = cov_full
            effective_box = detection.x1y1x2y2.copy()
        elif motion_gate == 0.0:
            self.mean = mean_prior
            self.covariance = covariance_prior
            effective_box = predicted_box.copy()
        else:
            mean_final = mean_prior + motion_gate * (mean_full - mean_prior)
            cov_final = (1 - motion_gate) * covariance_prior + motion_gate * cov_full
            cov_final = 0.5 * (cov_final + cov_final.T)
            self.mean = mean_final
            self.covariance = cov_final
            effective_box = (1 - motion_gate) * predicted_box + motion_gate * detection.x1y1x2y2.copy()

        # Step 4: Appearance gate with EXACT endpoint handling
        old_feat = self.feat.copy()
        if appearance_gate == 1.0:
            beta = self.alpha + (1 - self.alpha) * (1 - detection.score)
            self.feat = beta * old_feat + (1 - beta) * detection.feat.copy()
            self.feat /= np.linalg.norm(self.feat)
        elif appearance_gate == 0.0:
            self.feat = old_feat.copy()
        else:
            beta = self.alpha + (1 - self.alpha) * (1 - detection.score)
            feat_full = beta * old_feat + (1 - beta) * detection.feat.copy()
            feat_final = (1 - appearance_gate) * old_feat + appearance_gate * feat_full
            self.feat = feat_final / (np.linalg.norm(feat_final) + 1e-12)

        # Step 6: Other fields (same as original update)
        self._write_history(
            frame_id,
            effective_box,
            detection.score,
            self.mean,
            self.covariance,
            self.feat,
        )

        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, effective_box) / d_t
        self.velocity /= self.delta_t

        self.box = effective_box.copy()
        self.score = detection.score
        self.end_frame_id = frame_id
        self.state = (
            TrackState.Tracked if self.observation_count >= self.args.min_len else TrackState.New
        )

    @property
    def cxcywh(self):
        # Get current position in bounding box format `(center x, center y, width, height)`.
        if self.mean is None:
            cx = (self.box[0] + self.box[2]) / 2
            cy = (self.box[1] + self.box[3]) / 2
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            cx = self.mean[0]
            cy = self.mean[1]
            w = self.mean[2]
            h = self.mean[3]

        return np.array([cx, cy, w, h])

    @property
    def x1y1wh(self):
        # Get current position in bounding box format `(left top x, left top y, right bottom x, right bottom y)`.
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            w = self.mean[2]
            h = self.mean[3]

        return np.array([x1, y1, w, h])

    @property
    def x1y1x2y2(self):
        # Get current position in bounding box format `(left top x, left top y, right bottom x, right bottom y)`.
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            x2 = self.box[2]
            y2 = self.box[3]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            x2 = self.mean[0] + self.mean[2] / 2
            y2 = self.mean[1] + self.mean[3] / 2

        return np.array([x1, y1, x2, y2])
