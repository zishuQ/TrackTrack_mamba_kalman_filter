import numpy as np

from trackers.kalman_filter import create_kalman_filter
from trackers.track import BaseTrack, TrackState, get_vel
from trackers.utils import get_prev_box


class TrackAPU(BaseTrack):
    def __init__(self, args, detection):
        self.args = args
        self.box = detection[:4]
        self.score = detection[4]
        self.delta_t = 3
        self.history = {}
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))

        self.raw_feat = detection[6:][np.newaxis, :].copy()
        self.feat = self.raw_feat.copy()
        self.feat /= np.linalg.norm(self.feat)

        self.apu_local_queue = None
        self.apu_history_mask = None
        self.apu_pred_feat = None

    def update_features(self, feat, score):
        self.feat = feat
        self.feat /= np.linalg.norm(self.feat)

    def initiate(self, frame_id, counter, apu_adapter):
        self.track_id = counter.get_track_id()
        self.kalman_filter = create_kalman_filter(getattr(self.args, "kf_type", "nsa"))
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())
        self.apu_local_queue, self.apu_history_mask = apu_adapter.init_track_state(
            self.raw_feat.squeeze(0)
        )
        self.history[frame_id] = [self.box.copy(), self.score.copy(), self.mean.copy(), self.covariance.copy(), self.feat.copy()]
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self):
        if self.state != TrackState.Tracked and "Dance" in self.args.data_path:
            self.mean[6] = 0
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, frame_id, detection, apu_adapter):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            detection.cxcywh.copy(),
            detection.score,
        )
        self.raw_feat = detection.raw_feat.copy()
        self.update_features(detection.feat.copy(), detection.score)
        apu_adapter.update_track(self, detection)

        self.history[frame_id] = [detection.box.copy(), detection.score, self.mean.copy(), self.covariance.copy(), self.feat.copy()]
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t

        self.box = detection.box.copy()
        self.score = detection.score
        self.end_frame_id = frame_id
        self.state = TrackState.Tracked if len(self.history.keys()) >= self.args.min_len else TrackState.New

    @property
    def cxcywh(self):
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
        if self.mean is None:
            x1, y1, x2, y2 = self.box
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            x2 = self.mean[0] + self.mean[2] / 2
            y2 = self.mean[1] + self.mean[3] / 2
        return np.array([x1, y1, x2, y2])
