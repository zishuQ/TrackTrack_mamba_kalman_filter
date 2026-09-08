"""Batch C: bounded runtime track history vs the pre-change unbounded Track.

The ``PreChangeTrack`` class below is a frozen copy of ``track.py`` from
before this batch: every matched frame stored ``[box, score, mean, cov, feat]``
and New/Tracked used ``len(history)``. Comparisons treat that class as the
independent control, not ``retain_full_history=True``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agentguard" / "src"))
sys.path.insert(0, str(REPO_ROOT / "3. Tracker"))

from trackers.kalman_filter import create_kalman_filter
from trackers.track import Track, TrackCounter, TrackState, get_vel
from trackers.utils import (
    COMPACT_SNAPSHOT_HISTORY,
    ONLINE_HISTORY_KEEP,
    chronological_history_frame_ids,
    conf_distance,
    get_prev_box,
    second_recent_history_score,
    track_observation_count,
)


REID_DIM = 8


class PreChangeTrack:
    """Pre-batch-C Track: unbounded 5-tuple history, age = len(history)."""

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def __init__(self, args, detection):
        self.args = args
        self.box = detection[:4]
        self.score = detection[4]
        self.delta_t = 3
        self.history = {}
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))
        self.alpha = 0.95
        self.feat = detection[6:][np.newaxis, :].copy()

    def update_features(self, feat, score):
        beta = self.alpha + (1 - self.alpha) * (1 - score)
        self.feat = beta * self.feat + (1 - beta) * feat
        self.feat /= np.linalg.norm(self.feat)

    def initiate(self, frame_id, counter):
        self.track_id = counter.get_track_id()
        self.kalman_filter = create_kalman_filter(getattr(self.args, "kf_type", "nsa"))
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())
        self.history[frame_id] = [
            self.box.copy(),
            self.score.copy(),
            self.mean.copy(),
            self.covariance.copy(),
            self.feat.copy(),
        ]
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self):
        if self.state != TrackState.Tracked and "Dance" in self.args.data_path:
            self.mean[6] = 0
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, frame_id, detection):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, detection.cxcywh.copy(), detection.score
        )
        self.update_features(detection.feat.copy(), detection.score)
        self.history[frame_id] = [
            detection.box.copy(),
            detection.score,
            self.mean.copy(),
            self.covariance.copy(),
            self.feat.copy(),
        ]
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t
        self.box = detection.box.copy()
        self.score = detection.score
        self.end_frame_id = frame_id
        self.state = (
            TrackState.Tracked if len(self.history.keys()) >= self.args.min_len else TrackState.New
        )

    def snapshot_state(self, compact_history=False):
        from agentguard.contracts.states import TrackStateSnapshot

        history_copy = {}
        frame_ids = (
            sorted(self.history.keys())[-6:]
            if compact_history
            else self.history.keys()
        )
        for frame_id in frame_ids:
            hist_list = self.history[frame_id]
            if compact_history:
                history_copy[frame_id] = [hist_list[0].copy(), float(hist_list[1])]
            else:
                history_copy[frame_id] = [
                    hist_list[0].copy(),
                    hist_list[1],
                    hist_list[2].copy() if hist_list[2] is not None else None,
                    hist_list[3].copy() if hist_list[3] is not None else None,
                    hist_list[4].copy(),
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
        )

    def update_with_gates(self, frame_id, detection, motion_gate, appearance_gate):
        motion_gate = float(np.clip(motion_gate, 0.0, 1.0))
        appearance_gate = float(np.clip(appearance_gate, 0.0, 1.0))
        if not np.isfinite(motion_gate) or not np.isfinite(appearance_gate):
            raise ValueError(f"Non-finite gate: m={motion_gate}, a={appearance_gate}")
        mean_prior = self.mean.copy()
        covariance_prior = self.covariance.copy()
        predicted_box = self.x1y1x2y2.copy()
        mean_full, cov_full = self.kalman_filter.update(
            self.mean, self.covariance, detection.cxcywh.copy(), detection.score
        )
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
        self.history[frame_id] = [
            effective_box.copy(),
            detection.score,
            self.mean.copy(),
            self.covariance.copy(),
            self.feat.copy(),
        ]
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, effective_box) / d_t
        self.velocity /= self.delta_t
        self.box = effective_box.copy()
        self.score = detection.score
        self.end_frame_id = frame_id
        self.state = (
            TrackState.Tracked if len(self.history.keys()) >= self.args.min_len else TrackState.New
        )

    @property
    def cxcywh(self):
        if self.mean is None:
            cx = (self.box[0] + self.box[2]) / 2
            cy = (self.box[1] + self.box[3]) / 2
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            cx, cy, w, h = self.mean[0], self.mean[1], self.mean[2], self.mean[3]
        return np.array([cx, cy, w, h])

    @property
    def x1y1x2y2(self):
        if self.mean is None:
            return self.box.copy()
        return np.array(
            [
                self.mean[0] - self.mean[2] / 2,
                self.mean[1] - self.mean[3] / 2,
                self.mean[0] + self.mean[2] / 2,
                self.mean[1] + self.mean[3] / 2,
            ]
        )


def _args(min_len=3):
    return SimpleNamespace(min_len=min_len, data_path="/test", kf_type="nsa")


def _raw_det(box, score=0.9, feat=None, frame_shift=0.0):
    box = np.asarray(box, dtype=np.float64) + frame_shift
    if feat is None:
        feat = np.linspace(0.1, 1.0, REID_DIM, dtype=np.float64)
        feat = feat / np.linalg.norm(feat)
    return np.concatenate([box, [score, 1.0], feat])


def _paired_tracks(min_len=3):
    raw = _raw_det([10.0, 20.0, 40.0, 80.0], score=0.91)
    return Track(_args(min_len), raw.copy()), PreChangeTrack(_args(min_len), raw.copy())


def _as_detection(track_cls, box, score, feat=None, shift=0.0):
    return track_cls(_args(), _raw_det(box, score=score, feat=feat, frame_shift=shift))


def _assert_live_state_equal(new_track, old_track, atol=1e-12):
    assert new_track.state == old_track.state
    assert new_track.end_frame_id == old_track.end_frame_id
    assert new_track.observation_count == len(old_track.history)
    assert new_track.score == pytest.approx(float(old_track.score))
    np.testing.assert_allclose(new_track.box, old_track.box, atol=atol, rtol=0.0)
    np.testing.assert_allclose(new_track.mean, old_track.mean, atol=atol, rtol=0.0)
    np.testing.assert_allclose(new_track.covariance, old_track.covariance, atol=atol, rtol=0.0)
    np.testing.assert_allclose(new_track.feat, old_track.feat, atol=atol, rtol=0.0)
    np.testing.assert_allclose(new_track.velocity, old_track.velocity, atol=atol, rtol=0.0)


def _history_array_nbytes(history):
    total = 0
    for entry in history.values():
        for item in entry:
            if isinstance(item, np.ndarray):
                total += int(item.nbytes)
    return total


def test_first_create_short_and_mature_threshold():
    counter_a, counter_b = TrackCounter(), TrackCounter()
    new_track, old_track = _paired_tracks(min_len=3)
    new_track.initiate(1, counter_a)
    old_track.initiate(1, counter_b)
    _assert_live_state_equal(new_track, old_track)
    assert new_track.state == TrackState.New
    assert new_track.observation_count == 1
    assert len(new_track.history) == 1

    for frame_id in range(2, 5):
        det_new = _as_detection(Track, [10 + frame_id, 20, 40 + frame_id, 80], 0.8 + 0.01 * frame_id)
        det_old = _as_detection(PreChangeTrack, [10 + frame_id, 20, 40 + frame_id, 80], 0.8 + 0.01 * frame_id)
        new_track.predict()
        old_track.predict()
        new_track.update(frame_id, det_new)
        old_track.update(frame_id, det_old)
        _assert_live_state_equal(new_track, old_track)
        expected_state = TrackState.Tracked if frame_id >= 3 else TrackState.New
        assert new_track.state == expected_state
        assert new_track.observation_count == frame_id


def test_unmatched_does_not_write_history_and_rematch_preserves_counts():
    counter_a, counter_b = TrackCounter(), TrackCounter()
    new_track, old_track = _paired_tracks()
    new_track.initiate(1, counter_a)
    old_track.initiate(1, counter_b)
    for frame_id in range(2, 6):
        new_track.predict()
        old_track.predict()
        new_track.update(frame_id, _as_detection(Track, [10.0 + frame_id, 20, 40.0 + frame_id, 80], 0.9))
        old_track.update(frame_id, _as_detection(PreChangeTrack, [10.0 + frame_id, 20, 40.0 + frame_id, 80], 0.9))

    count_before = new_track.observation_count
    keys_before = set(new_track.history.keys())
    for frame_id in range(6, 9):
        new_track.predict()
        old_track.predict()
        new_track.mark_lost()
        old_track.mark_lost()
        assert new_track.observation_count == count_before
        assert set(new_track.history.keys()) == keys_before
        assert len(old_track.history) == count_before

    new_track.predict()
    old_track.predict()
    new_track.update(12, _as_detection(Track, [30.0, 24.0, 62.0, 88.0], 0.84))
    old_track.update(12, _as_detection(PreChangeTrack, [30.0, 24.0, 62.0, 88.0], 0.84))
    _assert_live_state_equal(new_track, old_track)
    assert new_track.observation_count == count_before + 1
    assert 12 in new_track.history
    assert 6 not in new_track.history
    assert 6 not in old_track.history


def test_noncontiguous_frames_keep_prev_box_and_second_score_rule():
    counter_a, counter_b = TrackCounter(), TrackCounter()
    new_track, old_track = _paired_tracks()
    frames = [1, 2, 5, 9, 10, 16]
    new_track.initiate(frames[0], counter_a)
    old_track.initiate(frames[0], counter_b)
    for frame_id in frames[1:]:
        new_track.predict()
        old_track.predict()
        box = [10.0 + frame_id, 20.0, 40.0 + frame_id, 80.0]
        new_track.update(frame_id, _as_detection(Track, box, 0.5 + 0.02 * frame_id))
        old_track.update(frame_id, _as_detection(PreChangeTrack, box, 0.5 + 0.02 * frame_id))
        _assert_live_state_equal(new_track, old_track)
        for dt in (1, 2, 3):
            np.testing.assert_allclose(
                get_prev_box(new_track.history, frame_id, dt),
                get_prev_box(old_track.history, frame_id, dt),
                atol=0.0,
                rtol=0.0,
            )
        assert second_recent_history_score(new_track.history) == pytest.approx(
            second_recent_history_score(old_track.history)
        )


@pytest.mark.parametrize("gate", [0.0, 1.0, 0.37])
def test_gated_update_matches_prechange_for_supported_gates(gate):
    counter_a, counter_b = TrackCounter(), TrackCounter()
    new_track, old_track = _paired_tracks()
    new_track.initiate(1, counter_a)
    old_track.initiate(1, counter_b)
    for frame_id in range(2, 8):
        new_track.predict()
        old_track.predict()
        box = [12.0 + frame_id, 21.0, 44.0 + frame_id, 83.0]
        new_track.update_with_gates(
            frame_id, _as_detection(Track, box, 0.77), gate, gate
        )
        old_track.update_with_gates(
            frame_id, _as_detection(PreChangeTrack, box, 0.77), gate, gate
        )
        _assert_live_state_equal(new_track, old_track)
        np.testing.assert_allclose(
            new_track.history[frame_id][0],
            old_track.history[frame_id][0],
            atol=1e-12,
            rtol=0.0,
        )
        assert new_track.history[frame_id][1] == pytest.approx(float(old_track.history[frame_id][1]))
        assert len(new_track.history[frame_id]) == 2


def test_long_track_bounds_history_but_keeps_observation_count():
    counter_a, counter_b = TrackCounter(), TrackCounter()
    new_track, old_track = _paired_tracks()
    new_track.initiate(1, counter_a)
    old_track.initiate(1, counter_b)
    n = 40
    for frame_id in range(2, n + 1):
        new_track.predict()
        old_track.predict()
        box = [10.0 + 0.4 * frame_id, 20.0, 40.0 + 0.4 * frame_id, 80.0]
        new_track.update(frame_id, _as_detection(Track, box, 0.85))
        old_track.update(frame_id, _as_detection(PreChangeTrack, box, 0.85))
    _assert_live_state_equal(new_track, old_track)
    assert new_track.observation_count == n
    assert len(new_track.history) == ONLINE_HISTORY_KEEP
    assert len(old_track.history) == n
    assert chronological_history_frame_ids(new_track.history) == list(range(n - ONLINE_HISTORY_KEEP + 1, n + 1))
    compact_new = new_track.snapshot_state(compact_history=True)
    compact_old = old_track.snapshot_state(compact_history=True)
    assert sorted(compact_new.history.keys()) == sorted(compact_old.history.keys())
    assert compact_new.observation_count == n
    for fid in compact_new.history:
        np.testing.assert_allclose(compact_new.history[fid][0], compact_old.history[fid][0])
        assert compact_new.history[fid][1] == pytest.approx(compact_old.history[fid][1])


@pytest.mark.parametrize("context_size", [6, 8])
def test_maturity_uses_observation_count_not_window_length(context_size):
    from agentguard.runtime.manager import AgentGuardRuntime

    runtime = AgentGuardRuntime({"mode": "capture", "iwg_context_size": context_size})
    counter = TrackCounter()
    track = Track(_args(), _raw_det([0, 0, 10, 20]))
    track.initiate(1, counter)
    for frame_id in range(2, context_size + 3):
        track.predict()
        track.update(frame_id, _as_detection(Track, [frame_id, 0, 10 + frame_id, 20], 0.9))
        mature = runtime.is_mature_track(track)
        expected = track.state in (1, 2) and track.observation_count >= context_size
        assert mature is expected
        assert len(track.history) <= ONLINE_HISTORY_KEEP


def test_compact_snapshot_false_raises_unless_full_history_retained():
    counter = TrackCounter()
    track = Track(_args(), _raw_det([0, 0, 10, 20]))
    track.initiate(1, counter)
    with pytest.raises(ValueError, match="compact_history=False"):
        track.snapshot_state(compact_history=False)

    full = Track(_args(), _raw_det([0, 0, 10, 20]), retain_full_history=True)
    full.initiate(1, TrackCounter())
    for frame_id in range(2, 12):
        full.predict()
        full.update(frame_id, _as_detection(Track, [frame_id, 0, 10 + frame_id, 20], 0.8))
    snap = full.snapshot_state(compact_history=False)
    assert sorted(snap.history.keys()) == list(range(1, 12))
    assert all(len(v) == 5 for v in snap.history.values())
    compact = full.snapshot_state(compact_history=True)
    assert sorted(compact.history.keys()) == list(range(12 - COMPACT_SNAPSHOT_HISTORY, 12))
    assert all(len(v) == 2 for v in compact.history.values())


def test_scalar_features_match_prechange_compact_snapshots():
    from agentguard.features.scalar import compute_scalar_features

    counter_a, counter_b = TrackCounter(), TrackCounter()
    new_track, old_track = _paired_tracks()
    new_track.initiate(1, counter_a)
    old_track.initiate(1, counter_b)
    for frame_id in range(2, 20):
        new_track.predict()
        old_track.predict()
        box = [8.0 + frame_id, 15.0, 30.0 + frame_id, 70.0]
        new_track.update(frame_id, _as_detection(Track, box, 0.6 + 0.01 * (frame_id % 5)))
        old_track.update(frame_id, _as_detection(PreChangeTrack, box, 0.6 + 0.01 * (frame_id % 5)))
        new_snap = new_track.snapshot_state(compact_history=True)
        old_snap = old_track.snapshot_state(compact_history=True)
        params = {
            "has_detection": True,
            "image_width": 1920,
            "image_height": 1080,
            "frame_id": frame_id + 1,
            "pre_update_state": {
                "mean": new_snap.mean,
                "covariance": new_snap.covariance,
                "velocity": new_snap.velocity,
                "score": new_snap.score,
                "history": new_snap.history,
                "end_frame_id": new_snap.end_frame_id,
                "state": new_snap.state,
            },
        }
        old_params = dict(params)
        old_params["pre_update_state"] = {
            "mean": old_snap.mean,
            "covariance": old_snap.covariance,
            "velocity": old_snap.velocity,
            "score": old_snap.score,
            "history": old_snap.history,
            "end_frame_id": old_snap.end_frame_id,
            "state": old_snap.state,
        }
        new_feat = compute_scalar_features(params)
        old_feat = compute_scalar_features(old_params)
        np.testing.assert_allclose(new_feat, old_feat, atol=0.0, rtol=0.0)
        # Feature 58 stays the compact-snapshot length, not the cumulative count.
        assert new_feat[58] == pytest.approx(min(len(new_snap.history), 50) / 50.0)


def test_conf_distance_matches_sorted_second_recent_without_full_sort():
    history = {1: [np.zeros(4), 0.1], 2: [np.zeros(4), 0.4], 9: [np.zeros(4), 0.9]}
    assert second_recent_history_score(history) == pytest.approx(0.4)
    shuffled = {9: [np.zeros(4), 0.9], 1: [np.zeros(4), 0.1], 2: [np.zeros(4), 0.4]}
    assert second_recent_history_score(shuffled) == pytest.approx(0.4)

    class Stub:
        def __init__(self, score, history):
            self.score = score
            self.history = history

    tracks = [Stub(0.8, history)]
    dets = [Stub(0.7, {})]
    dist = conf_distance(tracks, dets)
    prev = 0.4
    projected = 0.8 + (0.8 - prev)
    assert dist[0, 0] == pytest.approx(abs(projected - 0.7))


def test_get_prev_box_accepts_two_tuple_and_five_tuple_history():
    box_a = np.array([1.0, 2.0, 3.0, 4.0])
    box_b = np.array([5.0, 6.0, 7.0, 8.0])
    two = {10: [box_a, 0.2], 12: [box_b, 0.3]}
    five = {
        10: [box_a, 0.2, np.zeros(8), np.eye(8), np.zeros((1, 4))],
        12: [box_b, 0.3, np.zeros(8), np.eye(8), np.zeros((1, 4))],
    }
    np.testing.assert_array_equal(get_prev_box(two, 12, 2), box_a)
    np.testing.assert_array_equal(get_prev_box(five, 12, 2), box_a)
    np.testing.assert_array_equal(get_prev_box(two, 12, 1), box_b)


def test_capture_history_count_is_cumulative(tmp_path):
    from agentguard.data.compact_event_cache import CompactEventCacheSink, _compact_state
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    counter = TrackCounter()
    track = Track(_args(), _raw_det([0, 0, 10, 20], score=0.9))
    track.initiate(1, counter)
    for frame_id in range(2, 16):
        track.predict()
        track.update(frame_id, _as_detection(Track, [frame_id, 0, 10 + frame_id, 20], 0.8))
    snap = track.snapshot_state(compact_history=True)
    assert track.observation_count == 15
    assert sorted(snap.history.keys()) == list(range(16 - COMPACT_SNAPSHOT_HISTORY, 16))
    compact = AgentGuardTrackerAdapter(SimpleNamespace(dataset="MOT17"), "seq")._compact_snapshot_dict(snap)
    assert compact["observation_count"] == 15
    stored = _compact_state(compact)
    assert stored["history_count"] == 15
    assert stored["recent_history_boxes"].shape[0] == COMPACT_SNAPSHOT_HISTORY

    sink = CompactEventCacheSink(
        cache_root=tmp_path,
        dataset="MOT17",
        split="all",
        sequence="hist",
        reid_dim=REID_DIM,
        event_flush_size=1,
        frame_flush_size=1,
    )
    sink.on_sequence_start("hist", reid_dim=REID_DIM, image_width=64, image_height=48)
    sink.on_frame(
        {
            "frame_id": 16,
            "image_width": 64,
            "image_height": 48,
            "warp_matrix": np.eye(2, 3, dtype=np.float32),
            "detections": [],
            "association": None,
        },
        [
            {
                "event_id": "MOT17/hist/000016/000001",
                "frame_id": 16,
                "track_id": 1,
                "has_detection": True,
                "accepted_detection_index": -1,
                "frame_start_state": compact,
                "pre_update_state": compact,
                "scalar_features": np.zeros(63, dtype=np.float32),
                "track_feature": np.ones(REID_DIM, dtype=np.float32),
            }
        ],
    )
    sink.on_sequence_end()
    import torch

    events = torch.load(tmp_path / "MOT17" / "all" / "hist" / "events_00000.pt", weights_only=False)
    assert events[0]["history_count"] == 15


def test_bounded_history_memory_and_update_time():
    n_tracks = 20
    n_obs = 400

    def run(cls):
        tracks = []
        t0 = time.perf_counter()
        for i in range(n_tracks):
            raw = _raw_det([i, 0, 10 + i, 20], feat=np.ones(REID_DIM) / np.sqrt(REID_DIM))
            track = cls(_args(), raw)
            track.initiate(1, TrackCounter())
            for frame_id in range(2, n_obs + 1):
                track.predict()
                det = cls(_args(), _raw_det([i + 0.1 * frame_id, 0, 12 + i, 22], score=0.9))
                track.update(frame_id, det)
            tracks.append(track)
        elapsed = time.perf_counter() - t0
        entries = sum(len(t.history) for t in tracks)
        nbytes = sum(_history_array_nbytes(t.history) for t in tracks)
        return tracks, entries, nbytes, elapsed

    new_tracks, new_entries, new_bytes, new_time = run(Track)
    old_tracks, old_entries, old_bytes, old_time = run(PreChangeTrack)
    assert new_entries == n_tracks * ONLINE_HISTORY_KEEP
    assert old_entries == n_tracks * n_obs
    assert new_bytes < old_bytes / 10
    assert all(len(t.history[next(iter(t.history))]) == 2 for t in new_tracks)
    assert all(len(t.history[next(iter(t.history))]) == 5 for t in old_tracks)
    # Compact history must not introduce a stable slowdown on this path.
    assert new_time < old_time * 1.5
    print(
        f"history memory: new_entries={new_entries} old_entries={old_entries} "
        f"new_bytes={new_bytes} old_bytes={old_bytes} new_s={new_time:.4f} old_s={old_time:.4f}"
    )


def _mot17_cache_available():
    root = Path("/home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache/MOT17/all/MOT17-09-FRCNN")
    return root.is_dir() and (root / "boxes.npy").is_file()


@pytest.mark.skipif(not _mot17_cache_available(), reason="MOT17-09-FRCNN detection cache not present")
def test_mot17_tracker_outputs_match_prechange_track(monkeypatch):
    from agentguard.data.detection_cache import SequenceDetectionCache
    from trackers.tracker import Tracker
    import trackers.tracker as tracker_mod
    from utils.etc import set_parameters

    cache = SequenceDetectionCache(
        "/home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache/MOT17/all/MOT17-09-FRCNN"
    )
    n_frames = min(25, cache.num_frames)

    def run(track_cls):
        TrackCounter.track_count = 0
        monkeypatch.setattr(tracker_mod, "Track", track_cls)
        args = SimpleNamespace(
            pickle_dir="/tmp",
            data_dir="/home/shang/datasets/",
            min_len=3,
            min_box_area=100,
            max_time_lost=30,
            penalty_p=0.20,
            penalty_q=0.40,
            reduce_step=0.05,
            tai_thr=0.55,
            disable_gmc=False,
            kf_type="nsa",
            agentguard_mode="off",
            no_reid=False,
            dataset="MOT17",
        )
        set_parameters(args, "MOT17-09-FRCNN", "all")
        args.img_w, args.img_h = 1920, 1080
        tracker = Tracker(args, "MOT17-09-FRCNN")
        outputs = []
        t0 = time.perf_counter()
        for frame_idx in range(n_frames):
            target = cache.get_frame_array(frame_idx, view="target")
            source = cache.get_frame_array(frame_idx, view="source")
            if target is None:
                target = np.zeros((0, 6 + cache.reid_dim), dtype=np.float32)
            if source is None:
                source = np.zeros((0, 6 + cache.reid_dim), dtype=np.float32)
            tracker.update(target, source)
            frame_out = []
            for track in tracker.tracks:
                frame_out.append(
                    (
                        int(track.track_id),
                        int(track.state),
                        int(track.end_frame_id),
                        int(track_observation_count(track)),
                        np.asarray(track.x1y1x2y2, dtype=np.float64).copy(),
                        float(track.score),
                        np.asarray(track.mean, dtype=np.float64).copy(),
                        np.asarray(track.feat, dtype=np.float64).copy(),
                        np.asarray(track.velocity, dtype=np.float64).copy(),
                    )
                )
            outputs.append(sorted(frame_out, key=lambda item: item[0]))
        elapsed = time.perf_counter() - t0
        tracker.cmc.gmcFile.close()
        return outputs, elapsed

    new_out, new_time = run(Track)
    old_out, old_time = run(PreChangeTrack)
    assert len(new_out) == len(old_out) == n_frames
    max_box = 0.0
    max_mean = 0.0
    max_feat = 0.0
    for new_frame, old_frame in zip(new_out, old_out):
        assert len(new_frame) == len(old_frame)
        for new_t, old_t in zip(new_frame, old_frame):
            assert new_t[0] == old_t[0]
            assert new_t[1] == old_t[1]
            assert new_t[2] == old_t[2]
            assert new_t[3] == old_t[3]
            max_box = max(max_box, float(np.max(np.abs(new_t[4] - old_t[4]))))
            assert new_t[5] == pytest.approx(old_t[5])
            max_mean = max(max_mean, float(np.max(np.abs(new_t[6] - old_t[6]))))
            max_feat = max(max_feat, float(np.max(np.abs(new_t[7] - old_t[7]))))
            np.testing.assert_allclose(new_t[8], old_t[8], atol=1e-12, rtol=0.0)
    assert max_box == 0.0
    assert max_mean == 0.0
    assert max_feat == 0.0
    print(
        f"MOT17-09 {n_frames} frames: max_box={max_box} max_mean={max_mean} "
        f"max_feat={max_feat} new_s={new_time:.4f} old_s={old_time:.4f}"
    )


@pytest.mark.skipif(not _mot17_cache_available(), reason="MOT17-09-FRCNN detection cache not present")
def test_mot17_capture_history_count_is_cumulative(tmp_path):
    import torch
    from agentguard.data.compact_event_cache import CompactEventCacheSink
    from agentguard.data.detection_cache import SequenceDetectionCache
    from trackers.tracker import Tracker
    from utils.etc import set_parameters

    TrackCounter.track_count = 0
    cache = SequenceDetectionCache(
        "/home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache/MOT17/all/MOT17-09-FRCNN"
    )
    n_frames = min(20, cache.num_frames)
    sink = CompactEventCacheSink(
        cache_root=tmp_path,
        dataset="MOT17",
        split="all",
        sequence="MOT17-09-FRCNN",
        reid_dim=int(cache.reid_dim),
        event_flush_size=64,
        frame_flush_size=16,
    )
    args = SimpleNamespace(
        pickle_dir="/tmp",
        data_dir="/home/shang/datasets/",
        min_len=3,
        min_box_area=100,
        max_time_lost=30,
        penalty_p=0.20,
        penalty_q=0.40,
        reduce_step=0.05,
        tai_thr=0.55,
        disable_gmc=False,
        kf_type="nsa",
        agentguard_mode="capture",
        no_reid=False,
        dataset="MOT17",
        event_sink=sink,
        img_w=1920,
        img_h=1080,
    )
    set_parameters(args, "MOT17-09-FRCNN", "all")
    args.img_w, args.img_h = 1920, 1080
    sink.on_sequence_start(
        "MOT17-09-FRCNN",
        reid_dim=int(cache.reid_dim),
        image_width=1920,
        image_height=1080,
    )
    tracker = Tracker(args, "MOT17-09-FRCNN")
    try:
        for frame_idx in range(n_frames):
            target_frame = cache.get_frame(frame_idx, view="target")
            source_frame = cache.get_frame(frame_idx, view="source")
            args.agentguard_target_detection_indices = target_frame["detection_indices"]
            args.agentguard_source_detection_indices = source_frame["detection_indices"]
            target = cache.get_frame_array(frame_idx, view="target")
            source = cache.get_frame_array(frame_idx, view="source")
            if target is None:
                target = np.zeros((0, 6 + cache.reid_dim), dtype=np.float32)
            if source is None:
                source = np.zeros((0, 6 + cache.reid_dim), dtype=np.float32)
            tracker.update(target, source)
    finally:
        tracker.cmc.gmcFile.close()
        cache.close()
    sink.on_sequence_end()

    seq_dir = tmp_path / "MOT17" / "all" / "MOT17-09-FRCNN"
    events = []
    for path in sorted(seq_dir.glob("events_*.pt")):
        events.extend(torch.load(path, weights_only=False))
    assert events, "capture produced no events"
    counts = [int(ev["history_count"]) for ev in events]
    assert max(counts) > COMPACT_SNAPSHOT_HISTORY
    by_track = {}
    for ev in events:
        by_track.setdefault(int(ev["track_id"]), []).append(
            (int(ev["frame_id"]), int(ev["history_count"]))
        )
    saw_increase_past_six = False
    for rows in by_track.values():
        rows.sort()
        previous = None
        for _frame_id, hist in rows:
            if previous is not None:
                assert hist >= previous
            if hist > COMPACT_SNAPSHOT_HISTORY:
                saw_increase_past_six = True
            previous = hist
    assert saw_increase_past_six
    states = []
    for path in sorted(seq_dir.glob("states_*.pt")):
        states.extend(torch.load(path, weights_only=False))
    assert all("history" not in state for state in states)
    assert all(len(state.get("recent_history_frames", [])) <= COMPACT_SNAPSHOT_HISTORY for state in states)
