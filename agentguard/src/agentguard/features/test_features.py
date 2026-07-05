"""Tests for the features subpackage."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from agentguard.contracts.enums import DetectionSource, TrackLifecycle
from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.features import EventFeatureBuilder, NormalizationStats
from agentguard.features.scalar import compute_scalar_features


# ======================================================================
#  Fixtures
# ======================================================================


@pytest.fixture
def sample_detection() -> DetectionObservation:
    return DetectionObservation(
        detection_index=0,
        box=np.array([100, 200, 300, 400], dtype=np.float64),  # x1y1x2y2
        score=0.85,
        feature=np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        source=DetectionSource.HIGH,
        class_id=1,
    )


@pytest.fixture
def sample_association() -> AssociationPairFeatures:
    return AssociationPairFeatures(
        iou_similarity=0.75,
        iou_distance=0.25,
        cosine_distance=0.10,
        confidence_distance=0.05,
        angle_distance=0.08,
        raw_cost=0.30,
        final_cost=0.28,
        assignment_round=2,
        assignment_threshold=0.50,
        detection_source=0,
    )


@pytest.fixture
def sample_snapshot() -> TrackStateSnapshot:
    # KF mean: [cx, cy, w, h, vx, vy, vw, vh]
    mean = np.array([200.0, 300.0, 200.0, 200.0, 2.0, 1.0, 0.1, -0.05], dtype=np.float64)
    cov = np.eye(8, dtype=np.float64) * 0.01
    cov[0, 0] = 4.0  # cx variance
    cov[1, 1] = 4.0  # cy variance
    cov[2, 2] = 1.0  # w variance
    cov[3, 3] = 1.0  # h variance
    vel = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64
    )
    feat = np.array([[0.5, 0.6, 0.7]], dtype=np.float64)
    history = {
        90: [np.array([90, 190, 190, 290]), 0.7, mean.copy(), cov.copy(), feat.copy()],
        95: [np.array([95, 195, 195, 295]), 0.75, mean.copy(), cov.copy(), feat.copy()],
        98: [np.array([105, 205, 205, 305]), 0.8, mean.copy(), cov.copy(), feat.copy()],
    }
    return TrackStateSnapshot(
        track_id=42,
        box=np.array([105, 205, 205, 305], dtype=np.float64),
        score=0.80,
        mean=mean,
        covariance=cov,
        velocity=vel,
        feature=feat,
        history=history,
        end_frame_id=98,
        state=TrackLifecycle.TRACKED,
    )


@pytest.fixture
def sample_event(
    sample_detection, sample_association, sample_snapshot
) -> TrackEvent:
    ev = TrackEvent(
        event_id="test-001",
        dataset="test",
        sequence="seq1",
        frame_id=100,
        track_id=42,
        image_width=1920,
        image_height=1080,
        has_detection=True,
        pre_update_state=sample_snapshot,
        detection=sample_detection,
        association=sample_association,
        warp_matrix=np.eye(2, 3, dtype=np.float64),
    )
    # Attach cost matrix row/col for richer features
    ev.track_cost_row = [0.30, 0.50, 0.70]
    ev.detection_cost_col = [0.30, 0.45, 0.60, 0.90]
    ev.max_iou_with_other = 0.15
    return ev


# ======================================================================
#  Tests for compute_scalar_features
# ======================================================================


class TestComputeScalarFeatures:
    """Verify the 63-dim scalar feature vector."""

    def test_returns_63_dim(self, sample_event):
        feats = compute_scalar_features(sample_event)
        assert feats.shape == (63,)
        assert feats.dtype == np.float64

    def test_accepts_dict(self, sample_event):
        from agentguard.contracts.serialization import serialize_event

        data = serialize_event(sample_event)
        # Add extra fields not covered by serialization
        data["track_cost_row"] = [0.30, 0.50, 0.70]
        data["detection_cost_col"] = [0.30, 0.45, 0.60, 0.90]
        data["max_iou_with_other"] = 0.15

        feats = compute_scalar_features(data)
        assert feats.shape == (63,)

        # Both paths should produce the same result
        feats_event = compute_scalar_features(sample_event)
        np.testing.assert_array_almost_equal(feats, feats_event)

    def test_association_block(self, sample_event):
        feats = compute_scalar_features(sample_event)

        # 0: IoU similarity
        assert feats[0] == pytest.approx(0.75)
        # 1: IoU distance
        assert feats[1] == pytest.approx(0.25)
        # 2: cosine distance
        assert feats[2] == pytest.approx(0.10)
        # 3: confidence distance
        assert feats[3] == pytest.approx(0.05)
        # 4: angle distance
        assert feats[4] == pytest.approx(0.08)
        # 5: raw cost
        assert feats[5] == pytest.approx(0.30)
        # 6: final cost
        assert feats[6] == pytest.approx(0.28)
        # 7: assignment threshold
        assert feats[7] == pytest.approx(0.50)
        # 8: min(round, 10) / 10
        assert feats[8] == pytest.approx(0.20)  # round=2
        # 9: detection score
        assert feats[9] == pytest.approx(0.85)

        # 10-12: source one-hot (detection_source=0 => HIGH)
        assert feats[10] == 1.0
        assert feats[11] == 0.0
        assert feats[12] == 0.0

    def test_row_statistics(self, sample_event):
        feats = compute_scalar_features(sample_event)
        # track_cost_row = [0.30, 0.50, 0.70]
        # sorted: [0.30, 0.50, 0.70]
        assert feats[13] == pytest.approx(0.30)  # min
        assert feats[14] == pytest.approx(0.50)  # second min
        assert feats[15] == pytest.approx(0.20)  # diff
        assert feats[16] == pytest.approx(0.50)  # mean
        assert feats[17] == pytest.approx(0.163299, abs=1e-5)  # std

    def test_column_statistics(self, sample_event):
        feats = compute_scalar_features(sample_event)
        # detection_cost_col = [0.30, 0.45, 0.60, 0.90]
        assert feats[19] == pytest.approx(0.30)  # min
        assert feats[20] == pytest.approx(0.45)  # second min
        assert feats[21] == pytest.approx(0.15)  # diff
        assert feats[22] == pytest.approx(0.5625)  # mean
        assert feats[25] == pytest.approx(0.15)  # max_iou_with_other
        assert feats[26] == 1.0  # has_detection

    def test_column_entropy(self):
        """Verify normalised column entropy with synthetic data."""
        feats = compute_scalar_features(
            {
                "has_detection": True,
                "image_width": 1920,
                "image_height": 1080,
                "frame_id": 100,
                "detection": {
                    "box": [100, 200, 300, 400],
                    "score": 0.9,
                    "source": 0,
                },
                "association": {
                    "iou_similarity": 0.5,
                    "raw_cost": 0.3,
                },
                "detection_cost_col": [0.10, 0.30, 0.50],
            }
        )
        # p = exp(-c/0.1) => exp(-1) = 0.368, exp(-3) = 0.050, exp(-5) = 0.0067
        # sum = 0.424, norm: [0.868, 0.118, 0.016]
        # H = -(0.868*log(0.868) + 0.118*log(0.118) + 0.016*log(0.016)) / log(3)
        p = np.exp(-np.array([0.10, 0.30, 0.50]) / 0.1)
        p = p / p.sum()
        expected_h = -np.sum(p * np.log(p + 1e-12)) / np.log(3)
        assert feats[24] == pytest.approx(float(expected_h), abs=1e-5)

    def test_no_detection(self, sample_snapshot):
        """All association features (0-26) should be zero when has_detection=False."""
        ev = TrackEvent(
            event_id="no-det",
            dataset="test",
            sequence="seq1",
            frame_id=100,
            track_id=42,
            image_width=1920,
            image_height=1080,
            has_detection=False,
            pre_update_state=sample_snapshot,
        )
        feats = compute_scalar_features(ev)
        # Association block (0-26) all zeros
        np.testing.assert_array_equal(feats[0:27], np.zeros(27))
        # has_detection is at index 26, should be 0
        assert feats[26] == 0.0

    def test_geometry_motion_block_with_detection(self, sample_event):
        feats = compute_scalar_features(sample_event)

        # pred: cx=200, cy=300, w=200, h=200
        # det:  cx=200, cy=300, w=200, h=200 (from box [100,200,300,400])
        pred_cx, pred_cy, pred_w, pred_h = 200.0, 300.0, 200.0, 200.0
        det_cx, det_cy, det_w, det_h = 200.0, 300.0, 200.0, 200.0
        im_w, im_h = 1920.0, 1080.0

        # 27: (det_cx - pred_cx) / max(pred_w, 1)
        assert feats[27] == pytest.approx(0.0)
        # 28: (det_cy - pred_cy) / max(pred_h, 1)
        assert feats[28] == pytest.approx(0.0)
        # 29: log(max(det_w, 1) / max(pred_w, 1))
        assert feats[29] == pytest.approx(0.0)
        # 30: log(max(det_h, 1) / max(pred_h, 1))
        assert feats[30] == pytest.approx(0.0)

        # 31-34: normalised pred box
        assert feats[31] == pytest.approx(pred_cx / im_w)
        assert feats[32] == pytest.approx(pred_cy / im_h)
        assert feats[33] == pytest.approx(pred_w / im_w)
        assert feats[34] == pytest.approx(pred_h / im_h)

        # 35-38: normalised det box
        assert feats[35] == pytest.approx(det_cx / im_w)
        assert feats[36] == pytest.approx(det_cy / im_h)
        assert feats[37] == pytest.approx(det_w / im_w)
        assert feats[38] == pytest.approx(det_h / im_h)

        # 39-42: mean[4:8] / pred box dims
        assert feats[39] == pytest.approx(2.0 / pred_w)
        assert feats[40] == pytest.approx(1.0 / pred_h)
        assert feats[41] == pytest.approx(0.1 / pred_w)
        assert feats[42] == pytest.approx(-0.05 / pred_h)

    def test_covariance_features(self, sample_event):
        feats = compute_scalar_features(sample_event)
        # cov[0,0]=4.0 / (1920^2)
        assert feats[43] == pytest.approx(math.log(4.0 / (1920**2) + 1e-6))
        assert feats[44] == pytest.approx(math.log(4.0 / (1080**2) + 1e-6))
        assert feats[45] == pytest.approx(math.log(1.0 / (1920**2) + 1e-6))
        assert feats[46] == pytest.approx(math.log(1.0 / (1080**2) + 1e-6))

    def test_velocity_flatten(self, sample_event):
        feats = compute_scalar_features(sample_event)
        # velocity (4,2) column-major
        expected = np.array([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8])
        np.testing.assert_array_almost_equal(feats[47:55], expected)

    def test_track_context(self, sample_event):
        feats = compute_scalar_features(sample_event)
        # 55: track score
        assert feats[55] == pytest.approx(0.80)

        # 56: most recent history score (frame 98)
        assert feats[56] == pytest.approx(0.80)

        # 57: mean of last 3 history scores [0.7, 0.75, 0.8]
        assert feats[57] == pytest.approx(0.75)

        # 58: min(history len, 50) / 50
        assert feats[58] == pytest.approx(3 / 50.0)

        # 59: min(frame_id - end_frame_id, 30) / 30
        assert feats[59] == pytest.approx(min(100 - 98, 30) / 30.0)

        # 60-62: state one-hot (TRACKED = 1)
        assert feats[60] == 0.0  # New
        assert feats[61] == 1.0  # Tracked
        assert feats[62] == 0.0  # Lost

    def test_single_element_cost_row(self):
        """When only raw_cost is available (no full row), use single-element stats."""
        ev = TrackEvent(
            event_id="single-cost",
            dataset="test",
            sequence="seq1",
            frame_id=100,
            track_id=1,
            image_width=640,
            image_height=480,
            has_detection=True,
            detection=DetectionObservation(
                detection_index=0,
                box=np.array([0, 0, 10, 10], dtype=np.float64),
                score=0.9,
                feature=np.zeros((1, 64), dtype=np.float64),
                source=0,
                class_id=1,
            ),
            association=AssociationPairFeatures(
                iou_similarity=0.5,
                iou_distance=0.5,
                cosine_distance=0.3,
                confidence_distance=0.1,
                angle_distance=0.2,
                raw_cost=0.35,
                final_cost=0.35,
                assignment_round=0,
                assignment_threshold=0.5,
                detection_source=0,
            ),
            pre_update_state=TrackStateSnapshot(
                track_id=1,
                box=np.array([0, 0, 10, 10], dtype=np.float64),
                score=0.8,
                mean=np.array([5.0, 5.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
                covariance=np.eye(8) * 0.01,
                velocity=np.zeros((4, 2)),
                feature=np.zeros((1, 64)),
                history={95: [np.array([0,0,10,10]), 0.8, np.zeros(8), np.eye(8)*0.01, np.zeros((1,64))]},
                end_frame_id=95,
                state=TrackLifecycle.TRACKED,
            ),
        )
        feats = compute_scalar_features(ev)
        # Row has single element [0.35]
        assert feats[13] == pytest.approx(0.35)  # min
        assert feats[14] == 1.0  # second min (default)
        assert feats[15] == pytest.approx(0.65)  # diff = 1.0 - 0.35
        assert feats[16] == pytest.approx(0.35)  # mean
        assert feats[17] == pytest.approx(0.0)  # std (single element)
        assert feats[18] == pytest.approx(0.0)  # entropy (n=1)

    def test_no_association_or_pre_update(self):
        """Event with no association and no pre_update_state should not crash."""
        ev = TrackEvent(
            event_id="minimal",
            dataset="test",
            sequence="seq1",
            frame_id=0,
            track_id=0,
            image_width=640,
            image_height=480,
            has_detection=False,
        )
        feats = compute_scalar_features(ev)
        assert feats.shape == (63,)
        # All association features are 0
        np.testing.assert_array_equal(feats[0:27], np.zeros(27))
        # Geometry features for no-detection
        np.testing.assert_array_equal(feats[27:43], np.zeros(16))
        # Covariance defaults (identity * 0.01 won't be applied since track_covariance is None)
        # When covariance is None, _safe_array returns zeros(8,8), so log(0+1e-6) = log(1e-6)
        assert feats[43] == pytest.approx(math.log(1e-6))
        # Track context zeros
        assert feats[55] == 0.0  # no track score
        assert feats[56] == 0.0  # no history
        assert feats[58] == 0.0  # history length 0

    def test_state_one_hot(self):
        """Verify state one-hot encoding for all three states."""
        for state_val, expected in [
            (TrackLifecycle.NEW, (1, 0, 0)),
            (TrackLifecycle.TRACKED, (0, 1, 0)),
            (TrackLifecycle.LOST, (0, 0, 1)),
        ]:
            snap = TrackStateSnapshot(
                track_id=1,
                box=np.zeros(4),
                score=0.5,
                mean=np.zeros(8),
                covariance=np.eye(8),
                velocity=np.zeros((4, 2)),
                feature=np.zeros((1, 64)),
                history={},
                end_frame_id=0,
                state=state_val,
            )
            ev = TrackEvent(
                event_id=f"state-{state_val}",
                dataset="test",
                sequence="seq1",
                frame_id=0,
                track_id=1,
                image_width=640,
                image_height=480,
                has_detection=False,
                pre_update_state=snap,
            )
            feats = compute_scalar_features(ev)
            assert (feats[60], feats[61], feats[62]) == expected


# ======================================================================
#  Tests for NormalizationStats
# ======================================================================


class TestNormalizationStats:
    def test_fit_transform(self):
        rng = np.random.default_rng(42)
        features = [rng.normal(loc=i * 10, scale=2, size=63) for i in range(100)]
        stats = NormalizationStats()
        stats.fit(features)

        # Means should be roughly 0, 10, 20, ...
        mean_actual = stats.mean
        expected_mean = np.mean(np.stack(features), axis=0)
        np.testing.assert_array_almost_equal(mean_actual, expected_mean)

        # Transform a single feature
        f = features[0]
        normalized = stats.transform(f)
        expected_norm = (f - stats.mean) / stats.std
        np.testing.assert_array_almost_equal(normalized, expected_norm)

        # Inverse
        restored = stats.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(restored, f)

    def test_constant_feature(self):
        """Constant features should have std replaced with 1.0."""
        features = [np.ones(63) * i for i in range(5)]
        stats = NormalizationStats()
        stats.fit(features)
        # All std should be >= 1.0 (or exactly 1.0 if constant)
        assert np.all(stats.std >= 1.0)

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "norm.npz")
        rng = np.random.default_rng(42)
        features = [rng.normal(size=63) for _ in range(50)]
        stats = NormalizationStats()
        stats.fit(features)
        stats.save(path)

        loaded = NormalizationStats.load(path)
        np.testing.assert_array_almost_equal(loaded.mean, stats.mean)
        np.testing.assert_array_almost_equal(loaded.std, stats.std)

    def test_empty_fit(self):
        stats = NormalizationStats()
        stats.fit([])
        np.testing.assert_array_equal(stats.mean, np.zeros(63))
        np.testing.assert_array_equal(stats.std, np.ones(63))


# ======================================================================
#  Tests for EventFeatureBuilder
# ======================================================================


class TestEventFeatureBuilder:
    def test_build(self, sample_event):
        builder = EventFeatureBuilder(reid_dim=3)
        scalar, track_reid, det_reid = builder.build(sample_event)

        assert scalar.shape == (63,)
        assert track_reid.shape == (3,)
        assert det_reid.shape == (3,)

        # Track ReID from pre_update_state.feature = [[0.5, 0.6, 0.7]]
        np.testing.assert_array_almost_equal(track_reid, [0.5, 0.6, 0.7])
        # Detection ReID = [[0.1, 0.2, 0.3]]
        np.testing.assert_array_almost_equal(det_reid, [0.1, 0.2, 0.3])

    def test_build_padding(self, sample_event):
        """ReID features shorter than reid_dim get zero-padded."""
        builder = EventFeatureBuilder(reid_dim=5)
        _, track_reid, det_reid = builder.build(sample_event)
        assert track_reid.shape == (5,)
        np.testing.assert_array_almost_equal(track_reid[:3], [0.5, 0.6, 0.7])
        np.testing.assert_array_equal(track_reid[3:], [0.0, 0.0])

    def test_build_truncation(self, sample_event):
        """ReID features longer than reid_dim get truncated."""
        builder = EventFeatureBuilder(reid_dim=2)
        _, track_reid, det_reid = builder.build(sample_event)
        assert track_reid.shape == (2,)
        np.testing.assert_array_almost_equal(track_reid, [0.5, 0.6])

    def test_build_no_detection_reid(self, sample_snapshot):
        """When has_detection=False, detection_reid should be zeros."""
        ev = TrackEvent(
            event_id="no-det",
            dataset="test",
            sequence="seq1",
            frame_id=100,
            track_id=42,
            image_width=1920,
            image_height=1080,
            has_detection=False,
            pre_update_state=sample_snapshot,
        )
        builder = EventFeatureBuilder(reid_dim=3)
        _, _, det_reid = builder.build(ev)
        np.testing.assert_array_equal(det_reid, np.zeros(3))

    def test_build_no_snapshot(self):
        """When pre_update_state is None, track_reid should be zeros."""
        ev = TrackEvent(
            event_id="no-snap",
            dataset="test",
            sequence="seq1",
            frame_id=0,
            track_id=0,
            image_width=640,
            image_height=480,
            has_detection=False,
        )
        builder = EventFeatureBuilder(reid_dim=4)
        _, track_reid, _ = builder.build(ev)
        np.testing.assert_array_equal(track_reid, np.zeros(4))

    def test_build_normalized(self, sample_event):
        stats = NormalizationStats()
        # Fit on some made-up data
        rng = np.random.default_rng(0)
        stats.fit([rng.normal(size=63) for _ in range(50)])

        builder = EventFeatureBuilder(reid_dim=3, norm_stats=stats)
        scalar_norm, _, _ = builder.build_normalized(sample_event)

        scalar_raw, _, _ = builder.build(sample_event)
        expected_norm = stats.transform(scalar_raw)
        np.testing.assert_array_almost_equal(scalar_norm, expected_norm)

    def test_build_normalized_no_stats(self, sample_event):
        """Without norm_stats, build_normalized should return raw features."""
        builder = EventFeatureBuilder(reid_dim=3)
        scalar_norm, _, _ = builder.build_normalized(sample_event)
        scalar_raw, _, _ = builder.build(sample_event)
        np.testing.assert_array_almost_equal(scalar_norm, scalar_raw)


# ======================================================================
#  Integration: dict with nested sub-dicts
# ======================================================================


class TestDictInput:
    def test_nested_dict(self):
        data: Dict[str, Any] = {
            "has_detection": True,
            "image_width": 1920,
            "image_height": 1080,
            "frame_id": 100,
            "association": {
                "iou_similarity": 0.8,
                "iou_distance": 0.2,
                "cosine_distance": 0.15,
                "confidence_distance": 0.05,
                "angle_distance": 0.10,
                "raw_cost": 0.25,
                "final_cost": 0.25,
                "assignment_round": 1,
                "assignment_threshold": 0.6,
                "detection_source": 1,  # LOW
            },
            "detection": {
                "box": [50, 60, 150, 200],
                "score": 0.75,
                "source": 1,
            },
            "pre_update_state": {
                "mean": [100.0, 130.0, 100.0, 140.0, 0.5, 0.3, 0.0, 0.0],
                "covariance": [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)],
                "velocity": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                "score": 0.70,
                "history": {
                    95: [np.array([0, 0, 10, 10]), 0.6, np.zeros(8), np.eye(8), np.zeros((1, 64))],
                },
                "end_frame_id": 95,
                "state": 1,  # Tracked
            },
            "track_cost_row": [0.25, 0.40, 0.60],
            "detection_cost_col": [0.25, 0.35, 0.55],
            "max_iou_with_other": 0.12,
        }
        feats = compute_scalar_features(data)
        assert feats.shape == (63,)

        # Spot checks
        assert feats[0] == pytest.approx(0.80)  # iou_similarity
        assert feats[9] == pytest.approx(0.75)  # det score
        assert feats[11] == 1.0  # source_low (detection_source=1)
        assert feats[26] == 1.0  # has_detection
        assert feats[60] == 0.0  # not New
        assert feats[61] == 1.0  # Tracked
        assert feats[62] == 0.0  # not Lost

    def test_flat_dict(self):
        """Keys at top level (no sub-dicts)."""
        data: Dict[str, Any] = {
            "has_detection": True,
            "image_width": 640,
            "image_height": 480,
            "frame_id": 50,
            "iou_similarity": 0.6,
            "raw_cost": 0.4,
            "det_score": 0.9,
            "detection_source": 2,  # NMS_DELETED_HIGH
            "det_box": [0, 0, 10, 10],
            "track_score": 0.85,
            "track_end_frame_id": 45,
            "track_state": 0,  # New
        }
        feats = compute_scalar_features(data)
        assert feats.shape == (63,)
        assert feats[0] == pytest.approx(0.60)
        assert feats[12] == 1.0  # source_nms_deleted_high
        assert feats[60] == 1.0  # New
        assert feats[61] == 0.0
        assert feats[62] == 0.0

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            compute_scalar_features(42)  # type: ignore[arg-type]
