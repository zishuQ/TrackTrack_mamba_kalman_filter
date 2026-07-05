"""Test NSAKalmanFilter against TrackTrack's KalmanFilter for numeric parity.

The NSAKalmanFilter is designed to match TrackTrack's KalmanFilter.
We verify that initiate, predict, update, and bbox conversion produce
identical results within 1e-7 tolerance.
"""
from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.motion.nsa_numpy import NSAKalmanFilter


# ──────────────────────────────────────────────────────────────────────
#  A reference KalmanFilter matching TrackTrack's implementation
# ──────────────────────────────────────────────────────────────────────


class ReferenceKalmanFilter:
    """Reference implementation matching TrackTrack's KalmanFilter.

    Same motion model, same noise parameters, same math.
    """

    def __init__(self):
        self.dim_x = 8
        self.dim_z = 4

        # Motion matrix (constant velocity)
        self.motion_mat = np.eye(self.dim_x)
        for i in range(self.dim_x // 2):
            self.motion_mat[i, self.dim_z + i] = 1

        # Update matrix
        self.update_mat = np.eye(self.dim_z, self.dim_x)

        # Standard-deviation weights
        self.std_pos = 1.0 / 20.0
        self.std_vel = 1.0 / 160.0

        # Motion covariance
        self.motion_cov = np.eye(self.dim_x)
        self.motion_cov[:4, :4] *= self.std_pos
        self.motion_cov[4:, 4:] *= self.std_vel

        # Innovation covariance
        self.innovation_cov = np.eye(self.dim_z)
        self.innovation_cov *= self.std_pos

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        covariance = np.eye(self.dim_x)
        covariance[:4, :4] *= 2.0
        covariance[4:, 4:] *= 10.0
        covariance[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        covariance[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        covariance = np.square(covariance)

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.dot(self.motion_mat, mean)

        motion_cov = self.motion_cov.copy()
        motion_cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        motion_cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        motion_cov = np.square(motion_cov)

        covariance = np.linalg.multi_dot(
            (self.motion_mat, covariance, self.motion_mat.T)
        ) + motion_cov

        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import scipy.linalg

        projected_mean = np.dot(self.update_mat, mean)

        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= projected_mean[2]
        innovation_cov[[1, 3], [1, 3]] *= projected_mean[3]
        innovation_cov = np.square(innovation_cov)
        innovation_cov *= 1.0 - confidence

        projected_cov = np.linalg.multi_dot(
            (self.update_mat, covariance, self.update_mat.T)
        ) + innovation_cov

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self.update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance

    @staticmethod
    def mean_to_bbox(mean: np.ndarray) -> np.ndarray:
        cx, cy, w, h = mean[0], mean[1], mean[2], mean[3]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @staticmethod
    def bbox_to_measurement(box: np.ndarray) -> np.ndarray:
        return np.array([
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            box[2] - box[0],
            box[3] - box[1],
        ])


# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def nsa_kf() -> NSAKalmanFilter:
    return NSAKalmanFilter()


@pytest.fixture
def ref_kf() -> ReferenceKalmanFilter:
    return ReferenceKalmanFilter()


@pytest.fixture
def measurement() -> np.ndarray:
    """A realistic detection measurement [cx, cy, w, h]."""
    return np.array([200.0, 300.0, 200.0, 200.0], dtype=np.float64)


@pytest.fixture
def box_x1y1x2y2() -> np.ndarray:
    """A realistic bounding box."""
    return np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────


class TestInitiate:
    def test_initiate_parity(self, nsa_kf, ref_kf, measurement):
        """initiate should produce identical mean/covariance."""
        m1, c1 = nsa_kf.initiate(measurement)
        m2, c2 = ref_kf.initiate(measurement)

        np.testing.assert_array_almost_equal(m1, m2, decimal=7)
        np.testing.assert_array_almost_equal(c1, c2, decimal=7)

    def test_initiate_different_measurement(self, nsa_kf, ref_kf):
        """initiate should match for different box sizes."""
        meas = np.array([50.0, 100.0, 30.0, 60.0], dtype=np.float64)
        m1, c1 = nsa_kf.initiate(meas)
        m2, c2 = ref_kf.initiate(meas)

        np.testing.assert_array_almost_equal(m1, m2, decimal=7)
        np.testing.assert_array_almost_equal(c1, c2, decimal=7)


class TestPredict:
    def test_predict_parity(self, nsa_kf, ref_kf, measurement):
        """predict after initiate should match."""
        mean, cov = nsa_kf.initiate(measurement)
        m1, c1 = nsa_kf.predict(mean, cov)

        mean2, cov2 = ref_kf.initiate(measurement)
        m2, c2 = ref_kf.predict(mean2, cov2)

        np.testing.assert_array_almost_equal(m1, m2, decimal=7)
        np.testing.assert_array_almost_equal(c1, c2, decimal=7)

    def test_predict_multiple_steps(self, nsa_kf, ref_kf, measurement):
        """Multiple predict steps should remain in sync."""
        m_nsa, c_nsa = nsa_kf.initiate(measurement)
        m_ref, c_ref = ref_kf.initiate(measurement)

        for _ in range(5):
            m_nsa, c_nsa = nsa_kf.predict(m_nsa, c_nsa)
            m_ref, c_ref = ref_kf.predict(m_ref, c_ref)

        np.testing.assert_array_almost_equal(m_nsa, m_ref, decimal=7)
        np.testing.assert_array_almost_equal(c_nsa, c_ref, decimal=7)


class TestUpdate:
    def test_update_parity(self, nsa_kf, ref_kf, measurement):
        """Full initiate -> predict -> update cycle should match."""
        # NSA path
        m, c = nsa_kf.initiate(measurement)
        m, c = nsa_kf.predict(m, c)
        m1, c1 = nsa_kf.update(m, c, measurement, confidence=0.9)

        # Reference path
        m, c = ref_kf.initiate(measurement)
        m, c = ref_kf.predict(m, c)
        m2, c2 = ref_kf.update(m, c, measurement, confidence=0.9)

        np.testing.assert_array_almost_equal(m1, m2, decimal=7)
        np.testing.assert_array_almost_equal(c1, c2, decimal=7)

    def test_update_different_confidences(self, nsa_kf, ref_kf, measurement):
        """Update with different confidence values should match."""
        for conf in [0.0, 0.5, 1.0]:
            m_nsa, c_nsa = nsa_kf.initiate(measurement)
            m_ref, c_ref = ref_kf.initiate(measurement)

            m_nsa, c_nsa = nsa_kf.predict(m_nsa, c_nsa)
            m_ref, c_ref = ref_kf.predict(m_ref, c_ref)

            m_nsa_upd, c_nsa_upd = nsa_kf.update(m_nsa, c_nsa, measurement, conf)
            m_ref_upd, c_ref_upd = ref_kf.update(m_ref, c_ref, measurement, conf)

            np.testing.assert_array_almost_equal(m_nsa_upd, m_ref_upd, decimal=7)
            np.testing.assert_array_almost_equal(c_nsa_upd, c_ref_upd, decimal=7)


class TestBboxConversion:
    def test_bbox_roundtrip(self, nsa_kf, box_x1y1x2y2):
        """x1y1x2y2 -> measurement -> mean_to_bbox should recover the original box."""
        meas = nsa_kf.bbox_to_measurement(box_x1y1x2y2)
        recovered = nsa_kf.mean_to_bbox(np.array([meas[0], meas[1], meas[2], meas[3], 0, 0, 0, 0]))
        np.testing.assert_array_almost_equal(box_x1y1x2y2, recovered, decimal=10)

    def test_bbox_conversion_parity(self, nsa_kf, ref_kf, box_x1y1x2y2):
        """Both implementations should convert boxes identically."""
        meas1 = nsa_kf.bbox_to_measurement(box_x1y1x2y2)
        meas2 = ref_kf.bbox_to_measurement(box_x1y1x2y2)
        np.testing.assert_array_almost_equal(meas1, meas2, decimal=10)

        mean1 = nsa_kf.mean_to_bbox(np.array([100, 200, 50, 60, 0, 0, 0, 0]))
        mean2 = ref_kf.mean_to_bbox(np.array([100, 200, 50, 60, 0, 0, 0, 0]))
        np.testing.assert_array_almost_equal(mean1, mean2, decimal=10)

    def test_bbox_edge_cases(self, nsa_kf, ref_kf):
        """Edge cases: zero-size box."""
        box = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        meas = nsa_kf.bbox_to_measurement(box)
        assert meas[2] == 0.0  # width
        assert meas[3] == 0.0  # height

    def test_bbox_negative_values(self, nsa_kf):
        """Negative box coordinates should convert correctly."""
        box = np.array([-50.0, -50.0, 50.0, 50.0], dtype=np.float64)
        meas = nsa_kf.bbox_to_measurement(box)
        assert meas[0] == pytest.approx(0.0)
        assert meas[1] == pytest.approx(0.0)
        assert meas[2] == pytest.approx(100.0)
        assert meas[3] == pytest.approx(100.0)


class TestApplyWarp:
    def test_apply_warp_identity(self, nsa_kf, measurement):
        """Applying identity warp should not change state."""
        mean, cov = nsa_kf.initiate(measurement)
        warp = np.eye(2, 3, dtype=np.float64)
        m_new, c_new = nsa_kf.apply_warp(mean, cov, warp)

        np.testing.assert_array_almost_equal(mean, m_new, decimal=10)
        np.testing.assert_array_almost_equal(cov, c_new, decimal=10)

    def test_apply_warp_translation(self, nsa_kf, measurement):
        """Translation warp should shift position by the translation."""
        mean, cov = nsa_kf.initiate(measurement)
        dx, dy = 10.0, -5.0
        warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        m_new, c_new = nsa_kf.apply_warp(mean, cov, warp)

        np.testing.assert_array_almost_equal(m_new[:2], mean[:2] + [dx, dy], decimal=10)

    def test_apply_warp_rotation(self, nsa_kf, measurement):
        """Rotation warp should preserve state dimension."""
        mean, cov = nsa_kf.initiate(measurement)
        theta = np.radians(10.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        warp = np.column_stack([rot, [0.0, 0.0]])  # (2, 3)
        m_new, c_new = nsa_kf.apply_warp(mean, cov, warp)

        assert m_new.shape == (8,)
        assert c_new.shape == (8, 8)
