import numpy as np
import scipy.linalg


class NSAKalmanFilter:
    """Noise Scale Adaptive (NSA) Kalman Filter.

    Matches TrackTrack's ``KalmanFilter`` implementation in
    ``3. Tracker/trackers/kalman_filter.py``.

    State (dim_x = 8):  [cx, cy, w, h, vx, vy, vw, vh]
    Measurement (dim_z = 4):  [cx, cy, w, h]
    """

    def __init__(self):
        self.dim_x = 8
        self.dim_z = 4

        # Motion matrix F  (constant-velocity model)
        self.motion_mat = np.eye(self.dim_x)
        for i in range(self.dim_x // 2):
            self.motion_mat[i, self.dim_z + i] = 1

        # Update matrix H  (selects position components)
        self.update_mat = np.eye(self.dim_z, self.dim_x)

        # Standard-deviation weights
        self.std_pos = 1.0 / 20.0
        self.std_vel = 1.0 / 160.0

        # Base motion covariance (before square + scaling)
        self.motion_cov = np.eye(self.dim_x)
        self.motion_cov[:4, :4] *= self.std_pos
        self.motion_cov[4:, 4:] *= self.std_vel

        # Base innovation covariance (before square + scaling)
        self.innovation_cov = np.eye(self.dim_z)
        self.innovation_cov *= self.std_pos

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate(self, measurement):
        """Create a new track state from a first detection.

        Parameters
        ----------
        measurement : ndarray, shape (4,)
            Bounding box in ``[cx, cy, w, h]`` format.

        Returns
        -------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)
        """
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

    def predict(self, mean, covariance):
        """Run one Kalman-filter prediction step.

        Parameters
        ----------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)

        Returns
        -------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)
        """
        mean = np.dot(self.motion_mat, mean)

        motion_cov = self.motion_cov.copy()
        motion_cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        motion_cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        motion_cov = np.square(motion_cov)

        covariance = np.linalg.multi_dot(
            (self.motion_mat, covariance, self.motion_mat.T)
        ) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence):
        """Project state distribution into measurement space.

        The measurement noise is scaled by (1 - confidence) --- this is the
        Noise Scale Adaptive (NSA) core.

        Parameters
        ----------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)
        confidence : float
            Detection confidence in [0, 1].

        Returns
        -------
        projected_mean : ndarray, shape (4,)
        projected_cov : ndarray, shape (4, 4)
        """
        projected_mean = np.dot(self.update_mat, mean)

        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= projected_mean[2]
        innovation_cov[[1, 3], [1, 3]] *= projected_mean[3]
        innovation_cov = np.square(innovation_cov)

        # Noise Scale Adaptive: reduce measurement noise for confident detections
        innovation_cov *= 1.0 - confidence

        projected_cov = np.linalg.multi_dot(
            (self.update_mat, covariance, self.update_mat.T)
        ) + innovation_cov

        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement, confidence):
        """Run one Kalman-filter update step.

        Parameters
        ----------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)
        measurement : ndarray, shape (4,)
            Bounding box in ``[cx, cy, w, h]`` format.
        confidence : float
            Detection confidence in [0, 1].

        Returns
        -------
        new_mean : ndarray, shape (8,)
        new_covariance : ndarray, shape (8, 8)
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

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

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def mean_to_bbox(mean):
        """Convert state mean to ``[x1, y1, x2, y2]`` bounding box."""
        cx, cy, w, h = mean[0], mean[1], mean[2], mean[3]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @staticmethod
    def bbox_to_measurement(box):
        """Convert ``[x1, y1, x2, y2]`` bounding box to ``[cx, cy, w, h]``."""
        return np.array([
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            box[2] - box[0],
            box[3] - box[1],
        ])

    @staticmethod
    def apply_warp(mean, covariance, warp_matrix):
        """Apply a 2×3 warp (rotation + translation) to the state.

        This mirrors ``apply_cmc`` in TrackTrack's
        ``3. Tracker/trackers/cmc.py``.

        Parameters
        ----------
        mean : ndarray, shape (8,)
        covariance : ndarray, shape (8, 8)
        warp_matrix : ndarray, shape (2, 3)
            Affine warp matrix, typically produced by GMC / ECC.

        Returns
        -------
        mean_new : ndarray, shape (8,)
        cov_new : ndarray, shape (8, 8)
        """
        rot = warp_matrix[:, :2]  # (2, 2)
        rot_8x8 = np.kron(np.eye(4, dtype=float), rot)  # (8, 8)
        trans = warp_matrix[:, 2]  # (2,)

        mean_new = rot_8x8 @ mean.copy()
        mean_new[:2] += trans
        cov_new = rot_8x8 @ covariance @ rot_8x8.T

        return mean_new, cov_new
