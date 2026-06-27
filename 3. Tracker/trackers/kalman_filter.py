import numpy as np
import scipy.linalg


class KalmanFilter(object):
    def __init__(self):
        # Set dim
        self.dim_x = 8
        self.dim_z = 4

        # Set motion matrix (F)
        self.motion_mat = np.eye(self.dim_x)
        for i in range(self.dim_x // 2):
            self.motion_mat[i, self.dim_z + i] = 1

        # Set update matrix (H)
        self.update_mat = np.eye(self.dim_z, self.dim_x)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self.std_pos = 1. / 20.
        self.std_vel = 1. / 160.

        # Set motion covariance
        self.motion_cov = np.eye(self.dim_x)
        self.motion_cov[:4, :4] *= self.std_pos
        self.motion_cov[4:, 4:] *= self.std_vel

        # Set innovation cov
        self.innovation_cov = np.eye(self.dim_z)
        self.innovation_cov *= self.std_pos

    def initiate(self, measurement):
        # Initialize mean
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initialize covariance
        covariance = np.eye(self.dim_x)
        covariance[:4, :4] *= 2.
        covariance[4:, 4:] *= 10.
        covariance[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        covariance[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        covariance = np.square(covariance)

        return mean, covariance

    def predict(self, mean, covariance):
        # Predict mean
        mean = np.dot(self.motion_mat, mean)

        # Get motion covariance
        motion_cov = self.motion_cov.copy()
        motion_cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        motion_cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        motion_cov = np.square(motion_cov)

        # Predict covariance
        covariance = np.linalg.multi_dot((self.motion_mat, covariance, self.motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence):
        # Project mean
        mean = np.dot(self.update_mat, mean)

        # Get innovation covariance
        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= mean[2]
        innovation_cov[[1, 3], [1, 3]] *= mean[3]
        innovation_cov = np.square(innovation_cov)

        # Noise Scale Adaptive
        innovation_cov *= (1 - confidence)

        # Project covariance
        covariance = np.linalg.multi_dot((self.update_mat, covariance, self.update_mat.T)) + innovation_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, confidence):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)

        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self.update_mat.T).T,
                                             check_finite=False).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance


class StandardKalmanFilter(KalmanFilter):
    """Linear KF without the NSA confidence-based measurement-noise scaling."""

    def project(self, mean, covariance, confidence=None):
        mean = np.dot(self.update_mat, mean)

        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= mean[2]
        innovation_cov[[1, 3], [1, 3]] *= mean[3]
        innovation_cov = np.square(innovation_cov)

        covariance = np.linalg.multi_dot((self.update_mat, covariance, self.update_mat.T)) + innovation_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, confidence=None):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self.update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance


class ExtendedKalmanFilter(StandardKalmanFilter):
    """EKF on the current linear cxcywh constant-velocity model."""

    def _motion_function(self, mean):
        return np.dot(self.motion_mat, mean)

    def _motion_jacobian(self, mean):
        return self.motion_mat

    def _measurement_function(self, mean):
        return np.dot(self.update_mat, mean)

    def _measurement_jacobian(self, mean):
        return self.update_mat

    def predict(self, mean, covariance):
        motion_jacobian = self._motion_jacobian(mean)
        mean = self._motion_function(mean)

        motion_cov = self.motion_cov.copy()
        motion_cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        motion_cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        motion_cov = np.square(motion_cov)

        covariance = np.linalg.multi_dot((motion_jacobian, covariance, motion_jacobian.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=None):
        update_jacobian = self._measurement_jacobian(mean)
        projected_mean = self._measurement_function(mean)

        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= projected_mean[2]
        innovation_cov[[1, 3], [1, 3]] *= projected_mean[3]
        innovation_cov = np.square(innovation_cov)

        projected_cov = np.linalg.multi_dot((update_jacobian, covariance, update_jacobian.T)) + innovation_cov

        return projected_mean, projected_cov


class UnscentedKalmanFilter(StandardKalmanFilter):
    """UKF on the current linear cxcywh constant-velocity model."""

    def __init__(self, alpha=1e-3, beta=2.0, kappa=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = self.alpha ** 2 * (self.dim_x + self.kappa) - self.dim_x
        self._initialize_weights()

    def _initialize_weights(self):
        scale = self.dim_x + self.lambda_
        self.Wm = np.full(2 * self.dim_x + 1, 1.0 / (2.0 * scale))
        self.Wc = np.full(2 * self.dim_x + 1, 1.0 / (2.0 * scale))
        self.Wm[0] = self.lambda_ / scale
        self.Wc[0] = self.lambda_ / scale + (1.0 - self.alpha ** 2 + self.beta)

    def _sigma_points(self, mean, covariance):
        scale = self.dim_x + self.lambda_
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))

        try:
            sqrt_cov = scipy.linalg.cholesky(
                scale * covariance + 1e-9 * np.eye(self.dim_x),
                lower=True,
                check_finite=False,
            )
        except scipy.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(covariance)
            if np.any(eigvals < -1e-7):
                raise
            eigvals = np.clip(eigvals, 0.0, None)
            sqrt_cov = np.sqrt(scale) * eigvecs @ np.diag(np.sqrt(eigvals))

        sigmas[0] = mean
        for i in range(self.dim_x):
            sigmas[i + 1] = mean + sqrt_cov[:, i]
            sigmas[self.dim_x + i + 1] = mean - sqrt_cov[:, i]

        return sigmas

    def _motion_covariance(self, mean):
        motion_cov = self.motion_cov.copy()
        motion_cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        motion_cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        return np.square(motion_cov)

    def _innovation_covariance(self, projected_mean):
        innovation_cov = self.innovation_cov.copy()
        innovation_cov[[0, 2], [0, 2]] *= projected_mean[2]
        innovation_cov[[1, 3], [1, 3]] *= projected_mean[3]
        return np.square(innovation_cov)

    def _weighted_covariance(self, residuals):
        return (self.Wc[:, None] * residuals).T @ residuals

    def predict(self, mean, covariance):
        sigmas = self._sigma_points(mean, covariance)
        sigmas_pred = sigmas @ self.motion_mat.T
        mean_pred = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)
        residuals = sigmas_pred - mean_pred[None, :]
        covariance_pred = self._weighted_covariance(residuals) + self._motion_covariance(mean_pred)

        return mean_pred, covariance_pred

    def project(self, mean, covariance, confidence=None):
        sigmas = self._sigma_points(mean, covariance)
        sigmas_meas = sigmas @ self.update_mat.T
        projected_mean = np.sum(self.Wm[:, None] * sigmas_meas, axis=0)
        meas_residuals = sigmas_meas - projected_mean[None, :]
        projected_cov = self._weighted_covariance(meas_residuals) + self._innovation_covariance(projected_mean)

        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement, confidence=None):
        sigmas = self._sigma_points(mean, covariance)
        sigmas_meas = sigmas @ self.update_mat.T
        projected_mean = np.sum(self.Wm[:, None] * sigmas_meas, axis=0)

        meas_residuals = sigmas_meas - projected_mean[None, :]
        projected_cov = self._weighted_covariance(meas_residuals) + self._innovation_covariance(projected_mean)
        state_residuals = sigmas - mean[None, :]
        cross_cov = (self.Wc[:, None] * state_residuals).T @ meas_residuals

        try:
            chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        except scipy.linalg.LinAlgError:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov + 1e-7 * np.eye(self.dim_z),
                lower=True,
                check_finite=False,
            )
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), cross_cov.T, check_finite=False).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        new_covariance = 0.5 * (new_covariance + new_covariance.T)

        return new_mean, new_covariance


def create_kalman_filter(kf_type="nsa"):
    normalized = (kf_type or "nsa").lower()
    if normalized == "nsa":
        return KalmanFilter()
    if normalized == "kf":
        return StandardKalmanFilter()
    if normalized == "ekf":
        return ExtendedKalmanFilter()
    if normalized == "ukf":
        return UnscentedKalmanFilter()
    raise ValueError(f"Unsupported kf_type: {kf_type}")
