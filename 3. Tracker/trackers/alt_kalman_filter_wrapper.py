import os
import sys
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.dirname(THIS_DIR)
PROJECT_DIR = os.path.abspath(os.path.join(TRACKER_DIR, '..'))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..'))
for p in (PROJECT_DIR, WORKSPACE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from mamba_kalman_filter.config_alt import ConfigAlt
except ImportError:
    from config_alt import ConfigAlt


class AltKalmanFilterWrapper(object):
    """
    Wrapper for AltKalmanFilter to provide numpy-compatible interface.
    """

    _shared_instance = None

    @classmethod
    def get_shared_instance(cls, model_path=None, device='cuda', backbone='lstm'):
        """Get or create a shared instance of the alternative Kalman filter."""
        if cls._shared_instance is None:
            cls._shared_instance = cls(model_path=model_path, device=device, backbone=backbone)
        elif getattr(cls._shared_instance, '_backbone_name', None) != backbone:
            # Keep singleton semantics but avoid stale backbone configuration.
            cls._shared_instance = cls(model_path=model_path, device=device, backbone=backbone)
        return cls._shared_instance

    def __init__(self, model_path=None, device='cuda', backbone='lstm'):
        try:
            from mamba_kalman_filter.alt_kalmanfilter import AltKalmanFilter
        except ImportError:
            from alt_kalmanfilter import AltKalmanFilter

        ConfigAlt.set_alt_backbone(backbone)
        ConfigAlt.apply_param_match_preset()
        self._backbone_name = backbone
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.filter = AltKalmanFilter(ConfigAlt()).to(self.device)
        self.filter.eval()

        self.img_width = None
        self.img_height = None
        self._norm_factor_4_gpu = None
        self._norm_factor_8_gpu = None

        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.filter.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.filter.load_state_dict(checkpoint)
            print(f"Loaded AltKalmanFilter weights from {model_path}")
        else:
            print("Warning: No model weights loaded, using randomly initialized AltKalmanFilter")

    def set_image_size(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self._norm_factor_4_gpu = torch.tensor(
            [img_width, img_height, img_width, img_height],
            dtype=torch.float32,
            device=self.device,
        )
        self._norm_factor_8_gpu = torch.tensor(
            [img_width, img_height, img_width, img_height,
             img_width, img_height, img_width, img_height],
            dtype=torch.float32,
            device=self.device,
        )

    def _get_norm_factor_4(self):
        return np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height],
            dtype=np.float32,
        )

    def _get_norm_factor_8(self):
        return np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height,
             self.img_width, self.img_height, self.img_width, self.img_height],
            dtype=np.float32,
        )

    def initiate(self, measurement):
        with torch.no_grad():
            measurement_norm = measurement / self._get_norm_factor_4()
            measurement_torch = torch.from_numpy(measurement_norm).float().unsqueeze(0).to(self.device)
            mean_torch, covariance_torch = self.filter.initiate(measurement_torch)
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            mean = mean * self._get_norm_factor_8()
            return mean, covariance

    def delete_track(self, track_id):
        self.filter.delete_track(track_id)

    def batch_predict(self, means, covariances, measurements, track_ids):
        if len(track_ids) == 0:
            return means, covariances
        with torch.no_grad():
            means_gpu = torch.from_numpy(means).float().to(self.device)
            covariances_gpu = torch.from_numpy(covariances).float().to(self.device)
            measurements_gpu = torch.from_numpy(measurements).float().to(self.device)
            means_norm = means_gpu / self._norm_factor_8_gpu
            measurements_norm = measurements_gpu / self._norm_factor_4_gpu
            means_out, covariances_out = self.filter.batch_predict(
                means_norm,
                covariances_gpu,
                measurements_norm,
                track_ids,
            )
            means_out = means_out * self._norm_factor_8_gpu
            return means_out.cpu().numpy(), covariances_out.cpu().numpy()

    def batch_update(self, means, covariances, measurements, track_ids):
        if len(track_ids) == 0:
            return means, covariances
        with torch.no_grad():
            means_gpu = torch.from_numpy(means).float().to(self.device)
            covariances_gpu = torch.from_numpy(covariances).float().to(self.device)
            measurements_gpu = torch.from_numpy(measurements).float().to(self.device)
            means_norm = means_gpu / self._norm_factor_8_gpu
            measurements_norm = measurements_gpu / self._norm_factor_4_gpu
            means_out, covariances_out = self.filter.batch_update(
                means_norm,
                covariances_gpu,
                measurements_norm,
                track_ids,
            )
            means_out = means_out * self._norm_factor_8_gpu
            return means_out.cpu().numpy(), covariances_out.cpu().numpy()

    def batch_predict_gpu(self, means_gpu, covariances_gpu, measurements_gpu, track_ids):
        means_norm = means_gpu / self._norm_factor_8_gpu
        measurements_norm = measurements_gpu / self._norm_factor_4_gpu
        means_out, covariances_out = self.filter.batch_predict(
            means_norm, covariances_gpu, measurements_norm, track_ids
        )
        means_out = means_out * self._norm_factor_8_gpu
        return means_out, covariances_out

    def batch_update_gpu(self, means_gpu, covariances_gpu, measurements_gpu, track_ids):
        means_norm = means_gpu / self._norm_factor_8_gpu
        measurements_norm = measurements_gpu / self._norm_factor_4_gpu
        means_out, covariances_out = self.filter.batch_update(
            means_norm, covariances_gpu, measurements_norm, track_ids
        )
        means_out = means_out * self._norm_factor_8_gpu
        return means_out, covariances_out

    def batch_initiate(self, measurements):
        if len(measurements) == 0:
            return np.array([]).reshape(0, 8), np.array([]).reshape(0, 8, 8)
        with torch.no_grad():
            measurements_gpu = torch.from_numpy(measurements).float().to(self.device)
            measurements_norm = measurements_gpu / self._norm_factor_4_gpu
            means_out, covariances_out = self.filter.initiate(measurements_norm)
            if covariances_out.dim() == 2:
                covariances_out = covariances_out.unsqueeze(0)
            means_out = means_out * self._norm_factor_8_gpu
            return means_out.cpu().numpy(), covariances_out.cpu().numpy()
