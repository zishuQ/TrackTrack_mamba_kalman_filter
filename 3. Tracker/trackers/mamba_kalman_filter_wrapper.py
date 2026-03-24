import torch
import numpy as np
import os

from mamba_kalman_filter.config import Config


class MambaKalmanFilterWrapper(object):
    """
    Wrapper for MambaKalmanFilter to provide numpy-compatible interface.
    This class handles torch/numpy conversion, device management, and normalization.
    
    The model was trained with normalized inputs:
    - Position/size (x, y, w, h) normalized by (W, H, W, H)
    - Velocity (vx, vy, vw, vh) normalized by (W, H, W, H)
    - Coordinates are in center format (cx, cy, w, h)
    """
    _shared_instance = None
    
    @classmethod
    def get_shared_instance(cls, model_path=None, device='cuda'):
        """Get or create a shared instance of the Mamba Kalman Filter."""
        if cls._shared_instance is None:
            cls._shared_instance = cls(model_path=model_path, device=device)
        return cls._shared_instance
    
    def __init__(self, model_path=None, device='cuda'):
        from mamba_kalman_filter.mamba_kalmanfilter import MambaKalmanFilter
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.filter = MambaKalmanFilter(Config()).to(self.device)
        self.filter.eval()  # Set to evaluation mode
        # Image dimensions for normalization (will be set per video sequence)
        self.img_width = None
        self.img_height = None
        # Pre-computed normalization factors on GPU (for batch operations)
        self._norm_factor_4_gpu = None
        self._norm_factor_8_gpu = None
        # Load pretrained weights if provided
        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.filter.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.filter.load_state_dict(checkpoint)
            print(f"Loaded MambaKalmanFilter weights from {model_path}")
        else:
            print("Warning: No model weights loaded, using randomly initialized MambaKalmanFilter")
    
    def set_image_size(self, img_width, img_height):
        """
        Set image dimensions for normalization.
        Should be called when switching to a new video sequence.
        
        Args:
            img_width: Width of the image/video frame
            img_height: Height of the image/video frame
        """
        self.img_width = img_width
        self.img_height = img_height
        # Pre-compute normalization factors on GPU (avoid repeated creation)
        self._norm_factor_4_gpu = torch.tensor(
            [img_width, img_height, img_width, img_height],
            dtype=torch.float32, device=self.device
        )
        self._norm_factor_8_gpu = torch.tensor(
            [img_width, img_height, img_width, img_height,
             img_width, img_height, img_width, img_height],
            dtype=torch.float32, device=self.device
        )
    
    def _get_norm_factor_4(self):
        """Get normalization factor for measurement (4,): [W, H, W, H]"""
        return np.array([self.img_width, self.img_height, self.img_width, self.img_height], dtype=np.float32)
    
    def _get_norm_factor_8(self):
        """Get normalization factor for state (8,): [W, H, W, H, W, H, W, H]"""
        return np.array([self.img_width, self.img_height, self.img_width, self.img_height,
                         self.img_width, self.img_height, self.img_width, self.img_height], dtype=np.float32)
    
    def initiate(self, measurement):
        """
        Initialize a new track.
        
        Args:
            measurement: numpy array of shape (4,) - [cx, cy, w, h] in pixel coordinates
        
        Returns:
            mean: numpy array of shape (8,) in pixel coordinates
            covariance: numpy array of shape (8, 8)
        """
        with torch.no_grad():
            # Normalize measurement before passing to model
            norm_factor_4 = self._get_norm_factor_4()
            measurement_norm = measurement / norm_factor_4
            # Convert to torch and add batch dimension
            measurement_torch = torch.from_numpy(measurement_norm).float().unsqueeze(0).to(self.device)
            mean_torch, covariance_torch = self.filter.initiate(measurement_torch)
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            # Denormalize mean back to pixel coordinates
            norm_factor_8 = self._get_norm_factor_8()
            mean = mean * norm_factor_8
            return mean, covariance
    
    def delete_track(self, track_id):
        """
        Delete a track and clean up its associated Mamba hidden states.
        
        Args:
            track_id: unique identifier for the track to delete
        """
        self.filter.delete_track(track_id)
    
    # ========== Batch Operations (Optimized for reduced CPU-GPU transfers) ==========
    
    def batch_predict(self, means, covariances, measurements, track_ids):
        """
        Batch predict all tracks' next states (single GPU transfer + single Mamba forward).
        
        将所有 track 打包成一个 batch，单次 Mamba forward 完成全部 predict，
        消除原本逐 track 的 Python 循环和多次 GPU kernel launch。
        
        Args:
            means: numpy array (N, 8) - all tracks' current states in pixel coordinates
            covariances: numpy array (N, 8, 8) - all tracks' covariances
            measurements: numpy array (N, 4) - all tracks' current observations for innovation/DIoU
            track_ids: list[int] - track ID list for managing Mamba hidden states
        
        Returns:
            means: numpy array (N, 8) in pixel coordinates
            covariances: numpy array (N, 8, 8)
        """
        if len(track_ids) == 0:
            return means, covariances
        with torch.no_grad():
            # Single transfer: numpy -> GPU
            means_gpu = torch.from_numpy(means).float().to(self.device)
            covariances_gpu = torch.from_numpy(covariances).float().to(self.device)
            measurements_gpu = torch.from_numpy(measurements).float().to(self.device)
            # Normalize on GPU
            means_norm = means_gpu / self._norm_factor_8_gpu
            measurements_norm = measurements_gpu / self._norm_factor_4_gpu
            
            # 单次批量 predict（替代 N 次循环调用）
            means_out, covariances_out = self.filter.batch_predict(
                means_norm,         # (N, 8)
                covariances_gpu,    # (N, 8, 8)
                measurements_norm,  # (N, 4)
                track_ids,
            )
            # Denormalize on GPU
            means_out = means_out * self._norm_factor_8_gpu
            # Single transfer: GPU -> numpy
            return means_out.cpu().numpy(), covariances_out.cpu().numpy()
    
    def batch_update(self, means, covariances, measurements, track_ids):
        """
        Batch update all matched tracks (single GPU transfer + single Mamba forward).
        
        NIS gate 关闭时（默认），走批量路径；开启时回退到逐 track 循环。
        
        Args:
            means: numpy array (N, 8) in pixel coordinates
            covariances: numpy array (N, 8, 8)
            measurements: numpy array (N, 4) - new observations in pixel coordinates
            track_ids: list[int]
        
        Returns:
            means: numpy array (N, 8) in pixel coordinates
            covariances: numpy array (N, 8, 8)
        """
        if len(track_ids) == 0:
            return means, covariances
        with torch.no_grad():
            # Single transfer to GPU
            means_gpu = torch.from_numpy(means).float().to(self.device)
            covariances_gpu = torch.from_numpy(covariances).float().to(self.device)
            measurements_gpu = torch.from_numpy(measurements).float().to(self.device)
            # Normalize on GPU
            means_norm = means_gpu / self._norm_factor_8_gpu
            measurements_norm = measurements_gpu / self._norm_factor_4_gpu
            # 快速路径：单次批量 update（替代 N 次循环调用）
            means_out, covariances_out = self.filter.batch_update(
                means_norm,         # (N, 8)
                covariances_gpu,    # (N, 8, 8)
                measurements_norm,  # (N, 4)
                track_ids,
            )
            # Denormalize
            means_out = means_out * self._norm_factor_8_gpu
            # Single transfer back to CPU
            return means_out.cpu().numpy(), covariances_out.cpu().numpy()
    
    def batch_predict_gpu(self, means_gpu, covariances_gpu, measurements_gpu, track_ids):
        """
        GPU-native batch predict. Accepts/returns GPU tensors (pixel coords).
        
        与 batch_predict 逻辑相同，但跳过 numpy↔torch 转换，
        供 tracker 在 GPU 上缓存预测结果、减少 CPU↔GPU 传输次数。
        
        Args:
            means_gpu: torch tensor (N, 8) on GPU, pixel coordinates
            covariances_gpu: torch tensor (N, 8, 8) on GPU
            measurements_gpu: torch tensor (N, 4) on GPU, pixel coordinates
            track_ids: list[int]
        
        Returns:
            means_out: torch tensor (N, 8) on GPU, pixel coordinates
            covariances_out: torch tensor (N, 8, 8) on GPU
        """
        means_norm = means_gpu / self._norm_factor_8_gpu
        measurements_norm = measurements_gpu / self._norm_factor_4_gpu
        means_out, covariances_out = self.filter.batch_predict(
            means_norm, covariances_gpu, measurements_norm, track_ids,
        )
        means_out = means_out * self._norm_factor_8_gpu
        return means_out, covariances_out
    
    def batch_update_gpu(self, means_gpu, covariances_gpu, measurements_gpu, track_ids):
        """
        GPU-native batch update. Accepts/returns GPU tensors (pixel coords).
        
        Args:
            means_gpu: torch tensor (N, 8) on GPU, pixel coordinates
            covariances_gpu: torch tensor (N, 8, 8) on GPU
            measurements_gpu: torch tensor (N, 4) on GPU, pixel coordinates
            track_ids: list[int]
        
        Returns:
            means_out: torch tensor (N, 8) on GPU, pixel coordinates
            covariances_out: torch tensor (N, 8, 8) on GPU
        """
        means_norm = means_gpu / self._norm_factor_8_gpu
        measurements_norm = measurements_gpu / self._norm_factor_4_gpu
        means_out, covariances_out = self.filter.batch_update(
            means_norm, covariances_gpu, measurements_norm, track_ids,
        )
        means_out = means_out * self._norm_factor_8_gpu
        return means_out, covariances_out
    
    def batch_initiate(self, measurements):
        """
        Batch initialize new tracks (single GPU transfer).
        
        Args:
            measurements: numpy array (N, 4) in pixel coordinates
        
        Returns:
            means: numpy array (N, 8) in pixel coordinates
            covariances: numpy array (N, 8, 8)
        """
        if len(measurements) == 0:
            return np.array([]).reshape(0, 8), np.array([]).reshape(0, 8, 8)
        with torch.no_grad():
            measurements_gpu = torch.from_numpy(measurements).float().to(self.device)
            measurements_norm = measurements_gpu / self._norm_factor_4_gpu
            means_out, covariances_out = self.filter.initiate(measurements_norm)
            # initiate may return (8, 8) for single track, ensure (N, 8, 8)
            if covariances_out.dim() == 2:
                covariances_out = covariances_out.unsqueeze(0)
            means_out = means_out * self._norm_factor_8_gpu
            return means_out.cpu().numpy(), covariances_out.cpu().numpy()
