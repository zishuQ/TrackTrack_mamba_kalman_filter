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
    
    def predict(self, mean, covariance, measurement, track_id):
        """
        Predict next state using motion model.
        
        Args:
            mean: numpy array of shape (8,) in pixel coordinates
            covariance: numpy array of shape (8, 8)
            measurement: numpy array of shape (4,) - current observation for computing innovation/DIoU
            track_id: unique identifier for the track
        
        Returns:
            mean: numpy array of shape (8,) in pixel coordinates
            covariance: numpy array of shape (8, 8)
        """
        with torch.no_grad():
            # Normalize inputs before passing to model
            norm_factor_4 = self._get_norm_factor_4()
            norm_factor_8 = self._get_norm_factor_8()
            mean_norm = mean / norm_factor_8
            measurement_norm = measurement / norm_factor_4
            
            # Convert to torch and add batch dimension
            mean_torch = torch.from_numpy(mean_norm).float().unsqueeze(0).to(self.device)
            covariance_torch = torch.from_numpy(covariance).float().to(self.device)
            measurement_torch = torch.from_numpy(measurement_norm).float().unsqueeze(0).to(self.device)
            
            mean_torch, covariance_torch = self.filter.predict(
                mean_torch, covariance_torch, measurement_torch, track_id
            )
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            # Denormalize mean back to pixel coordinates
            mean = mean * norm_factor_8
            
            return mean, covariance
    
    def project(self, mean, covariance, measurement, track_id):
        """
        Project state to measurement space.
        
        Args:
            mean: numpy array of shape (8,) in pixel coordinates
            covariance: numpy array of shape (8, 8)
            measurement: numpy array of shape (4,) - current observation for computing innovation/DIoU
            track_id: unique identifier for the track
        
        Returns:
            mean: numpy array of shape (4,) in pixel coordinates
            covariance: numpy array of shape (4, 4)
        """
        with torch.no_grad():
            # Normalize inputs before passing to model
            norm_factor_4 = self._get_norm_factor_4()
            norm_factor_8 = self._get_norm_factor_8()
            mean_norm = mean / norm_factor_8
            measurement_norm = measurement / norm_factor_4
            
            # Convert to torch and add batch dimension
            mean_torch = torch.from_numpy(mean_norm).float().unsqueeze(0).to(self.device)
            covariance_torch = torch.from_numpy(covariance).float().to(self.device)
            measurement_torch = torch.from_numpy(measurement_norm).float().unsqueeze(0).to(self.device)
            
            mean_torch, covariance_torch = self.filter.project(
                mean_torch, covariance_torch, measurement_torch, track_id
            )
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            # Denormalize mean back to pixel coordinates
            mean = mean * norm_factor_4
            
            return mean, covariance
    
    def update(self, mean, covariance, measurement, track_id):
        """
        Update state with new measurement.
        
        Args:
            mean: numpy array of shape (8,) in pixel coordinates
            covariance: numpy array of shape (8, 8)
            measurement: numpy array of shape (4,) - [cx, cy, w, h] in pixel coordinates
            track_id: unique identifier for the track
        
        Returns:
            mean: numpy array of shape (8,) in pixel coordinates
            covariance: numpy array of shape (8, 8)
        """
        with torch.no_grad():
            # Normalize inputs before passing to model
            norm_factor_4 = self._get_norm_factor_4()
            norm_factor_8 = self._get_norm_factor_8()
            mean_norm = mean / norm_factor_8
            measurement_norm = measurement / norm_factor_4
            
            # Convert to torch and add batch dimension
            mean_torch = torch.from_numpy(mean_norm).float().unsqueeze(0).to(self.device)
            covariance_torch = torch.from_numpy(covariance).float().to(self.device)
            measurement_torch = torch.from_numpy(measurement_norm).float().unsqueeze(0).to(self.device)
            
            mean_torch, covariance_torch = self.filter.update(mean_torch, covariance_torch, measurement_torch, track_id)
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            # Denormalize mean back to pixel coordinates
            mean = mean * norm_factor_8
            
            return mean, covariance
    
    def delete_track(self, track_id):
        """
        Delete a track and clean up its associated Mamba hidden states.
        
        Args:
            track_id: unique identifier for the track to delete
        """
        self.filter.delete_track(track_id)
