import numpy as np
import torch
import os


class MambaKalmanFilterWrapper(object):
    """
    Wrapper for MambaKalmanFilter to provide numpy-compatible interface.
    This class handles torch/numpy conversion and device management.
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
        self.filter = MambaKalmanFilter().to(self.device)
        self.filter.eval()  # Set to evaluation mode
        
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
        
        # Cache for standard KF parameters
        self.dim_x = 8
        self.dim_z = 4
        self.std_pos = 1. / 20.
        self.std_vel = 1. / 160.
    
    def initiate(self, measurement):
        """
        Initialize a new track.
        
        Args:
            measurement: numpy array of shape (4,) - [x, y, w, h]
        
        Returns:
            mean: numpy array of shape (8,)
            covariance: numpy array of shape (8, 8)
        """
        with torch.no_grad():
            # Convert to torch and add batch dimension
            measurement_torch = torch.from_numpy(measurement).float().unsqueeze(0).to(self.device)
            mean_torch, covariance_torch = self.filter.initiate(measurement_torch)
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            return mean, covariance
    
    def predict(self, mean, covariance, track_id):
        """
        Predict next state using motion model.
        
        Args:
            mean: numpy array of shape (8,)
            covariance: numpy array of shape (8, 8)
            track_id: unique identifier for the track
        
        Returns:
            mean: numpy array of shape (8,)
            covariance: numpy array of shape (8, 8)
        """
        with torch.no_grad():
            # Convert to torch and add batch dimension
            mean_torch = torch.from_numpy(mean).float().unsqueeze(0).to(self.device)
            covariance_torch = torch.from_numpy(covariance).float().to(self.device)
            
            mean_torch, covariance_torch = self.filter.predict(mean_torch, covariance_torch, track_id)
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            return mean, covariance
    
    def project(self, mean, covariance, track_id):
        """
        Project state to measurement space.
        
        Args:
            mean: numpy array of shape (8,)
            covariance: numpy array of shape (8, 8)
            track_id: unique identifier for the track
        
        Returns:
            mean: numpy array of shape (4,)
            covariance: numpy array of shape (4, 4)
        """
        with torch.no_grad():
            # Convert to torch and add batch dimension
            mean_torch = torch.from_numpy(mean).float().unsqueeze(0).to(self.device)
            covariance_torch = torch.from_numpy(covariance).float().to(self.device)
            
            mean_torch, covariance_torch = self.filter.project(mean_torch, covariance_torch, track_id)
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            return mean, covariance
    
    def update(self, mean, covariance, measurement, track_id):
        """
        Update state with new measurement.
        
        Args:
            mean: numpy array of shape (8,)
            covariance: numpy array of shape (8, 8)
            measurement: numpy array of shape (4,)
            track_id: unique identifier for the track
        
        Returns:
            mean: numpy array of shape (8,)
            covariance: numpy array of shape (8, 8)
        """
        with torch.no_grad():
            # Convert to torch and add batch dimension
            mean_torch = torch.from_numpy(mean).float().unsqueeze(0).to(self.device)
            covariance_torch = torch.from_numpy(covariance).float().to(self.device)
            measurement_torch = torch.from_numpy(measurement).float().unsqueeze(0).to(self.device)
            
            mean_torch, covariance_torch = self.filter.update(mean_torch, covariance_torch, measurement_torch, track_id)
            
            # Convert back to numpy
            mean = mean_torch.squeeze(0).cpu().numpy()
            covariance = covariance_torch.cpu().numpy()
            
            return mean, covariance
    
    def delete_track(self, track_id):
        """
        Delete a track and clean up its associated Mamba hidden states.
        
        Args:
            track_id: unique identifier for the track to delete
        """
        self.filter.delete_track(track_id)
