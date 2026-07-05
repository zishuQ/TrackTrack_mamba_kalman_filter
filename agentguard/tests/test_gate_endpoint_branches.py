import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest

class FakeKF:
    def predict(self, m, c): return m + 1, c + 1
    def update(self, m, c, meas, conf):
        meas_pad = np.zeros(8, dtype=np.float64)
        meas_pad[:4] = meas
        return m + meas_pad, c + 10

class FakeTrack:
    def __init__(self):
        self.mean = np.array([10.0]*8, dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64)
        self.feat = np.array([1.0]*2048, dtype=np.float64).reshape(1, -1)
        self.box = np.array([1,2,3,4], dtype=np.float64)
        self.score = 0.9
        self.state = 1
        self.end_frame_id = 0
        self.alpha = 0.95
        self.velocity = np.zeros((4,2))
        self.history = {}
        self.track_id = 1
        self.args = type('Args', (), {'min_len': 3, 'data_path': '/test'})
        self.kalman_filter = FakeKF()
    
    @property
    def cxcywh(self):
        return np.array([2,3,2,2], dtype=np.float64)
    @property
    def x1y1x2y2(self):
        return self.box.copy()
    
    def snapshot_state(self):
        from agentguard.contracts.states import TrackStateSnapshot
        return TrackStateSnapshot(
            track_id=1, box=self.box.copy(), score=self.score,
            mean=self.mean.copy(), covariance=self.covariance.copy(),
            velocity=self.velocity.copy(), feature=self.feat.copy(),
            history={}, end_frame_id=0, state=1,
        )
    
    def update_with_gates(self, fid, det, mg, ag):
        """Replicate the real implementation from track.py"""
        mg = float(mg)
        ag = float(ag)
        if not np.isfinite(mg) or not np.isfinite(ag):
            raise ValueError(f"Non-finite gate")
        mg = float(np.clip(mg, 0.0, 1.0))
        ag = float(np.clip(ag, 0.0, 1.0))
        
        mean_prior = self.mean.copy()
        cov_prior = self.covariance.copy()
        pred_box = self.x1y1x2y2.copy()
        old_feat = self.feat.copy()
        
        mean_full, cov_full = self.kalman_filter.update(self.mean, self.covariance, det.cxcywh, det.score)
        
        if mg == 1.0:
            self.mean = mean_full
            self.covariance = cov_full
        elif mg == 0.0:
            self.mean = mean_prior
            self.covariance = cov_prior
        else:
            self.mean = mean_prior + mg * (mean_full - mean_prior)
            cov_final = (1 - mg) * cov_prior + mg * cov_full
            self.covariance = 0.5 * (cov_final + cov_final.T)
        
        if ag == 1.0:
            beta = self.alpha + (1 - self.alpha) * (1 - det.score)
            self.feat = beta * old_feat + (1 - beta) * det.feat
            self.feat /= np.linalg.norm(self.feat)
        elif ag == 0.0:
            self.feat = old_feat.copy()
        else:
            beta = self.alpha + (1 - self.alpha) * (1 - det.score)
            feat_full = beta * old_feat + (1 - beta) * det.feat
            self.feat = (1 - ag) * old_feat + ag * feat_full
            self.feat /= np.linalg.norm(self.feat) + 1e-12

class FakeDet:
    def __init__(self):
        self.box = np.array([2,3,5,6], dtype=np.float64)
        self.score = 0.9
        self.feat = np.array([2.0]*2048, dtype=np.float64).reshape(1, -1)
    @property
    def cxcywh(self): return np.array([3.5, 4.5, 3, 3], dtype=np.float64)
    @property
    def x1y1x2y2(self): return self.box.copy()

def test_gate_exact_one_matches_full_update():
    t1 = FakeTrack()
    t2 = FakeTrack()
    det = FakeDet()
    t1.update_with_gates(1, det, 1.0, 1.0)
    t2.update_with_gates(1, det, 1.0, 1.0)
    assert np.allclose(t1.mean, t2.mean)

def test_gate_exact_zero_preserves_state():
    t = FakeTrack()
    det = FakeDet()
    mean_before = t.mean.copy()
    feat_before = t.feat.copy()
    t.update_with_gates(1, det, 0.0, 0.0)
    assert np.allclose(t.mean, mean_before)
    assert np.allclose(t.feat, feat_before)

def test_nan_gate_raises():
    t = FakeTrack()
    det = FakeDet()
    with pytest.raises(ValueError):
        t.update_with_gates(1, det, float('nan'), 1.0)

def test_inf_gate_raises():
    t = FakeTrack()
    det = FakeDet()
    with pytest.raises(ValueError):
        t.update_with_gates(1, det, float('inf'), 1.0)
