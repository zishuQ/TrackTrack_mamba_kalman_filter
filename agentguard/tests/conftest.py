"""Shared fixtures for AgentGuard tests."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.enums import DetectionSource
from agentguard.contracts.events import TrackEvent
from agentguard.contracts.outputs import GateDecision
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.motion.nsa_numpy import NSAKalmanFilter


# ──────────────────────────────────────────────────────────────────────
#  Mock Track object  (mimics TrackTrack's Track interface)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_track_id() -> int:
    return 42


@pytest.fixture
def mock_box() -> np.ndarray:
    """A realistic bounding box [x1, y1, x2, y2]."""
    return np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)


@pytest.fixture
def mock_kf_mean() -> np.ndarray:
    """KF state mean: [cx, cy, w, h, vx, vy, vw, vh]."""
    return np.array([200.0, 300.0, 200.0, 200.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float64)


@pytest.fixture
def mock_kf_cov() -> np.ndarray:
    """KF state covariance (8, 8)."""
    cov = np.eye(8, dtype=np.float64) * 0.1
    cov[0, 0] = 4.0
    cov[1, 1] = 4.0
    cov[2, 2] = 1.0
    cov[3, 3] = 1.0
    return cov


@pytest.fixture
def mock_velocity() -> np.ndarray:
    """Velocity (4, 2) — column 0 = position, column 1 = velocity."""
    return np.array(
        [[1.0, 0.5], [2.0, 0.3], [0.5, 0.1], [0.8, 0.2]], dtype=np.float64
    )


@pytest.fixture
def mock_feature() -> np.ndarray:
    """ReID feature (1, D) with D=64."""
    rng = np.random.RandomState(42)
    feat = rng.randn(1, 64).astype(np.float64)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    return feat


@pytest.fixture
def mock_history(mock_box, mock_kf_mean, mock_kf_cov, mock_feature) -> Dict[int, List]:
    """Track history with 3 entries."""
    return {
        90: [mock_box.copy(), 0.7, mock_kf_mean.copy(), mock_kf_cov.copy(), mock_feature.copy()],
        95: [mock_box.copy(), 0.75, mock_kf_mean.copy(), mock_kf_cov.copy(), mock_feature.copy()],
        98: [mock_box.copy(), 0.8, mock_kf_mean.copy(), mock_kf_cov.copy(), mock_feature.copy()],
    }


@pytest.fixture
def mock_track_snapshot(
    mock_track_id: int,
    mock_box: np.ndarray,
    mock_kf_mean: np.ndarray,
    mock_kf_cov: np.ndarray,
    mock_velocity: np.ndarray,
    mock_feature: np.ndarray,
    mock_history: Dict[int, List],
) -> TrackStateSnapshot:
    """A realistic TrackStateSnapshot."""
    return TrackStateSnapshot(
        track_id=mock_track_id,
        box=mock_box.copy(),
        score=0.85,
        mean=mock_kf_mean.copy(),
        covariance=mock_kf_cov.copy(),
        velocity=mock_velocity.copy(),
        feature=mock_feature.copy(),
        history=mock_history,
        end_frame_id=98,
        state=1,
    )


@pytest.fixture
def mock_detection_obs() -> DetectionObservation:
    """A realistic DetectionObservation."""
    rng = np.random.RandomState(7)
    feat = rng.randn(1, 64).astype(np.float64)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    return DetectionObservation(
        detection_index=0,
        box=np.array([105.0, 205.0, 310.0, 410.0], dtype=np.float64),
        score=0.9,
        feature=feat,
        source=DetectionSource.HIGH,
        class_id=1,
    )


@pytest.fixture
def mock_association() -> AssociationPairFeatures:
    """A realistic AssociationPairFeatures."""
    return AssociationPairFeatures(
        iou_similarity=0.75,
        iou_distance=0.25,
        cosine_distance=0.12,
        confidence_distance=0.05,
        angle_distance=0.03,
        raw_cost=0.30,
        final_cost=0.28,
        assignment_round=0,
        assignment_threshold=0.5,
        detection_source=DetectionSource.HIGH,
    )


@pytest.fixture
def mock_event(
    mock_track_snapshot: TrackStateSnapshot,
    mock_detection_obs: DetectionObservation,
    mock_association: AssociationPairFeatures,
) -> TrackEvent:
    """A realistic TrackEvent with detection."""
    ev = TrackEvent(
        event_id="test_event_001",
        dataset="test_dataset",
        sequence="test_sequence",
        frame_id=100,
        track_id=mock_track_snapshot.track_id,
        image_width=1920,
        image_height=1080,
        has_detection=True,
        frame_start_state=mock_track_snapshot,
        pre_update_state=mock_track_snapshot,
        detection=mock_detection_obs,
        association=mock_association,
        warp_matrix=np.eye(2, 3, dtype=np.float64),
        scalar_features=np.zeros(63, dtype=np.float64),
        track_feature=mock_track_snapshot.feature.ravel().copy(),
        detection_feature=mock_detection_obs.feature.ravel().copy(),
        iwg_policy_probs=np.ones(5, dtype=np.float64) / 5.0,
        iwg_gate=np.array([1.0, 1.0], dtype=np.float64),
    )
    ev.track_cost_row = [0.30, 0.50, 0.70]
    ev.detection_cost_col = [0.30, 0.45, 0.60]
    ev.max_iou_with_other = 0.15
    return ev


@pytest.fixture
def mock_event_no_detection(
    mock_track_snapshot: TrackStateSnapshot,
) -> TrackEvent:
    """A TrackEvent with no detection (unmatched)."""
    return TrackEvent(
        event_id="test_event_no_det",
        dataset="test_dataset",
        sequence="test_sequence",
        frame_id=100,
        track_id=mock_track_snapshot.track_id,
        image_width=1920,
        image_height=1080,
        has_detection=False,
        frame_start_state=mock_track_snapshot,
        pre_update_state=mock_track_snapshot,
        warp_matrix=np.eye(2, 3, dtype=np.float64),
        scalar_features=np.zeros(63, dtype=np.float64),
        track_feature=mock_track_snapshot.feature.ravel().copy(),
        detection_feature=np.zeros(0, dtype=np.float64),
    )


@pytest.fixture
def mock_kf() -> NSAKalmanFilter:
    """A fresh NSAKalmanFilter instance."""
    return NSAKalmanFilter()


@pytest.fixture
def mock_gate_decision() -> GateDecision:
    """A realistic GateDecision."""
    return GateDecision(
        motion_gate=0.8,
        appearance_gate=0.6,
        policy_probs=np.array([0.3, 0.2, 0.2, 0.15, 0.15], dtype=np.float64),
        confidence=0.7,
    )


@pytest.fixture
def mock_identity_prototype() -> np.ndarray:
    """Random unit-norm prototype of dimension 64."""
    rng = np.random.RandomState(0)
    v = rng.randn(64).astype(np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def mock_future_oracle_data() -> Dict[str, Any]:
    """Standard future oracle data for 5 frames."""
    rng = np.random.RandomState(1)
    boxes: List[np.ndarray] = []
    features: List[np.ndarray] = []
    scores: List[float] = []
    warp_mats: List[np.ndarray] = []
    for i in range(5):
        shift = (i + 1) * 2.0
        boxes.append(
            np.array(
                [75.0 + shift, 75.0 + shift, 125.0 + shift, 125.0 + shift],
                dtype=np.float64,
            )
        )
        feat = rng.randn(1, 64).astype(np.float64)
        feat /= np.linalg.norm(feat)
        features.append(feat)
        scores.append(0.9 - i * 0.02)
        warp_mats.append(np.eye(2, 3, dtype=np.float64))

    return {
        "future_oracle_detections": boxes,
        "future_oracle_features": features,
        "future_oracle_scores": scores,
        "future_warp_matrices": warp_mats,
    }


class FakeTrack:
    """Minimal track stub that mimics TrackTrack's Track interface."""

    def __init__(
        self,
        track_id: int = 0,
        state: int = 1,
        box: Optional[np.ndarray] = None,
        score: float = 0.9,
    ):
        self.track_id = track_id
        self.state = state  # 1 = Tracked
        self.score = score
        self.box = box if box is not None else np.array([100, 200, 300, 400], dtype=np.float64)
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.feature: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.end_frame_id: int = 0
        self.history: Dict[int, List] = {}
        # Ensure at least 6 history entries so is_mature_track passes
        if len(self.history) < 6:
            self.history = {
                i: [self.box.copy(), score, None, None, np.zeros((1, 64))]
                for i in range(6)
            }

    def update(self, frame_id: int, detection: Any, **kwargs) -> None:
        """Original update: apply KF update with full detection."""
        self.end_frame_id = frame_id
        if hasattr(detection, "box"):
            self.box = detection.box.copy()
        if hasattr(detection, "score"):
            self.score = detection.score

    def update_with_gates(
        self,
        frame_id: int,
        detection: Any,
        motion_gate: float,
        appearance_gate: float,
    ) -> None:
        """Gated update (may be overridden in tests)."""
        self.end_frame_id = frame_id
        # Default: same as update
        if hasattr(detection, "box"):
            self.box = detection.box.copy()
        if hasattr(detection, "score"):
            self.score = detection.score

    def mark_lost(self) -> None:
        self.state = 2  # LOST


@pytest.fixture
def mock_track(mock_track_id: int) -> FakeTrack:
    """A realistic FakeTrack fixture."""
    return FakeTrack(track_id=mock_track_id)
