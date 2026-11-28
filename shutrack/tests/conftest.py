"""Shared fixtures for SHUTrack v3 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shutrack.data.evidence import EventEvidence
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.schema import CONTEXT_SIZE
from shutrack.training.checkpoint import build_checkpoint_payload, save_checkpoint_atomic


def make_evidence(
    *,
    track_id: int = 1,
    segment_id: int = 0,
    frame_id: int = 1,
    observation_count: int = 1,
    reid_dim: int = 8,
    has_detection: bool = True,
    seed: int = 0,
    evidence_scale: float = 1.0,
    competition_scale: float = 1.0,
) -> EventEvidence:
    rng = np.random.RandomState(seed)
    track = rng.randn(reid_dim).astype(np.float32)
    track /= np.linalg.norm(track) + 1e-6
    detection = rng.randn(reid_dim).astype(np.float32)
    detection /= np.linalg.norm(detection) + 1e-6
    if not has_detection:
        detection = np.zeros_like(detection)
    return EventEvidence(
        evidence=(rng.randn(13) * evidence_scale).astype(np.float32),
        competition=(rng.randn(3) * competition_scale).astype(np.float32),
        track_feature=track,
        detection_feature=detection,
        has_detection=has_detection,
        frame_id=frame_id,
        track_id=track_id,
        segment_id=segment_id,
        observation_count=observation_count,
    )


def make_model(reid_dim: int = 8, seed: int = 7, competition_enabled: bool = True) -> ShutrackV2Model:
    torch.manual_seed(seed)
    return ShutrackV2Model(
        ShutrackV2Config(
            reid_dim=reid_dim,
            d_model=32,
            n_heads=4,
            hce_layers=2,
            ff_dim=64,
            dropout=0.0,
            appearance_proj_dim=16,
            competition_enabled=competition_enabled,
        )
    )


def write_tiny_checkpoint(
    path: Path,
    *,
    reid_dim: int = 8,
    max_frame_gap: int = 30,
    competition_enabled: bool = True,
) -> ShutrackV2Model:
    model = make_model(reid_dim=reid_dim, competition_enabled=competition_enabled)
    payload = build_checkpoint_payload(
        model=model,
        training_config={"smoke": True},
        dataset_descriptor={
            "dataset": "Tiny",
            "split": "train",
            "dataset_dir": "fixture",
            "max_frame_gap": max_frame_gap,
            "competition_enabled": competition_enabled,
            "label_source": "packed_train_data_safe_gate_target",
            "feature_schema_version": "shutrack_v3_evidence13_competition3",
        },
        epoch=1,
        global_step=1,
        best_metric=0.5,
        optimizer=None,
        scheduler=None,
        normalization_mean=np.zeros(63),
        normalization_std=np.ones(63),
        include_optimizer=False,
    )
    save_checkpoint_atomic(path, payload)
    return model


class FakeSnapshot:
    def __init__(
        self,
        *,
        feature: np.ndarray | None = None,
        frame_id: int = 1,
        state: int = 1,
        history: dict | None = None,
    ) -> None:
        self.mean = np.asarray([100.0, 100.0, 40.0, 80.0, 1.0, 0.5, 0.0, 0.0])
        self.covariance = np.eye(8) * 2.0
        self.velocity = np.asarray([[0.5, 0.1]] * 4)
        self.feature = (
            np.asarray(feature if feature is not None else np.ones(8), dtype=np.float32)
            .reshape(1, -1)
        )
        self.history = history if history is not None else {1: [np.array([90, 90, 110, 120]), 0.9]}
        self.end_frame_id = frame_id
        self.state = state
        self.score = 0.9


class FakeDetection:
    def __init__(
        self,
        feature: np.ndarray | None = None,
        box=None,
        score=0.9,
        frame_detection_index: int = 0,
    ) -> None:
        self.feat = np.asarray(feature if feature is not None else np.ones(8), dtype=np.float32)
        self.box = np.asarray(box if box is not None else [95, 95, 115, 125], dtype=np.float64)
        self.score = float(score)
        self.source = 0
        self.frame_detection_index = int(frame_detection_index)

    @property
    def x1y1x2y2(self) -> np.ndarray:
        return self.box

    @property
    def cxcywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
            dtype=np.float64,
        )


class FakeTrack:
    def __init__(self, track_id: int = 1, observation_count: int = 3, state: int = 1) -> None:
        self.track_id = track_id
        self.observation_count = observation_count
        self.state = state
        self.history = {i: [np.array([90, 90, 110, 120]), 0.9] for i in range(observation_count)}
        self.feat = np.ones(8, dtype=np.float32)
        self.calls: list[tuple] = []

    def update_with_gates(self, frame_id, detection, motion_gate, appearance_gate):
        self.calls.append((frame_id, detection, motion_gate, appearance_gate))


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "shutrack_best.pt"
    write_tiny_checkpoint(path)
    return path
