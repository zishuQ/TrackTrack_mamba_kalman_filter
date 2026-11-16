"""Shared fixtures for SHUTrack v2 tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shutrack.data.evidence import EventEvidence
from shutrack.features.margins import DEFAULT_MARGIN_STATS
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
    motion_scale: float = 1.0,
    context_scale: float = 1.0,
) -> EventEvidence:
    rng = np.random.RandomState(seed)
    track = rng.randn(reid_dim).astype(np.float32)
    track /= np.linalg.norm(track) + 1e-6
    detection = rng.randn(reid_dim).astype(np.float32)
    detection /= np.linalg.norm(detection) + 1e-6
    if not has_detection:
        detection = np.zeros_like(detection)
    return EventEvidence(
        motion=(rng.randn(12) * motion_scale).astype(np.float32),
        context=(rng.randn(8) * context_scale).astype(np.float32),
        track_feature=track,
        detection_feature=detection,
        has_detection=has_detection,
        frame_id=frame_id,
        track_id=track_id,
        segment_id=segment_id,
        observation_count=observation_count,
    )


def make_model(reid_dim: int = 8, seed: int = 7) -> ShutrackV2Model:
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
        )
    )


def model_inputs(events: list[EventEvidence], current) -> dict[str, torch.Tensor]:
    from shutrack.features.windows import build_window, stack_windows

    window = build_window(events, current)
    batch = stack_windows([window])
    return {key: torch.from_numpy(value) for key, value in batch.items()}


def write_tiny_packed_dataset(root: Path, *, reid_dim: int = 8) -> Path:
    """Create a small self-contained packed train_data fixture."""

    sequences = ["seqA", "seqB"]
    timelines = {
        "seqA": 10,
        "seqB": 8,
    }
    samples = {
        "seqA": [
            # two left-padded cold-start samples and one full window
            {"indices": [-1, -1, -1, 0, 1, 2], "track": 10, "segment": 0},
            {"indices": [-1, 0, 1, 2, 3, 4], "track": 10, "segment": 0},
            {"indices": [2, 3, 4, 5, 6, 7], "track": 11, "segment": 1},
        ],
        "seqB": [
            {"indices": [-1, -1, -1, -1, 0, 1], "track": 20, "segment": 0},
        ],
    }
    mean = [0.0] * 63
    std = [1.0] * 63
    metadata = {
        "format": "train_data",
        "index_format": "train_data",
        "dataset": "Tiny",
        "split": "train",
        "train_sequences": sequences,
        "context_size": CONTEXT_SIZE,
        "max_frame_gap": 30,
        "reid_dim": reid_dim,
        "scalar_dim": 63,
        "num_train_samples": sum(len(value) for value in samples.values()),
        "timeline_counts": {
            sequence: {"labeled_endpoints": len(samples[sequence])}
            for sequence in sequences
        },
        "normalization_mean": mean,
        "normalization_std": std,
        "reid_feature_file": "reid_features.npy",
    }
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    rng = np.random.RandomState(123)
    for sequence in sequences:
        timeline_count = timelines[sequence]
        scalar = rng.randn(timeline_count, 63).astype(np.float32) * 0.25
        # Give the context indices recognizable values.
        scalar[:, 9] = np.linspace(0.1, 0.9, timeline_count)
        scalar[:, 26] = 1.0
        scalar[:, 58] = np.linspace(0.0, 1.0, timeline_count)
        scalar[:, 59] = np.linspace(1.0, 0.0, timeline_count)
        has_detection = np.ones(timeline_count, dtype=np.bool_)
        event_indices = np.asarray(
            [item["indices"] for item in samples[sequence]], dtype=np.int32
        )
        count = event_indices.shape[0]
        arrays = {
            "event_indices": torch.from_numpy(event_indices),
            "track_ids": torch.tensor(
                [item["track"] for item in samples[sequence]], dtype=torch.int64
            ),
            "segment_ids": torch.tensor(
                [item["segment"] for item in samples[sequence]], dtype=torch.int64
            ),
            "safe_gate_target": torch.tensor(
                [[0.8, 0.2], [0.4, 0.6], [0.9, 0.1], [0.3, 0.7]][:count],
                dtype=torch.float32,
            )[:count],
            "valid_channels": torch.ones((count, 2), dtype=torch.bool),
            "sample_weight": torch.ones(count, dtype=torch.float32),
            "timeline_scalar_feats": torch.from_numpy(scalar),
            "timeline_has_detection": torch.from_numpy(has_detection),
        }
        sequence_dir = root / sequence
        sequence_dir.mkdir()
        torch.save({"arrays": arrays, "metadata": metadata}, sequence_dir / "data.pt")
        reid = rng.randn(timeline_count, 2, reid_dim).astype(np.float32)
        reid /= np.linalg.norm(reid, axis=-1, keepdims=True) + 1e-6
        np.save(sequence_dir / "reid_features.npy", reid)
    return root


def write_tiny_checkpoint(
    path: Path,
    *,
    reid_dim: int = 8,
    max_frame_gap: int = 30,
) -> ShutrackV2Model:
    model = make_model(reid_dim=reid_dim)
    payload = build_checkpoint_payload(
        model=model,
        training_config={"smoke": True},
        dataset_descriptor={
            "dataset": "Tiny",
            "split": "train",
            "dataset_dir": "fixture",
            "max_frame_gap": max_frame_gap,
            "margin_source": "unavailable_packed_train_data",
            "margin_valid_rate": 0.0,
        },
        epoch=1,
        global_step=1,
        best_metric=0.5,
        optimizer=None,
        scheduler=None,
        normalization_mean=np.zeros(63),
        normalization_std=np.ones(63),
        margin={
            "source": "unavailable_packed_train_data",
            "valid_rate": 0.0,
            "stats": DEFAULT_MARGIN_STATS,
        },
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
    def __init__(self, feature: np.ndarray | None = None, box=None, score=0.9) -> None:
        self.feat = np.asarray(feature if feature is not None else np.ones(8), dtype=np.float32)
        self.box = np.asarray(box if box is not None else [95, 95, 115, 125], dtype=np.float64)
        self.score = float(score)
        self.source = 0


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
def tiny_packed_dir(tmp_path: Path) -> Path:
    return write_tiny_packed_dataset(tmp_path / "packed")


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "shutrack_best.pt"
    write_tiny_checkpoint(path)
    return path
