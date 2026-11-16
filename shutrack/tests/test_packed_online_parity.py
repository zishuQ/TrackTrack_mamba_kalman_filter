"""Packed offline adapter vs real online builder parity on one controlled event."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from shutrack.data.legacy_adapter import legacy_shutrack_scalar63
from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.features.core import normalize_scalar63
from shutrack.integration.tracktrack import ShutrackTrackerAdapter
from shutrack.runtime.adapter import ShutrackRuntime, ShutrackRuntimeConfig
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.tests.conftest import (
    FakeDetection,
    FakeSnapshot,
    FakeTrack,
    write_tiny_checkpoint,
)
from shutrack.training.train import MODEL_INPUT_KEYS


def _association_meta() -> dict:
    raw = np.array([[0.9, 0.2, 0.8], [0.1, 0.7, 0.6]], dtype=np.float64)
    return {
        "raw_cost": raw,
        "final_cost": raw.copy(),
        "iou_similarity": np.ones_like(raw),
        "iou_distance": np.zeros_like(raw),
        "cosine_distance": np.zeros_like(raw),
        "confidence_distance": np.zeros_like(raw),
        "angle_distance": np.zeros_like(raw),
        "assignment_round": np.zeros_like(raw, dtype=np.int16),
        "assignment_threshold": np.full_like(raw, 0.5),
        "detection_source": np.zeros_like(raw, dtype=np.int8),
    }


def _write_single_event_packed(
    root: Path,
    *,
    normalized_scalar: np.ndarray,
    track_feature: np.ndarray,
    detection_feature: np.ndarray,
) -> Path:
    reid_dim = int(track_feature.shape[0])
    sequence = "seq1"
    metadata = {
        "format": "train_data",
        "index_format": "train_data",
        "dataset": "Tiny",
        "split": "train",
        "train_sequences": [sequence],
        "context_size": 6,
        "max_frame_gap": 30,
        "reid_dim": reid_dim,
        "scalar_dim": 63,
        "num_train_samples": 1,
        "timeline_counts": {sequence: {"labeled_endpoints": 1}},
        "normalization_mean": [0.0] * 63,
        "normalization_std": [1.0] * 63,
        "reid_feature_file": "reid_features.npy",
    }
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    sequence_dir = root / sequence
    sequence_dir.mkdir()
    arrays = {
        "event_indices": torch.tensor(
            [[-1, -1, -1, -1, -1, 0]], dtype=torch.int32
        ),
        "track_ids": torch.tensor([1], dtype=torch.int64),
        "segment_ids": torch.tensor([0], dtype=torch.int64),
        "safe_gate_target": torch.tensor([[0.7, 0.3]], dtype=torch.float32),
        "valid_channels": torch.ones((1, 2), dtype=torch.bool),
        "sample_weight": torch.ones(1, dtype=torch.float32),
        "timeline_scalar_feats": torch.from_numpy(
            np.asarray(normalized_scalar, dtype=np.float32).reshape(1, 63)
        ),
        "timeline_has_detection": torch.ones(1, dtype=torch.bool),
    }
    torch.save({"arrays": arrays, "metadata": metadata}, sequence_dir / "data.pt")
    reid = np.stack(
        [
            np.asarray(track_feature, dtype=np.float32),
            np.asarray(detection_feature, dtype=np.float32),
        ]
    ).reshape(1, 2, reid_dim)
    np.save(sequence_dir / "reid_features.npy", reid)
    return root


def test_packed_offline_matches_online_features_and_gates(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path, margin_enabled=False)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    runtime = ShutrackRuntime(
        model,
        checkpoint,
        device="cpu",
        config=ShutrackRuntimeConfig(min_observations=1),
    )
    adapter = ShutrackTrackerAdapter(
        SimpleNamespace(img_w=640, img_h=480), "tiny", runtime=runtime
    )
    adapter.begin_frame(1, 640, 480)
    track = FakeTrack(track_id=1, observation_count=6)
    detection = FakeDetection()
    snapshot = FakeSnapshot(frame_id=1)
    association_meta = _association_meta()

    event = adapter.build_matched_event(
        track,
        detection,
        track_idx=0,
        det_idx=1,
        association_meta=association_meta,
        warp_matrix=np.eye(2, 3),
        frame_start_state_snapshot=snapshot,
        pre_update_state_snapshot=snapshot,
    )
    evidence = event.evidence

    params = runtime.scalar_params_for(
        frame_id=1,
        image_width=640,
        image_height=480,
        has_detection=True,
        pre_update_state=snapshot,
        detection_box=detection.box,
        detection_score=detection.score,
        detection_source=0,
        association_meta=association_meta,
        track_row=0,
        detection_col=1,
    )
    raw = legacy_shutrack_scalar63(params)
    normalized = normalize_scalar63(
        raw,
        np.asarray(checkpoint["normalization_mean"], dtype=np.float64),
        np.asarray(checkpoint["normalization_std"], dtype=np.float64),
    )

    dataset_dir = _write_single_event_packed(
        tmp_path / "packed",
        normalized_scalar=normalized,
        track_feature=evidence.track_feature,
        detection_feature=evidence.detection_feature,
    )
    dataset = PackedShutrackDataset(dataset_dir, margin_mode="disabled")
    try:
        sample = dataset[0]
        np.testing.assert_allclose(
            sample["current_motion"].numpy(), evidence.motion, atol=1e-6
        )
        np.testing.assert_allclose(
            sample["current_context"].numpy(), evidence.context, atol=1e-6
        )
        np.testing.assert_allclose(
            sample["current_track"].numpy(), evidence.track_feature, atol=1e-6
        )
        np.testing.assert_allclose(
            sample["current_detection"].numpy(),
            evidence.detection_feature,
            atol=1e-6,
        )
        assert sample["history_mask"].all()
        assert not sample["history_has_detection"].any()

        offline_inputs = {
            key: sample[key].unsqueeze(0) for key in MODEL_INPUT_KEYS
        }
        with torch.no_grad():
            offline_gates = model(**offline_inputs)["gates"].numpy()[0]
        online_gates = runtime.infer_batch([(1, evidence)])[0]
        np.testing.assert_allclose(offline_gates, online_gates, atol=1e-6)
    finally:
        dataset.close()
