"""Test that EventCacheWriter -> EventCacheReader round-trips correctly.

Events written by EventCacheWriter should be read back by EventCacheReader
with identical fields and arrays matching within tolerance.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

sys.path.insert(0, "src")

from agentguard.contracts.events import TrackEvent
from agentguard.contracts.states import (
    AssociationPairFeatures,
    DetectionObservation,
    TrackStateSnapshot,
)
from agentguard.data.cache_reader import EventCacheReader
from agentguard.data.cache_schema import CacheManifest
from agentguard.data.cache_writer import EventCacheWriter


@pytest.fixture
def cache_dir() -> str:
    """Temporary directory for cache output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def manifest() -> CacheManifest:
    return CacheManifest(
        schema_version=1,
        dataset="test_dataset",
        split="train",
        sequence="test_sequence",
        num_frames=5,
        num_events=3,
        num_candidates=2,
        reid_dim=64,
        image_width=1920,
        image_height=1080,
    )


@pytest.fixture
def sample_events() -> List[TrackEvent]:
    """Create 3 sample events for cache round-trip testing."""
    rng = np.random.RandomState(42)
    events = []
    for i in range(3):
        feat = rng.randn(1, 64).astype(np.float64)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        det_feat = rng.randn(1, 64).astype(np.float64)
        det_feat /= np.linalg.norm(det_feat, axis=1, keepdims=True)

        mean = np.array([200.0, 300.0, 200.0, 200.0, 0.5, 0.3, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(8, dtype=np.float64) * 0.1
        box = np.array([100.0 + i, 200.0 + i, 300.0 + i, 400.0 + i], dtype=np.float64)

        state = TrackStateSnapshot(
            track_id=i,
            box=box,
            score=0.85 - i * 0.05,
            mean=mean,
            covariance=cov,
            velocity=np.zeros((4, 2), dtype=np.float64),
            feature=feat,
            history={},
            end_frame_id=99 + i,
            state=1,
        )

        det = DetectionObservation(
            detection_index=i,
            box=box + 5.0,
            score=0.9 - i * 0.05,
            feature=det_feat,
            source=0,
            class_id=1,
        )

        assoc = AssociationPairFeatures(
            iou_similarity=0.8 - i * 0.1,
            iou_distance=0.2 + i * 0.1,
            cosine_distance=0.1 + i * 0.05,
            confidence_distance=0.05,
            angle_distance=0.02,
            raw_cost=0.3 + i * 0.05,
            final_cost=0.28 + i * 0.05,
            assignment_round=i,
            assignment_threshold=0.5,
            detection_source=0,
        )

        ev = TrackEvent(
            event_id=f"evt_{i:04d}",
            dataset="test_dataset",
            sequence="test_sequence",
            frame_id=100 + i,
            track_id=i,
            image_width=1920,
            image_height=1080,
            has_detection=True,
            frame_start_state=state,
            pre_update_state=state,
            detection=det,
            association=assoc,
            warp_matrix=np.eye(2, 3, dtype=np.float64),
            scalar_features=np.arange(63, dtype=np.float64) + i,
            track_feature=feat.ravel(),
            detection_feature=det_feat.ravel(),
        )
        events.append(ev)
    return events


@pytest.fixture
def sample_candidates() -> List[TrackEvent]:
    """Create 2 candidate events."""
    rng = np.random.RandomState(123)
    candidates = []
    for i in range(2):
        feat = rng.randn(1, 64).astype(np.float64)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        mean = np.array([210.0, 310.0, 190.0, 190.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(8, dtype=np.float64) * 0.2
        box = np.array([110.0 + i * 5, 210.0 + i * 5, 290.0 + i * 5, 390.0 + i * 5], dtype=np.float64)

        state = TrackStateSnapshot(
            track_id=10 + i,
            box=box,
            score=0.7,
            mean=mean,
            covariance=cov,
            velocity=np.zeros((4, 2), dtype=np.float64),
            feature=feat,
            history={},
            end_frame_id=100,
            state=1,
        )
        ev = TrackEvent(
            event_id=f"candidate_{i}",
            dataset="test_dataset",
            sequence="test_sequence",
            frame_id=100,
            track_id=10 + i,
            image_width=1920,
            image_height=1080,
            has_detection=True,
            pre_update_state=state,
            detection=DetectionObservation(
                detection_index=10 + i,
                box=box,
                score=0.7,
                feature=feat,
                source=0,
                class_id=1,
            ),
            warp_matrix=np.eye(2, 3, dtype=np.float64),
            scalar_features=np.ones(63, dtype=np.float64) * 0.5,
            track_feature=feat.ravel(),
            detection_feature=feat.ravel(),
        )
        candidates.append(ev)
    return candidates


@pytest.fixture
def sample_frames() -> Dict[int, Any]:
    """Sample frame metadata."""
    return {
        100: {"frame_index": 0, "timestamp": 0.0, "has_gt": True},
        101: {"frame_index": 1, "timestamp": 0.033, "has_gt": True},
        102: {"frame_index": 2, "timestamp": 0.067, "has_gt": False},
    }


@pytest.fixture
def sample_prototypes() -> Dict[int, np.ndarray]:
    """Sample identity prototypes."""
    rng = np.random.RandomState(999)
    protos = {}
    for tid in range(3):
        v = rng.randn(64).astype(np.float64)
        protos[tid] = v / np.linalg.norm(v)
    return protos


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────


class TestCacheRoundTrip:
    def test_manifest_roundtrip(self, cache_dir, manifest):
        """Manifest written and read should have identical fields."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)
        manifest_read = reader.read_manifest()

        assert manifest_read.schema_version == manifest.schema_version
        assert manifest_read.dataset == manifest.dataset
        assert manifest_read.split == manifest.split
        assert manifest_read.sequence == manifest.sequence
        assert manifest_read.num_frames == manifest.num_frames
        assert manifest_read.num_events == manifest.num_events
        assert manifest_read.num_candidates == manifest.num_candidates
        assert manifest_read.reid_dim == manifest.reid_dim
        assert manifest_read.image_width == manifest.image_width
        assert manifest_read.image_height == manifest.image_height

    def test_events_roundtrip(self, cache_dir, manifest, sample_events):
        """Events written and read back should have identical fields."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)
        writer.write_events(sample_events, shard_size=2, manifest=manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)
        events_read = reader.read_events()

        assert len(events_read) == len(sample_events)
        for evt_orig, evt_read in zip(sample_events, events_read):
            assert evt_read.event_id == evt_orig.event_id
            assert evt_read.dataset == evt_orig.dataset
            assert evt_read.sequence == evt_orig.sequence
            assert evt_read.frame_id == evt_orig.frame_id
            assert evt_read.track_id == evt_orig.track_id
            assert evt_read.image_width == evt_orig.image_width
            assert evt_read.image_height == evt_orig.image_height
            assert evt_read.has_detection == evt_orig.has_detection

            # Compare scalar_features arrays
            np.testing.assert_array_almost_equal(
                evt_read.scalar_features, evt_orig.scalar_features, decimal=6
            )
            np.testing.assert_array_almost_equal(
                evt_read.track_feature, evt_orig.track_feature, decimal=6
            )
            np.testing.assert_array_almost_equal(
                evt_read.detection_feature, evt_orig.detection_feature, decimal=6
            )
            np.testing.assert_array_almost_equal(
                evt_read.warp_matrix, evt_orig.warp_matrix, decimal=10
            )

    def test_candidates_roundtrip(self, cache_dir, manifest, sample_candidates):
        """Candidates written and read back should be identical."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)
        writer.write_candidates(sample_candidates, shard_size=10, manifest=manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)
        candidates_read = reader.read_candidates()

        assert len(candidates_read) == len(sample_candidates)
        for c_orig, c_read in zip(sample_candidates, candidates_read):
            assert c_read.event_id == c_orig.event_id
            np.testing.assert_array_almost_equal(
                c_read.scalar_features, c_orig.scalar_features, decimal=6
            )

    def test_frames_roundtrip(self, cache_dir, manifest, sample_frames):
        """Frame data written and read should be identical."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)
        writer.write_frames(sample_frames, manifest=manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)
        frames_read = reader.read_frames()

        assert frames_read.keys() == sample_frames.keys()
        for fid in sample_frames:
            assert frames_read[fid] == sample_frames[fid]

    def test_prototypes_roundtrip(self, cache_dir, manifest, sample_prototypes):
        """Identity prototypes written and read should be identical."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)
        writer.write_identity_prototypes(sample_prototypes, manifest=manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)
        protos_read = reader.read_identity_prototypes()

        assert protos_read.keys() == sample_prototypes.keys()
        for tid in sample_prototypes:
            np.testing.assert_array_almost_equal(
                protos_read[tid], sample_prototypes[tid], decimal=6
            )

    def test_full_roundtrip(self, cache_dir, manifest, sample_events, sample_candidates,
                            sample_frames, sample_prototypes):
        """All data types survive a full write + read cycle."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)
        writer.write_events(sample_events, shard_size=2, manifest=manifest)
        writer.write_candidates(sample_candidates, shard_size=10, manifest=manifest)
        writer.write_frames(sample_frames, manifest=manifest)
        writer.write_identity_prototypes(sample_prototypes, manifest=manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)

        assert len(reader.read_events()) == len(sample_events)
        assert len(reader.read_candidates()) == len(sample_candidates)
        assert reader.read_frames().keys() == sample_frames.keys()
        assert reader.read_identity_prototypes().keys() == sample_prototypes.keys()

    def test_manifest_json_exists(self, cache_dir, manifest):
        """manifest.json should be valid JSON with correct keys."""
        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        manifest_path = os.path.join(seq_dir, "manifest.json")
        assert os.path.isfile(manifest_path)

        with open(manifest_path, "r") as f:
            data = json.load(f)

        assert data["dataset"] == manifest.dataset
        assert data["sequence"] == manifest.sequence
        assert data["schema_version"] == 1

    def test_multiple_event_shards(self, cache_dir, manifest):
        """Events are split across multiple shard files."""
        many_events = []
        for i in range(25):
            ev = TrackEvent(
                event_id=f"evt_{i:04d}",
                dataset=manifest.dataset,
                sequence=manifest.sequence,
                frame_id=i,
                track_id=i,
                image_width=640,
                image_height=480,
                has_detection=False,
            )
            many_events.append(ev)

        writer = EventCacheWriter(output_dir=cache_dir)
        writer.write_manifest(manifest)
        paths = writer.write_events(many_events, shard_size=10, manifest=manifest)

        assert len(paths) == 3  # 25 events / 10 = 3 shards

        seq_dir = os.path.join(cache_dir, manifest.dataset, manifest.split, manifest.sequence)
        reader = EventCacheReader(cache_dir=seq_dir)
        all_read = reader.read_events()
        assert len(all_read) == 25
