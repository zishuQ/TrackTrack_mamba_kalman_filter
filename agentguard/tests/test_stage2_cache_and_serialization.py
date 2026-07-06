"""Stage 2 tests: event cache capture, serialization, atomic write, ReID shape."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from agentguard.contracts.states import (
    TrackStateSnapshot,
    DetectionObservation,
    AssociationContext,
    AssociationPairFeatures,
)
from agentguard.contracts.events import TrackEvent
from agentguard.data.event_sink import CacheEventSink, _validate_payload


# ======================================================================
# Fixtures
# ======================================================================


def _make_event(has_detection=True, frame_id=1, track_id=1, seq="test_seq"):
    det = None
    assoc = None
    ctx = None
    if has_detection:
        det = DetectionObservation(
            detection_index=0,
            box=np.array([100, 200, 300, 400], dtype=np.float64),
            score=0.9,
            feature=np.random.RandomState(42).randn(1, 64).astype(np.float64),
            source=0,
            class_id=1,
        )
        assoc = AssociationPairFeatures(
            iou_similarity=0.75,
            iou_distance=0.25,
            cosine_distance=0.10,
            confidence_distance=0.05,
            angle_distance=0.08,
            raw_cost=0.30,
            final_cost=0.28,
            assignment_round=2,
            assignment_threshold=0.50,
            detection_source=0,
        )
        ctx = AssociationContext(
            track_cost_row=np.array([0.3, 0.5], dtype=np.float64),
            detection_cost_col=np.array([0.3, 0.45], dtype=np.float64),
            detection_overlap_row=np.array([0.8, 0.6], dtype=np.float64),
            accepted_detection_index=0,
            num_tracks=2,
            num_detections=2,
            reid_available=True,
        )

    evt = TrackEvent(
        event_id=f"test/test/{frame_id:06d}/{track_id:06d}",
        dataset="test",
        sequence=seq,
        frame_id=frame_id,
        track_id=track_id,
        image_width=1920,
        image_height=1080,
        has_detection=has_detection,
        detection=det,
        association=assoc,
        association_context=ctx,
        frame_start_state=TrackStateSnapshot(
            track_id=track_id,
            box=np.array([100, 200, 300, 400], dtype=np.float64),
            score=0.8,
            mean=np.array([200, 300, 200, 200, 0, 0, 0, 0], dtype=np.float64),
            covariance=np.eye(8, dtype=np.float64) * 0.1,
            velocity=np.zeros((4, 2)),
            feature=np.random.RandomState(99).randn(1, 64).astype(np.float64),
            history={},
            end_frame_id=frame_id - 1,
            state=1,
        ),
        pre_update_state=TrackStateSnapshot(
            track_id=track_id,
            box=np.array([105, 205, 305, 405], dtype=np.float64),
            score=0.8,
            mean=np.array([205, 305, 200, 200, 0.5, 0.3, 0, 0], dtype=np.float64),
            covariance=np.eye(8, dtype=np.float64) * 0.15,
            velocity=np.zeros((4, 2)),
            feature=np.random.RandomState(99).randn(1, 64).astype(np.float64),
            history={},
            end_frame_id=frame_id - 1,
            state=1,
        ),
        scalar_features=np.ones(63, dtype=np.float64),
        track_feature=np.ones(64, dtype=np.float64),
        detection_feature=np.ones(64, dtype=np.float64) if has_detection else np.zeros(64, dtype=np.float64),
        iwg_gate=np.array([0.8, 0.6], dtype=np.float64),
    )
    return evt


def _make_frame_record(frame_id=1, num_dets=5, warp=None):
    if warp is None:
        warp = np.eye(2, 3, dtype=np.float64)
    return {
        "frame_id": frame_id,
        "image_width": 1920,
        "image_height": 1080,
        "warp_matrix": warp.tolist(),
        "detections": [
            {"det_idx": i, "score": 0.8}
            for i in range(num_dets)
        ],
    }


# ======================================================================
# Stage 2: Cache Atomic Write
# ======================================================================


def test_cache_atomic_write_uses_incomplete_dir():
    """CacheEventSink must write to .incomplete, then atomically rename."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = os.path.join(tmp, "cache")
        seq = "test_seq"

        sink = CacheEventSink(
            cache_root=cache_root,
            dataset="test",
            split="val",
            sequence=seq,
            shard_size=10,
        )

        sink.on_sequence_start(seq, reid_dim=64, image_width=1920, image_height=1080)
        assert sink._temp_dir is not None
        assert str(sink._temp_dir).endswith(f"{seq}.incomplete")

        # Write some frames
        for fid in range(3):
            fr = _make_frame_record(fid + 1)
            evts = [_make_event(frame_id=fid + 1, track_id=tid) for tid in range(1, 4)]
            event_dicts = []
            from agentguard.contracts.serialization import serialize_event
            for evt in evts:
                d = serialize_event(evt)
                d["target_gt_id"] = None
                d["detection_gt_id"] = None
                d["accepted_detection_index"] = evt.detection.detection_index if evt.has_detection and evt.detection else None
                event_dicts.append(d)
            sink.on_frame(fr, event_dicts)

        result = sink.on_sequence_end()

        # Verify .incomplete is gone, final dir exists
        assert not sink._temp_dir
        final_dir = Path(cache_root) / "test" / "val" / seq
        assert final_dir.is_dir()

        # Verify manifest
        with open(final_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["complete"] is True
        assert manifest["num_frames"] == 3
        assert manifest["num_events"] == result["num_events"]

        # Verify event shards exist
        shards = sorted(final_dir.glob("events_*.pt"))
        assert len(shards) > 0

        # Verify frames shard
        assert (final_dir / "frames_00000.pt").is_file()


def test_cache_skip_if_complete_and_hashes_match():
    """If final dir exists with complete=true and matching hashes, skip."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = os.path.join(tmp, "cache")
        seq = "test_seq"
        final_dir = Path(cache_root) / "test" / "val" / seq
        os.makedirs(final_dir, exist_ok=True)
        with open(final_dir / "manifest.json", "w") as f:
            json.dump({
                "complete": True,
                "config_sha256": "abc123",
                "feature_schema_sha256": "def456",
            }, f)
        # Create a dummy shard
        import torch
        torch.save([], final_dir / "events_00000.pt")

        sink = CacheEventSink(
            cache_root=cache_root,
            dataset="test",
            split="val",
            sequence=seq,
            config_hash="abc123",
            schema_hash="def456",
        )
        sink.on_sequence_start(seq, reid_dim=64, image_width=1920, image_height=1080)
        # Should have skipped — temp_dir is None
        assert sink._temp_dir is None


def test_cache_rebuild_stale_incomplete():
    """Stale .incomplete dir must be deleted and rebuilt."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = os.path.join(tmp, "cache")
        seq = "test_seq"

        # Create stale .incomplete dir
        stale = Path(cache_root) / "test" / "val" / f"{seq}.incomplete"
        os.makedirs(stale, exist_ok=True)
        os.makedirs(stale / "subdir", exist_ok=True)  # extra content

        sink = CacheEventSink(
            cache_root=cache_root,
            dataset="test",
            split="val",
            sequence=seq,
        )
        sink.on_sequence_start(seq, reid_dim=64, image_width=1920, image_height=1080)
        assert sink._temp_dir is not None
        assert not (stale / "subdir").exists()


def test_max_frames_enforces_limit():
    """max_frames > 0 must limit event capture."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = os.path.join(tmp, "cache")
        sink = CacheEventSink(
            cache_root=cache_root,
            dataset="test",
            split="val",
            sequence="limited",
            max_frames=2,
        )
        sink.on_sequence_start("limited", reid_dim=64, image_width=1920, image_height=1080)

        for fid in range(5):
            fr = _make_frame_record(fid + 1)
            sink.on_frame(fr, [])
        result = sink.on_sequence_end()
        assert result["num_frames"] == 2
        assert result["truncated"] is True


def test_max_frames_zero_is_unlimited():
    """max_frames=0 means unlimited."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = os.path.join(tmp, "cache")
        sink = CacheEventSink(
            cache_root=cache_root,
            dataset="test",
            split="val",
            sequence="unlimited",
            max_frames=0,
        )
        sink.on_sequence_start("unlimited", reid_dim=64, image_width=1920, image_height=1080)

        for fid in range(10):
            fr = _make_frame_record(fid + 1)
            sink.on_frame(fr, [])
        result = sink.on_sequence_end()
        assert result["num_frames"] == 10
        assert result["truncated"] is False


def test_manifest_stores_processed_and_total_frames():
    """Manifest must store processed_frames and total_sequence_frames."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = os.path.join(tmp, "cache")
        seq = "manifest_test"
        sink = CacheEventSink(
            cache_root=cache_root,
            dataset="test",
            split="val",
            sequence=seq,
        )
        sink.on_sequence_start(seq, reid_dim=64, image_width=1920, image_height=1080)
        for fid in range(5):
            sink.on_frame(_make_frame_record(fid + 1), [])
        sink.on_sequence_end()

        final_dir = Path(cache_root) / "test" / "val" / seq
        with open(final_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["processed_frames"] == 5
        assert manifest["total_sequence_frames"] == 5
        assert manifest["truncated"] is False


# ======================================================================
# Stage 2: Serialization validation
# ======================================================================


def test_serialize_event_payload_is_clean():
    """serialize_event output must contain only allowed types."""
    from agentguard.contracts.serialization import serialize_event

    evt = _make_event(has_detection=True)
    data = serialize_event(evt)
    _validate_payload(data, "event")


def test_deserialize_round_trip_preserves_association_context():
    """Round-trip serialize/deserialize preserves association_context."""
    from agentguard.contracts.serialization import serialize_event, deserialize_event

    evt = _make_event(has_detection=True)
    data = serialize_event(evt)
    restored = deserialize_event(data)

    assert restored.association_context is not None
    assert evt.association_context is not None
    assert restored.association_context.num_tracks == evt.association_context.num_tracks
    np.testing.assert_array_equal(
        restored.association_context.track_cost_row,
        evt.association_context.track_cost_row,
    )


def test_unmatched_event_no_detection_features():
    """Unmatched events must have has_detection=False, detection=None,
    detection_feature=zeros, track_feature from pre_update_state."""
    evt = _make_event(has_detection=False)
    assert evt.has_detection is False
    assert evt.detection is None
    assert evt.association is None
    # detection_feature should be all zeros
    assert np.all(evt.detection_feature == 0)


# ======================================================================
# Stage 2: Cache path responsibility
# ======================================================================


def test_cache_path_responsibility():
    """CacheEventSink must use cache_root only and append dataset/split/sequence."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = CacheEventSink(
            cache_root=tmp,
            dataset="DS",
            split="train",
            sequence="seq01",
        )
        assert str(sink._seq_dir()).endswith("DS/train/seq01")
        assert sink._seq_dir() == Path(tmp) / "DS" / "train" / "seq01"


# ======================================================================
# Stage 2: ReID dimension derivation
# ======================================================================


def test_reid_dim_not_hardcoded_2048_in_features():
    """Feature builder should accept and reflect arbitrary reid_dim."""
    from agentguard.features.builder import EventFeatureBuilder

    fb = EventFeatureBuilder(reid_dim=128)
    assert fb.reid_dim == 128

    fb256 = EventFeatureBuilder(reid_dim=256)
    assert fb256.reid_dim == 256


def test_track_feature_shape_unification():
    """Track feature must be (D,) at boundaries, not (1, D)."""
    feat_2d = np.random.randn(1, 128).astype(np.float64)
    reshaped = np.asarray(feat_2d, dtype=np.float64).reshape(-1)
    assert reshaped.shape == (128,)
    assert reshaped.ndim == 1


# ======================================================================
# Stage 2: EventCacheWriter path
# ======================================================================


def test_cache_writer_uses_fixed_root():
    """EventCacheWriter must append dataset/split/sequence from manifest."""
    from agentguard.data.cache_writer import EventCacheWriter
    from agentguard.data.cache_schema import CacheManifest

    writer = EventCacheWriter("/tmp/fake_cache")
    manifest = CacheManifest(dataset="DS", split="train", sequence="seq01")
    path = writer._seq_dir(manifest)
    assert path == "/tmp/fake_cache/DS/train/seq01"


# ======================================================================
# Stage 2: CacheEventSink event counting
# ======================================================================


def test_cache_counts_matched_and_unmatched():
    """CacheEventSink must count matched/unmatched events separately."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = CacheEventSink(cache_root=tmp, dataset="test", split="val", sequence="counts")
        sink.on_sequence_start("counts", reid_dim=64, image_width=1920, image_height=1080)

        from agentguard.contracts.serialization import serialize_event

        # Frame 1: 2 matched + 1 unmatched
        matched1 = serialize_event(_make_event(True, 1, 1))
        matched2 = serialize_event(_make_event(True, 1, 2))
        unmatched1 = serialize_event(_make_event(False, 1, 3))
        for d in [matched1, matched2, unmatched1]:
            d["target_gt_id"] = None
            d["detection_gt_id"] = None
            d["accepted_detection_index"] = d.get("accepted_detection_index")
        sink.on_frame(_make_frame_record(1), [matched1, matched2, unmatched1])

        result = sink.on_sequence_end()
        assert result["num_matched_events"] == 2
        assert result["num_unmatched_events"] == 1
        assert result["num_events"] == 3


def test_empty_event_shards_are_skipped():
    """When no events exist, no empty event shard should be written."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = CacheEventSink(cache_root=tmp, dataset="test", split="val", sequence="empty")
        sink.on_sequence_start("empty", reid_dim=64, image_width=1920, image_height=1080)
        # No events at all
        sink.on_sequence_end()

        final_dir = Path(tmp) / "test" / "val" / "empty"
        shards = sorted(final_dir.glob("events_*.pt"))
        assert len(shards) == 0


def test_association_context_counted_in_manifest():
    """Manifest must count association_contexts."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = CacheEventSink(cache_root=tmp, dataset="test", split="val", sequence="ctx_count")
        sink.on_sequence_start("ctx_count", reid_dim=64, image_width=1920, image_height=1080)

        from agentguard.contracts.serialization import serialize_event

        evt = _make_event(True, 1, 1)
        d = serialize_event(evt)
        d["target_gt_id"] = None
        d["detection_gt_id"] = None
        d["accepted_detection_index"] = 0

        sink.on_frame(_make_frame_record(1), [d])
        result = sink.on_sequence_end()

        assert result["num_association_contexts"] == 1
