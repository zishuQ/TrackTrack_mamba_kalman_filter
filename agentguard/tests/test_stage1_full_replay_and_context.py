"""Stage 1 tests: AgentGuard Full-mode TGR Replay and all-one gating."""

import numpy as np
import pytest

from agentguard.contracts.states import (
    TrackStateSnapshot,
    DetectionObservation,
    AssociationContext,
    AssociationPairFeatures,
)
from agentguard.contracts.events import TrackEvent
from agentguard.runtime.replay import ReplayPlan, ReplayStep


# ======================================================================
# Fixtures
# ======================================================================

def _make_snapshot(track_id=1, end_frame_id=5, box=None, mean=None, feat=None):
    return TrackStateSnapshot(
        track_id=track_id,
        box=box if box is not None else np.array([100, 100, 200, 200], dtype=np.float64),
        score=0.9,
        mean=mean if mean is not None else np.array([150, 150, 100, 100, 0, 0, 0, 0], dtype=np.float64),
        covariance=np.eye(8, dtype=np.float64) * 0.1,
        velocity=np.zeros((4, 2), dtype=np.float64),
        feature=feat if feat is not None else np.random.RandomState(42).randn(1, 64).astype(np.float64),
        history={},
        end_frame_id=end_frame_id,
        state=1,
    )


def _make_detection(det_idx=0, score=0.85):
    return DetectionObservation(
        detection_index=det_idx,
        box=np.array([105, 205, 310, 410], dtype=np.float64),
        score=score,
        feature=np.random.RandomState(7).randn(1, 64).astype(np.float64),
        source=0,
        class_id=1,
    )


class _SpyTrack:
    """A Track-like spy that records calls and tracks state mutations."""

    def __init__(self):
        self.calls = []
        self.mean = None
        self.covariance = None
        self.box = np.zeros(4)
        self.score = 0.0
        self.feat = np.zeros((1, 64))
        self.history = {}
        self.end_frame_id = 0
        self.state = 1
        self.track_id = 1
        self.velocity = np.zeros((4, 2))
        self.kalman_filter = _DummyKF()

    def predict(self):
        self.calls.append("predict")

    def update_with_gates(self, frame_id, det, mg, ag):
        self.calls.append(("update_with_gates", frame_id, mg, ag))
        self.end_frame_id = frame_id
        self.state = 1

    def mark_lost(self):
        self.calls.append("mark_lost")
        self.state = 2

    def restore_state(self, snapshot):
        self.calls.append("restore_state")
        self.box = snapshot.box.copy()
        self.score = snapshot.score
        self.mean = snapshot.mean.copy() if snapshot.mean is not None else None
        self.covariance = snapshot.covariance.copy() if snapshot.covariance is not None else None
        self.velocity = snapshot.velocity.copy()
        self.feat = snapshot.feature.copy()
        self.history = dict(snapshot.history)
        self.end_frame_id = snapshot.end_frame_id
        self.state = snapshot.state
        self.track_id = snapshot.track_id


class _DummyKF:
    def predict(self, m, c):
        return m, c

    def update(self, m, c, meas, score):
        return m, c

    def initiate(self, cxcywh):
        return np.zeros(8), np.eye(8)


# ======================================================================
# Stage 1: TGR Replay mutates live Track
# ======================================================================


def test_full_mode_tgr_replay_mutates_live_track():
    """TGR full mode replay must restore checkpoint, apply steps, and
    leave live Track in post-replay state (not restore replay-pre state)."""
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    backend = TrackTrackReplayBackend()

    snapshot = _make_snapshot(end_frame_id=5, box=np.array([90, 90, 190, 190], dtype=np.float64))
    det = _make_detection()

    steps = [
        ReplayStep(6, np.eye(2, 3), True, det, 0.8, 0.6),
        ReplayStep(7, np.eye(2, 3), True, det, 0.5, 0.9),
    ]
    plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=steps)

    spy = _SpyTrack()
    backend.replay_into_live_track(spy, plan)

    assert spy.calls[0] == "restore_state"
    assert "predict" in spy.calls
    update_calls = [c for c in spy.calls if isinstance(c, tuple) and c[0] == "update_with_gates"]
    assert len(update_calls) == 2
    assert update_calls[0][1] == 6
    assert update_calls[0][2] == 0.8
    assert update_calls[0][3] == 0.6
    assert update_calls[1][1] == 7
    assert update_calls[1][2] == 0.5
    assert update_calls[1][3] == 0.9
    # Track ends at post-replay state: end_frame_id from last matched step
    assert spy.end_frame_id == 7
    assert spy.state == 1


def test_full_mode_unmatched_replay_only_calls_mark_lost():
    """Unmatched steps must call mark_lost only, never write end_frame_id."""
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    backend = TrackTrackReplayBackend()

    snapshot = _make_snapshot(end_frame_id=3)
    steps = [
        ReplayStep(10, np.eye(2, 3), False, None, 0.0, 0.0),
    ]
    plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=steps)

    spy = _SpyTrack()
    spy.end_frame_id = 99  # pre-set, restore_state will overwrite to 3
    backend.replay_into_live_track(spy, plan)

    assert spy.state == 2
    assert spy.calls.count("mark_lost") == 1
    # end_frame_id must be the snapshot value (3), not the unmatched frame (10)
    assert spy.end_frame_id == snapshot.end_frame_id


def test_all_one_gating_reproduces_baseline():
    """Gate = (1.0, 1.0) with matched step should produce same end state as baseline."""
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    backend = TrackTrackReplayBackend()

    det = _make_detection(score=0.9)
    snapshot = _make_snapshot(box=det.box, mean=np.array([150, 150, 100, 100, 0, 0, 0, 0], dtype=np.float64))
    steps = [ReplayStep(6, np.eye(2, 3), True, det, 1.0, 1.0)]
    plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=steps)

    spy = _SpyTrack()
    backend.replay_into_live_track(spy, plan)

    # With gate=(1.0, 1.0), update_with_gates should apply the full update
    assert spy.end_frame_id == 6
    assert spy.state == 1
    assert ("update_with_gates", 6, 1.0, 1.0) in spy.calls


def test_no_replay_engine_in_backend_source():
    """TrackTrackReplayBackend.replay_into_live_track must not import or use ReplayEngine."""
    import integrations.agentguard.replay_backend as rb
    import inspect
    source = inspect.getsource(rb.TrackTrackReplayBackend.replay_into_live_track)
    assert "ReplayEngine" not in source
    assert "replay_event" not in source


def test_checkpoint_roll_preserves_live_fields():
    """After checkpoint roll, live Track box/mean/cov/feature/velocity must not change."""
    from integrations.agentguard.replay_backend import TrackTrackReplayBackend

    backend = TrackTrackReplayBackend()

    snapshot = _make_snapshot(
        end_frame_id=3,
        box=np.array([100, 200, 300, 400], dtype=np.float64),
        mean=np.array([200, 300, 200, 200, 1, 0.5, 0, 0], dtype=np.float64),
    )
    det = _make_detection()
    oldest_step = ReplayStep(99, np.eye(2, 3), True, det, 0.4, 0.7)
    roll_plan = ReplayPlan(track_id=1, checkpoint=snapshot, steps=[oldest_step])

    spy = _SpyTrack()
    # Set live track to a known "current" state
    live_snapshot = _make_snapshot(
        end_frame_id=50,
        box=np.array([500, 600, 700, 800], dtype=np.float64),
        mean=np.array([650, 700, 200, 200, 5, 2, 0, 0], dtype=np.float64),
        feat=np.random.RandomState(99).randn(1, 64).astype(np.float64),
    )
    spy.restore_state(live_snapshot)

    # Save current live state
    saved_box = spy.box.copy()
    saved_mean = spy.mean.copy()
    saved_feat = spy.feat.copy()
    saved_vel = spy.velocity.copy()
    saved_efid = spy.end_frame_id

    # Execute roll: replay oldest event, save new checkpoint
    backend.replay_into_live_track(spy, roll_plan)
    # spy is now at checkpoint + oldest_step state

    # Restore saved live state
    spy.restore_state(live_snapshot)

    # After restore, live state must match saved values exactly
    assert np.allclose(spy.box, saved_box)
    assert np.allclose(spy.mean, saved_mean)
    assert np.allclose(spy.feat, saved_feat)
    assert np.allclose(spy.velocity, saved_vel)
    assert spy.end_frame_id == saved_efid


# ======================================================================
# Stage 1: AssociationContext wiring
# ======================================================================


def test_association_context_is_built_and_wired():
    """AssociationContext must be created with full matrix snapshots and wired into TrackEvent."""
    from integrations.agentguard.converters import build_association_context

    meta = {
        "final_cost": np.array([[0.3, 0.5, 0.7], [0.4, 0.2, 0.6], [0.9, 0.8, 0.1]], dtype=np.float64),
        "cosine_distance": np.array([[0.1, 0.2, 0.3], [0.15, 0.05, 0.25], [0.3, 0.2, 0.4]], dtype=np.float64),
        "iou_similarity": np.array([[0.8, 0.6, 0.4], [0.7, 0.9, 0.5], [0.3, 0.2, 0.85]], dtype=np.float64),
    }
    det_overlap = np.array([0.2, 0.0, 0.4], dtype=np.float64)

    ctx = build_association_context(
        meta, track_index=1, detection_index=1,
        num_tracks=3, num_detections=3, no_reid=False,
        detection_overlap_row=det_overlap,
    )

    assert ctx.track_cost_row.shape == (3,)
    assert ctx.detection_cost_col.shape == (3,)
    assert ctx.detection_overlap_row.shape == (3,)
    assert ctx.accepted_detection_index == 1
    assert ctx.num_tracks == 3
    assert ctx.num_detections == 3
    assert ctx.reid_available is True
    # Track-to-detection costs still use row 1, while detection overlap is
    # supplied independently from the detection pool.
    np.testing.assert_array_equal(ctx.track_cost_row, np.array([0.4, 0.2, 0.6]))
    np.testing.assert_array_equal(ctx.detection_cost_col, np.array([0.5, 0.2, 0.8]))
    np.testing.assert_array_equal(ctx.detection_overlap_row, det_overlap)


def test_association_context_without_detection_overlap_is_safe():
    from integrations.agentguard.converters import build_association_context

    ctx = build_association_context(
        {"final_cost": np.zeros((1, 2), dtype=np.float64)},
        track_index=0,
        detection_index=0,
        num_tracks=1,
        num_detections=2,
    )

    np.testing.assert_array_equal(ctx.detection_overlap_row, np.zeros(2))


def test_adapter_detection_overlap_uses_detection_pool_and_excludes_self():
    from types import SimpleNamespace

    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    class Detection:
        def __init__(self, box, score):
            self.box = np.asarray(box, dtype=np.float64)
            self.score = score

        @property
        def x1y1x2y2(self):
            return self.box

    first = Detection([0, 0, 9, 9], 0.9)
    second = Detection([5, 0, 14, 9], 0.8)
    adapter = AgentGuardTrackerAdapter(SimpleNamespace(dataset="MOT17"), "seq")

    adapter.begin_frame(
        frame_id=1,
        img_width=1920,
        img_height=1080,
        detection_pool=[first, second],
    )

    overlap = adapter._frame_detection_overlap
    assert overlap.shape == (2, 2)
    np.testing.assert_array_equal(np.diag(overlap), np.zeros(2))
    assert overlap[0, 1] == overlap[1, 0]
    assert overlap[0, 1] > 0.0


def test_association_context_no_reid_fallback():
    """When no_reid=True, cosine distance is zeros and reid_available=False."""
    from integrations.agentguard.converters import build_association_context

    meta = {
        "final_cost": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
    }

    ctx = build_association_context(
        meta, track_index=0, detection_index=0,
        num_tracks=2, num_detections=2, no_reid=True,
    )

    assert ctx.reid_available is False


def test_detection_to_update_input_shape():
    """detection_to_update_input must produce flat feature with shape (D,)."""
    from integrations.agentguard.converters import detection_to_update_input

    det = DetectionObservation(
        detection_index=0,
        box=np.array([10, 20, 30, 40], dtype=np.float64),
        score=0.9,
        feature=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64),
        source=0,
        class_id=1,
    )
    result = detection_to_update_input(det)
    assert result["feat"].ndim == 1
    assert result["feat"].shape == (4,)
    np.testing.assert_array_equal(result["box"], det.box)
    assert result["score"] == 0.9


# ======================================================================
# Stage 1: EventSink validation
# ======================================================================


def test_event_sink_validates_payload_types():
    """CacheEventSink must reject custom objects in shard payloads."""
    from agentguard.data.event_sink import _validate_payload

    # Valid payloads
    _validate_payload({"a": 1, "b": "hello", "c": [1, 2, 3], "d": None})
    _validate_payload({"arr": np.array([1.0, 2.0])})

    # Invalid: custom object
    class _Custom:
        pass
    with pytest.raises(TypeError, match="Disallowed type"):
        _validate_payload({"bad": _Custom()})


def test_serialize_deserialize_includes_association_context():
    """serialize_event / deserialize_event round-trips association_context."""
    from agentguard.contracts.serialization import serialize_event, deserialize_event
    from agentguard.contracts.states import AssociationContext
    from agentguard.contracts.events import TrackEvent

    ctx = AssociationContext(
        track_cost_row=np.array([0.3, 0.5, 0.7], dtype=np.float64),
        detection_cost_col=np.array([0.3, 0.45], dtype=np.float64),
        detection_overlap_row=np.array([0.8, 0.6, 0.4], dtype=np.float64),
        accepted_detection_index=1,
        num_tracks=2,
        num_detections=3,
        reid_available=True,
    )

    evt = TrackEvent(
        event_id="test/seq/000001/000042",
        dataset="test",
        sequence="seq",
        frame_id=1,
        track_id=42,
        image_width=1920,
        image_height=1080,
        has_detection=True,
        association_context=ctx,
    )

    serialized = serialize_event(evt)
    assert serialized["association_context"] is not None
    assert serialized["association_context"]["num_tracks"] == 2

    restored = deserialize_event(serialized)
    assert restored.association_context is not None
    assert restored.association_context.num_tracks == 2
    np.testing.assert_array_equal(
        restored.association_context.track_cost_row, ctx.track_cost_row
    )
