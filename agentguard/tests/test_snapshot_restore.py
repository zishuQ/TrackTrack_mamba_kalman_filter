"""Tests for TrackStateSnapshot copy semantics and restore_track_state."""
from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.contracts.states import TrackStateSnapshot


def test_snapshot_copies_all_fields(
    mock_box, mock_kf_mean, mock_kf_cov, mock_velocity, mock_feature, mock_history,
    mock_track_id,
):
    """TrackStateSnapshot should deep-copy all numpy arrays from constructor args."""
    # Create a snapshot from mutable originals
    orig_box = mock_box.copy()
    orig_mean = mock_kf_mean.copy()
    orig_cov = mock_kf_cov.copy()
    orig_vel = mock_velocity.copy()
    orig_feat = mock_feature.copy()
    orig_hist = mock_history.copy()

    snap = TrackStateSnapshot(
        track_id=mock_track_id,
        box=orig_box,
        score=0.85,
        mean=orig_mean,
        covariance=orig_cov,
        velocity=orig_vel,
        feature=orig_feat,
        history=orig_hist,
        end_frame_id=98,
        state=1,
    )

    # Mutating the ORIGINAL arrays should NOT affect the snapshot
    orig_box[:] = 0.0
    orig_mean[:] = 0.0
    orig_cov[:] = 0.0
    orig_vel[:] = 0.0
    orig_feat[:] = 0.0
    for fid in orig_hist:
        for j, item in enumerate(orig_hist[fid]):
            if isinstance(item, np.ndarray):
                item[:] = 0.0

    # Snapshot should retain pre-mutation values
    np.testing.assert_array_equal(snap.box, mock_box)
    if snap.mean is not None:
        np.testing.assert_array_equal(snap.mean, mock_kf_mean)
    if snap.covariance is not None:
        np.testing.assert_array_equal(snap.covariance, mock_kf_cov)
    np.testing.assert_array_equal(snap.velocity, mock_velocity)
    np.testing.assert_array_equal(snap.feature, mock_feature)
    for fid in snap.history:
        for j, item in enumerate(snap.history[fid]):
            if isinstance(item, np.ndarray):
                assert not np.allclose(item, 0.0), f"History entry [{fid}][{j}] was zeroed"


def test_snapshot_copies_history_arrays(
    mock_box, mock_kf_mean, mock_kf_cov, mock_feature,
):
    """History entries containing arrays should be deep-copied from constructor."""
    # Build a history dict with mutable arrays
    orig_hist = {
        90: [mock_box.copy(), 0.7, mock_kf_mean.copy(), mock_kf_cov.copy(), mock_feature.copy()],
    }

    snap = TrackStateSnapshot(
        track_id=1,
        box=mock_box.copy(),
        score=0.8,
        mean=mock_kf_mean.copy(),
        covariance=mock_kf_cov.copy(),
        velocity=np.zeros((4, 2)),
        feature=mock_feature.copy(),
        history=orig_hist,
        end_frame_id=90,
        state=1,
    )

    # Mutate the ORIGINAL history arrays
    orig_hist[90][0][:] = -999.0
    orig_hist[90][2][:] = -999.0
    orig_hist[90][3][:] = -999.0
    orig_hist[90][4][:] = -999.0

    # Snapshot's history should be unchanged
    np.testing.assert_array_equal(snap.history[90][0], mock_box)
    np.testing.assert_array_equal(snap.history[90][2], mock_kf_mean)


def test_snapshot_scalar_fields_unchanged(mock_track_snapshot: TrackStateSnapshot):
    """Scalar fields (track_id, score, end_frame_id, state) should be preserved."""
    snap = mock_track_snapshot
    assert snap.track_id == 42
    assert snap.score == 0.85
    assert snap.end_frame_id == 98
    assert snap.state == 1  # TRACKED


def test_snapshot_with_none_mean_covariance(mock_track_snapshot, mock_track_id, mock_box, mock_velocity, mock_feature, mock_history):
    """Snapshot handles None mean / covariance gracefully."""
    snap = TrackStateSnapshot(
        track_id=mock_track_id,
        box=mock_box.copy(),
        score=0.5,
        mean=None,
        covariance=None,
        velocity=mock_velocity.copy(),
        feature=mock_feature.copy(),
        history=mock_history,
        end_frame_id=0,
        state=0,  # NEW
    )
    assert snap.mean is None
    assert snap.covariance is None
    # No mutation through copying should occur
    snap.box[:] = 0.0
    assert not np.all(mock_box == 0.0)


def test_snapshot_immutability_via_frozen(mock_track_snapshot: TrackStateSnapshot):
    """Frozen dataclass should prevent setting attributes directly."""
    with pytest.raises(AttributeError):
        mock_track_snapshot.track_id = 999  # type: ignore[misc]


def test_snapshot_roundtrip_values(mock_track_snapshot: TrackStateSnapshot):
    """Verify all values survive a construction round-trip."""
    snap = mock_track_snapshot
    snap2 = TrackStateSnapshot(
        track_id=snap.track_id,
        box=snap.box.copy(),
        score=snap.score,
        mean=snap.mean.copy() if snap.mean is not None else None,
        covariance=snap.covariance.copy() if snap.covariance is not None else None,
        velocity=snap.velocity.copy(),
        feature=snap.feature.copy(),
        history=snap.history,
        end_frame_id=snap.end_frame_id,
        state=snap.state,
    )
    assert snap.track_id == snap2.track_id
    assert snap.score == pytest.approx(snap2.score)
    assert snap.end_frame_id == snap2.end_frame_id
    assert snap.state == snap2.state
    np.testing.assert_array_equal(snap.box, snap2.box)
    if snap.mean is not None and snap2.mean is not None:
        np.testing.assert_array_equal(snap.mean, snap2.mean)
    if snap.covariance is not None and snap2.covariance is not None:
        np.testing.assert_array_equal(snap.covariance, snap2.covariance)
    np.testing.assert_array_equal(snap.velocity, snap2.velocity)
    np.testing.assert_array_equal(snap.feature, snap2.feature)
    assert snap.history.keys() == snap2.history.keys()


def test_snapshot_copy_independence():
    """Snapshots constructed from the same source data should be independent."""
    box = np.array([10.0, 20.0, 100.0, 200.0])
    mean = np.zeros(8)
    cov = np.eye(8) * 0.01
    vel = np.zeros((4, 2))
    feat = np.ones((1, 16))
    hist = {0: [box.copy(), 1.0, mean.copy(), cov.copy(), feat.copy()]}

    s1 = TrackStateSnapshot(
        track_id=1, box=box, score=1.0, mean=mean, covariance=cov,
        velocity=vel, feature=feat, history=hist, end_frame_id=0, state=0,
    )
    s2 = TrackStateSnapshot(
        track_id=1, box=box, score=1.0, mean=mean, covariance=cov,
        velocity=vel, feature=feat, history=hist, end_frame_id=0, state=0,
    )

    # Mutate s1
    s1.box[:] = 999.0
    # s2 should be unaffected
    assert s2.box[0] == 10.0


def test_snapshot_history_deep_copy():
    """History entries with nested arrays are deeply independent."""
    box = np.array([10.0, 20.0, 100.0, 200.0])
    box2 = np.array([50.0, 60.0, 150.0, 250.0])
    mean = np.zeros(8)
    cov = np.eye(8)
    feat = np.ones((1, 16))
    hist = {0: [box.copy(), 1.0, mean.copy(), cov.copy(), feat.copy()]}

    snap = TrackStateSnapshot(
        track_id=1, box=box, score=1.0, mean=mean, covariance=cov,
        velocity=np.zeros((4, 2)), feature=feat, history=hist,
        end_frame_id=0, state=0,
    )
    # Modify original hist
    hist[0][0][:] = 999.0
    # Snapshot history should be unchanged
    np.testing.assert_array_equal(snap.history[0][0], box)
