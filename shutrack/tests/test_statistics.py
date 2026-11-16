"""Runtime gate-statistics semantics."""

from __future__ import annotations

import numpy as np

from shutrack.runtime.statistics import ShutrackRuntimeStatistics


def test_gate_stats_only_count_matched_applied_events():
    stats = ShutrackRuntimeStatistics()
    stats.record_event()
    stats.record_gate_update()
    stats.record_gates(
        np.array([0.8, 0.6]), matched=True, used_model=True
    )
    stats.record_event()
    stats.record_gates(
        np.zeros(2), matched=False, used_model=False
    )
    summary = stats.summary()

    assert summary["avg_motion_gate"] == 0.8
    assert summary["avg_appearance_gate"] == 0.6
    assert summary["min_motion_gate"] == 0.8
    assert summary["max_motion_gate"] == 0.8
    assert summary["min_appearance_gate"] == 0.6
    assert summary["max_appearance_gate"] == 0.6
    assert summary["motion_gate_observations"] == 1
    assert summary["appearance_gate_observations"] == 1
    assert summary["matched_events"] == 1
    assert summary["unmatched_events"] == 1
    assert summary["no_detection_events"] == 1
    assert summary["model_inference_count"] == 1
    assert summary["gate_update_count"] == 1


def test_gate_stats_without_matches_is_finite():
    stats = ShutrackRuntimeStatistics()
    summary = stats.summary()
    for key, value in summary.items():
        assert np.isfinite(value), key
    assert summary["avg_motion_gate"] == 0.0
    assert summary["avg_appearance_gate"] == 0.0
    assert summary["min_motion_gate"] == 0.0
    assert summary["max_motion_gate"] == 0.0
    assert summary["motion_gate_observations"] == 0
    assert summary["appearance_gate_observations"] == 0


def test_per_channel_averages_use_each_channel_count():
    stats = ShutrackRuntimeStatistics()
    stats.record_gates(np.array([1.0, 0.2]), matched=True, used_model=True)
    stats.record_gates(np.array([0.0, 0.8]), matched=True, used_model=True)
    summary = stats.summary()
    assert summary["avg_motion_gate"] == 0.5
    assert summary["avg_appearance_gate"] == 0.5
    assert summary["min_motion_gate"] == 0.0
    assert summary["max_motion_gate"] == 1.0
    assert summary["min_appearance_gate"] == 0.2
    assert summary["max_appearance_gate"] == 0.8
    assert summary["motion_gate_observations"] == 2
    assert summary["appearance_gate_observations"] == 2
