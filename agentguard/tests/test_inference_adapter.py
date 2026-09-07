from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "../3. Tracker"))

from integrations.agentguard.adapter import AgentGuardTrackerAdapter
from integrations.agentguard.config_bridge import build_runtime_config
from integrations.agentguard.converters import build_association_context


class _Det:
    def __init__(self, box, score=0.9, index=0):
        self.x1y1x2y2 = np.asarray(box, dtype=np.float64)
        self.score = score
        self.class_id = 0
        self.frame_detection_index = index


def test_attention_diagnostics_config_bridge():
    default = build_runtime_config(SimpleNamespace())
    assert default["return_attention_diagnostics"] is False
    enabled = build_runtime_config(
        SimpleNamespace(agentguard_attention_diagnostics=True)
    )
    assert enabled["return_attention_diagnostics"] is True


@pytest.mark.parametrize("ablation", ["kf", "ema"])
def test_diagnostics_config_keeps_new_ablation_and_fallback_options(ablation):
    args = SimpleNamespace(
        agentguard_mode="iwg-rg-cma",
        rg_cma_output="final",
        rg_cma_alpha=1.0,
        agentguard_disable_kf_gate=ablation == "kf",
        agentguard_disable_ema_gate=ablation == "ema",
        agentguard_fallback_threshold=0.4,
        agentguard_attention_diagnostics=True,
        agentguard_device="cpu",
    )
    config = build_runtime_config(args)
    assert config["return_attention_diagnostics"] is True
    assert config["disable_kf_gate"] is (ablation == "kf")
    assert config["disable_ema_gate"] is (ablation == "ema")
    assert config["fallback_threshold"] == 0.4


def test_no_sink_keeps_online_detection_context_without_serializing_matrices():
    adapter = AgentGuardTrackerAdapter(
        SimpleNamespace(dataset="test", agentguard_mode="iwg-rg-cma"),
        "seq",
        agentguard_runtime=None,
    )
    dets = [
        _Det([0, 0, 10, 10], index=0),
        _Det([8, 0, 18, 10], index=1),
    ]
    adapter.begin_frame(3, 64, 48, detection_pool=dets, detection_sources=[0, 0])
    assert adapter._pending_frame_record is None
    assert adapter._frame_detection_overlap.shape == (2, 2)
    assert adapter._frame_detection_overlap[0, 1] > 0
    adapter.set_association_record(
        [1],
        dets,
        {
            "raw_cost": np.ones((1, 2), dtype=np.float32),
            "final_cost": np.ones((1, 2), dtype=np.float32),
            "iou_similarity": np.ones((1, 2), dtype=np.float32),
            "iou_distance": np.zeros((1, 2), dtype=np.float32),
            "cosine_distance": np.zeros((1, 2), dtype=np.float32),
            "confidence_distance": np.zeros((1, 2), dtype=np.float32),
            "angle_distance": np.zeros((1, 2), dtype=np.float32),
            "assignment_round": np.zeros((1, 2), dtype=np.int16),
            "assignment_threshold": np.zeros((1, 2), dtype=np.float32),
            "detection_source": np.zeros((1, 2), dtype=np.int8),
        },
    )
    assert adapter._pending_frame_record is None


def test_sink_still_receives_independent_frame_and_association_arrays():
    runtime = SimpleNamespace(event_sink=SimpleNamespace(compact=True), mode="capture")
    adapter = AgentGuardTrackerAdapter(
        SimpleNamespace(dataset="test", agentguard_mode="capture"),
        "seq",
        agentguard_runtime=runtime,
    )
    dets = [_Det([0, 0, 4, 4], index=7)]
    adapter.begin_frame(4, 32, 24, detection_pool=dets, detection_sources=[1])
    assert adapter._pending_frame_record is not None
    assert adapter._pending_frame_record["detections"][0]["detection_index"] == 7
    final_cost = np.array([[0.3]], dtype=np.float32)
    adapter.set_association_record(
        [9],
        dets,
        {
            "raw_cost": np.array([[0.2]], dtype=np.float32),
            "final_cost": final_cost,
            "iou_similarity": np.array([[0.8]], dtype=np.float32),
            "iou_distance": np.array([[0.2]], dtype=np.float32),
            "cosine_distance": np.array([[0.1]], dtype=np.float32),
            "confidence_distance": np.array([[0.0]], dtype=np.float32),
            "angle_distance": np.array([[0.0]], dtype=np.float32),
            "assignment_round": np.array([[0]], dtype=np.int16),
            "assignment_threshold": np.array([[0.7]], dtype=np.float32),
            "detection_source": np.array([[1]], dtype=np.int8),
        },
    )
    association = adapter._pending_frame_record["association"]
    assert association is not None
    assert association["track_ids"].tolist() == [9]
    assert association["detection_indices"].tolist() == [7]
    association["final_cost"][0, 0] = 9.0
    assert final_cost[0, 0] == pytest.approx(0.3)


@pytest.mark.parametrize("no_reid", [False, True])
def test_association_context_does_not_read_unused_cosine_or_raw_cost(no_reid):
    meta = {"final_cost": np.array([[0.4, 0.9], [0.7, 0.2]], dtype=np.float64)}
    overlap = np.array([0.0, 0.15], dtype=np.float64)
    context = build_association_context(
        meta,
        track_index=1,
        detection_index=1,
        num_tracks=2,
        num_detections=2,
        no_reid=no_reid,
        detection_overlap_row=overlap,
    )
    np.testing.assert_allclose(context.track_cost_row, [0.7, 0.2])
    np.testing.assert_allclose(context.detection_cost_col, [0.9, 0.2])
    np.testing.assert_allclose(context.detection_overlap_row, overlap)
    assert context.reid_available is (not no_reid)
