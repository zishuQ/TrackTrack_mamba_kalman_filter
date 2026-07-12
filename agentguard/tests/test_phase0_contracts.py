from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
import torch

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)


def test_cli_uses_central_cache_schema_version():
    from agentguard import cli

    source = inspect.getsource(cli._cmd_cache_events)
    assert "COMPACT_CACHE_SCHEMA_VERSION" in source
    assert "FEATURE_SCHEMA_SHA256" in source
    assert '"schema_version": 2' not in source


def test_shared_iou_matches_tracktrack_bbox_overlaps():
    from agentguard.features.geometry import pairwise_iou_xyxy
    from trackers.utils import bbox_overlaps

    rng = np.random.default_rng(1234)
    starts_a = rng.uniform(-20.0, 100.0, size=(50, 2))
    starts_b = rng.uniform(-20.0, 100.0, size=(37, 2))
    boxes_a = np.concatenate(
        [starts_a, starts_a + rng.uniform(0.0, 80.0, size=(50, 2))],
        axis=1,
    )
    boxes_b = np.concatenate(
        [starts_b, starts_b + rng.uniform(0.0, 80.0, size=(37, 2))],
        axis=1,
    )

    expected = bbox_overlaps(boxes_a, boxes_b)
    actual = pairwise_iou_xyxy(boxes_a, boxes_b)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert pairwise_iou_xyxy(np.empty((0, 4)), boxes_b).shape == (0, 37)
    assert pairwise_iou_xyxy(boxes_a, np.empty((0, 4))).shape == (50, 0)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"track_cost_row": np.zeros(1)}, "track_cost_row"),
        ({"detection_cost_col": np.zeros(1)}, "detection_cost_col"),
        ({"detection_overlap_row": np.zeros(1)}, "detection_overlap_row"),
        ({"accepted_detection_index": 2}, "local association column"),
        ({"track_cost_row": np.array([0.1, np.nan])}, "finite"),
    ],
)
def test_association_context_rejects_invalid_shapes(overrides, match):
    from agentguard.contracts.states import AssociationContext

    values = {
        "track_cost_row": np.zeros(2),
        "detection_cost_col": np.zeros(3),
        "detection_overlap_row": np.zeros(2),
        "accepted_detection_index": 0,
        "num_tracks": 3,
        "num_detections": 2,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        AssociationContext(**values)


@pytest.mark.parametrize(
    "method_name,batch",
    [
        ("build_iwg_input", False),
        ("build_iwg_batch_input", True),
        ("build_tgr_input", False),
        ("build_tgr_batch_input", True),
    ],
)
def test_feature_builder_recomputes_missing_nonpadding_scalar(
    mock_event,
    method_name,
    batch,
):
    from agentguard.features.builder import EventFeatureBuilder
    from agentguard.features.scalar import compute_scalar_features

    event = copy.deepcopy(mock_event)
    event.scalar_features = None
    expected = compute_scalar_features(event)
    builder = EventFeatureBuilder(reid_dim=event.track_feature.size)
    method = getattr(builder, method_name)
    inputs = method([[event]] if batch else [event])

    np.testing.assert_allclose(
        inputs["scalar_feats"].detach().cpu().numpy().reshape(-1, 63)[0],
        expected,
        rtol=0.0,
        atol=1e-6,
    )
    assert event.scalar_features is not None


def test_checkpoint_uses_current_feature_schema(tmp_path):
    from agentguard.training.checkpointing import save_checkpoint

    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        metadata={"reid_dim": 4, "scalar_dim": 63, "event_dim": 128},
        path=str(path),
    )
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
    assert checkpoint["cache_schema_version"] == COMPACT_CACHE_SCHEMA_VERSION
