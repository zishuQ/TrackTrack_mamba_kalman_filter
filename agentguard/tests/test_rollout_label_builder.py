from __future__ import annotations

import numpy as np

from agentguard.data import rollout_label_builder


def test_current_rollout_builder_golden_fixture(monkeypatch):
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    detection = type(
        "Detection",
        (),
        {"box": box, "score": 0.9, "feature": np.ones((1, 4), dtype=np.float32)},
    )()
    event = type(
        "Event",
        (),
        {
            "event_id": "event-1",
            "sequence": "seq",
            "track_id": 3,
            "frame_id": 10,
            "has_detection": True,
            "detection": detection,
            "pre_update_state": object(),
            "frame_start_state": object(),
        },
    )()
    record = {
        "event_id": "event-1",
        "event_shard_id": 2,
        "event_offset": 4,
        "state_shard_id": 2,
        "frame_start_state_offset": 5,
        "pre_update_state_offset": 6,
        "association_shard_id": 2,
        "association_offset": 7,
        "frame_index": 9,
        "accepted_detection_index": 0,
    }
    warmup_records = [
        {**record, "event_offset": 5, "accepted_detection_index": -1},
        {**record, "event_offset": 6, "accepted_detection_index": -1},
        record,
    ]

    class Reader:
        def __init__(self, *args, **kwargs):
            self.manifest = {"sequence": "seq"}

        def iter_event_records(self):
            return warmup_records

        def materialize_training_event(self, value):
            return event

        def close(self):
            pass

    class GTReader:
        def __init__(self, *args):
            pass

        def get_gt_for_frame(self, frame_id):
            return [(box, 7)]

    class VoteState:
        def resolve_before_current(self, sequence, track_id):
            return 7

        def get_target_identity_key(self, track_id):
            return ("seq", 7)

        def add_current_observation(self, sequence, track_id, detection_gt_id):
            pass

    class PrototypeBuilder:
        @staticmethod
        def build_prototype(features):
            assert len(features) == 3
            return np.ones(4, dtype=np.float64) / 2.0

    monkeypatch.setattr(rollout_label_builder, "CompactEventCacheReader", Reader)
    monkeypatch.setattr(rollout_label_builder, "GTReader", GTReader)
    monkeypatch.setattr(rollout_label_builder, "TrackIdentityVoteState", VoteState)
    monkeypatch.setattr(rollout_label_builder, "IdentityPrototypeBuilder", PrototypeBuilder)
    monkeypatch.setattr(
        rollout_label_builder,
        "_build_future_context",
        lambda *args, **kwargs: (box, [box, box], [None, None], [np.eye(2, 3)] * 2, 2, 0),
    )
    monkeypatch.setattr(
        rollout_label_builder,
        "compute_motion_benefit",
        lambda *args, **kwargs: (0.25, [0.0, 0.0], [0.0, 0.0], np.ones(2, dtype=bool)),
    )
    def fake_appearance(*args, **kwargs):
        return (
            -0.5,
            [0.0, 0.0],
            [0.0, 0.0],
            [],
            [],
            np.ones(2, dtype=bool),
        )

    monkeypatch.setattr(rollout_label_builder, "compute_appearance_benefit", fake_appearance)
    monkeypatch.setattr(
        rollout_label_builder,
        "compute_dataset_stats",
        lambda values: {"tau_motion": 0.1, "tau_appearance": 0.1},
    )

    labels, summary = rollout_label_builder.build_compact_rollout_labels_for_sequence(
        "event-cache",
        "detection-cache",
        "gt-root",
        future_frames=2,
    )

    assert len(labels) == 1
    label = labels[0]
    assert {
        key: label[key]
        for key in (
            "event_id",
            "event_shard_id",
            "event_offset",
            "candidate_type",
            "target_gt_id",
            "target_identity_key",
            "motion_benefit",
            "appearance_benefit",
            "valid_motion",
            "valid_appearance",
            "valid_horizon_count",
            "gt_coverage",
            "oracle_detection_coverage",
        )
    } == {
        "event_id": "event-1",
        "event_shard_id": 2,
        "event_offset": 4,
        "candidate_type": "A",
        "target_gt_id": 7,
        "target_identity_key": ["seq", 7],
        "motion_benefit": 0.25,
        "appearance_benefit": -0.5,
        "valid_motion": True,
        "valid_appearance": True,
        "valid_horizon_count": 2,
        "gt_coverage": 2.0 / 3.0,
        "oracle_detection_coverage": 0.0,
    }
    np.testing.assert_allclose(label["cue_target"], [2.0 / 3.0, 0.0, 2.0 / 3.0])
    assert summary["num_labels"] == 1
    assert summary["candidate_a_count"] == 1
    assert summary["valid_motion_labels"] == 1
    assert summary["valid_appearance_labels"] == 1
    assert summary["prototype_count"] == 1
