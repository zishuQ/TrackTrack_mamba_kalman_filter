from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentguard.data.cache_schema import (
    COMPACT_CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
)
from agentguard.data.compact_iwg_labels import (
    COMPACT_IWG_LABEL_SCHEMA_SHA256,
    COMPACT_IWG_LABEL_SCHEMA_VERSION,
    _write_arrays,
    load_compact_label_arrays,
)
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
)
from agentguard.data.mamba_distill import build_mamba_distill_labels
from agentguard.motion.mamba_shadow import (
    MambaShadow,
    MambaShadowReader,
    MambaShadowWriter,
    apply_warp_to_state,
    canonical_sha256,
)
from agentguard.v0_pipeline import _mamba_native_current_motion_benefit


class _FilterStub:
    def __init__(self):
        self.deleted = []
        self.image_size = None

    def set_image_size(self, width, height):
        self.image_size = (width, height)

    def initiate(self, measurement):
        mean = np.concatenate([np.asarray(measurement), np.zeros(4)])
        return mean, np.eye(8)

    def batch_predict(self, means, covariances, measurements, track_ids):
        predicted = np.asarray(means).copy()
        predicted[:, :4] += predicted[:, 4:]
        return predicted, np.asarray(covariances) + np.eye(8)[None] * 0.1

    def batch_update(self, means, covariances, measurements, track_ids):
        updated = np.asarray(means).copy()
        updated[:, :4] = (
            updated[:, :4] + np.asarray(measurements)
        ) / 2.0
        return updated, np.asarray(covariances) * 0.5

    def delete_track(self, track_id):
        self.deleted.append(int(track_id))


def _writer(tmp_path: Path):
    config = {"checkpoint_sha256": "a" * 64, "profile": "test"}
    return MambaShadowWriter(
        tmp_path / "teacher" / "seq",
        sequence="seq",
        teacher_config=config,
        event_cache_manifest_sha256="b" * 64,
        detection_cache_manifest_sha256="c" * 64,
        shard_size=1,
    )


def test_shadow_lifecycle_gmc_streaming_and_cleanup(tmp_path):
    filter_stub = _FilterStub()
    writer = _writer(tmp_path)
    shadow = MambaShadow(
        sequence="seq",
        image_width=100,
        image_height=50,
        checkpoint_path="unused",
        writer=writer,
        device="cpu",
        filter_factory=lambda _path, _device: filter_stub,
    )
    shadow.initiate_track(7, np.asarray([20.0, 15.0, 10.0, 8.0]))
    warp = np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]])
    expected_mean, _ = apply_warp_to_state(
        shadow.states[7].mean, shadow.states[7].covariance, warp
    )
    shadow.begin_frame([7], warp)
    assert np.allclose(shadow.states[7].prior_mean, expected_mean)
    shadow.update_matches([(7, np.asarray([24.0, 13.0, 10.0, 8.0]))])

    detection = SimpleNamespace(
        detection_index=11,
        box=np.asarray([19.0, 9.0, 29.0, 17.0]),
    )
    event = SimpleNamespace(
        event_id="dataset/seq/000001/000007",
        frame_id=1,
        track_id=7,
        has_detection=True,
        detection=detection,
        pre_update_state=SimpleNamespace(mean=np.arange(8, dtype=np.float64)),
    )
    shadow.record_event(event, np.asarray([18.0, 8.0, 28.0, 18.0]))
    recorded_prior = shadow.states[7].prior_mean.copy()
    shadow.begin_frame([7], np.eye(2, 3))
    shadow.update_matches([(7, np.asarray([90.0, 40.0, 5.0, 5.0]))])
    manifest = shadow.close()

    assert manifest["num_records"] == 1
    assert manifest["num_matched"] == 1
    assert filter_stub.deleted == [7]
    reader = MambaShadowReader(tmp_path / "teacher" / "seq")
    record = next(reader.iter_records())
    assert str(record["event_id"]) == event.event_id
    assert int(record["accepted_detection_index"]) == 11
    assert int(record["gap"]) == 1
    assert np.array_equal(record["mamba_prior_mean"], recorded_prior.astype(np.float32))
    assert str(record["teacher_config_sha256"]) == canonical_sha256(
        {"checkpoint_sha256": "a" * 64, "profile": "test"}
    )
    assert np.isfinite(record["mamba_posterior_covariance"]).all()


def _base_labels(root: Path, sequence: str) -> None:
    sequence_dir = root / sequence
    sequence_dir.mkdir(parents=True)
    safe = np.asarray([[0.2, 0.8], [0.7, 0.4]], dtype=np.float32)
    hashes = _write_arrays(
        sequence_dir,
        event_keys=np.asarray([1, 2], dtype=np.int64),
        safe_gate=safe,
        policy=np.full((2, 5), 0.2, dtype=np.float32),
        cue=np.asarray([[0.1, 0.2, 0.2], [0.3, 0.4, 0.4]], dtype=np.float32),
        risk=np.full((2, 4), 0.25, dtype=np.float32),
        valid=np.ones((2, 2), dtype=np.bool_),
        sample_weight=np.ones(2, dtype=np.float32),
    )
    manifest = {
        "complete": True,
        "dataset": "MOT20",
        "sequence": sequence,
        "source_labels": 2,
        "retained_labels": 2,
        "compact_label_schema_version": COMPACT_IWG_LABEL_SCHEMA_VERSION,
        "compact_label_schema_sha256": COMPACT_IWG_LABEL_SCHEMA_SHA256,
        "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
        "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "array_sha256": hashes,
    }
    (sequence_dir / "manifest.json").write_text(json.dumps(manifest))


def test_hybrid_labels_cap_weight_and_exactly_fallback(tmp_path):
    sequence = "MOT20-01"
    source = tmp_path / "nsa"
    _base_labels(source, sequence)
    results = {
        sequence: (
            {
                "event_keys": np.asarray([1, 2], dtype=np.int64),
                "advantage": np.asarray([-0.1, 2.0], dtype=np.float32),
                "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
                "projection_gate": np.asarray([0.9, 0.1], dtype=np.float32),
            },
            {
                "teacher_config_sha256": "d" * 64,
                "teacher_checkpoint_path": "/teacher.pt",
                "teacher_checkpoint_sha256": "e" * 64,
                "teacher_config": {"profile": "test"},
                "teacher_manifest_sha256": "f" * 64,
                "event_cache_manifest_sha256": "1" * 64,
                "detection_cache_manifest_sha256": "2" * 64,
            },
        )
    }
    output = tmp_path / "hybrid"
    build_mamba_distill_labels(
        sequence_results=results,
        nsa_label_root=source,
        output_root=output,
        tau_adv=1.0,
    )
    weight = np.load(output / sequence / "teacher_weight.npy")
    nsa = np.load(output / sequence / "nsa_motion_target.npy")
    hybrid = np.load(output / sequence / "hybrid_motion_target.npy")
    safe = np.load(output / sequence / "safe_gate_target.npy")
    policy = np.load(output / sequence / "policy_safe_soft_target.npy")

    assert weight.tolist() == [0.0, 0.5]
    assert hybrid[0].tobytes() == nsa[0].tobytes()
    assert hybrid[1] == pytest.approx(0.4)
    assert np.array_equal(safe[:, 0], hybrid)
    assert np.allclose(policy.sum(axis=1), 1.0)
    assert np.all((policy >= 0.0) & (policy <= 1.0))
    manifest, loaded = load_compact_label_arrays(output / sequence)
    assert manifest["motion_target_mode"] == "mamba_hybrid"
    assert np.array_equal(loaded["teacher_weight"], weight)
    assert np.array_equal(loaded["nsa_motion_target"], nsa)


def test_mamba_native_current_motion_benefit_prefers_better_source():
    gt = np.asarray([10.0, 10.0, 20.0, 30.0])
    exact_prior = np.asarray([15.0, 20.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    bad_detection = np.asarray([40.0, 40.0, 50.0, 60.0])
    assert (
        _mamba_native_current_motion_benefit(exact_prior, bad_detection, gt) < 0.0
    )

    bad_prior = np.asarray([45.0, 50.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    assert _mamba_native_current_motion_benefit(bad_prior, gt, gt) > 0.0
