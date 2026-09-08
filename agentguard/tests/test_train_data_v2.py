from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.compact_iwg_labels import build_current_rollout_compact_labels_for_sequence
from agentguard.data.identity_prototype import IdentityPrototypeBuilder
from agentguard.datasets.iwg_rg_cma_dataset import build_iwg_rg_cma_dataset, StreamingIWGRGCMADataset
from agentguard.rollout.motion import compute_motion_benefit
from agentguard.rollout.context import RolloutContext
from agentguard.rollout.losses import motion_frame_loss


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_v2", ROOT / "scripts/agentguard/build_mot17_train_data_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_motion_current_and_future_use_gt_dimensions(mock_event, mock_kf):
    motion_kf = mock_kf
    gt = np.array([80.0, 180.0, 360.0, 470.0])
    ctx = RolloutContext(1, 1, mock_event.pre_update_state, mock_event.detection,
                         gt, [gt], [None], [np.eye(2, 3)], None)
    _, _, skip, _ = compute_motion_benefit(ctx, motion_kf, include_current=True, normalization="gt")
    state = ctx.pre_update_state
    box = motion_kf.mean_to_bbox(state.mean)
    assert skip[0] == pytest.approx(motion_frame_loss(box, gt))
    assert not np.isclose(skip[0], motion_frame_loss(gt, box))
    w_mean, w_cov = motion_kf.apply_warp(state.mean, state.covariance, np.eye(2, 3))
    pred, _ = motion_kf.predict(w_mean, w_cov)
    pred_box = motion_kf.mean_to_bbox(pred)
    assert skip[1] == pytest.approx(motion_frame_loss(pred_box, gt))
    legacy = compute_motion_benefit(ctx, motion_kf, include_current=True)
    assert legacy[2][0] == pytest.approx(motion_frame_loss(gt, box))
    assert legacy[2][1] == pytest.approx(motion_frame_loss(gt, pred_box))


def test_loo_excludes_before_mean_and_trim_and_requires_three():
    rng = np.random.default_rng(8)
    observations = {i: rng.normal(size=8) for i in range(10)}
    before = {k: v.copy() for k, v in observations.items()}
    actual = IdentityPrototypeBuilder.build_leave_one_out(observations, 4)
    expected = IdentityPrototypeBuilder.build_prototype([v for k, v in observations.items() if k != 4])
    np.testing.assert_array_equal(actual, expected)
    # Repeated observations of one detection stay one reference.
    unique = {0: observations[0], 1: observations[1], 2: observations[2]}
    for _ in range(5):
        unique.setdefault(1, observations[1])
    assert IdentityPrototypeBuilder.build_leave_one_out(unique, 0) is None
    assert IdentityPrototypeBuilder.build_leave_one_out(unique, 99) is not None
    assert all(np.array_equal(observations[k], before[k]) for k in observations)


def make_fixture(tmp_path):
    sequence = "MOT17-09-FRCNN"
    event_root = tmp_path / "events"
    event_dir = event_root / "MOT17/all" / sequence
    detection_root = tmp_path / "detection"
    detection_dir = detection_root / "MOT17/all" / sequence
    gt_parent = tmp_path / "datasets/MOT17/train"
    gt_dir = gt_parent / sequence / "gt"
    for p in (event_dir, detection_dir, gt_dir):
        p.mkdir(parents=True)
    n, dim = 8, 4
    features = np.array([[1.0, i * 0.05, (i % 2) * 0.1, 0.0] for i in range(n)], dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    boxes = np.tile(np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32), (n, 1))
    for name, value in {
        "boxes": boxes, "features": features, "scores": np.ones(n, dtype=np.float32) * 0.9,
        "sources": np.zeros(n, dtype=np.int8), "class_ids": np.ones(n, dtype=np.int16),
        "frame_offsets": np.arange(n + 1, dtype=np.int64),
        "target_frame_offsets": np.arange(n + 1, dtype=np.int64),
        "target_detection_indices": np.arange(n, dtype=np.int64),
    }.items():
        np.save(detection_dir / f"{name}.npy", value)
    detection_manifest = {"complete": True, "schema_version": 1, "dataset": "MOT17",
                          "split": "all", "sequence": sequence, "num_frames": n,
                          "num_detections": n, "reid_dim": dim}
    (detection_dir / "manifest.json").write_text(json.dumps(detection_manifest))
    (gt_dir / "gt.txt").write_text("".join(f"{i+1},1,0,0,10,20,1,1,1\n" for i in range(n)))
    (gt_dir.parent / "seqinfo.ini").write_text("[Sequence]\nimWidth=100\nimHeight=100\nframeRate=30\nseqLength=8\n")
    states, events, frames, associations = [], [], [], []
    state = {"mean": np.array([6.0, 10.0, 12.0, 21.0, 0, 0, 0, 0]), "covariance": np.eye(8),
             "box": np.array([0.0, -0.5, 12.0, 20.5]), "score": 0.9, "state": 1,
             "velocity": np.zeros((4, 2)), "end_frame_id": 0,
             "recent_history_frames": np.zeros(0, dtype=np.int32),
             "recent_history_boxes": np.zeros((0, 4)), "recent_history_scores": np.zeros(0)}
    for i in range(n):
        states.extend([state, state])
        events.append({"event_id": f"MOT17/{sequence}/{i+1:06d}/000001", "frame_id": i+1,
                       "frame_index": i, "track_id": 1, "matched": True, "accepted_detection_index": i,
                       "association_shard_id": 0, "association_offset": i, "association_track_row": 0,
                       "state_shard_id": 0, "frame_start_state_offset": 2*i, "pre_update_state_offset": 2*i+1,
                       "event_shard_id": 0, "event_offset": i, "scalar_features": np.ones(63, dtype=np.float32)*i,
                       "track_feature": np.array([1., 0., 0., 0.], dtype=np.float32), "history_count": min(i+1,6)})
        frames.append({"frame_id": i+1, "frame_index": i, "effective_warp": np.eye(2,3),
                       "image_width": 100, "image_height": 100})
        associations.append({"frame_id": i+1, "track_ids": np.array([1]), "detection_indices": np.array([i]),
                             "final_cost": np.zeros((1,1)), "assignment_threshold": np.ones((1,1)),
                             "detection_source": np.zeros((1,1), dtype=np.int8)})
    for kind, records in {"events": events, "states": states, "frames": frames, "associations": associations}.items():
        torch.save(records, event_dir / f"{kind}_00000.pt")
    manifest = {"complete": True, "truncated": False, "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256, "dataset": "MOT17", "sequence": sequence,
                "num_frames": n, "num_events": n, "reid_dim": dim,
                "detection_cache_manifest_sha256": runner.sha256(detection_dir / "manifest.json")}
    (event_dir / "manifest.json").write_text(json.dumps(manifest))
    label_root = tmp_path / "old_labels"
    build_current_rollout_compact_labels_for_sequence(sequence=sequence, event_cache_dir=event_dir,
        detection_cache_dir=detection_dir, gt_root=gt_parent, output_root=label_root)
    old_root = tmp_path / "old"
    build_iwg_rg_cma_dataset(dataset="MOT17", split="all", event_cache_root=event_root,
        detection_cache_root=detection_root, label_dir=label_root, output_dir=old_root / "MOT17",
        sequence_subset=[sequence])
    return sequence, event_root, detection_root, old_root



from agentguard.data.train_data_v2 import PackedReplayReader, table_row, pack_table
from agentguard.data.rollout_label_builder import build_compact_rollout_labels_for_sequence
import shutil
import pickle


def invoke(tmp_path, monkeypatch, sequence, event_root, detection_root, output, relabel=None):
    argv = ["build_v2", "--sequence", sequence, "--output-root", str(output),
            "--detection-cache-root", str(detection_root), "--data-dir", str(tmp_path / "datasets")]
    argv += ["--relabel-from", str(relabel)] if relabel else ["--event-cache-root", str(event_root)]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()


def test_v2_roundtrip_relabel_without_raw_cache_and_old_compatibility(tmp_path, monkeypatch):
    sequence, event_root, detection_root, old_root = make_fixture(tmp_path)
    old_hashes = {p: runner.sha256(p) for p in old_root.rglob("*") if p.is_file()}
    output = tmp_path / "v2"
    invoke(tmp_path, monkeypatch, sequence, event_root, detection_root, output)
    seq = output / "MOT17" / sequence
    assert {p.name for p in seq.iterdir()} == {"data.pt", "manifest.json", "reid_features.npy"}
    assert np.load(seq / "reid_features.npy").shape == (8, 4)
    packed = torch.load(seq / "data.pt", mmap=True, weights_only=True)
    assert packed["metadata"]["generation_rules"] == runner.RULES
    assert set(packed["replay"]["tables"]) == {"events", "states", "frames", "associations"}
    assert all(runner.sha256(p) == h for p, h in old_hashes.items())
    old = StreamingIWGRGCMADataset(old_root / "MOT17")
    new = StreamingIWGRGCMADataset(output / "MOT17")
    assert len(old) == len(new) > 0
    # Same captured features, changed supervision. Packed input fields agree.
    for i in range(len(new)):
        for key in ("track_feats", "det_feats", "scalar_feats", "padding_mask", "has_detection_mask"):
            torch.testing.assert_close(new[i][key], old[i][key])
    # Workers reopen mmap files instead of pickling the replay payload.
    restored = pickle.loads(pickle.dumps(new))
    torch.testing.assert_close(restored[0]["det_feats"], new[0]["det_feats"])
    restored.close()
    new.close()
    old.close()
    before = runner.sha256(seq / "data.pt")
    invoke(tmp_path, monkeypatch, sequence, event_root, detection_root, output)
    assert runner.sha256(seq / "data.pt") == before
    shutil.rmtree(event_root)
    shutil.rmtree(old_root)
    reader = PackedReplayReader(seq)
    labels, _ = build_compact_rollout_labels_for_sequence(seq, detection_root / "MOT17/all" / sequence,
        tmp_path / "datasets/MOT17/train", reader=reader, persist_stats=False, **runner.RULES)
    for i, label in enumerate(labels):
        original = table_row(packed["labels_raw"], i)
        assert original == label
    second = tmp_path / "v2_relabel"
    invoke(tmp_path, monkeypatch, sequence, None, detection_root, second, relabel=output)
    other = torch.load(second / "MOT17" / sequence / "data.pt", weights_only=True)
    for key, value in packed["arrays"].items():
        torch.testing.assert_close(other["arrays"][key], value, rtol=0, atol=0)


def test_ragged_state_and_association_tables_are_lossless():
    records = [{"matrix": np.arange(i * 3, dtype=np.float32).reshape(i, 3),
                "history": np.arange(i, dtype=np.int32), "name": f"row{i}"} for i in range(4)]
    table = pack_table(records)
    for i, row in enumerate(records):
        actual = table_row(table, i)
        for key in row:
            np.testing.assert_array_equal(actual[key], row[key])


def test_v2_relocation_and_wrong_detection_cache_rejected(tmp_path, monkeypatch):
    sequence, events, detection_root, _ = make_fixture(tmp_path)
    output = tmp_path / "v2"
    invoke(tmp_path, monkeypatch, sequence, events, detection_root, output)
    moved = tmp_path / "moved_detection"
    detection_root.rename(moved)
    dataset = StreamingIWGRGCMADataset(output / "MOT17", detection_cache_root=moved)
    assert dataset[0]["det_feats"].shape[-1] == 4
    dataset.close()
    path = moved / "MOT17/all" / sequence / "features.npy"
    features = np.load(path)
    features[0, 0] += 0.01
    np.save(path, features)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner.validate_final(output / "MOT17" / sequence, moved / "MOT17/all" / sequence)


def test_unmatched_without_association_and_resume_labels(tmp_path, monkeypatch):
    sequence, events, detection_root, _ = make_fixture(tmp_path)
    path = events / "MOT17/all" / sequence / "events_00000.pt"
    records = torch.load(path, weights_only=False)
    records[4].update(matched=False, accepted_detection_index=-1,
                      association_shard_id=-1, association_offset=-1, association_track_row=-1)
    torch.save(records, path)
    output = tmp_path / "v2"
    original_pack = runner.write_packed_sequence
    def fail_pack(*args, **kwargs):
        raise RuntimeError("simulated interrupted packing")
    monkeypatch.setattr(runner, "write_packed_sequence", fail_pack)
    with pytest.raises(RuntimeError, match="interrupted packing"):
        invoke(tmp_path, monkeypatch, sequence, events, detection_root, output)
    assert (output / "MOT17/.work" / sequence / "labels.pt").is_file()
    assert not (output / "MOT17" / sequence).exists()
    monkeypatch.setattr(runner, "write_packed_sequence", original_pack)
    def forbid_labels(*args, **kwargs):
        raise AssertionError("completed label stage must be reused")
    monkeypatch.setattr(runner, "build_compact_rollout_labels_for_sequence", forbid_labels)
    invoke(tmp_path, monkeypatch, sequence, events, detection_root, output)
    dataset = StreamingIWGRGCMADataset(output / "MOT17")
    samples = [dataset[i] for i in range(len(dataset))]
    containing_unmatched = [s for s in samples if ((~s["has_detection_mask"]) & (~s["padding_mask"])).any()]
    assert containing_unmatched
    for sample in containing_unmatched:
        missing = (~sample["has_detection_mask"]) & (~sample["padding_mask"])
        assert sample["det_feats"][missing].count_nonzero() == 0
        assert sample["track_feats"][missing].count_nonzero() > 0
    dataset.close()
    assert not (output / "MOT17/.work" / sequence).exists()
    assert path.is_file()  # An explicitly supplied source is never cleaned up.


def test_v2_cpu_training_checkpoint_and_validation(tmp_path, monkeypatch):
    from agentguard.training.train_iwg_rg_cma import smoke_train_iwg_rg_cma, validate_iwg_rg_cma_checkpoint
    torch.set_num_threads(1)
    sequence, events, detection_root, _ = make_fixture(tmp_path)
    output = tmp_path / "v2"
    invoke(tmp_path, monkeypatch, sequence, events, detection_root, output)
    checkpoint = tmp_path / "checkpoints"
    smoke_train_iwg_rg_cma({"dataset_dir": str(output / "MOT17"), "checkpoint_dir": str(checkpoint),
                           "device": "cpu", "epochs": 1, "batch_size": 2, "num_workers": 0,
                           "context_size": 6, "correction_bound": 0.05})
    result = validate_iwg_rg_cma_checkpoint(checkpoint_path=checkpoint / "iwg_rg_cma_last.pt",
                                           dataset_dir=output / "MOT17", device="cpu")
    assert (checkpoint / "iwg_rg_cma_last.pt").is_file()
    assert result


def test_capture_uses_shared_cache_and_fresh_process(tmp_path, monkeypatch):
    from argparse import Namespace
    args = Namespace(event_cache_root=None, data_dir=tmp_path / "datasets", detection_cache_root=tmp_path / "det")
    work = tmp_path / "work"
    work.mkdir()
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    expected = runner.capture(args, "MOT17-09-FRCNN", work)
    command, kw = calls[0]
    assert command[:4] == [sys.executable, "-m", "agentguard.cli", "cache_events"]
    assert command[command.index("--detection-cache-root") + 1] == str(args.detection_cache_root)
    assert command[command.index("--mode") + 1] == "all"
    assert command[command.index("--detector") + 1] == "FRCNN"
    assert kw["check"] is True
    assert expected == work / "capture/MOT17/all/MOT17-09-FRCNN"


def test_cache_events_cli_enables_capture_and_writes_real_sink(tmp_path, monkeypatch):
    from argparse import Namespace
    from types import ModuleType
    import agentguard.cli as cli
    sequence, events, detection_root, _ = make_fixture(tmp_path)
    source = events / "MOT17/all" / sequence
    records = torch.load(source / "events_00000.pt", weights_only=False)
    states = torch.load(source / "states_00000.pt", weights_only=False)
    associations = torch.load(source / "associations_00000.pt", weights_only=False)
    for a in associations:
        for key in ("raw_cost", "iou_similarity", "iou_distance", "cosine_distance", "confidence_distance", "angle_distance"):
            a[key] = np.zeros((1, 1), dtype=np.float32)
        a["assignment_round"] = np.zeros((1, 1), dtype=np.int16)
    tracker_module = ModuleType("trackers.tracker")
    etc_module = ModuleType("utils.etc")
    def parameters(args, seq, mode):
        args.data_path = str(tmp_path / "datasets/MOT17/train")
        args.det_thr, args.init_thr, args.match_thr = 0.6, 0.7, 0.7
    etc_module.set_parameters = parameters
    class FakeTracker:
        def __init__(self, args, seq):
            assert args.agentguard_mode == "capture"
            assert args.kf_type == "nsa" and args.disable_gmc is False
            self.args, self.index = args, 0
        def update(self, target, source):
            i = self.index
            r = records[i]
            state = {**states[2*i], "history": {j: [states[2*i]["box"], .9] for j in range(1,i+2)}}
            event = {**r, "has_detection": True, "frame_start_state": state, "pre_update_state": state}
            self.args.event_sink.on_frame({"frame_id": i+1, "frame_index": i, "warp_matrix": np.eye(2,3),
                "image_width": 100, "image_height": 100, "association": associations[i],
                "detections": [{"detection_index": i}]}, [event])
            self.index += 1
    tracker_module.Tracker = FakeTracker
    monkeypatch.setitem(sys.modules, "trackers.tracker", tracker_module)
    monkeypatch.setitem(sys.modules, "utils.etc", etc_module)
    cache_root = tmp_path / "captured"
    cli._cmd_cache_events(Namespace(dataset="MOT17", mode="all", sequence=sequence, detector="FRCNN",
        data_dir=str(tmp_path / "datasets"), max_frames=0, detection_cache_root=str(detection_root),
        event_cache_root=str(cache_root)))
    manifest = json.loads((cache_root / "MOT17/all" / sequence / "manifest.json").read_text())
    assert manifest["complete"] and manifest["num_events"] == 8
    assert manifest["tracker_config"]["agentguard_mode"] == "capture"
    # Exercise real sink output through v2, including ragged recent histories.
    invoke(tmp_path, monkeypatch, sequence, cache_root, detection_root, tmp_path / "v2_sink")


def test_two_sequences_update_global_normalization_without_rewriting_first(tmp_path, monkeypatch):
    sequence, events, detection_root, _ = make_fixture(tmp_path)
    output = tmp_path / "v2"
    invoke(tmp_path, monkeypatch, sequence, events, detection_root, output)
    first_path = output / "MOT17" / sequence / "data.pt"
    first_hash = runner.sha256(first_path)
    second = "MOT17-02-FRCNN"
    for root in (events / "MOT17/all", detection_root / "MOT17/all", tmp_path / "datasets/MOT17/train"):
        shutil.copytree(root / sequence, root / second)
    for root in (events / "MOT17/all", detection_root / "MOT17/all"):
        manifest_path = root / second / "manifest.json"
        m = json.loads(manifest_path.read_text())
        m["sequence"] = second
        manifest_path.write_text(json.dumps(m))
    manifest_path = events / "MOT17/all" / second / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["detection_cache_manifest_sha256"] = runner.sha256(detection_root / "MOT17/all" / second / "manifest.json")
    manifest_path.write_text(json.dumps(m))
    event_path = events / "MOT17/all" / second / "events_00000.pt"
    records = torch.load(event_path, weights_only=False)
    for r in records:
        r["event_id"] = r["event_id"].replace(sequence, second)
        r["scalar_features"] += 10
    torch.save(records, event_path)
    invoke(tmp_path, monkeypatch, second, events, detection_root, output)
    assert runner.sha256(first_path) == first_hash
    ds = StreamingIWGRGCMADataset(output / "MOT17")
    assert set(ds.metadata["train_sequences"]) == {sequence, second}
    assert len(ds) == 10
    np.testing.assert_allclose(ds.norm_stats.mean, 8.5)
    expected_std = np.std(np.concatenate([np.arange(8), np.arange(8)+10]))
    np.testing.assert_allclose(ds.norm_stats.std, expected_std)
    ds[0]
    ds[len(ds)-1]
    ds.close()
