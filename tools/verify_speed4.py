#!/usr/bin/env python3
"""Real-data validation for the 4-D (vx, vy, vw, vh) speed experiment.

Checks
  * the regenerated samples reproduce the previous round's rows and vx/vy
    endpoints bit-for-bit, and that adjacent/central size labels carry
    independent masks;
  * g = 0 / 0.5 / 1 endpoint consistency against the online KF path, including
    events whose next frame has a non-identity GMC warp (online predicted
    states are captured from a real replay);
  * real-batch gradients of the size-velocity L1 vs the centre term and the
    box DIoU, used to pick the starting size coefficient;
  * with size_weight = 0 the 4-D loss, gate gradient and parameter gradients
    are exactly equal to the existing centre-speed loss.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE / "tools"),
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_supervision_groups import (  # noqa: E402
    INIT_CKPT,
    SequenceData,
    batch_tensors,
    derive_targets,
    load_global_taus,
    model_inputs,
    to_device,
)
from sportsmot_supervision_pilot import SequencePilot  # noqa: E402
from shutrack.training.checkpoint import load_shutrack_checkpoint  # noqa: E402
from shutrack.training.loss import (  # noqa: E402
    motion_box_diou_loss,
    motion_speed4_l1_loss,
    motion_speed_l1_loss,
)

OLD = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119/data"
NEW_ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_speed4_20261119"
NEW = NEW_ROOT / "data"
SEQUENCES = (
    "v_9MHDmAMxO5I_c003",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c011",
)
ALPHA = 0.5  # target: size term contributes half of the centre term's gradient


def split_of(sequence: str) -> str:
    summary = json.loads((NEW / sequence / "summary.json").read_text())
    split = summary.get("split")
    if split in ("train", "val"):
        return split
    val_dir = MAIN / "outputs/agentguard/detection_cache/SportsMOT/val"
    return "val" if (val_dir / sequence).is_dir() else "train"


class PredictCapturePilot(SequencePilot):
    """Replay capturing each track's KF mean right after its per-frame predict."""

    def __init__(self, sequence, split, checkpoint, out_dir):
        super().__init__(sequence, split, checkpoint, out_dir)
        self.predicted: dict[tuple[int, int], np.ndarray] = {}
        self.current_frame = 0

    def build_prototypes(self) -> None:
        self.prototypes = {}

    def install_hooks(self) -> None:
        from trackers.track import Track
        from trackers import tracker as tracker_module

        super().install_hooks()
        original_predict = Track.predict
        original_update = tracker_module.Tracker.update
        original_no_det = tracker_module.Tracker.update_without_detections
        pilot = self

        def predict(track_self):
            result = original_predict(track_self)
            if track_self.mean is not None:
                pilot.predicted[
                    (int(pilot.current_frame), int(track_self.track_id))
                ] = np.asarray(track_self.mean, dtype=np.float64).copy()
            return result

        def update(tracker_self, dets, dets_95):
            pilot.current_frame = int(tracker_self.frame_id) + 1
            return original_update(tracker_self, dets, dets_95)

        def update_no_det(tracker_self):
            pilot.current_frame = int(tracker_self.frame_id) + 1
            return original_no_det(tracker_self)

        Track.predict = predict
        tracker_module.Tracker.update = update
        tracker_module.Tracker.update_without_detections = update_no_det
        self._restore = [
            (Track, "predict", original_predict),
            (tracker_module.Tracker, "update", original_update),
            (tracker_module.Tracker, "update_without_detections", original_no_det),
        ] + self._restore


def sequence_list(root: Path) -> set[str]:
    return {path.parent.name for path in root.glob("*/samples.npz")}


def official_trainval() -> set[str]:
    metadata_path = MAIN / "outputs/agentguard/train_data/SportsMOT/metadata.json"
    if not metadata_path.is_file():
        return set()
    metadata = json.loads(metadata_path.read_text())
    val_map = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
    val = {
        line.strip().split()[0]
        for line in val_map.read_text().splitlines()
        if line.strip() and line.strip() != "name"
    }
    return set(metadata["train_sequences"]) | val


def data_alignment() -> dict:
    expected = official_trainval()
    new_list = sequence_list(NEW)
    old_list = sequence_list(OLD)
    reasons: list[str] = []
    if not expected:
        reasons.append("official trainval sequence list is empty or missing")
    missing_new = sorted(expected - new_list)
    extra_new = sorted(new_list - expected)
    missing_old = sorted(expected - old_list)
    extra_old = sorted(old_list - expected)
    if not new_list:
        reasons.append("new data directory has no sequences")
    if missing_new:
        reasons.append(f"new data missing {len(missing_new)} sequences: {missing_new[:5]}")
    if extra_new:
        reasons.append(f"new data has {len(extra_new)} unexpected sequences: {extra_new[:5]}")
    if missing_old:
        reasons.append(f"old data missing {len(missing_old)} sequences: {missing_old[:5]}")
    if extra_old:
        reasons.append(f"old data has {len(extra_old)} unexpected sequences: {extra_old[:5]}")
    if len(new_list) != len(expected):
        reasons.append(f"new sequence count {len(new_list)} != expected {len(expected)}")

    totals = {
        "expected_sequences": len(expected),
        "new_sequences": len(new_list),
        "old_sequences": len(old_list),
        "sequences": 0,
        "samples": 0,
        "row_mismatch": 0,
        "label_mismatch": 0,
        "center_endpoint_mismatch": 0,
        "gt_center_mismatch": 0,
        "mask_mismatch": 0,
        "key_mismatch": 0,
        "size_adj_valid": 0,
        "size_central_valid": 0,
        "size_shared": 0,
        "size_label_abs_sum": [0.0, 0.0],
        "velocity4_mid_error_max": 0.0,
        "box_mid_error_max": 0.0,
        "reasons": reasons,
    }
    required_files = (
        "samples.npz",
        "event_evidence.npy",
        "event_track.npy",
        "event_detection_index.npy",
        "event_has_detection.npy",
        "summary.json",
    )
    for sequence in sorted(expected & new_list & old_list):
        missing_files = [
            name for name in required_files if not (NEW / sequence / name).is_file()
        ]
        if missing_files:
            reasons.append(f"{sequence}: missing shard files {missing_files}")
            continue
        old = np.load(OLD / sequence / "samples.npz")
        new = np.load(NEW / sequence / "samples.npz")
        totals["sequences"] += 1
        totals["samples"] += int(new["frame"].size)
        if old["frame"].size != new["frame"].size:
            reasons.append(
                f"{sequence}: sample count {new['frame'].size} != old {old['frame'].size}"
            )
            continue
        if not np.array_equal(old["keys_json"], new["keys_json"]):
            totals["key_mismatch"] += 1
            reasons.append(f"{sequence}: event keys differ")
        same_rows = all(
            np.array_equal(old[key], new[key])
            for key in ("frame", "track_id", "detection_index", "gt_id", "window_indices")
        )
        totals["row_mismatch"] += 0 if same_rows else 1
        label_keys = (
            "motion_benefit",
            "appearance_benefit",
            "valid_motion",
            "valid_appearance",
            "gt_coverage",
            "oracle_detection_coverage",
            "motion_gt_valid",
            "pred_box_g0",
            "pred_box_g1",
            "gt_box_t1",
            "appearance_delta",
            "appearance_diag_valid",
            "appearance_score_template0",
            "appearance_score_template1",
            "motion_gate",
            "appearance_gate",
        )
        same_labels = all(np.array_equal(old[key], new[key]) for key in label_keys)
        totals["label_mismatch"] += 0 if same_labels else 1
        same_center = (
            np.array_equal(old["kf_velocity_g0"], new["kf_velocity4_g0"][:, :2])
            and np.array_equal(old["kf_velocity_g1"], new["kf_velocity4_g1"][:, :2])
        )
        same_gt = np.array_equal(old["gt_velocity"], new["gt_velocity4"][:, :2])
        same_mask = np.array_equal(old["speed_valid"], new["size_adj_valid"])
        totals["center_endpoint_mismatch"] += 0 if same_center else 1
        totals["gt_center_mismatch"] += 0 if same_gt else 1
        totals["mask_mismatch"] += 0 if same_mask else 1
        mismatched = [
            name
            for name, failed in (
                ("rows", not same_rows),
                ("labels", not same_labels),
                ("centre endpoints", not same_center),
                ("GT centres", not same_gt),
                ("masks", not same_mask),
            )
            if failed
        ]
        if mismatched:
            reasons.append(
                f"{sequence}: "
                + ", ".join(mismatched)
                + " differ from the previous round"
            )
        adj = new["size_adj_valid"].astype(bool)
        central = new["size_central_valid"].astype(bool)
        shared = adj & central
        totals["size_adj_valid"] += int(adj.sum())
        totals["size_central_valid"] += int(central.sum())
        totals["size_shared"] += int(shared.sum())
        if shared.any():
            difference = np.abs(
                new["gt_velocity4"][shared][:, 2:]
                - new["gt_size_velocity_central"][shared]
            )
            totals["size_label_abs_sum"][0] += float(difference[:, 0].sum())
            totals["size_label_abs_sum"][1] += float(difference[:, 1].sum())
        summary = json.loads((NEW / sequence / "summary.json").read_text())
        if summary.get("kf_velocity4_mid_error_max") is not None:
            totals["velocity4_mid_error_max"] = max(
                totals["velocity4_mid_error_max"], summary["kf_velocity4_mid_error_max"]
            )
        if summary.get("kf_box_mid_error_max") is not None:
            totals["box_mid_error_max"] = max(
                totals["box_mid_error_max"], summary["kf_box_mid_error_max"]
            )
    if totals["size_shared"]:
        totals["size_label_mean_abs_diff"] = [
            value / totals["size_shared"] for value in totals["size_label_abs_sum"]
        ]
    if totals["samples"] == 0:
        reasons.append("validated shard set contains zero samples")
    if len(new_list) != len(expected):
        reasons.append(
            f"validated {len(new_list)} new sequences, expected {len(expected)}"
        )
    totals["data_aligned"] = not reasons and totals["sequences"] == len(expected)
    if not totals["data_aligned"] and not reasons:
        reasons.append("sequence validation incomplete")
    return totals


def online_spot_check(checkpoint: Path) -> dict:
    report = {}
    for sequence in SEQUENCES:
        split = split_of(sequence)
        pilot = PredictCapturePilot(
            sequence, split, checkpoint, NEW_ROOT / "logs" / "speed4_spot"
        )
        pilot.run_replay()
        samples = np.load(NEW / sequence / "samples.npz")
        frame = samples["frame"]
        track = samples["track_id"]
        gate = samples["motion_gate"].astype(np.float64)
        v0 = samples["kf_velocity4_g0"].astype(np.float64)
        v1 = samples["kf_velocity4_g1"].astype(np.float64)
        box0 = samples["pred_box_g0"].astype(np.float64)
        box1 = samples["pred_box_g1"].astype(np.float64)
        scale = np.array(
            [
                float(pilot.img_w),
                float(pilot.img_h),
                float(pilot.img_w),
                float(pilot.img_h),
            ]
        )
        velocity_error = 0.0
        box_error = 0.0
        compared = 0
        gmc_count = 0
        for index in range(frame.size):
            key = (int(frame[index]) + 1, int(track[index]))
            mean = pilot.predicted.get(key)
            if mean is None:
                continue
            g = 1.0 if gate[index] < 0 else gate[index]
            predicted_velocity = mean[4:8] / scale
            expected_velocity = v0[index] + g * (v1[index] - v0[index])
            velocity_error = max(
                velocity_error,
                float(np.max(np.abs(predicted_velocity - expected_velocity))),
            )
            predicted_box = np.array(
                [
                    mean[0] - mean[2] / 2,
                    mean[1] - mean[3] / 2,
                    mean[0] + mean[2] / 2,
                    mean[1] + mean[3] / 2,
                ]
            )
            expected_box = box0[index] + g * (box1[index] - box0[index])
            box_error = max(box_error, float(np.max(np.abs(predicted_box - expected_box))))
            warp = pilot.warps.get(int(frame[index]) + 1)
            if warp is not None and (
                np.max(np.abs(np.asarray(warp)[:, :2] - np.eye(2))) > 1e-9
                or np.max(np.abs(np.asarray(warp)[:, 2])) > 1e-9
            ):
                gmc_count += 1
            compared += 1
        pilot.close()
        report[sequence] = {
            "compared_events": compared,
            "non_identity_gmc_events": gmc_count,
            "velocity4_max_abs_error": velocity_error,
            "box_max_abs_error": box_error,
        }
    return report


def grad_norm(model, inputs, loss_fn):
    model.zero_grad(set_to_none=True)
    outputs = model(**inputs)
    loss, _ = loss_fn(outputs)
    loss.backward()
    total = 0.0
    grads = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm() ** 2)
            grads.append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return math.sqrt(total), float(loss.detach()), grads


def batch_checks(checkpoint: Path) -> dict:
    torch.set_num_threads(4)
    model, _ = load_shutrack_checkpoint(INIT_CKPT, map_location="cpu")
    model.eval()
    taus = load_global_taus(list(SEQUENCES), NEW)
    totals = {
        "rows": 0,
        "box_grad_abs_sum": 0.0,
        "center_grad_abs_sum": 0.0,
        "size_grad_abs_sum": 0.0,
        "box_param_norm": 0.0,
        "center_param_norm": 0.0,
        "size_param_norm": 0.0,
        "center_at_g0_sum": 0.0,
        "center_at_g1_sum": 0.0,
        "size_at_g0_sum": 0.0,
        "size_at_g1_sum": 0.0,
        "size_target_abs_sum": 0.0,
        "size_delta_abs_sum": 0.0,
        "zero_equiv_loss_max": 0.0,
        "zero_equiv_grad_max": 0.0,
        "zero_equiv_param_max": 0.0,
        "sequences": {},
    }
    for sequence in SEQUENCES:
        split = split_of(sequence)
        data = SequenceData(sequence, split, NEW)
        data.open_cache()
        targets = derive_targets(data.samples, taus)
        rows = np.arange(min(data.n, 4096), dtype=np.int64)
        batch = to_device(
            batch_tensors(data, rows, targets, "motion_diou_speed4"), torch.device("cpu")
        )
        inputs = model_inputs(batch)
        # Two-dimensional views for the centre-only loss equivalence check;
        # these are bit-identical to the separate 2-D arrays in the shards.
        batch["kf_velocity_g0"] = batch["kf_velocity4_g0"][:, :2]
        batch["kf_velocity_g1"] = batch["kf_velocity4_g1"][:, :2]
        batch["gt_velocity"] = batch["gt_velocity4"][:, :2]
        batch["speed_valid"] = batch["size_adj_valid"]
        logits = model(**inputs)["gate_logits"].clone().requires_grad_(True)
        box_loss, _ = motion_box_diou_loss({"gate_logits": logits}, batch)
        center_loss, center_components = motion_speed4_l1_loss(
            {"gate_logits": logits}, batch, center_weight=1.0, size_weight=0.0
        )
        size_loss, size_components = motion_speed4_l1_loss(
            {"gate_logits": logits}, batch, center_weight=0.0, size_weight=1.0
        )
        box_grad = torch.autograd.grad(box_loss, logits, retain_graph=True)[0][:, 0]
        center_grad = torch.autograd.grad(center_loss, logits, retain_graph=True)[0][:, 0]
        size_grad = torch.autograd.grad(size_loss, logits)[0][:, 0]
        valid = batch["size_adj_valid"]

        def masked_mean(values):
            return float(
                (values * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            )

        totals["box_grad_abs_sum"] += masked_mean(box_grad.abs()) * int(valid.sum())
        totals["center_grad_abs_sum"] += masked_mean(center_grad.abs()) * int(valid.sum())
        totals["size_grad_abs_sum"] += masked_mean(size_grad.abs()) * int(valid.sum())
        totals["center_at_g0_sum"] += float(center_components["center_at_g0"]) * int(valid.sum())
        totals["center_at_g1_sum"] += float(center_components["center_at_g1"]) * int(valid.sum())
        totals["size_at_g0_sum"] += float(size_components["size_at_g0"]) * int(valid.sum())
        totals["size_at_g1_sum"] += float(size_components["size_at_g1"]) * int(valid.sum())
        target = batch["gt_velocity4"][:, 2:]
        delta = (batch["kf_velocity4_g1"] - batch["kf_velocity4_g0"])[:, 2:]
        totals["size_target_abs_sum"] += masked_mean(target.abs().sum(dim=-1)) * int(valid.sum())
        totals["size_delta_abs_sum"] += masked_mean(delta.abs().sum(dim=-1)) * int(valid.sum())
        totals["rows"] += int(valid.sum())

        # Coefficient-0 equivalence against the existing centre-speed loss.
        logits_a = model(**inputs)["gate_logits"].clone().requires_grad_(True)
        logits_b = model(**inputs)["gate_logits"].clone().requires_grad_(True)
        loss_a, _ = motion_speed_l1_loss({"gate_logits": logits_a}, batch, weight=50.0)
        loss_b, _ = motion_speed4_l1_loss(
            {"gate_logits": logits_b}, batch, center_weight=50.0, size_weight=0.0
        )
        grad_a = torch.autograd.grad(loss_a, logits_a)[0]
        grad_b = torch.autograd.grad(loss_b, logits_b)[0]
        totals["zero_equiv_loss_max"] = max(
            totals["zero_equiv_loss_max"], float((loss_a - loss_b).abs())
        )
        totals["zero_equiv_grad_max"] = max(
            totals["zero_equiv_grad_max"], float((grad_a - grad_b).abs().max())
        )
        norm_a, _, grads_a = grad_norm(
            model, inputs, lambda out: motion_speed_l1_loss(out, batch, weight=50.0)
        )
        norm_b, _, grads_b = grad_norm(
            model,
            inputs,
            lambda out: motion_speed4_l1_loss(
                out, batch, center_weight=50.0, size_weight=0.0
            ),
        )
        param_max = 0.0
        for first, second in zip(grads_a, grads_b):
            param_max = max(param_max, float((first - second).abs().max()))
        totals["zero_equiv_param_max"] = max(totals["zero_equiv_param_max"], param_max)

        box_norm, _, _ = grad_norm(
            model, inputs, lambda out: motion_box_diou_loss(out, batch)
        )
        center_norm, _, _ = grad_norm(
            model,
            inputs,
            lambda out: motion_speed4_l1_loss(
                out, batch, center_weight=1.0, size_weight=0.0
            ),
        )
        size_norm, _, _ = grad_norm(
            model,
            inputs,
            lambda out: motion_speed4_l1_loss(
                out, batch, center_weight=0.0, size_weight=1.0
            ),
        )
        totals["box_param_norm"] += box_norm * int(valid.sum())
        totals["center_param_norm"] += center_norm * int(valid.sum())
        totals["size_param_norm"] += size_norm * int(valid.sum())
        totals["sequences"][sequence] = {
            "rows": int(rows.size),
            "valid": int(valid.sum()),
            "box_grad_abs_mean": masked_mean(box_grad.abs()),
            "center_grad_abs_mean_unit": masked_mean(center_grad.abs()),
            "size_grad_abs_mean_unit": masked_mean(size_grad.abs()),
            "box_param_norm": box_norm,
            "center_param_norm_unit": center_norm,
            "size_param_norm_unit": size_norm,
            "center_at_g0": float(center_components["center_at_g0"]),
            "center_at_g1": float(center_components["center_at_g1"]),
            "size_at_g0": float(size_components["size_at_g0"]),
            "size_at_g1": float(size_components["size_at_g1"]),
            "size_target_abs_mean": masked_mean(target.abs().sum(dim=-1)),
            "size_delta_abs_mean": masked_mean(delta.abs().sum(dim=-1)),
            "zero_equiv_param_max": param_max,
            "zero_equiv_norm_a": norm_a,
            "zero_equiv_norm_b": norm_b,
        }
        data.close_cache()
    rows = max(totals["rows"], 1)
    totals["box_grad_abs_mean"] = totals["box_grad_abs_sum"] / rows
    totals["center_grad_abs_mean_unit"] = totals["center_grad_abs_sum"] / rows
    totals["size_grad_abs_mean_unit"] = totals["size_grad_abs_sum"] / rows
    totals["center_at_g0"] = totals["center_at_g0_sum"] / rows
    totals["center_at_g1"] = totals["center_at_g1_sum"] / rows
    totals["size_at_g0"] = totals["size_at_g0_sum"] / rows
    totals["size_at_g1"] = totals["size_at_g1_sum"] / rows
    totals["size_target_abs_mean"] = totals["size_target_abs_sum"] / rows
    totals["size_delta_abs_mean"] = totals["size_delta_abs_sum"] / rows
    totals["box_param_norm_mean"] = totals["box_param_norm"] / rows
    totals["center_param_norm_unit_mean"] = totals["center_param_norm"] / rows
    totals["size_param_norm_unit_mean"] = totals["size_param_norm"] / rows
    center_operating_grad = 50.0 * totals["center_grad_abs_mean_unit"]
    totals["center_grad_abs_mean_at_weight"] = center_operating_grad
    totals["suggested_size_weight_alpha50_gate"] = (
        ALPHA * center_operating_grad / max(totals["size_grad_abs_mean_unit"], 1e-12)
    )
    center_operating_norm = 50.0 * totals["center_param_norm_unit_mean"]
    totals["suggested_size_weight_alpha50_param"] = (
        ALPHA * center_operating_norm / max(totals["size_param_norm_unit_mean"], 1e-12)
    )
    totals["zero_equivalence_exact"] = (
        totals["zero_equiv_loss_max"] == 0.0
        and totals["zero_equiv_grad_max"] == 0.0
        and totals["zero_equiv_param_max"] == 0.0
    )
    return totals


def main() -> None:
    torch.set_num_threads(4)
    data = data_alignment()
    out = NEW_ROOT / "speed4_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not data["data_aligned"]:
        out.write_text(json.dumps({"data": data}, indent=2, sort_keys=True) + "\n")
        print(json.dumps(data, indent=2, sort_keys=True))
        raise SystemExit(
            "data alignment failed: " + "; ".join(data["reasons"])
        )
    checkpoint = Path(INIT_CKPT)
    report = {
        "data": data,
        "online_spot": online_spot_check(checkpoint),
        "batch": batch_checks(checkpoint),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("saved", out)


if __name__ == "__main__":
    main()
