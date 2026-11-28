#!/usr/bin/env python3
"""21-level per-match gate oracle diagnostics (raw val, inference only).

For each dataset this runs up to three complete raw-val groups with the same
checkpoint: ``baseline`` (the normal dual-gate model), ``motion`` (21-level KF
write-back oracle) and ``appearance`` (21-level EMA write-back oracle).

Both oracles evaluate every matched, mature-track event on 21 candidate gates
0.00, 0.05, ..., 1.00 against the same pre-update state.  Candidate evaluation
is pure (state copies only); exactly one chosen gate is executed through the
original ``Track.update_with_gates`` path, and the other gate keeps the
model's own value.  A route falls back to the learned gate when the track's
GT identity is not reliably bound or the reference is missing, and counts the
reason.

Motion rule   : minimise ``1 - DIoU(predicted box at t+1, GT box of the
                track's bound identity at t+1)`` using the tracker's GMC warp
                and KF conventions.
Appearance rule: build 21 EMA write-back candidate templates with the exact
                online update/normalisation rule, then score each template on
                the first frame in t+1..t+3 that has both a same-identity and a
                different-identity detection (best same minus best other
                cosine similarity).

Ties (within 1e-6 of the optimum) choose the level closest to the learned
gate.  The tracking entry point (``3. Tracker/run.py``) and the training entry
point are untouched; this tool installs its hooks in-process only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Tiny per-event matvecs otherwise pay OpenBLAS thread-team overhead
# (~1-4 ms per call with the default 64-thread pool on this host).  Torch
# keeps its own thread count from run.py's --torch-num-threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (
    str(ROOT / "tools"),
    str(ROOT),
    str(ROOT / "agentguard/src"),
    str(ROOT / "3. Tracker"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PY = ROOT / ".venv/bin/python"
TRACKER = ROOT / "3. Tracker"
CACHE_ROOT = ROOT / "outputs/agentguard/detection_cache"
DEFAULT_OUT_ROOT = ROOT / "outputs/agentguard/experiments/gate_oracle_21level_20261126"

ROUTES = ("baseline", "motion", "appearance")
GRID = np.linspace(1.0, 0.0, 21, dtype=np.float64)
TIE_TOLERANCE = 1e-6
GT_BIND_IOU = 0.5


def _sportsmot_val_sequences() -> list[str]:
    path = TRACKER / "trackeval/seqmap/sportsmot/val.txt"
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and line.strip() != "name"]


DATASETS = {
    "SportsMOT": {
        "mode": "val",
        "gt_split": "val",
        "sequences": _sportsmot_val_sequences,
        "checkpoint": (
            ROOT
            / "outputs/agentguard/experiments/mot_sportsmot_serial_20261124"
            / "harm4/groups/appearance_row_column/checkpoints/epoch_100.pt"
        ),
        "note": (
            "Held-out checkpoint: official train45 (val45 excluded), harm=4, "
            "dropout=0.05, e100 -- the closest held-out recipe to the current "
            "best trainval90 harm=4 e100 (official 76.320, #1068003); the "
            "trainval90 checkpoint itself saw val45 during training and is "
            "therefore not usable for this diagnostic."
        ),
    },
    "MOT20": {
        "mode": "all",
        "gt_split": "all",
        "sequences": ["MOT20-01", "MOT20-05"],
        "checkpoint": (
            ROOT / "outputs/agentguard/keep/MOT20_66.637_1063086/weights/epoch_100.pt"
        ),
        "note": (
            "Current best submission recipe (sqrt-size, harm=1, dropout=0, "
            "e100, official 66.637, #1063086). No dual-gate checkpoint held out "
            "MOT20-01/05 exists on this mainline; both validation sequences are "
            "inside the training split, so val numbers are optimistic."
        ),
    },
    "MOT17": {
        "mode": "all",
        "gt_split": "all",
        "sequences": ["MOT17-09-FRCNN", "MOT17-10-FRCNN"],
        "checkpoint": (
            ROOT / "outputs/agentguard/keep/MOT17_68.023_1068665/weights/epoch_050.pt"
        ),
        "note": (
            "Current best submission recipe (init = round-1 MOT20 harm2 e100, "
            "harm=2, dropout=0, 50 fine-tune epochs, official 68.023, "
            "#1068665). No dual-gate checkpoint held out MOT17-09/10 exists on "
            "this mainline; both validation sequences are inside the training "
            "split, so val numbers are optimistic."
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_config(name: str) -> dict:
    cfg = dict(DATASETS[name])
    sequences = cfg["sequences"]
    cfg["sequences"] = list(sequences() if callable(sequences) else sequences)
    cfg["checkpoint"] = Path(cfg["checkpoint"])
    return cfg


def tracker_argv(dataset: str, sequences, checkpoint: Path, out: Path,
                 suffix: str, max_frames: int) -> list[str]:
    cfg = DATASETS[dataset]
    argv = [
        "--dataset", dataset,
        "--mode", cfg["mode"],
        "--sequences", *sequences,
        "--seed", "10000",
        "--kf-type", "nsa",
        "--agentguard-mode", "shutrack-v2",
        "--shutrack-checkpoint", str(checkpoint),
        "--agentguard-device", "cpu",
        "--detection-cache-root", str(CACHE_ROOT),
        "--output_dir", str(out),
        "--tracker-suffix", suffix,
        "--agentguard-stats-output", str(out / "gates.jsonl"),
        "--print-per-sequence-metrics",
        "--eval-json", str(out / "summary.json"),
        "--torch-num-threads", "4",
        "--torch-num-interop-threads", "1",
    ]
    if max_frames > 0:
        argv += ["--max-frames", str(max_frames)]
    return argv


def _base_manifest(dataset: str, route: str, checkpoint: Path, sequences,
                   max_frames: int) -> dict:
    return {
        "dataset": dataset,
        "route": route,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sequences": list(sequences),
        "max_frames": int(max_frames),
        "grid": [float(g) for g in GRID],
        "tie_tolerance": TIE_TOLERANCE,
        "gt_bind_iou": GT_BIND_IOU,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_note": DATASETS[dataset]["note"],
        "started_at": time.strftime("%F %T"),
    }


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------
def run_baseline(dataset: str, out: Path, sequences, checkpoint: Path,
                 max_frames: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    argv = tracker_argv(dataset, sequences, checkpoint, out, "baseline", max_frames)
    manifest = _base_manifest(dataset, "baseline", checkpoint, sequences, max_frames)
    (out / "command.sh").write_text(
        "cd " + str(TRACKER) + "\n"
        + " ".join([str(PY), "-u", "run.py", *argv]) + "\n"
    )
    log = out / "run.log"
    started = time.monotonic()
    with log.open("wb") as handle:
        code = subprocess.call(
            [str(PY), "-u", "run.py", *argv], cwd=str(TRACKER),
            stdout=handle, stderr=subprocess.STDOUT,
        )
    manifest.update({
        "seconds": round(time.monotonic() - started, 2),
        "exit_code": int(code),
        "finished_at": time.strftime("%F %T"),
    })
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    if code != 0:
        raise SystemExit(f"baseline {dataset} failed with exit code {code}; see {log}")
    return manifest


# ----------------------------------------------------------------------
# Oracle
# ----------------------------------------------------------------------
def run_oracle(dataset: str, route: str, out: Path, sequences, checkpoint: Path,
               max_frames: int, event_log: bool) -> dict:
    import run as run_module  # noqa: PLC0415  (3. Tracker/run.py)
    from trackers import tracker as tracker_module  # noqa: PLC0415
    from trackers.track import Track  # noqa: PLC0415
    from trackers.kalman_filter import create_kalman_filter  # noqa: PLC0415
    from agentguard.data.detection_cache import SequenceDetectionCache  # noqa: PLC0415
    from sportsmot_supervision_pilot import (  # noqa: PLC0415
        SequencePilot, mean_to_box, warp_predict,
    )
    from shutrack.training.motion_targets import numpy_diou  # noqa: PLC0415

    cfg = DATASETS[dataset]
    out.mkdir(parents=True, exist_ok=True)
    argv = tracker_argv(dataset, sequences, checkpoint, out, f"oracle_{route}", max_frames)
    args = run_module.make_parser().parse_args(argv)
    run_module.args = args
    (out / "command.sh").write_text(
        "cd " + str(TRACKER) + "\n"
        + " ".join([str(PY), "-u", str(Path(__file__).resolve()),
                    "run", "--dataset", dataset, "--route", route,
                    "--out-root", str(out.parents[1]),
                    *(["--max-frames", str(max_frames)] if max_frames > 0 else []),
                    *(["--event-log"] if event_log else [])]) + "\n"
    )

    state = {
        "pilot": None,
        "seq": None,
        "seq_started": 0.0,
        "stats": {},
        "event_file": None,
        "event_log": bool(event_log),
        "max_state_err": 0.0,
        "max_feature_err": 0.0,
        "max_side_effect": 0.0,
        "endpoint": {
            "0.00": {"events": 0, "max_state_err": 0.0, "max_feature_err": 0.0},
            "1.00": {"events": 0, "max_state_err": 0.0, "max_feature_err": 0.0},
        },
        "feature_frame_cache": OrderedDict(),
        "tim": {},
    }
    if event_log:
        state["event_file"] = (out / "events.jsonl").open("w")

    def _mark(tim, name, started):
        now = time.perf_counter()
        tim[name] = tim.get(name, 0.0) + (now - started)
        return now

    def bucket(seq):
        return state["stats"].setdefault(seq, {
            "match_events": 0,
            "gated_events": 0,
            "ungated_events": 0,
            "searchable": 0,
            "reasons": {},
            "chosen_hist": {},
            "learned_sum": 0.0,
            "learned_count": 0,
            "target_bound_events": 0,
            "target_unbound_events": 0,
            "det_identity_mismatch_events": 0,
            "state_err_max": 0.0,
            "feature_err_max": 0.0,
            "side_effect_max": 0.0,
            "seconds": 0.0,
        })

    def frame_features(pilot, frame_index: int, entries):
        """One contiguous float64 feature block per detection-cache frame.

        The MOT20-05 cache is 7.4 GB; per-row random mmap reads thrashed the
        page cache, so each reference frame is materialised once (rows are
        frame-contiguous in ``features.npy``) and kept in a small LRU."""
        cache = state["feature_frame_cache"]
        cached = cache.get(frame_index)
        if cached is not None:
            cache.move_to_end(frame_index)
            return cached
        indices = np.asarray([entry["index"] for entry in entries], dtype=np.int64)
        block = np.asarray(pilot.det_cache.features[indices], dtype=np.float64)
        row_of = {int(index): row for row, index in enumerate(indices)}
        cache[frame_index] = (row_of, block)
        if len(cache) > 32:
            cache.popitem(last=False)
        return row_of, block

    def close_pilot():
        pilot = state.get("pilot")
        if pilot is None:
            return
        if state.get("seq"):
            elapsed = time.monotonic() - state["seq_started"]
            bucket(state["seq"])["seconds"] += elapsed
            print(f"[{dataset}/{route}] {state['seq']}: {elapsed:.1f}s "
                  f"events={bucket(state['seq'])['match_events']} "
                  f"searchable={bucket(state['seq'])['searchable']}", flush=True)
        if getattr(pilot, "det_cache", None) is not None:
            try:
                pilot.det_cache.close()
            except Exception:  # noqa: BLE001
                pass
        state["pilot"] = None
        state["seq"] = None
        state["feature_frame_cache"].clear()

    def make_pilot(sequence: str):
        split = cfg["gt_split"]
        pilot = SequencePilot(
            sequence, split, checkpoint, out / "pilot_scratch" / sequence,
            dataset=dataset,
        )
        pilot.lean = False
        pilot.store_rank_features = False
        pilot.img_w = int(run_module.args.img_w)
        pilot.img_h = int(run_module.args.img_h)
        pilot.det_thr = float(getattr(run_module.args, "det_thr", 0.55))
        pilot.kf = create_kalman_filter("nsa")
        pilot.det_cache = SequenceDetectionCache(
            run_module._sequence_detection_cache_dir(run_module.args, sequence),
            frame_array_cache_size=4,
        )
        pilot.frame_matches_cache = {}
        return pilot

    def prepare_event(track, frame_id, detection, motion_gate, appearance_gate, gated):
        pilot = state["pilot"]
        stats = bucket(pilot.sequence)
        stats["match_events"] += 1
        stats["gated_events" if gated else "ungated_events"] += 1
        info = {
            "sequence": pilot.sequence,
            "frame": int(frame_id),
            "track_id": int(track.track_id),
            "gated": bool(gated),
            "learned_motion_gate": None if motion_gate is None else float(motion_gate),
            "learned_appearance_gate": None if appearance_gate is None else float(appearance_gate),
        }
        if motion_gate is not None:
            stats["learned_sum"] += float(motion_gate)
            stats["learned_count"] += 1
        pilot.frame_matches(frame_id - 1)
        det_idx = int(getattr(detection, "frame_detection_index", -1))
        det_gt_id, det_iou = pilot.det_gt.get((frame_id, det_idx), (-1, 0.0))
        target = pilot.vote.resolve_before_current(pilot.sequence, track.track_id)
        pilot.vote.add_current_observation(
            pilot.sequence, track.track_id, int(det_gt_id)
        )
        info.update({
            "detection_index": det_idx,
            "detection_gt_id": int(det_gt_id),
            "detection_gt_iou": float(det_iou),
            "target_gt_id": None if target is None else int(target),
        })
        if target is None:
            stats["target_unbound_events"] += 1
        else:
            stats["target_bound_events"] += 1
            if det_gt_id >= 0 and int(det_gt_id) != int(target):
                stats["det_identity_mismatch_events"] += 1
        return info

    def choose_motion(track, frame_id, detection, learned, pilot, info):
        stats = bucket(pilot.sequence)
        if track.mean is None or track.covariance is None:
            return learned, {"reason": "no_state"}
        target = info["target_gt_id"]
        if target is None:
            return learned, {"reason": "no_target"}
        warp_next = pilot.warps.get(frame_id + 1)
        if warp_next is None:
            return learned, {"reason": "no_warp"}
        gt_t1 = pilot.gt.get(frame_id + 1, {}).get(int(target))
        if gt_t1 is None:
            return learned, {"reason": "no_gt_t1"}
        prior = np.array(track.mean, dtype=np.float64, copy=True)
        cov_prior = np.array(track.covariance, dtype=np.float64, copy=True)
        measurement = np.asarray(detection.cxcywh, dtype=np.float64)
        mean_full, cov_full = pilot.kf.update(
            prior, cov_prior, measurement, float(detection.score)
        )
        losses = np.empty(GRID.size, dtype=np.float64)
        for index, gate in enumerate(GRID):
            mean_g = prior + float(gate) * (mean_full - prior)
            cov_g = (1.0 - float(gate)) * cov_prior + float(gate) * cov_full
            pred_mean, _ = warp_predict(mean_g, cov_g, warp_next, pilot.kf)
            box = mean_to_box(pred_mean)
            losses[index] = 1.0 - float(
                numpy_diou(box[None], np.asarray(gt_t1, dtype=np.float64)[None])[0]
            )
        best = float(losses.min())
        eligible = np.flatnonzero(losses <= best + TIE_TOLERANCE)
        index = int(eligible[np.argmin(np.abs(GRID[eligible] - float(learned)))])
        chosen = float(GRID[index])
        if chosen == 1.0:
            expected_mean = mean_full
        elif chosen == 0.0:
            expected_mean = prior
        else:
            expected_mean = prior + chosen * (mean_full - prior)
        return chosen, {
            "expected_mean": expected_mean,
            "losses": losses,
            "min_loss": best,
            "loss_full": float(losses[0]),
            "loss_zero": float(losses[-1]),
            "chosen_index": index,
        }

    def choose_appearance(track, frame_id, detection, learned, pilot, info):
        tim = state["tim"]
        _t = time.perf_counter()
        target = info["target_gt_id"]
        if target is None:
            return learned, {"reason": "no_target"}
        old = np.asarray(track.feat, dtype=np.float32).reshape(-1).copy()
        det_feat = np.asarray(detection.feat, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(det_feat)) <= 0.0:
            return learned, {"reason": "no_detection_feature"}
        alpha = track.alpha
        beta = alpha + (1 - alpha) * (1 - detection.score)
        # The online rule blends the *unnormalised* full EMA vector for
        # intermediate gates and only renormalises afterwards (mirrored by
        # shutrack.training.loss.gated_appearance_template).
        raw_full = beta * old + (1.0 - beta) * det_feat
        _t = _mark(tim, "setup", _t)
        reference = None
        for step in (1, 2, 3):
            frame_next = frame_id + step
            if frame_next > pilot.det_cache.num_frames:
                break
            entries = pilot.frame_matches(frame_next - 1)
            same = [entry for entry in entries
                    if entry["gt_id"] == int(target) and entry["iou"] >= GT_BIND_IOU]
            other = [entry for entry in entries
                     if entry["gt_id"] >= 0 and entry["gt_id"] != int(target)
                     and entry["iou"] >= GT_BIND_IOU]
            if same and other:
                reference = (frame_next, frame_next - 1, entries, same, other)
                break
        _t = _mark(tim, "reference", _t)
        if reference is None:
            return learned, {"reason": "no_appearance_pair"}
        frame_next, frame_index, entries, same, other = reference
        row_of, block = frame_features(pilot, frame_index, entries)
        same_rows = np.asarray([row_of[int(entry["index"])] for entry in same], dtype=np.int64)
        other_rows = np.asarray([row_of[int(entry["index"])] for entry in other], dtype=np.int64)
        _t = _mark(tim, "rows", _t)
        old64 = old.astype(np.float64)
        raw64 = raw_full.astype(np.float64)
        # Candidate templates are normalised blends of ``old`` and ``raw_full``;
        # candidate_g . det = ((1-g) * old.det + g * raw.det) / ||blend_g||, so
        # two dot products per candidate detection replace 21 x N per event.
        a = block @ old64
        b = block @ raw64
        _t = _mark(tim, "dot", _t)
        norms = np.empty(GRID.size, dtype=np.float64)
        for index, gate in enumerate(GRID):
            gate = float(gate)
            if gate == 1.0:
                norms[index] = np.linalg.norm(raw64)
            elif gate == 0.0:
                norms[index] = 1.0
            else:
                norms[index] = np.linalg.norm((1.0 - gate) * old64 + gate * raw64) + 1e-12
        _t = _mark(tim, "norms", _t)
        scores = np.empty(GRID.size, dtype=np.float64)
        for index, gate in enumerate(GRID):
            gate = float(gate)
            sim_same = ((1.0 - gate) * a[same_rows] + gate * b[same_rows]).max()
            sim_other = ((1.0 - gate) * a[other_rows] + gate * b[other_rows]).max()
            scores[index] = (sim_same - sim_other) / norms[index]
        _t = _mark(tim, "scores", _t)
        best = float(scores.max())
        eligible = np.flatnonzero(scores >= best - TIE_TOLERANCE)
        index = int(eligible[np.argmin(np.abs(GRID[eligible] - float(learned)))])
        chosen = float(GRID[index])
        gate = float(GRID[index])
        if gate == 1.0:
            expected = raw_full / np.linalg.norm(raw_full)
        elif gate == 0.0:
            expected = old.copy()
        else:
            value = (1.0 - gate) * old + gate * raw_full
            expected = value / (np.linalg.norm(value) + 1e-12)
        _mark(tim, "tail", _t)
        return chosen, {
            "expected_feat": np.asarray(expected, dtype=np.float32),
            "scores": scores,
            "reference_frame": int(frame_next),
            "n_same": len(same),
            "n_other": len(other),
            "best_separation": best,
            "chosen_index": index,
        }

    def record(info, detail, chosen):
        pilot = state["pilot"]
        stats = bucket(pilot.sequence)
        reason = detail.get("reason")
        if reason is not None:
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            info.update({"searchable": False, "reason": reason, "chosen": float(chosen)})
        else:
            stats["searchable"] += 1
            key = f"{float(chosen):.2f}"
            stats["chosen_hist"][key] = stats["chosen_hist"].get(key, 0) + 1
            info.update({"searchable": True, "chosen": float(chosen)})
            if route == "motion":
                info.update({
                    "min_loss": detail["min_loss"],
                    "loss_full": detail["loss_full"],
                    "loss_zero": detail["loss_zero"],
                })
                if state["event_log"]:
                    info["losses"] = detail["losses"].tolist()
            else:
                info.update({
                    "reference_frame": detail["reference_frame"],
                    "n_same": detail["n_same"],
                    "n_other": detail["n_other"],
                    "best_separation": detail["best_separation"],
                })
                if state["event_log"]:
                    info["scores"] = detail["scores"].tolist()
        if state["event_file"] is not None:
            state["event_file"].write(json.dumps(info, ensure_ascii=False) + "\n")
            state["event_file"].flush()

    def verify_execution(track, chosen, detail, info):
        stats = bucket(state["pilot"].sequence)
        endpoint_key = f"{float(chosen):.2f}" if float(chosen) in (0.0, 1.0) else None
        if route == "motion" and "expected_mean" in detail:
            error = float(np.max(np.abs(
                np.asarray(track.mean, dtype=np.float64) - detail["expected_mean"]
            )))
            stats["state_err_max"] = max(stats["state_err_max"], error)
            state["max_state_err"] = max(state["max_state_err"], error)
            info["state_err"] = error
            if endpoint_key is not None:
                entry = state["endpoint"][endpoint_key]
                entry["events"] += 1
                entry["max_state_err"] = max(entry["max_state_err"], error)
        if route == "appearance" and "expected_feat" in detail:
            error = float(np.max(np.abs(
                np.asarray(track.feat, dtype=np.float32).reshape(-1)
                - detail["expected_feat"]
            )))
            stats["feature_err_max"] = max(stats["feature_err_max"], error)
            state["max_feature_err"] = max(state["max_feature_err"], error)
            info["feature_err"] = error
            if endpoint_key is not None:
                entry = state["endpoint"][endpoint_key]
                entry["events"] += 1
                entry["max_feature_err"] = max(entry["max_feature_err"], error)

    original_init = tracker_module.Tracker.__init__
    original_update_with_gates = Track.update_with_gates
    original_update = Track.update

    def patched_init(self, args, vid_name):  # noqa: ANN001
        close_pilot()
        state["pilot"] = make_pilot(vid_name)
        state["seq"] = vid_name
        state["seq_started"] = time.monotonic()
        return original_init(self, args, vid_name)

    def patched_update_with_gates(self, frame_id, detection, motion_gate, appearance_gate):
        pilot = state["pilot"]
        info = prepare_event(self, frame_id, detection, motion_gate, appearance_gate, True)
        pre_mean = np.array(self.mean, dtype=np.float64, copy=True) if self.mean is not None else None
        pre_feat = np.array(self.feat, dtype=np.float32, copy=True)
        if route == "motion":
            chosen, detail = choose_motion(
                self, frame_id, detection, float(motion_gate), pilot, info
            )
        else:
            chosen, detail = choose_appearance(
                self, frame_id, detection, float(appearance_gate), pilot, info
            )
        if pre_mean is not None:
            side = float(np.max(np.abs(np.asarray(self.mean, dtype=np.float64) - pre_mean)))
            side = max(side, float(np.max(np.abs(
                np.asarray(self.feat, dtype=np.float32) - pre_feat
            ))))
            stats = bucket(pilot.sequence)
            stats["side_effect_max"] = max(stats["side_effect_max"], side)
            state["max_side_effect"] = max(state["max_side_effect"], side)
        if route == "motion":
            result = original_update_with_gates(
                self, frame_id, detection, chosen, float(appearance_gate)
            )
        else:
            result = original_update_with_gates(
                self, frame_id, detection, float(motion_gate), chosen
            )
        if detail.get("reason") is None:
            verify_execution(self, chosen, detail, info)
        record(info, detail, chosen)
        return result

    def patched_update(self, frame_id, detection):
        pilot = state["pilot"]
        if pilot is not None:
            prepare_event(self, frame_id, detection, None, None, False)
        return original_update(self, frame_id, detection)

    started = time.monotonic()
    code = 0
    error_text = None
    Tracker = tracker_module.Tracker
    Tracker.__init__ = patched_init
    Track.update_with_gates = patched_update_with_gates
    Track.update = patched_update
    try:
        run_module.run()
    except Exception as exc:  # noqa: BLE001
        code = 1
        error_text = f"{type(exc).__name__}: {exc}"
        (out / "error.txt").write_text(error_text + "\n")
        raise
    finally:
        Tracker.__init__ = original_init
        Track.update_with_gates = original_update_with_gates
        Track.update = original_update
        close_pilot()
        if state["event_file"] is not None:
            state["event_file"].close()

    totals = {
        "match_events": 0, "gated_events": 0, "ungated_events": 0,
        "searchable": 0, "target_bound_events": 0, "target_unbound_events": 0,
        "det_identity_mismatch_events": 0, "learned_sum": 0.0, "learned_count": 0,
        "reasons": {}, "chosen_hist": {},
    }
    for stats in state["stats"].values():
        for key in ("match_events", "gated_events", "ungated_events", "searchable",
                    "target_bound_events", "target_unbound_events",
                    "det_identity_mismatch_events", "learned_sum", "learned_count"):
            totals[key] += stats[key]
        for reason, count in stats["reasons"].items():
            totals["reasons"][reason] = totals["reasons"].get(reason, 0) + count
        for gate, count in stats["chosen_hist"].items():
            totals["chosen_hist"][gate] = totals["chosen_hist"].get(gate, 0) + count
    totals["searchable_ratio_gated"] = (
        totals["searchable"] / totals["gated_events"] if totals["gated_events"] else None
    )
    totals["searchable_ratio_all_matches"] = (
        totals["searchable"] / totals["match_events"] if totals["match_events"] else None
    )
    totals["learned_motion_mean"] = (
        totals["learned_sum"] / totals["learned_count"] if totals["learned_count"] else None
    )
    coverage = {
        "dataset": dataset,
        "route": route,
        "sequences": state["stats"],
        "totals": totals,
        "max_state_error": state["max_state_err"],
        "max_feature_error": state["max_feature_err"],
        "max_candidate_side_effect": state["max_side_effect"],
        "endpoints": state["endpoint"],
        "seconds": round(time.monotonic() - started, 2),
    }
    if state["tim"]:
        print("timing:", {key: round(value, 2) for key, value in state["tim"].items()},
              "sum", round(sum(state["tim"].values()), 2), flush=True)
    (out / "coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n"
    )
    manifest = _base_manifest(dataset, route, checkpoint, sequences, max_frames)
    manifest.update({
        "seconds": coverage["seconds"],
        "exit_code": int(code),
        "error": error_text,
        "finished_at": time.strftime("%F %T"),
        "max_state_error": state["max_state_err"],
        "max_feature_error": state["max_feature_err"],
        "max_candidate_side_effect": state["max_side_effect"],
    })
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
METRIC_KEYS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")


def _load_summary(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _metrics(payload, sequence=None):
    if payload is None:
        return {key: None for key in METRIC_KEYS}
    if sequence is None:
        source = payload.get("combined") or {}
    else:
        source = (payload.get("per_sequence") or {}).get(sequence) or {}
    return {key: source.get(key) for key in METRIC_KEYS}


def _delta(after, before):
    out = {}
    for key in METRIC_KEYS:
        a, b = after.get(key), before.get(key)
        out[key] = None if a is None or b is None else (a - b if key == "IDSW" else (a - b) * 100.0)
    return out


def build_report(out_root: Path) -> dict:
    report = {"created_at": time.strftime("%F %T"), "datasets": {}}
    for dataset in ("SportsMOT", "MOT20", "MOT17"):
        cfg = dataset_config(dataset)
        base_dir = out_root / dataset
        summaries = {
            route: _load_summary(base_dir / f"{route}/summary.json")
            for route in ROUTES
        }
        available = {route: payload for route, payload in summaries.items() if payload}
        if not available:
            continue
        info = {"note": cfg["note"], "checkpoint": str(cfg["checkpoint"]), "rows": {},
                "deltas": {}, "per_sequence": {}, "coverage": {}}
        for route, payload in available.items():
            info["rows"][route] = {
                "combined": _metrics(payload),
                "per_sequence": {
                    seq: _metrics(payload, seq) for seq in cfg["sequences"]
                },
            }
        for route in ("motion", "appearance"):
            if "baseline" in info["rows"] and route in info["rows"]:
                info["deltas"][route] = _delta(
                    info["rows"][route]["combined"],
                    info["rows"]["baseline"]["combined"],
                )
        for seq in cfg["sequences"]:
            info["per_sequence"][seq] = {}
            for route in info["rows"]:
                info["per_sequence"][seq][route] = info["rows"][route]["per_sequence"][seq]
            for route in info["deltas"]:
                info["per_sequence"][seq][f"delta_{route}"] = _delta(
                    info["rows"][route]["per_sequence"][seq],
                    info["rows"]["baseline"]["per_sequence"][seq],
                )
        for route in ("motion", "appearance"):
            path = base_dir / f"{route}/coverage.json"
            if path.is_file():
                coverage = json.loads(path.read_text())
                info["coverage"][route] = coverage["totals"]
                info["coverage"][route]["seconds"] = coverage.get("seconds")
                info["coverage"][route]["max_state_error"] = coverage.get("max_state_error")
                info["coverage"][route]["max_feature_error"] = coverage.get("max_feature_error")
                info["coverage"][route]["max_candidate_side_effect"] = coverage.get("max_candidate_side_effect")
        # single-sequence dominance on HOTA deltas
        info["dominance"] = {}
        for route in info["deltas"]:
            deltas = {
                seq: info["per_sequence"][seq][f"delta_{route}"]["HOTA"]
                for seq in cfg["sequences"]
                if info["per_sequence"][seq][f"delta_{route}"]["HOTA"] is not None
            }
            if not deltas:
                continue
            ordered = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)
            top_seq, top_delta = ordered[0]
            rest = [value for seq, value in ordered[1:]]
            rest_mean = sum(rest) / len(rest) if rest else None
            info["dominance"][route] = {
                "top_sequence": top_seq,
                "top_delta_pp": top_delta,
                "other_mean_pp": rest_mean,
                "sign_flip_without_top": (
                    None if rest_mean is None or top_delta == 0
                    else (rest_mean > 0) != (top_delta > 0)
                ),
            }
        report["datasets"][dataset] = info
    (out_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    lines = ["# 21-level gate oracle diagnostics (raw val)", "",
             f"created: {report['created_at']}", ""]
    for dataset, info in report["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- checkpoint: `{info['checkpoint']}`")
        lines.append(f"- note: {info['note']}")
        lines.append("")
        lines.append("| group | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for route in ("baseline", "motion", "appearance"):
            if route not in info["rows"]:
                continue
            values = info["rows"][route]["combined"]

            def fmt(key):
                value = values.get(key)
                if value is None:
                    return "-"
                return f"{value:.0f}" if key == "IDSW" else f"{value * 100:.3f}"
            lines.append(f"| {route} | " + " | ".join(fmt(key) for key in METRIC_KEYS) + " |")
        for route in ("motion", "appearance"):
            if route not in info["deltas"]:
                continue
            values = info["deltas"][route]

            def dfmt(key):
                value = values.get(key)
                if value is None:
                    return "-"
                return f"{value:+.0f}" if key == "IDSW" else f"{value:+.3f} pp"
            lines.append(f"| Δ {route} − baseline | " + " | ".join(dfmt(key) for key in METRIC_KEYS) + " |")
        lines.append("")
        for route, coverage in info["coverage"].items():
            ratio = coverage.get("searchable_ratio_gated")
            lines.append(
                f"- {route} coverage: searchable {coverage['searchable']}/"
                f"{coverage['gated_events']} gated"
                + (f" ({ratio * 100:.1f}%)" if ratio is not None else "")
                + f"; all matches {coverage['match_events']};"
                f" fallback {coverage['reasons']}; chosen {coverage['chosen_hist']};"
                f" wall {coverage['seconds']}s"
                f"; max state err {coverage['max_state_error']}"
                f"; max feature err {coverage['max_feature_error']}"
                f"; max candidate side effect {coverage['max_candidate_side_effect']}"
            )
        lines.append("")
        lines.append("Per-sequence HOTA / IDSW deltas (pp / events):")
        lines.append("")
        lines.append("| sequence | motion ΔHOTA | motion ΔIDSW | appearance ΔHOTA | appearance ΔIDSW |")
        lines.append("|---|---:|---:|---:|---:|")
        for seq in info["per_sequence"]:
            parts = []
            for route in ("motion", "appearance"):
                delta = info["per_sequence"][seq].get(f"delta_{route}")
                if delta is None:
                    parts += ["-", "-"]
                else:
                    hota = delta["HOTA"]
                    idsw = delta["IDSW"]
                    parts.append("-" if hota is None else f"{hota:+.3f}")
                    parts.append("-" if idsw is None else f"{idsw:+.0f}")
            lines.append(f"| {seq} | " + " | ".join(parts) + " |")
        lines.append("")
        if info["dominance"]:
            for route, item in info["dominance"].items():
                lines.append(
                    f"- {route} dominance: top `{item['top_sequence']}` "
                    f"{item['top_delta_pp']:+.3f} pp; others mean "
                    + ("-" if item["other_mean_pp"] is None else f"{item['other_mean_pp']:+.3f} pp")
                    + ("; sign flips without the top sequence"
                       if item["sign_flip_without_top"] else "")
                )
            lines.append("")
    (out_root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run one diagnostic group")
    run_parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    run_parser.add_argument("--route", choices=ROUTES, required=True)
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    run_parser.add_argument("--checkpoint", default=None)
    run_parser.add_argument("--sequences", nargs="*", default=None)
    run_parser.add_argument("--max-frames", type=int, default=0)
    run_parser.add_argument("--event-log", action="store_true")
    report_parser = sub.add_parser("report", help="build the side-by-side report")
    report_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    if args.command == "report":
        report = build_report(out_root)
        print(json.dumps(report["datasets"], indent=2)[:2000])
        return

    cfg = dataset_config(args.dataset)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else cfg["checkpoint"]
    sequences = args.sequences or cfg["sequences"]
    out = out_root / args.dataset / args.route
    if args.route == "baseline":
        manifest = run_baseline(args.dataset, out, sequences, checkpoint, args.max_frames)
    else:
        manifest = run_oracle(
            args.dataset, args.route, out, sequences, checkpoint,
            args.max_frames, args.event_log,
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
