#!/usr/bin/env python3
"""Appearance-template counterfactual: online EMA (A) vs pre-contention (B).

For every tracker association call the same detections, motion states and
matching thresholds are used twice: once with each track's live online EMA
template (A, the actual run) and once with a snapshot template left by the
track's most recent reliable match (B).  GT is used only to score the two
counterfactual assignments and to select cases; a GT-free variant of the
reliable-match rule is evaluated for deployment feasibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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

from sportsmot_supervision_pilot import SequencePilot, iou_xyxy  # noqa: E402

VAL_MAP = WORKTREE / "3. Tracker/trackeval/seqmap/sportsmot/val.txt"
ONLINE_SCORE = 0.80
ONLINE_MOTION_IOU = 0.60
ONLINE_COS = 0.35
CONTEST_IOU = 0.50


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-12)


def cos_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))


class TemplateExperimentPilot(SequencePilot):
    def __init__(self, sequence, split, checkpoint, out_dir):
        super().__init__(sequence, split, checkpoint, out_dir)
        self.b_oracle: dict[int, np.ndarray] = {}
        self.b_online: dict[int, np.ndarray] = {}
        self.b_oracle_source: dict[int, int] = {}
        self.b_online_source: dict[int, int] = {}
        self.pending_oracle: dict[int, int] = {}
        self.pending_online: dict[int, int] = {}
        self.identity: dict[int, int] = {}
        self.records: list[dict] = []

    def install_hooks(self) -> None:
        from trackers import tracker as tracker_module

        super().install_hooks()
        original = tracker_module.iterative_assignment
        pilot = self

        def wrapped(
            tracks,
            dets_high,
            dets_low,
            dets_del_high,
            match_thr,
            penalty_p,
            penalty_q,
            reduce_step,
            frame_id,
            d_t=3,
            no_reid=False,
            return_meta=False,
        ):
            pilot._apply_pending(tracks, int(frame_id))
            dets = list(dets_high) + list(dets_low) + list(dets_del_high)

            def run_with(feature_map):
                if not feature_map or not any(f is not None for f in feature_map):
                    return None
                saved = [np.asarray(track.feat).copy() for track in tracks]
                for track, feature in zip(tracks, feature_map):
                    if feature is not None:
                        track.feat = np.asarray(feature, dtype=saved[0].dtype).copy()
                try:
                    return original(
                        tracks,
                        dets_high,
                        dets_low,
                        dets_del_high,
                        match_thr,
                        penalty_p,
                        penalty_q,
                        reduce_step,
                        frame_id,
                        d_t=d_t,
                        no_reid=no_reid,
                        return_meta=return_meta,
                    )
                finally:
                    for track, feature in zip(tracks, saved):
                        track.feat = feature

            oracle_features = [pilot.b_oracle.get(int(t.track_id)) for t in tracks]
            online_features = [pilot.b_online.get(int(t.track_id)) for t in tracks]
            oracle_result = run_with(oracle_features)
            online_result = run_with(online_features)
            actual = original(
                tracks,
                dets_high,
                dets_low,
                dets_del_high,
                match_thr,
                penalty_p,
                penalty_q,
                reduce_step,
                frame_id,
                d_t=d_t,
                no_reid=no_reid,
                return_meta=return_meta,
            )
            pilot._record(
                int(frame_id),
                tracks,
                dets,
                actual,
                oracle_result,
                online_result,
                bool(no_reid),
            )
            return actual

        tracker_module.iterative_assignment = wrapped
        self._restore = [
            (tracker_module, "iterative_assignment", original)
        ] + self._restore

    def _apply_pending(self, tracks, frame_id: int) -> None:
        for track in tracks:
            track_id = int(track.track_id)
            if track_id in self.pending_oracle:
                source = self.pending_oracle[track_id]
                if source < frame_id:
                    self.b_oracle[track_id] = np.asarray(track.feat).copy()
                    self.b_oracle_source[track_id] = source
                    self.pending_oracle.pop(track_id)
            if track_id in self.pending_online:
                source = self.pending_online[track_id]
                if source < frame_id:
                    self.b_online[track_id] = np.asarray(track.feat).copy()
                    self.b_online_source[track_id] = source
                    self.pending_online.pop(track_id)

    def _det_labels(self, frame_id: int, dets):
        """Return (best GT with IoU>=0.5, unambiguous GT with margin>=0.1)."""
        boxes = self.gt.get(frame_id, {})
        best_labels: list[int | None] = []
        clean_labels: list[int | None] = []
        for det in dets:
            det_box = np.asarray(det.x1y1x2y2, dtype=np.float64)
            scored = sorted(
                ((iou_xyxy(det_box, gt_box), gt_id) for gt_id, gt_box in boxes.items()),
                reverse=True,
            )
            if not scored:
                best_labels.append(None)
                clean_labels.append(None)
                continue
            best_iou, best_id = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            best_labels.append(int(best_id) if best_iou >= 0.5 else None)
            clean_labels.append(
                int(best_id) if best_iou >= 0.5 and best_iou - second >= 0.1 else None
            )
        return best_labels, clean_labels

    def _record(self, frame_id, tracks, dets, actual, oracle_result, online_result, no_reid) -> None:
        if no_reid or not dets or not tracks:
            return
        best_labels, clean_labels = self._det_labels(frame_id, dets)
        track_boxes = [np.asarray(t.x1y1x2y2, dtype=np.float64) for t in tracks]
        det_boxes = [np.asarray(d.x1y1x2y2, dtype=np.float64) for d in dets]
        motion = np.zeros((len(tracks), len(dets)), dtype=np.float64)
        for i, track_box in enumerate(track_boxes):
            for j, det_box in enumerate(det_boxes):
                motion[i, j] = iou_xyxy(track_box, det_box)
        contested = [
            int((motion[:, j] >= CONTEST_IOU).sum()) > 1 for j in range(len(dets))
        ]
        track_contested = [
            any(
                iou_xyxy(track_boxes[i], track_boxes[j]) >= 0.30
                for j in range(len(tracks))
                if j != i
            )
            for i in range(len(tracks))
        ]
        gt_boxes = self.gt.get(frame_id, {})

        def match_map(result):
            if result is None:
                return None
            matches = result[0]
            return {int(tracks[t].track_id): int(d) for t, d in matches}

        actual_map = match_map(actual)
        oracle_map = match_map(oracle_result)
        online_map = match_map(online_result)
        entry = {
            "frame": frame_id,
            "actual": {},
            "oracle": oracle_map,
            "online": online_map,
            "labels": best_labels,
            "clean_labels": clean_labels,
            "scores": [float(d.score) for d in dets],
        }
        for t_index, track in enumerate(tracks):
            track_id = int(track.track_id)
            det_index = actual_map.get(track_id)
            label = best_labels[det_index] if det_index is not None else None
            clean = clean_labels[det_index] if det_index is not None else None
            identity_before = self.identity.get(track_id)
            reliable_oracle = False
            reliable_online = False
            if det_index is not None:
                if identity_before is None and clean is not None:
                    self.identity[track_id] = clean
                    identity_before = clean
                gt_contested = False
                own_box = (
                    gt_boxes.get(identity_before)
                    if identity_before is not None
                    else None
                )
                if own_box is not None:
                    gt_contested = any(
                        iou_xyxy(det_box, own_box) >= 0.5
                        and sum(
                            1
                            for other_id, other_box in gt_boxes.items()
                            if int(other_id) != identity_before
                            and iou_xyxy(det_box, other_box) >= 0.5
                        )
                        >= 1
                        for det_box in det_boxes
                    )
                reliable_oracle = (
                    clean is not None
                    and (identity_before is None or clean == identity_before)
                    and not gt_contested
                )
                det = dets[det_index]
                reliable_online = (
                    float(det.score) >= ONLINE_SCORE
                    and motion[t_index, det_index] >= ONLINE_MOTION_IOU
                    and cos_distance(track.feat, det.feat) <= ONLINE_COS
                    and not contested[det_index]
                    and not track_contested[t_index]
                )
            if reliable_oracle:
                self.pending_oracle[track_id] = frame_id
            if reliable_online:
                self.pending_online[track_id] = frame_id
            entry["actual"][track_id] = {
                "det": det_index,
                "label": label,
                "clean": clean,
                "identity": self.identity.get(track_id),
                "reliable_oracle": reliable_oracle,
                "reliable_online": reliable_online,
                "b_oracle_source": self.b_oracle_source.get(track_id),
                "b_online_source": self.b_online_source.get(track_id),
            }
        self.records.append(entry)


def summarize_sequence(pilot) -> dict:
    def status(info, mapping, labels, track_id):
        if mapping is None:
            return "no_b"
        det = mapping.get(track_id)
        if det is None:
            return "unmatched"
        label = labels[det]
        if label is None:
            return "unknown"
        return "correct" if label == info["identity"] else "wrong"

    stats = {
        "actual": {"correct": 0, "wrong": 0, "unmatched": 0, "unknown": 0},
        "oracle": {"correct": 0, "wrong": 0, "unmatched": 0, "unknown": 0, "no_b": 0},
        "online": {"correct": 0, "wrong": 0, "unmatched": 0, "unknown": 0, "no_b": 0},
    }
    changes = {
        "oracle": {"fix": 0, "break": 0, "fix_miss": 0, "break_miss": 0, "comparable": 0},
        "online": {"fix": 0, "break": 0, "fix_miss": 0, "break_miss": 0, "comparable": 0},
    }
    identity_tracks = 0
    identity_frames = 0
    fix_events: list[dict] = []
    break_events: list[dict] = []
    miss_events: list[dict] = []
    for entry in pilot.records:
        labels = entry["labels"]
        oracle_map = entry["oracle"]
        online_map = entry["online"]
        for track_id, info in entry["actual"].items():
            identity = info["identity"]
            if identity is None:
                continue
            identity_tracks += 1
            a_det = info["det"]
            if a_det is None:
                a_status = "unmatched"
            elif labels[a_det] is None:
                a_status = "unknown"
            else:
                a_status = "correct" if labels[a_det] == identity else "wrong"
            stats["actual"][a_status] += 1
            if a_status in ("correct", "wrong"):
                identity_frames += 1
            for variant, mapping in (("oracle", oracle_map), ("online", online_map)):
                b_status = status(info, mapping, labels, track_id)
                stats[variant][b_status] += 1
                if b_status == "no_b":
                    continue
                if a_status in ("correct", "wrong") and b_status in ("correct", "wrong"):
                    changes[variant]["comparable"] += 1
                if a_status == "wrong" and b_status == "correct":
                    changes[variant]["fix"] += 1
                    if len(fix_events) < 20:
                        fix_events.append(
                            {
                                "frame": entry["frame"],
                                "track_id": track_id,
                                "identity": identity,
                                "variant": variant,
                                "a_det": a_det,
                                "a_label": labels[a_det],
                                "b_det": mapping.get(track_id),
                                "b_label": labels[mapping.get(track_id)],
                                "b_source": info.get(f"b_{variant}_source"),
                            }
                        )
                elif a_status == "correct" and b_status == "wrong":
                    changes[variant]["break"] += 1
                    if len(break_events) < 20:
                        break_events.append(
                            {
                                "frame": entry["frame"],
                                "track_id": track_id,
                                "identity": identity,
                                "variant": variant,
                                "a_det": a_det,
                                "a_label": labels[a_det],
                                "b_det": mapping.get(track_id),
                                "b_label": labels[mapping.get(track_id)],
                                "b_source": info.get(f"b_{variant}_source"),
                            }
                        )
                elif a_status == "unmatched" and b_status == "correct":
                    changes[variant]["fix_miss"] += 1
                    if len(miss_events) < 20:
                        miss_events.append(
                            {
                                "kind": "fix_miss",
                                "frame": entry["frame"],
                                "track_id": track_id,
                                "identity": identity,
                                "variant": variant,
                                "a_det": None,
                                "b_det": mapping.get(track_id),
                                "b_label": labels[mapping.get(track_id)],
                                "b_source": info.get(f"b_{variant}_source"),
                            }
                        )
                elif a_status == "correct" and b_status == "unmatched":
                    changes[variant]["break_miss"] += 1
                    if len(miss_events) < 20:
                        miss_events.append(
                            {
                                "kind": "break_miss",
                                "frame": entry["frame"],
                                "track_id": track_id,
                                "identity": identity,
                                "variant": variant,
                                "a_det": a_det,
                                "a_label": labels[a_det],
                                "b_det": None,
                                "b_source": info.get(f"b_{variant}_source"),
                            }
                        )
    return {
        "frames": len(pilot.records),
        "identity_tracks": identity_tracks,
        "identity_frames": identity_frames,
        "status": stats,
        "changes": changes,
        "oracle_fix": changes["oracle"]["fix"],
        "oracle_break": changes["oracle"]["break"],
        "online_fix": changes["online"]["fix"],
        "online_break": changes["online"]["break"],
        "fix_events": fix_events,
        "break_events": break_events,
        "miss_events": miss_events,
    }


def writes_before(pilot, track_id: int, before_frame: int) -> list[dict]:
    writes = []
    for record in pilot.events:
        if int(record["track_id"]) != track_id or int(record["frame_t"]) >= before_frame:
            continue
        pre = np.asarray(record["pre_feature"], dtype=np.float64)
        det = np.asarray(record["det_feature"], dtype=np.float64)
        score = float(record["detection_score"])
        gate = (
            1.0
            if record.get("appearance_gate") is None
            else float(record["appearance_gate"])
        )
        beta = 0.95 + 0.05 * (1.0 - score)
        full = beta * pre + (1.0 - beta) * det
        if gate <= 0.0:
            post = pre.copy()
        elif gate >= 1.0:
            post = full / np.linalg.norm(full)
        else:
            blended = (1.0 - gate) * pre + gate * full
            post = blended / (np.linalg.norm(blended) + 1e-12)
        writes.append(
            {
                "frame": int(record["frame_t"]),
                "appearance_gate": (
                    None
                    if record.get("appearance_gate") is None
                    else float(record["appearance_gate"])
                ),
                "cos_pre_post": cos_distance(pre, post),
                "detection_gt": int(record.get("detection_gt_id", -1)),
            }
        )
    return writes


def event_timeline(pilot, sequence: str, event: dict) -> dict:
    track_id = int(event["track_id"])
    identity = int(event["identity"])
    frame = int(event["frame"])
    recovery = None
    for entry in pilot.records:
        if entry["frame"] <= frame:
            continue
        info = entry["actual"].get(track_id)
        if info is None or info["identity"] != identity:
            continue
        det = info["det"]
        label = entry["labels"][det] if det is not None else None
        if label == identity:
            recovery = entry["frame"]
            break
    return {
        **event,
        "sequence": sequence,
        "writes_before": writes_before(pilot, track_id, frame)[-6:],
        "recovery_frame": recovery,
    }


def case_timeline(pilot, sequence, track_id, identity) -> dict:
    wrong_frame = None
    recovery_frame = None
    wrong_info = None
    for entry in pilot.records:
        info = entry["actual"].get(track_id)
        if info is None or info["identity"] != identity:
            continue
        det = info["det"]
        if det is None:
            continue
        label = entry["labels"][det]
        if label is None:
            continue
        if label != identity and wrong_frame is None:
            wrong_frame = entry["frame"]
            wrong_info = info
        if wrong_frame is not None and label == identity:
            recovery_frame = entry["frame"]
            break
    writes = writes_before(pilot, track_id, wrong_frame) if wrong_frame else []
    return {
        "sequence": sequence,
        "track_id": track_id,
        "identity": identity,
        "first_wrong_frame": wrong_frame,
        "recovery_frame": recovery_frame,
        "b_oracle_source_frame": (
            wrong_info.get("b_oracle_source") if wrong_info else None
        ),
        "b_online_source_frame": (
            wrong_info.get("b_online_source") if wrong_info else None
        ),
        "last_writes": writes[-4:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-root", default=str(MAIN / "outputs/agentguard/experiments/shutrack_a_appearance_template_20261119/logs"))
    args = parser.parse_args()
    val = {
        line.strip().split()[0]
        for line in VAL_MAP.read_text().splitlines()
        if line.strip() and line.strip() != "name"
    }
    results = {}
    timelines = {}
    event_timelines = {}
    for sequence in args.sequences:
        split = "val" if sequence in val else "train"
        pilot = TemplateExperimentPilot(
            sequence, split, args.checkpoint, Path(args.log_root) / sequence
        )
        pilot.run_replay()
        results[sequence] = summarize_sequence(pilot)
        timelines[sequence] = [
            case_timeline(pilot, sequence, track_id, identity)
            for track_id, identity in sorted(pilot.identity.items())
            if track_id is not None
        ]
        selected = results[sequence]["fix_events"][:3] + results[sequence]["break_events"][:2]
        event_timelines[sequence] = [
            event_timeline(pilot, sequence, event) for event in selected
        ]
        pilot.close()
        summary = results[sequence]
        print(
            sequence,
            {
                key: summary[key]
                for key in (
                    "identity_tracks",
                    "identity_frames",
                    "oracle_fix",
                    "oracle_break",
                    "online_fix",
                    "online_break",
                )
            },
            flush=True,
        )

    def aggregate_subset(names) -> dict:
        entries = [results[name] for name in names]
        agg = {
            key: sum(entry[key] for entry in entries)
            for key in (
                "oracle_fix",
                "oracle_break",
                "online_fix",
                "online_break",
                "identity_frames",
                "identity_tracks",
            )
        }
        for variant in ("actual", "oracle", "online"):
            for status_key in ("correct", "wrong", "unmatched", "unknown", "no_b"):
                if variant == "actual" and status_key == "no_b":
                    continue
                agg[f"status_{variant}_{status_key}"] = sum(
                    entry["status"][variant].get(status_key, 0) for entry in entries
                )
        for variant in ("oracle", "online"):
            for change_key in ("fix", "break", "fix_miss", "break_miss", "comparable"):
                agg[f"changes_{variant}_{change_key}"] = sum(
                    entry["changes"][variant][change_key] for entry in entries
                )
        agg["sequences"] = len(entries)
        agg["sources"] = len({name.split("_c")[0] for name in names})
        return agg

    names = list(results)
    aggregate = aggregate_subset(names)
    by_split = {
        "train": aggregate_subset([name for name in names if name not in val]),
        "val": aggregate_subset([name for name in names if name in val]),
    }
    payload = {
        "settings": {
            "online_score": ONLINE_SCORE,
            "online_motion_iou": ONLINE_MOTION_IOU,
            "online_cos": ONLINE_COS,
            "contest_iou": CONTEST_IOU,
        },
        "per_sequence": results,
        "aggregate": aggregate,
        "aggregate_by_split": by_split,
        "timelines": timelines,
        "event_timelines": event_timelines,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
