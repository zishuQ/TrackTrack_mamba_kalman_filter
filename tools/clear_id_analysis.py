#!/usr/bin/env python3
"""CLEAR-style identity error spans for the key SportsMOT sequences.

Matches tracker boxes to GT per frame with an IoU>=0.5 Hungarian assignment
(the CLEAR matching rule), then reports runs where a GT identity is carried
by a different tracker id, including the recovery frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GT_ROOT = Path("/home/shang/datasets/SportsMOT/dataset/val")
KEY_SEQUENCES = (
    "v_9MHDmAMxO5I_c003",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c011",
    "v_BgwzTUxJaeU_c012",
    "v_0kUtTtmLaJA_c008",
    "v_cC2mHWqMcjk_c008",
)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter = (
        np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]))
        * np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]))
    )
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_gt(sequence: str) -> dict[int, dict[int, np.ndarray]]:
    frames: dict[int, dict[int, np.ndarray]] = {}
    path = GT_ROOT / sequence / "gt" / "gt.txt"
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        gt_id = int(float(fields[1]))
        if len(fields) >= 8 and (float(fields[6]) <= 0 or float(fields[7]) != 1):
            continue
        x, y, w, h = map(float, fields[2:6])
        frames.setdefault(frame, {})[gt_id] = np.array([x, y, x + w, y + h])
    return frames


def load_tracks(path: Path) -> dict[int, dict[int, np.ndarray]]:
    frames: dict[int, dict[int, np.ndarray]] = {}
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        track_id = int(float(fields[1]))
        x, y, w, h = map(float, fields[2:6])
        frames.setdefault(frame, {})[track_id] = np.array([x, y, x + w, y + h])
    return frames


def assignments(
    gt_frames: dict[int, dict[int, np.ndarray]],
    track_frames: dict[int, dict[int, np.ndarray]],
    threshold: float = 0.5,
) -> dict[int, dict[int, int]]:
    """frame -> {gt_id: track_id} under TrackEval's CLEAR matching.

    Mirrors ``trackeval/metrics/clear.py``: the score matrix is
    ``1000 * (track_id == previous frame's track for this gt) + IoU`` with
    entries below the IoU threshold zeroed, so matching first minimises
    identity switches from the previous frame and then maximises IoU.
    """
    eps = np.finfo("float").eps
    result: dict[int, dict[int, int]] = {}
    previous: dict[int, int] = {}
    for frame in sorted(gt_frames):
        gt_ids = sorted(gt_frames[frame])
        tracks = track_frames.get(frame, {})
        track_ids = sorted(tracks)
        gt_boxes = np.stack([gt_frames[frame][g] for g in gt_ids]) if gt_ids else np.zeros((0, 4))
        track_boxes = np.stack([tracks[t] for t in track_ids]) if track_ids else np.zeros((0, 4))
        ious = iou_matrix(gt_boxes, track_boxes)
        mapping: dict[int, int] = {}
        if ious.size:
            previous_ids = np.asarray(
                [previous.get(gt_id, -1) for gt_id in gt_ids], dtype=np.int64
            )
            track_array = np.asarray(track_ids, dtype=np.int64)
            continuity = (
                track_array[np.newaxis, :] == previous_ids[:, np.newaxis]
            ) if previous_ids.size else np.zeros_like(ious, dtype=bool)
            score = 1000.0 * continuity.astype(np.float64) + ious
            score[ious < threshold - eps] = 0.0
            rows, cols = linear_sum_assignment(-score)
            for row, col in zip(rows, cols):
                if score[row, col] > 0.0 + eps:
                    mapping[gt_ids[row]] = track_ids[col]
        result[frame] = mapping
        previous = mapping
    return result


def identity_spans(assign: dict[int, dict[int, int]]) -> list[dict]:
    per_gt: dict[int, list[tuple[int, int]]] = {}
    for frame in sorted(assign):
        for gt_id, track_id in assign[frame].items():
            per_gt.setdefault(gt_id, []).append((frame, track_id))
    spans: list[dict] = []
    for gt_id, entries in sorted(per_gt.items()):
        run_start, run_track = entries[0]
        previous_frame = entries[0][0]
        for frame, track_id in entries[1:]:
            if track_id != run_track:
                spans.append(
                    {
                        "gt_id": gt_id,
                        "start": run_start,
                        "end": previous_frame,
                        "duration": previous_frame - run_start + 1,
                        "track_id": run_track,
                        "next_track_id": track_id,
                        "next_start": frame,
                    }
                )
                run_start, run_track = frame, track_id
            previous_frame = frame
        spans.append(
            {
                "gt_id": gt_id,
                "start": run_start,
                "end": previous_frame,
                "duration": previous_frame - run_start + 1,
                "track_id": run_track,
                "next_track_id": None,
                "next_start": None,
            }
        )
    # A span is an identity error when a GT id changes tracker id but does not
    # start at its first observation and later returns to the previous id.
    errors = []
    for index, span in enumerate(spans):
        if span["next_track_id"] is None:
            continue
        following = spans[index + 1] if index + 1 < len(spans) else None
        if following is None or following["gt_id"] != span["gt_id"]:
            continue
        errors.append(
            {
                "gt_id": span["gt_id"],
                "wrong_from": span["next_start"],
                "wrong_track_id": span["next_track_id"],
                "previous_track_id": span["track_id"],
                "previous_span_end": span["end"],
            }
        )
    for error in errors:
        gt_id = error["gt_id"]
        recovery = None
        for frame, track_id in per_gt[gt_id]:
            if frame >= error["wrong_from"] and track_id == error["previous_track_id"]:
                recovery = frame
                break
        error["recovery_frame"] = recovery
        if recovery is not None:
            error["wrong_duration"] = recovery - error["wrong_from"]
        else:
            error["wrong_duration"] = max(
                frame for frame, _ in per_gt[gt_id]
            ) - error["wrong_from"] + 1
    return errors


def wrong_identity_runs(assign: dict[int, dict[int, int]]) -> list[dict]:
    """Runs where a GT is carried by a track other than its first track.

    The first observed track of each GT is the reference identity; any
    contiguous run with a different track is a wrong-identity run.  This
    counts both recovered swaps and swaps that never recover, without the
    overlapping-span double counting of ``identity_spans``.
    """
    per_gt: dict[int, list[tuple[int, int]]] = {}
    for frame in sorted(assign):
        for gt_id, track_id in assign[frame].items():
            per_gt.setdefault(gt_id, []).append((frame, track_id))
    runs: list[dict] = []
    for gt_id, entries in sorted(per_gt.items()):
        reference = entries[0][1]
        current = None
        for frame, track_id in entries:
            if track_id != reference:
                if (
                    current is not None
                    and track_id == current["track_id"]
                    and frame == current["end"] + 1
                ):
                    current["end"] = frame
                    current["duration"] += 1
                else:
                    if current is not None:
                        runs.append(current)
                    current = {
                        "gt_id": gt_id,
                        "track_id": track_id,
                        "reference_track": reference,
                        "start": frame,
                        "end": frame,
                        "duration": 1,
                    }
            elif current is not None:
                runs.append(current)
                current = None
        if current is not None:
            runs.append(current)
    for run in runs:
        run["recovered"] = any(
            frame > run["end"] and track_id == run["reference_track"]
            for frame, track_id in per_gt[run["gt_id"]]
        )
    return runs


def window_report(assign: dict[int, dict[int, int]], start: int, end: int, gt_ids: tuple[int, ...]):
    rows = []
    for frame in range(start, end + 1):
        mapping = assign.get(frame, {})
        rows.append(
            {
                "frame": frame,
                **{f"gt{gt_id}": mapping.get(gt_id) for gt_id in gt_ids},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, help="name=dir")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sequences-file",
        default="",
        help="optional seqmap; when set, analyse all listed sequences",
    )
    args = parser.parse_args()

    groups = {}
    for item in args.result:
        name, directory = item.split("=", 1)
        groups[name] = Path(directory)

    sequences = KEY_SEQUENCES
    if args.sequences_file:
        sequences = tuple(
            line.strip().split()[0]
            for line in Path(args.sequences_file).read_text().splitlines()
            if line.strip() and line.strip() != "name"
        )

    output: dict = {"sequences": {}}
    for sequence in sequences:
        gt_frames = load_gt(sequence)
        output["sequences"][sequence] = {}
        for name, directory in groups.items():
            track_path = directory / f"{sequence}.txt"
            if not track_path.is_file():
                output["sequences"][sequence][name] = {"missing": str(track_path)}
                continue
            track_frames = load_tracks(track_path)
            assign = assignments(gt_frames, track_frames)
            errors = identity_spans(assign)
            if sequence == "v_9MHDmAMxO5I_c003":
                window = window_report(assign, 100, 140, (3, 4))
            elif sequence == "v_00HRwkvvjtQ_c007":
                window = window_report(assign, 550, 707, (2, 3, 4, 7))
            else:
                window = None
            output["sequences"][sequence][name] = {
                "identity_errors": errors,
                "wrong_identity_runs": wrong_identity_runs(assign),
                "window": window,
                "id_switches": len(errors),
            }
    Path(args.out).write_text(json.dumps(output, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "groups": list(groups)}, indent=1))


if __name__ == "__main__":
    main()
