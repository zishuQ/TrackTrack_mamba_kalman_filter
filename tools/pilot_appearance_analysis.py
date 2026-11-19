#!/usr/bin/env python3
"""Key-window DIoU comparison and appearance-score explanatory analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/shang/workspace/TrackTrack/outputs/agentguard/experiments"
    "/shutrack_a_supervision_pilot_20261118"
)
SEQUENCES = ("v_9MHDmAMxO5I_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")
WINDOWS = {
    "v_9MHDmAMxO5I_c003": ((100, 140),),
    "v_00HRwkvvjtQ_c007": ((555, 600),),
    "v_00HRwkvvjtQ_c011": ((380, 410), (680, 700)),
}


def diou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    iou = inter / max(area_a + area_b - inter, 1e-9)
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    c2 = max((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2, 1e-9)
    return iou - (((ax1 + ax2) - (bx1 + bx2)) ** 2 / 4 + ((ay1 + ay2) - (by1 + by2)) ** 2 / 4) / c2


def main() -> None:
    for sequence in SEQUENCES:
        events = [
            json.loads(line)
            for line in (ROOT / sequence / "events.jsonl").read_text().splitlines()
        ]
        data = np.load(ROOT / sequence / "samples.npz", allow_pickle=False)
        keys = json.loads(str(data["keys_json"]))
        index_of = {
            (key["frame"], key["track_id"]): row
            for row, key in enumerate(keys)
        }
        print("=" * 24, sequence)
        for start, end in WINDOWS[sequence]:
            rows = [
                event
                for event in events
                if start <= event["frame_t"] <= end
                and event.get("motion_gt_valid")
                and event.get("pred_box_g0") is not None
            ]
            if not rows:
                print(f"  window {start}-{end}: no valid GT events")
                continue
            print(f"  window {start}-{end}: {len(rows)} valid events")
            for event in rows:
                row = index_of.get((event["frame_t"], event["track_id"]))
                gate = event.get("motion_gate")
                if gate is None and row is not None:
                    gate = float(np.asarray(data["motion_safe_target"][row]))
                box0 = np.asarray(event["pred_box_g0"])
                box1 = np.asarray(event["pred_box_g1"])
                gt = np.asarray(event["gt_box_t1"])
                g = 0.0 if gate is None else float(np.clip(gate, 0.0, 1.0))
                box_gate = box0 + g * (box1 - box0)
                print(
                    f"    f{event['frame_t']:4d} t{event['track_id']:3d} "
                    f"gt{event.get('target_gt_before')} det_gt{event.get('detection_gt_id')} "
                    f"gate={g:.3f} d0={1-diou(box0, gt):.4f} "
                    f"dgate={1-diou(box_gate, gt):.4f} d1={1-diou(box1, gt):.4f}"
                )
        # Appearance-score explanatory stats over diag-valid events.
        valid = [
            event
            for event in events
            if event.get("appearance_diag_valid") and event.get("valid_appearance")
        ]
        with_mismatch = [
            event for event in valid if event.get("identity_mismatch_frame") is not None
        ]
        without = [
            event for event in valid if event.get("identity_mismatch_frame") is None
        ]

        def mean(field, subset):
            values = [float(event[field]) for event in subset if event.get(field) is not None]
            return float(np.mean(values)) if values else None

        print(
            "  appearance diag-valid:",
            len(valid),
            "later-mismatch:",
            len(with_mismatch),
            "no-mismatch:",
            len(without),
        )
        for field in (
            "appearance_score_template0",
            "appearance_score_template1",
        ):
            print(
                f"    {field}: mismatch={mean(field, with_mismatch)} "
                f"no_mismatch={mean(field, without)}"
            )
        improved = [
            event
            for event in valid
            if event["appearance_score_template1"] > event["appearance_score_template0"]
        ]
        print(f"    template1 improves score in {len(improved)}/{len(valid)}")
        negative = [
            event for event in valid if event["appearance_score_template0"] < 0
        ]
        negative_mismatch = [
            event for event in negative if event.get("identity_mismatch_frame") is not None
        ]
        print(
            f"    T0 score<0: {len(negative)} ({len(negative_mismatch)} later mismatch)"
        )
        contested = [event for event in events if event.get("appearance_contested")]
        print(f"    contested detections: {len(contested)}")
        # Key-window appearance rows.
        for start, end in WINDOWS[sequence]:
            rows = [
                event
                for event in events
                if start <= event["frame_t"] <= end
                and event.get("appearance_diag_valid")
            ]
            print(f"    appearance rows in {start}-{end}: {len(rows)}")
            for event in rows:
                print(
                    f"      f{event['frame_t']:4d} t{event['track_id']:3d} "
                    f"gt{event.get('target_gt_before')} pos@{event.get('appearance_positive_frame')} "
                    f"score0={event['appearance_score_template0']:.4f} "
                    f"score1={event['appearance_score_template1']:.4f} "
                    f"soft={event.get('appearance_soft_target')} "
                    f"gate={event.get('appearance_gate')} "
                    f"mismatch={event.get('identity_mismatch_frame')} "
                    f"recovery={event.get('identity_recovery_frame')}"
                )


if __name__ == "__main__":
    main()
