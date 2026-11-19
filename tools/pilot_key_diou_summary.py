#!/usr/bin/env python3
"""Key-window DIoU preference summary from the three pilot sequences."""

import json
from pathlib import Path

import numpy as np

PILOT = Path(
    "/home/shang/workspace/TrackTrack/outputs/agentguard/experiments"
    "/shutrack_a_supervision_pilot_20261118"
)
WINDOWS = {
    "v_9MHDmAMxO5I_c003": [(100, 140)],
    "v_00HRwkvvjtQ_c007": [(555, 600)],
    "v_00HRwkvvjtQ_c011": [(380, 410), (680, 700)],
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
    return iou - (
        ((ax1 + ax2 - bx1 - bx2) ** 2 + (ay1 + ay2 - by1 - by2) ** 2) / 4
    ) / c2


def main() -> None:
    report = {}
    for sequence, windows in WINDOWS.items():
        events = [
            json.loads(line)
            for line in (PILOT / sequence / "events.jsonl").read_text().splitlines()
        ]
        report[sequence] = []
        for start, end in windows:
            rows = []
            for event in events:
                if not (start <= event["frame_t"] <= end):
                    continue
                if not event.get("motion_gt_valid"):
                    continue
                gate = event.get("motion_gate")
                if gate is None or gate < 0:
                    continue
                box0 = np.asarray(event["pred_box_g0"])
                box1 = np.asarray(event["pred_box_g1"])
                gt = np.asarray(event["gt_box_t1"])
                box_gate = box0 + gate * (box1 - box0)
                rows.append(
                    {
                        "frame": event["frame_t"],
                        "track_id": event["track_id"],
                        "gt_id": event.get("target_gt_before"),
                        "detection_gt_id": event.get("detection_gt_id"),
                        "gate": float(gate),
                        "loss_g0": float(1 - diou(box0, gt)),
                        "loss_gate": float(1 - diou(box_gate, gt)),
                        "loss_g1": float(1 - diou(box1, gt)),
                        "preferred": "g0" if diou(box0, gt) >= diou(box1, gt) else "g1",
                    }
                )
            if not rows:
                continue
            report[sequence].append(
                {
                    "start": start,
                    "end": end,
                    "events": len(rows),
                    "prefers_g0": sum(1 for row in rows if row["preferred"] == "g0"),
                    "prefers_g1": sum(1 for row in rows if row["preferred"] == "g1"),
                    "mean_loss_g0": float(np.mean([row["loss_g0"] for row in rows])),
                    "mean_loss_gate": float(
                        np.mean([row["loss_gate"] for row in rows])
                    ),
                    "mean_loss_g1": float(np.mean([row["loss_g1"] for row in rows])),
                    "gate_matches_best": sum(
                        1
                        for row in rows
                        if (row["gate"] < 0.5 and row["preferred"] == "g0")
                        or (row["gate"] >= 0.5 and row["preferred"] == "g1")
                    ),
                    "rows": rows,
                }
            )
    out = PILOT / "key_diou_summary.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for sequence, windows in report.items():
        for window in windows:
            print(
                sequence,
                f"{window['start']}-{window['end']}",
                "n=",
                window["events"],
                "g0/g1 preferred=",
                window["prefers_g0"],
                window["prefers_g1"],
                "mean d0/gate/d1=",
                round(window["mean_loss_g0"], 4),
                round(window["mean_loss_gate"], 4),
                round(window["mean_loss_g1"], 4),
                "gate_matches_best=",
                window["gate_matches_best"],
            )
    print("saved", out)


if __name__ == "__main__":
    main()
