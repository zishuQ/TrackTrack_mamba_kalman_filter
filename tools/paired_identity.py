"""Conservative training-only labels for two established tracks crossing."""

from collections import defaultdict
from itertools import combinations

from sportsmot_supervision_pilot import iou_xyxy


def recover_identity_labels(pilot):
    by_frame = defaultdict(dict)
    for record in pilot.events:
        by_frame[record["frame_t"]][record["track_id"]] = record
    streak = {}
    established = {}
    active = {}
    recovered = []
    for frame in sorted(by_frame):
        rows = by_frame[frame]
        boxes = pilot.gt.get(frame, {})
        observed = {}
        for tid, row in rows.items():
            scores = sorted(((iou_xyxy(row["detection_box"], box), gid)
                             for gid, box in boxes.items()), reverse=True)
            if scores and scores[0][0] >= 0.5 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.1):
                observed[tid] = scores[0][1]

        # A GT interruption invalidates every piece of identity evidence that
        # depends on that GT, including frames without write events.  After
        # recovery the streak/establishment rules must be satisfied again.
        def gt_missing_since(gt_id, after_frame):
            return any(
                gt_id not in pilot.gt.get(f, {})
                for f in range(after_frame + 1, frame + 1)
            )

        for tid, (gt_id, est_frame) in list(established.items()):
            if gt_missing_since(int(gt_id), est_frame):
                del established[tid]
        for tid, (gt_id, streak_frame, _count) in list(streak.items()):
            if gt_missing_since(int(gt_id), streak_frame):
                streak.pop(tid, None)

        # Establish both identities before entry. Check that the two people
        # recently overlapped and that two distinct, unambiguous detections
        # now support a reciprocal exchange.
        fresh = [tid for tid in observed if tid in established and frame - established[tid][1] <= 12]
        for left, right in combinations(sorted(fresh), 2):
            a, b = established[left][0], established[right][0]
            if a == b or observed[left] != b or observed[right] != a:
                continue
            pair = (left, right)
            if pair in active or any(left in key or right in key for key in active):
                continue
            recent = [pilot.gt.get(f, {}) for f in range(max(1, frame - 12), frame + 1)]
            if any(a in gt and b in gt and iou_xyxy(gt[a], gt[b]) >= 0.1 for gt in recent):
                active[pair] = (a, b, frame, frame)

        for (left, right), (a, b, start, last) in list(active.items()):
            pair = left, right
            # The anchor may not be extended across any real frame in which
            # either person's GT is missing, even if no write event exists on
            # that frame.  Reappearance must re-accumulate evidence first.
            gt_broken = any(
                a not in pilot.gt.get(f, {}) or b not in pilot.gt.get(f, {})
                for f in range(last + 1, frame + 1)
            )
            if (gt_broken or frame - start >= 64 or frame - last > 5 or a not in boxes
                or b not in boxes
                or any(t in observed and observed[t] != other for t, other in ((left, b), (right, a)))
                or any(t in rows and rows[t]["target_gt_before"] not in (None, own)
                       for t, own in ((left, a), (right, b)))):
                del active[pair]
                continue
            if observed.get(left) != b or observed.get(right) != a:
                continue
            if sum(identity in (a, b) for identity in observed.values()) != 2:
                continue
            active[pair] = (a, b, start, frame)
            for tid, original in ((left, a), (right, b)):
                row = rows[tid]
                if row["target_gt_before"] is None:
                    row["target_gt_before"] = original
                    row["anchor_identity"] = True
                    pilot._compute_motion_endpoints(row)
                    recovered.append((frame, tid, original))

        for tid, record in rows.items():
            gt = observed.get(tid)
            if gt is None or record["target_gt_before"] != gt or record.get("anchor_identity"):
                streak.pop(tid, None)
                continue
            old_gt, old_frame, count = streak.get(tid, (None, 0, 0))
            count = count + 1 if old_gt == gt and old_frame == frame - 1 else 1
            streak[tid] = (gt, frame, count)
            if count >= 3:
                established[tid] = (gt, frame)
    return recovered
