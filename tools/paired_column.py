"""Inspect a previous appearance write using the actual next-frame matching pool."""

import copy

import numpy as np

from sportsmot_supervision_pilot import SequencePilot, iou_xyxy, pairwise_gt_iou


def unique_detection_identities(dets, gt):
    identities = list(gt)
    if not identities:
        return [None] * len(dets)
    overlaps = pairwise_gt_iou([det.x1y1x2y2 for det in dets], list(gt.values()))
    best = overlaps.argmax(axis=1)
    top = overlaps[np.arange(len(dets)), best]
    if len(identities) > 1:
        second = np.partition(overlaps, -2, axis=1)[:, -2]
    else:
        second = np.full(len(dets), -np.inf)
    return [identities[index] if score >= 0.5 and score - rival >= 0.1 else None
            for index, score, rival in zip(best, top, second)]


def alternate_matches(meta, row_index, track, dets, threshold, p, q, step):
    """Recompute only the changed appearance row; keep the original matcher."""
    from trackers.utils import associate, cos_distance
    row = 0.50 * meta["iou_distance"][row_index] + 0.50 * cos_distance([track], dets)[0]
    row += (0.10 * meta["confidence_distance"][row_index]
            + 0.05 * meta["angle_distance"][row_index])
    sources = meta["detection_source"][row_index]
    row[sources == 1] += p
    row[sources == 2] += q
    row[meta["iou_similarity"][row_index] <= 0.10] = 1.0
    cost = meta["final_cost"].copy()
    cost[row_index] = np.clip(row, 0.0, 1.0)
    matches = []
    while True:
        selected = associate(cost, threshold)
        threshold -= step
        if not selected:
            break
        matches += selected
        for t, d in matches:
            cost[t, :] = 1.0
            cost[:, d] = 1.0
    return matches


class ColumnPilot(SequencePilot):
    def install_hooks(self):
        from trackers import tracker as tracker_module

        super().install_hooks()
        original = tracker_module.iterative_assignment
        latest = {}
        cursor = 0
        self.column_counts = {"writes": 0, "changed_assignment": 0, "actual_mismatch": 0}

        def assignment(tracks, high, low, extra, threshold, p, q, step, frame, **kwargs):
            nonlocal cursor
            result = original(tracks, high, low, extra, threshold, p, q, step, frame, **kwargs)
            if not kwargs.get("return_meta") or kwargs.get("no_reid") or not tracks:
                return result
            while cursor < len(self.events):
                event = self.events[cursor]
                latest[event["track_id"]] = event
                cursor += 1
            dets = high + low + extra
            matches, _, _, meta = result
            current = dict(matches)
            owners = [self.vote.resolve_before_current(self.sequence, t.track_id) for t in tracks]
            gt = self.gt.get(frame, {})
            identities = unique_detection_identities(dets, gt)
            for i, track in enumerate(tracks):
                rec = latest.get(track.track_id)
                if rec is None or rec["frame_t"] != frame - 1 or rec["appearance_gate"] is None:
                    continue
                own = rec["target_gt_before"]
                if own is None or owners[i] != own or identities.count(own) != 1:
                    continue
                j = identities.index(own)
                if meta["iou_similarity"][i,j] <= 0.1:
                    continue
                rivals = [k for k, identity in enumerate(owners)
                          if identity is not None and identity != own and meta["final_cost"][k,j] < threshold
                          and meta["iou_similarity"][k,j] > 0.1]
                if not rivals:
                    continue
                rival_cost = float(min(meta["final_cost"][k,j] for k in rivals))
                source = int(meta["detection_source"][i,j])
                offset = float(meta["raw_cost"][i,j] - 0.5 * meta["cosine_distance"][i,j])
                offset += (0.0, p, q)[source]
                before = np.asarray(rec["pre_feature"]).reshape(1,-1)
                obs = np.asarray(rec["det_feature"]).reshape(1,-1)
                score = float(rec["detection_score"])
                endpoints = []
                for gate in (0.0, 1.0, float(rec["appearance_gate"])):
                    clone = copy.copy(track)
                    feat = before + (1.0 - track.alpha) * score * gate * (obs - before)
                    clone.feat = feat / max(np.linalg.norm(feat), 1e-12) if gate else before
                    alternate = alternate_matches(meta, i, clone, dets, threshold, p, q, step)
                    endpoints.append(dict(alternate).get(i))
                self.column_counts["writes"] += 1
                if endpoints[2] != current.get(i):
                    self.column_counts["actual_mismatch"] += 1
                    continue
                self.column_counts["changed_assignment"] += int((endpoints[0] == j) != (endpoints[1] == j))
                rec["column_det"] = int(dets[j].frame_detection_index)
                rec["column_offset"] = offset
                rec["column_rival"] = rival_cost
            return result

        tracker_module.iterative_assignment = assignment
        self._restore.insert(0, (tracker_module, "iterative_assignment", original))
