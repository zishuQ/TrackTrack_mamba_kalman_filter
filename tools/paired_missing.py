"""Shared detector-visibility convention for offline gap supervision and evaluation."""

import numpy as np


def capture_detection_pool(pilot, adapter_class):
    from sportsmot_supervision_pilot import iou_xyxy

    pilot.visible = {}
    pilot.unique_gt = {}
    pilot.clear_visible = {}
    original = adapter_class.begin_frame

    def begin(adapter, frame_id, img_width, img_height, detection_pool=None, detection_sources=None):
        present, clear = set(), set()
        for det in detection_pool or ():
            ids = [gt for gt, box in pilot.gt.get(frame_id, {}).items()
                   if iou_xyxy(det.x1y1x2y2, box) >= 0.5]
            present.update(ids)
            if len(ids) == 1:
                pilot.unique_gt[(frame_id, int(det.frame_detection_index))] = ids[0]
                clear.add(ids[0])
        pilot.visible[frame_id] = present
        pilot.clear_visible[frame_id] = clear
        return original(adapter, frame_id, img_width, img_height, detection_pool, detection_sources)

    adapter_class.begin_frame = begin
    pilot._restore.insert(0, (adapter_class, "begin_frame", original))


def third_frame_targets(pilot, events):
    from sportsmot_supervision_pilot import mean_to_box, warp_predict

    boxes = np.zeros((len(events), 3, 4), dtype=np.float32)
    valid = np.zeros(len(events), dtype=np.bool_)
    for i, rec in enumerate(events):
        t, gt = rec["frame_t"], rec["target_gt_before"]
        if gt is None or pilot.unique_gt.get((t, rec["detection_index"])) != gt:
            continue
        future = range(t + 1, t + 4)
        if any(gt not in pilot.gt.get(f, {}) or f not in pilot.warps or
               gt in pilot.visible.get(f, set()) for f in future):
            continue
        prior = (rec["mean_prior"], rec["cov_prior"])
        full = pilot.kf.update(*[x.copy() for x in prior],
                               np.asarray(rec["detection_measurement"]), rec["detection_score"])
        for index, (mean, cov) in enumerate((prior, full)):
            for f in future:
                mean, cov = warp_predict(mean, cov, pilot.warps[f], pilot.kf)
            boxes[i, index] = mean_to_box(mean)
        boxes[i, 2] = pilot.gt[t + 3][gt]
        valid[i] = True
    return {"gap3_boxes": boxes, "gap3_valid": valid}
