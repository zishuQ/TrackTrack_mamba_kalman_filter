import lap
import numpy as np


# Compact event snapshots persist the last 6 observation boxes/scores.
COMPACT_SNAPSHOT_HISTORY = 6

# Runtime Track history is a bounded box/score window. It must cover:
#   * velocity / angle lookback (dt = 1, 2, 3) plus the current write
#   * conf_distance's second-most-recent score
#   * scalar-feature last-3 scores
#   * compact snapshots of the last 6 observations
# It is not the IWG context size (6 or 8). Age and maturity use
# observation_count, not this container length.
ONLINE_HISTORY_KEEP = 8


def chronological_history_frame_ids(history):
    """Return observation frame ids in increasing order.

    TrackTrack writes increasing frame ids and never back-fills, so dict
    insertion order already matches sorted order. Fall back to a sort only
    when a caller mutates keys out of order.
    """
    frame_ids = list(history)
    for index in range(1, len(frame_ids)):
        if frame_ids[index] < frame_ids[index - 1]:
            return sorted(history)
    return frame_ids


def recent_history_frame_ids(history, count):
    """Return the last ``count`` observation frame ids in chronological order."""
    frame_ids = chronological_history_frame_ids(history)
    if count is None or count >= len(frame_ids):
        return frame_ids
    return frame_ids[-count:]


def second_recent_history_score(history):
    """Score of the second-most-recent observation, or the only one if n == 1.

    Matches ``sorted(keys, reverse=True)[min(1, len-1)]`` without sorting a
    long history on the monotonic write path.
    """
    frame_ids = chronological_history_frame_ids(history)
    frame_id = frame_ids[-2] if len(frame_ids) > 1 else frame_ids[-1]
    return history[frame_id][1]


def track_observation_count(track):
    """Cumulative matched observations, independent of the bounded history window."""
    count = getattr(track, "observation_count", None)
    if count is None:
        return len(getattr(track, "history", {}) or {})
    return int(count)


def bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2):
    """Pairwise IoU with TrackTrack's inclusive ``+1`` pixel convention.

    Intersection is written only when width and height are both strictly
    positive; the rest stay 0. Areas are not clipped, matching the original
    loop, including inverted boxes. Output is always float64.
    """
    boxes_a = np.asarray(a_x1y1x2y2, dtype=np.float64)
    boxes_b = np.asarray(b_x1y1x2y2, dtype=np.float64)
    num_a = boxes_a.shape[0]
    num_b = boxes_b.shape[0]
    if num_a == 0 or num_b == 0:
        return np.zeros((num_a, num_b), dtype=np.float64)

    intersection_w = (
        np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        - np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        + 1.0
    )
    intersection_h = (
        np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        - np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        + 1.0
    )
    overlap_mask = (intersection_w > 0.0) & (intersection_h > 0.0)
    intersection = np.where(overlap_mask, intersection_w * intersection_h, 0.0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0] + 1.0) * (boxes_a[:, 3] - boxes_a[:, 1] + 1.0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0] + 1.0) * (boxes_b[:, 3] - boxes_b[:, 1] + 1.0)
    union = area_a[:, None] + area_b[None, :] - intersection

    overlaps = np.zeros((num_a, num_b), dtype=np.float64)
    np.divide(intersection, union, out=overlaps, where=overlap_mask)
    return overlaps


def find_deleted_detections(dets, dets_95, source_indices=None, return_indices=False):
    # Get boxes
    a_x1y1x2y2 = np.ascontiguousarray(dets[:, :4], dtype=np.float64)
    b_x1y1x2y2 = np.ascontiguousarray(dets_95[:, :4], dtype=np.float64)

    # Calculate IoU
    ious = bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2)

    # Find deleted detections
    keep_mask = np.max(ious, axis=0) < 0.97
    dets_del = dets_95[keep_mask]

    if return_indices:
        if source_indices is None:
            return dets_del, np.flatnonzero(keep_mask)
        return dets_del, np.asarray(source_indices, dtype=np.int64)[keep_mask]
    return dets_del


def iou_distance(a_tracks, b_tracks):
    # Get boxes
    a_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU distance
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        iou_sim = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float64)
        iou_dist = 1 - iou_sim
    else:
        # Calculate HIoU
        h_iou = (np.minimum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.maximum(a_boxes[:, 1:2], b_boxes[:, 1:2].T))
        h_iou /= (np.maximum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.minimum(a_boxes[:, 1:2], b_boxes[:, 1:2].T))

        # Calculate HMIoU
        iou_sim = bbox_overlaps(a_boxes, b_boxes)
        iou_dist = 1 - h_iou * iou_sim

    return iou_sim, iou_dist


def cos_distance(tracks, dets):
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Calculate cosine distance
    t_feat = np.concatenate([t.feat for t in tracks], axis=0)
    d_feat = np.concatenate([d.feat for d in dets], axis=0)
    cos_dist = np.clip(1 - np.dot(t_feat, d_feat.T), a_min=0., a_max=1.)

    return cos_dist


def conf_distance(tracks, dets):
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get previous scores (second-most-recent observation; no full-history sort)
    t_score_prev = [second_recent_history_score(t.history) for t in tracks]

    # Linear projection
    t_score_prev = np.array(t_score_prev)
    t_score = np.array([t.score for t in tracks])
    t_score += (t_score - t_score_prev)

    # Calculate confidence similarity
    d_score = np.array([d.score for d in dets])
    conf_dist = np.abs(t_score[:, None] - d_score[None, :])

    return conf_dist


def get_prev_box(history, frame_id, dt):
    # Prefer the exact frame_id - dt observation when it still exists.
    target_key = frame_id - dt
    if target_key in history:
        return history[target_key][0]

    # If there is no observation at that offset, use the latest remaining one.
    return history[max(history)][0]


def get_vel_t_d(b_1, b_2):
    # Expand boxes
    b_1, b_2 = b_1[:, np.newaxis, :], b_2[np.newaxis, :, :]

    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_lb = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 3:4]**2) + 1e-5
    norm_rt = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_rb = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 3:4]**2) + 1e-5

    # Get velocities
    vel_lt = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_lt
    vel_lb = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 3] - b_1[:, :, 3]], axis=-1) / norm_lb
    vel_rt = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_rt
    vel_rb = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 3] - b_1[:, :, 3]], axis=-1) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=2)


def calc_angle(vel_t, vel_t_d):
    angle_ = 0
    for vdx in range(vel_t.shape[2]):
        # Divide & Repeat
        vel_t_x = np.repeat(vel_t[:, :, vdx, 0], vel_t_d.shape[1], axis=1)
        vel_t_y = np.repeat(vel_t[:, :, vdx, 1], vel_t_d.shape[1], axis=1)

        # Calculate angle, Normalize to range (0 ~ 1)
        angle = vel_t_x * vel_t_d[:, :, vdx, 0] + vel_t_y * vel_t_d[:, :, vdx, 1]
        angle = np.abs(np.arccos(np.clip(angle, a_min=-1, a_max=1))) / np.pi
        angle_ += angle / 4

    return angle_


def angle_distance(tracks, dets, frame_id, d_t=3):
    # Initialization
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get velocity between track and detections
    track_boxes = np.stack([get_prev_box(t.history, frame_id, d_t) for t in tracks], axis=0)
    vel_t_d = get_vel_t_d(track_boxes, np.stack([d.x1y1x2y2 for d in dets], axis=0))

    # Get angle distance
    angle_dist = calc_angle(np.stack([t.velocity for t in tracks], axis=0)[:, np.newaxis], vel_t_d)

    # Fuse score
    scores = np.array([d.score for d in dets])[np.newaxis, :]
    angle_dist *= scores

    return angle_dist


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def associate(cost, match_thr):
    # Initialization
    matches = []

    # Run
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(cost, axis=1)
        min_tdx = np.argmin(cost, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < match_thr:
                matches.append([tdx, ddx])

    return matches


def iterative_assignment(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q,
                        reduce_step, frame_id, d_t=3, no_reid=False, return_meta=False):
    # Initialization
    matches = []
    dets = dets_high + dets_low + dets_del_high

    # Calculate preliminaries
    iou_sim, iou_dist = iou_distance(tracks, dets)

    # Calculate confidence and angle distance (compute once for both branches)
    conf_dist_arr = conf_distance(tracks, dets)
    angle_dist_arr = angle_distance(tracks, dets, frame_id, d_t)

    # Calculate cost
    cos_dist = None
    if no_reid:
        cost = iou_dist + 0.10 * conf_dist_arr + 0.05 * angle_dist_arr
    else:
        cos_dist = cos_distance(tracks, dets)
        cost = 0.50 * iou_dist + 0.50 * cos_dist
        cost += 0.10 * conf_dist_arr + 0.05 * angle_dist_arr

    # Build association metadata (capture before any modification)
    if return_meta:
        iou_sim_meta = iou_sim.copy()
        iou_dist_meta = iou_dist.copy()
        cos_dist_meta = cos_dist.copy() if cos_dist is not None else np.zeros_like(iou_dist)
        conf_dist_meta = conf_dist_arr.copy()
        angle_dist_meta = angle_dist_arr.copy()
        raw_cost = cost.copy()

    # Give penalty
    cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low):] += penalty_q

    # Constraint & Clip
    cost[iou_sim <= 0.10] = 1.
    cost = np.clip(cost, 0, 1)

    # Capture final cost before iteration begins
    if return_meta:
        final_cost = cost.copy()

        # Build detection source matrix
        num_tracks = len(tracks)
        num_dets = len(dets)
        detection_source = np.full((num_tracks, num_dets), -1, dtype=int)
        n_high = len(dets_high)
        detection_source[:, :n_high] = 0
        detection_source[:, n_high:n_high + len(dets_low)] = 1
        detection_source[:, n_high + len(dets_low):] = 2

        # Initialize assignment tracking
        assignment_round = np.full((num_tracks, num_dets), -1, dtype=int)
        assignment_threshold = np.full((num_tracks, num_dets), -1.0, dtype=np.float64)

    # Match
    round_idx = 0
    while True:
        # Match tracks with detections
        matches_ = associate(cost, match_thr)
        current_threshold = match_thr
        match_thr -= reduce_step

        # Check (if there are no more matchable pairs)
        if len(matches_) == 0:
            break

        # Track assignment metadata for this round
        if return_meta:
            for t, d in matches_:
                assignment_round[t, d] = round_idx
                assignment_threshold[t, d] = current_threshold

        # Append
        matches += matches_
        round_idx += 1

        # Update cost matrix
        for t, d in matches:
            cost[t, :] = 1.
            cost[:, d] = 1.

    # Find indices of unmatched tracks and detections
    m_tracks = set(t for t, _ in matches)
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = set(d for _, d in matches)
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    if return_meta:
        association_meta = {
            "iou_similarity": iou_sim_meta,
            "iou_distance": iou_dist_meta,
            "cosine_distance": cos_dist_meta,
            "confidence_distance": conf_dist_meta,
            "angle_distance": angle_dist_meta,
            "raw_cost": raw_cost,
            "final_cost": final_cost,
            "assignment_round": assignment_round,
            "assignment_threshold": assignment_threshold,
            "detection_source": detection_source,
        }
        return matches, u_tracks, u_dets, association_meta

    return matches, u_tracks, u_dets


def track_aware_nms(pair_sims, scores, num_tracks, nms_thresh, score_thresh):
    # Initialization
    num_dets = len(pair_sims) - num_tracks
    allow_indices = np.ones(num_dets) * (scores > score_thresh)

    # Run
    for idx in range(num_dets):
        # Check 1
        if allow_indices[idx] == 0:
            continue

        # Check 2
        if num_tracks > 0:
            if np.max(pair_sims[num_tracks + idx, :num_tracks]) > nms_thresh:
                allow_indices[idx] = 0
                continue

        # Check 3
        for jdx in range(num_dets):
            if idx != jdx and allow_indices[jdx] == 1 and scores[idx] > scores[jdx]:
                if pair_sims[num_tracks + idx, num_tracks + jdx] > nms_thresh:
                    allow_indices[jdx] = 0

    return allow_indices == 1
