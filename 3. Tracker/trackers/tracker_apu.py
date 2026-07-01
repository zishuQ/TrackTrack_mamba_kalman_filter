import numpy as np

from trackers.cmc import CMC, apply_cmc
from trackers.track import TrackCounter, TrackState
from trackers.track_apu import TrackAPU
from trackers.utils import (
    angle_distance,
    associate,
    conf_distance,
    find_deleted_detections,
    iou_distance,
    track_aware_nms,
)


class TrackerAPU(object):
    def __init__(self, args, vid_name, apu_adapter):
        self.args = args
        self.max_time_lost = args.max_time_lost
        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()
        self.cmc = CMC(vid_name)
        self.disable_gmc = getattr(args, "disable_gmc", False)
        self.apu_adapter = apu_adapter

    def init_tracks(self, dets):
        tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])
        allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)
        for idx, flag in enumerate(allow_indices):
            if flag:
                dets[idx].initiate(self.frame_id, self.counter, self.apu_adapter)
                self.tracks.append(dets[idx])

    def _assignment(self, tracks, dets_high, dets_low, dets_del_high):
        return iterative_assignment_apu(
            tracks,
            dets_high,
            dets_low,
            dets_del_high,
            self.args.match_thr,
            self.args.penalty_p,
            self.args.penalty_q,
            self.args.reduce_step,
            self.frame_id,
            self.apu_adapter,
            no_reid=getattr(self.args, "no_reid", False),
        )

    def update(self, dets, dets_95):
        self.frame_id += 1
        dets_del = find_deleted_detections(dets, dets_95)
        dets = [TrackAPU(self.args, d) for d in dets]
        dets_del = [TrackAPU(self.args, d) for d in dets_del]

        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        warp_matrix = self.cmc.get_warp_matrix()
        if not self.disable_gmc:
            apply_cmc(tracked_lost, warp_matrix)
            apply_cmc(new, warp_matrix)

        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]
        self.apu_adapter.predict_tracks(tracked_lost + new)

        dets_all = dets_high + dets_low + dets_del_high
        matches, u_tracks, u_dets = self._assignment(tracked_lost, dets_high, dets_low, dets_del_high)
        for t, d in matches:
            tracked_lost[t].update(self.frame_id, dets_all[d], self.apu_adapter)
        for t in u_tracks:
            tracked_lost[t].mark_lost()

        dets_high_left = [dets_all[i] for i in u_dets if i < len(dets_high)]
        matches, u_tracks, u_dets = self._assignment(new, dets_high_left, [], [])
        for t, d in matches:
            new[t].update(self.frame_id, dets_high_left[d], self.apu_adapter)
        for t in u_tracks:
            new[t].mark_removed()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        self.init_tracks([dets_high_left[udx] for udx in u_dets])
        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        self.frame_id += 1
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]
        warp_matrix = self.cmc.get_warp_matrix()
        if not self.disable_gmc:
            apply_cmc(self.tracks, warp_matrix)
        [t.predict() for t in self.tracks]
        self.apu_adapter.predict_tracks(self.tracks)
        for t in self.tracks:
            t.mark_lost()
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        return []


def iterative_assignment_apu(
    tracks,
    dets_high,
    dets_low,
    dets_del_high,
    match_thr,
    penalty_p,
    penalty_q,
    reduce_step,
    frame_id,
    apu_adapter,
    d_t=3,
    no_reid=False,
):
    matches = []
    dets = dets_high + dets_low + dets_del_high
    iou_sim, iou_dist = iou_distance(tracks, dets)
    if no_reid:
        cost = iou_dist + 0.10 * conf_distance(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, d_t)
    else:
        app_dist = apu_adapter.appearance_cost(tracks, dets)
        cost = 0.50 * iou_dist + 0.50 * app_dist
        cost += 0.10 * conf_distance(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, d_t)

    cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low):] += penalty_q
    cost[iou_sim <= 0.10] = 1.0
    cost = np.clip(cost, 0, 1)

    while True:
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step
        if len(matches_) == 0:
            break
        matches += matches_
        for t, d in matches:
            cost[t, :] = 1.0
            cost[:, d] = 1.0

    m_tracks = set(t for t, _ in matches)
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = set(d for _, d in matches)
    u_dets = [d for d in range(len(dets)) if d not in m_dets]
    return matches, u_tracks, u_dets
