import torch
import numpy as np
from trackers.cmc import *
from trackers.utils import *
from trackers.track_mamba import *


class TrackerMamba(object):
    def __init__(self, args, vid_name):
        # Initialize
        self.args = args
        self.max_time_lost = args.max_time_lost

        # Initialize
        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()

        # Set global motion compensation model
        self.cmc = CMC(vid_name)
        
        # Initialize shared Kalman filter (MambaKalmanFilter)
        self.shared_kalman_filter = None
        
        # GPU cache: predicted states from batch_predict, reused in batch_update
        # to skip redundant CPU→GPU transfers for matched tracks
        self._predicted_gpu_cache = None  # (means_gpu, covs_gpu, {track_id: idx})

    def _batch_predict(self, tracks):
        """
        Batch predict all tracks' next states (optimized: single GPU transfer).
        
        Caches GPU tensors internally so that subsequent _batch_update calls
        can reuse them, avoiding a redundant CPU→GPU round-trip.
        
        Args:
            tracks: List of TrackMamba objects to predict
        """
        if len(tracks) == 0:
            self._predicted_gpu_cache = None
            return
        
        # Handle DanceTrack special case: zero out velocity for non-tracked
        for t in tracks:
            if t.state != TrackState.Tracked and 'Dance' in self.args.data_path:
                t.mean[6] = 0
                t.mean[7] = 0
        
        # Collect all track states into numpy arrays
        means = np.stack([t.mean for t in tracks], axis=0)  # (N, 8)
        covariances = np.stack([t.covariance for t in tracks], axis=0)  # (N, 8, 8)
        
        # Collect measurements (used for computing innovation and DIoU)
        measurements = []
        for t in tracks:
            meas = t.last_observation if t.last_observation is not None else t.mean[:4].copy()
            measurements.append(meas)
        measurements = np.stack(measurements, axis=0)  # (N, 4)
        
        track_ids = [t.track_id for t in tracks]
        
        # Transfer to GPU once
        device = self.shared_kalman_filter.device
        means_gpu = torch.from_numpy(means).float().to(device)
        covs_gpu = torch.from_numpy(covariances).float().to(device)
        meas_gpu = torch.from_numpy(measurements).float().to(device)
        
        # Predict on GPU (returns GPU tensors, no CPU transfer inside)
        with torch.no_grad():
            means_pred_gpu, covs_pred_gpu = self.shared_kalman_filter.batch_predict_gpu(
                means_gpu, covs_gpu, meas_gpu, track_ids
            )
        
        # Transfer results to CPU and update tracks
        means_np = means_pred_gpu.cpu().numpy()
        covs_np = covs_pred_gpu.cpu().numpy()
        for i, t in enumerate(tracks):
            t.mean = means_np[i]
            t.covariance = covs_np[i]
        
        # Cache GPU tensors for subsequent _batch_update (skip CPU→GPU round-trip)
        self._predicted_gpu_cache = (
            means_pred_gpu,   # (N, 8) on GPU, pixel coords
            covs_pred_gpu,    # (N, 8, 8) on GPU
            {t.track_id: i for i, t in enumerate(tracks)},
        )

    def _batch_update(self, track_det_pairs):
        """
        Batch update all matched track-detection pairs (optimized: reuses GPU cache).
        
        If _batch_predict was called earlier in this frame, matched tracks' states
        are indexed from the cached GPU tensors directly (no CPU→GPU transfer needed).
        
        Args:
            track_det_pairs: List of (track, detection) tuples
        """
        if len(track_det_pairs) == 0:
            return
        
        tracks = [pair[0] for pair in track_det_pairs]
        detections = [pair[1] for pair in track_det_pairs]
        
        track_ids = [t.track_id for t in tracks]
        measurements = np.stack([d.cxcywh for d in detections], axis=0)  # (N, 4)
        
        device = self.shared_kalman_filter.device
        meas_gpu = torch.from_numpy(measurements).float().to(device)
        
        # Try to reuse cached GPU tensors from _batch_predict (skip CPU→GPU)
        cache = self._predicted_gpu_cache
        use_cache = False
        if cache is not None:
            cached_means, cached_covs, tid_map = cache
            try:
                indices = [tid_map[t.track_id] for t in tracks]
                means_gpu = cached_means[indices]  # GPU indexing, no CPU transfer
                covs_gpu = cached_covs[indices]
                use_cache = True
            except KeyError:
                pass
        
        if not use_cache:
            # Fallback: transfer from CPU (e.g. if predict cache was invalidated)
            means = np.stack([t.mean for t in tracks], axis=0)
            covariances = np.stack([t.covariance for t in tracks], axis=0)
            means_gpu = torch.from_numpy(means).float().to(device)
            covs_gpu = torch.from_numpy(covariances).float().to(device)
        
        # Update on GPU (returns GPU tensors)
        with torch.no_grad():
            means_upd_gpu, covs_upd_gpu = self.shared_kalman_filter.batch_update_gpu(
                means_gpu, covs_gpu, meas_gpu, track_ids
            )
        
        # Transfer results to CPU
        means_np = means_upd_gpu.cpu().numpy()
        covs_np = covs_upd_gpu.cpu().numpy()
        for i, t in enumerate(tracks):
            t.mean = means_np[i]
            t.covariance = covs_np[i]

    def init_tracks(self, dets):
        # Get alive tracks, iou_similarity, and scores
        tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])

        # Run track aware NMS
        allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)

        for idx, flag in enumerate(allow_indices):
            if flag:
                dets[idx].initiate(self.frame_id, self.counter, self.shared_kalman_filter)
                self.tracks.append(dets[idx])

    def update(self, dets, dets_95):
        # ==============================================================================================================
        # Update frame id
        self.frame_id += 1

        # Get deleted detections &  Encode
        dets_del = find_deleted_detections(dets, dets_95)
        dets = [TrackMamba(self.args, d) for d in dets]
        dets_del = [TrackMamba(self.args, d) for d in dets_del]

        # Divide detections
        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        # Split tracks
        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(tracked_lost, warp_matrix)
        apply_cmc(new, warp_matrix)

        # Predict the current location with KF (OPTIMIZED: batch operation)
        self._batch_predict(tracked_lost + new)

        # ==============================================================================================================
        # Association between (tracked and lost tracks) & (high confidence detections)
        dets = dets_high + dets_low + dets_del_high
        matches, u_tracks, u_dets = iterative_assignment(tracked_lost, dets_high, dets_low, dets_del_high,
                                                         self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        # Update matched tracks (OPTIMIZED: batch KF update + individual attribute updates)
        matched_pairs = [(tracked_lost[t], dets[d]) for t, d in matches]
        self._batch_update(matched_pairs)
        for t, d in matches:
            tracked_lost[t].update_after_kf(self.frame_id, dets[d])

        # Mark "lost" to unmatched tracks
        for t in u_tracks:
            tracked_lost[t].mark_lost()

        # ==============================================================================================================
        # Get remained high confidence detections
        dets_high_left = [dets[i] for i in u_dets if i < len(dets_high)]

        # Association between (new tracks) & (left high confidence detections)
        matches, u_tracks, u_dets = iterative_assignment(new, dets_high_left, [], [], self.args.match_thr,
                                                         self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        # Update matched tracks (OPTIMIZED: batch KF update + individual attribute updates)
        matched_pairs_new = [(new[t], dets_high_left[d]) for t, d in matches]
        self._batch_update(matched_pairs_new)
        for t, d in matches:
            new[t].update_after_kf(self.frame_id, dets_high_left[d])

        # Mark "remove" to unmatched tracks
        for t in u_tracks:
            new[t].mark_removed()

        # ==============================================================================================================
        # Mark "remove" lost tracks which are too old and add to finished
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks and clean up Mamba states
        removed_tracks = [t for t in self.tracks if t.state == TrackState.Removed]
        for track in removed_tracks:
            self.shared_kalman_filter.delete_track(track.track_id)
        
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Init new tracks
        self.init_tracks([dets_high_left[udx] for udx in u_dets])

        # Clear GPU cache (free memory, no longer needed this frame)
        self._predicted_gpu_cache = None

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        # Update frame id
        self.frame_id += 1

        # Only maintain already tracked and new tracks, Drop all the new tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(self.tracks, warp_matrix)

        # Predict the current location with KF (OPTIMIZED: batch operation)
        self._batch_predict(self.tracks)

        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks and clean up Mamba states
        removed_tracks = [t for t in self.tracks if t.state == TrackState.Removed]
        for track in removed_tracks:
            self.shared_kalman_filter.delete_track(track.track_id)
        
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Clear GPU cache
        self._predicted_gpu_cache = None

        return [t for t in self.tracks if t.state == TrackState.Tracked]
