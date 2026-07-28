import torch
import numpy as np
from trackers.cmc import *
from trackers.utils import *
from trackers.track_mamba import *

try:
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter
    from agentguard.contracts.outputs import GateDecision
    _AGENTGUARD_AVAILABLE = True
except ImportError:
    _AGENTGUARD_AVAILABLE = False


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
        self.disable_gmc = getattr(args, 'disable_gmc', False)
        
        # Initialize shared Kalman filter (MambaKalmanFilter)
        self.shared_kalman_filter = None
        
        # GPU cache: predicted states from batch_predict, reused in batch_update
        # to skip redundant CPU→GPU transfers for matched tracks
        self._predicted_gpu_cache = None  # (means_gpu, covs_gpu, {track_id: idx})

        # Native Mamba events are captured for offline training only. Online
        # AgentGuard inference stays attached to the NSA tracker.
        self.agentguard_adapter = None
        if _AGENTGUARD_AVAILABLE and getattr(args, 'agentguard_mode', 'off') == 'capture':
            from agentguard.runtime.manager import AgentGuardRuntime

            runtime = AgentGuardRuntime({"mode": "capture"}, device="cpu")
            runtime.event_sink = getattr(args, 'event_sink', None)
            self.agentguard_adapter = AgentGuardTrackerAdapter(
                args, vid_name, agentguard_runtime=runtime
            )

    def _initialize_capture_features(self, detections):
        if not self.agentguard_adapter or not self.agentguard_adapter.capture_only:
            return
        runtime = self.agentguard_adapter.runtime
        if runtime is None or runtime.feature_builder is not None:
            return
        first_det = next(iter(detections), None)
        if first_det is None or getattr(first_det, "feat", None) is None:
            return
        reid_dim = int(np.asarray(first_det.feat).reshape(-1).shape[0])
        runtime.init_feature_builder(reid_dim=reid_dim)
        if self.agentguard_adapter.event_sink is not None:
            self.agentguard_adapter.event_sink._reid_dim = reid_dim

    def _populate_capture_features(self, event, track, detection=None):
        runtime = self.agentguard_adapter.runtime
        feature_builder = runtime.feature_builder if runtime is not None else None
        if feature_builder is None:
            return
        source = event.pre_update_state or event.frame_start_state
        event.track_feature = np.asarray(
            source.feature if source is not None else track.feat,
            dtype=np.float64,
        ).reshape(-1)
        if detection is None:
            event.detection_feature = np.zeros(
                feature_builder.reid_dim, dtype=np.float64
            )
        else:
            event.detection_feature = np.asarray(
                detection.feat, dtype=np.float64
            ).reshape(-1)
        event.scalar_features = feature_builder.compute_scalar(event)

    def _batch_predict(self, tracks, force_missing=False):
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
        
        # Q-net uses the last real observation, matching the training-time predict input.
        measurements = []
        for t in tracks:
            last_obs = t.last_observation if t.last_observation is not None else t.mean[:4].copy()
            measurements.append(last_obs.copy())
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
        self.frame_id += 1

        target_detection_indices = getattr(
            self.args, "agentguard_target_detection_indices", None
        )
        source_detection_indices = getattr(
            self.args, "agentguard_source_detection_indices", None
        )
        if source_detection_indices is not None:
            dets_del, dets_del_indices = find_deleted_detections(
                dets,
                dets_95,
                source_indices=source_detection_indices,
                return_indices=True,
            )
        else:
            dets_del = find_deleted_detections(dets, dets_95)
            dets_del_indices = None
        dets = [TrackMamba(self.args, detection) for detection in dets]
        dets_del = [TrackMamba(self.args, detection) for detection in dets_del]
        if target_detection_indices is not None:
            for detection, detection_index in zip(
                dets, target_detection_indices
            ):
                detection.frame_detection_index = int(detection_index)
        else:
            for detection_index, detection in enumerate(dets):
                detection.frame_detection_index = int(detection_index)
        if dets_del_indices is not None:
            for detection, detection_index in zip(
                dets_del, dets_del_indices
            ):
                detection.frame_detection_index = int(detection_index)

        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]
        detection_pool = dets_high + dets_low + dets_del_high

        self._initialize_capture_features(detection_pool)
        if self.agentguard_adapter:
            self.agentguard_adapter.begin_frame(
                self.frame_id,
                getattr(self.args, 'img_w', 1920),
                getattr(self.args, 'img_h', 1080),
                detection_pool=detection_pool,
                detection_sources=[0] * len(dets_high)
                + [1] * len(dets_low)
                + [2] * len(dets_del_high),
            )

        tracked_lost = [
            track
            for track in self.tracks
            if track.state in (TrackState.Tracked, TrackState.Lost)
        ]
        new = [track for track in self.tracks if track.state == TrackState.New]

        frame_start_snapshots = {}
        if self.agentguard_adapter:
            runtime = self.agentguard_adapter.runtime
            for track in tracked_lost:
                if runtime and runtime.is_mature_track(track):
                    frame_start_snapshots[track.track_id] = track.snapshot_state(
                        compact_history=True
                    )

        if self.disable_gmc:
            effective_warp = np.eye(2, 3, dtype=np.float64)
        else:
            effective_warp = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, effective_warp)
            apply_cmc(new, effective_warp)
        if self.agentguard_adapter:
            self.agentguard_adapter.set_frame_warp(effective_warp)

        no_current_detections = not detection_pool
        self._batch_predict(
            tracked_lost + new, force_missing=no_current_detections
        )

        pre_update_snapshots = {}
        if self.agentguard_adapter:
            runtime = self.agentguard_adapter.runtime
            for track in tracked_lost:
                if runtime and runtime.is_mature_track(track):
                    pre_update_snapshots[track.track_id] = track.snapshot_state(
                        compact_history=True
                    )

        use_meta = self.agentguard_adapter is not None
        result = iterative_assignment(
            tracked_lost,
            dets_high,
            dets_low,
            dets_del_high,
            self.args.match_thr,
            self.args.penalty_p,
            self.args.penalty_q,
            self.args.reduce_step,
            self.frame_id,
            no_reid=getattr(self.args, 'no_reid', False),
            return_meta=use_meta,
        )
        if use_meta:
            matches, u_tracks, u_dets, association_meta = result
            self.agentguard_adapter.set_association_record(
                track_ids=[track.track_id for track in tracked_lost],
                detection_pool=detection_pool,
                association_meta=association_meta,
                no_reid=getattr(self.args, 'no_reid', False),
            )
        else:
            matches, u_tracks, u_dets = result
            association_meta = None

        captured_matches = {}
        if self.agentguard_adapter and association_meta is not None:
            runtime = self.agentguard_adapter.runtime
            for track_index, detection_index in matches:
                track = tracked_lost[track_index]
                if runtime and runtime.is_mature_track(track):
                    detection = detection_pool[detection_index]
                    event = self.agentguard_adapter.build_matched_event(
                        track,
                        detection,
                        track_index,
                        detection_index,
                        association_meta,
                        effective_warp,
                        frame_start_snapshots.get(track.track_id),
                        pre_update_snapshots.get(track.track_id),
                        num_tracks=len(tracked_lost),
                        num_detections=len(detection_pool),
                        no_reid=getattr(self.args, 'no_reid', False),
                    )
                    self._populate_capture_features(event, track, detection)
                    captured_matches[track.track_id] = event

        matched_pairs = [
            (tracked_lost[t], detection_pool[d]) for t, d in matches
        ]
        self._batch_update(matched_pairs)
        for track_index, detection_index in matches:
            track = tracked_lost[track_index]
            track.update_after_kf(
                self.frame_id, detection_pool[detection_index]
            )
            event = captured_matches.get(track.track_id)
            if event is not None:
                decision = GateDecision(
                    1.0,
                    1.0,
                    np.ones(5, dtype=np.float64) / 5.0,
                )
                self.agentguard_adapter.record_event(
                    track.track_id, event, decision
                )

        for track_index in u_tracks:
            track = tracked_lost[track_index]
            if self.agentguard_adapter:
                runtime = self.agentguard_adapter.runtime
                if runtime and runtime.is_mature_track(track):
                    event = self.agentguard_adapter.build_unmatched_event(
                        track,
                        effective_warp,
                        frame_start_snapshots.get(track.track_id),
                        pre_update_snapshots.get(track.track_id),
                    )
                    self._populate_capture_features(event, track)
                    self.agentguard_adapter.record_unmatched_event(
                        track.track_id, event
                    )
            track.mark_lost()

        if self.agentguard_adapter:
            self.agentguard_adapter.end_frame()

        dets_high_left = [
            detection_pool[index]
            for index in u_dets
            if index < len(dets_high)
        ]
        matches, u_tracks, u_dets = iterative_assignment(
            new,
            dets_high_left,
            [],
            [],
            self.args.match_thr,
            self.args.penalty_p,
            self.args.penalty_q,
            self.args.reduce_step,
            self.frame_id,
            no_reid=getattr(self.args, 'no_reid', False),
        )

        matched_pairs_new = [
            (new[t], dets_high_left[d]) for t, d in matches
        ]
        self._batch_update(matched_pairs_new)
        for track_index, detection_index in matches:
            new[track_index].update_after_kf(
                self.frame_id, dets_high_left[detection_index]
            )

        for track_index in u_tracks:
            new[track_index].mark_removed()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        removed_tracks = [
            track
            for track in self.tracks
            if track.state == TrackState.Removed
        ]
        for track in removed_tracks:
            if self.agentguard_adapter:
                self.agentguard_adapter.remove_track(track.track_id)
            self.shared_kalman_filter.delete_track(track.track_id)

        self.tracks = [
            track
            for track in self.tracks
            if track.state != TrackState.Removed
        ]
        self.init_tracks([dets_high_left[index] for index in u_dets])
        self._predicted_gpu_cache = None
        return [
            track
            for track in self.tracks
            if track.state == TrackState.Tracked
        ]

    def update_without_detections(self):
        self.frame_id += 1

        if self.agentguard_adapter:
            self.agentguard_adapter.begin_frame(
                self.frame_id,
                getattr(self.args, 'img_w', 1920),
                getattr(self.args, 'img_h', 1080),
            )

        self.tracks = [
            track for track in self.tracks if track.state != TrackState.New
        ]
        frame_start_snapshots = {}
        if self.agentguard_adapter:
            runtime = self.agentguard_adapter.runtime
            for track in self.tracks:
                if runtime and runtime.is_mature_track(track):
                    frame_start_snapshots[track.track_id] = track.snapshot_state(
                        compact_history=True
                    )

        if self.disable_gmc:
            effective_warp = np.eye(2, 3, dtype=np.float64)
        else:
            effective_warp = self.cmc.get_warp_matrix()
            apply_cmc(self.tracks, effective_warp)
        if self.agentguard_adapter:
            self.agentguard_adapter.set_frame_warp(effective_warp)

        self._batch_predict(self.tracks, force_missing=True)
        pre_update_snapshots = {}
        if self.agentguard_adapter:
            runtime = self.agentguard_adapter.runtime
            for track in self.tracks:
                if runtime and runtime.is_mature_track(track):
                    pre_update_snapshots[track.track_id] = track.snapshot_state(
                        compact_history=True
                    )

        for track in self.tracks:
            if self.agentguard_adapter:
                runtime = self.agentguard_adapter.runtime
                if runtime and runtime.is_mature_track(track):
                    event = self.agentguard_adapter.build_unmatched_event(
                        track,
                        effective_warp,
                        frame_start_snapshots.get(track.track_id),
                        pre_update_snapshots.get(track.track_id),
                    )
                    self._populate_capture_features(event, track)
                    self.agentguard_adapter.record_unmatched_event(
                        track.track_id, event
                    )
            track.mark_lost()

        if self.agentguard_adapter:
            self.agentguard_adapter.end_frame()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        removed_tracks = [
            track
            for track in self.tracks
            if track.state == TrackState.Removed
        ]
        for track in removed_tracks:
            if self.agentguard_adapter:
                self.agentguard_adapter.remove_track(track.track_id)
            self.shared_kalman_filter.delete_track(track.track_id)

        self.tracks = [
            track
            for track in self.tracks
            if track.state != TrackState.Removed
        ]
        self._predicted_gpu_cache = None
        return []
