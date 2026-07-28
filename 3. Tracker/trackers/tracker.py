from typing import Optional

from trackers.cmc import *
from trackers.utils import *
from trackers.track import *

# AgentGuard integration (optional, only loaded when enabled)
try:
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter
    from integrations.agentguard.config_bridge import build_runtime_config
    from agentguard.contracts.outputs import GateDecision
    _AGENTGUARD_AVAILABLE = True
except ImportError:
    _AGENTGUARD_AVAILABLE = False


class Tracker(object):
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

        # Per-frame warp matrix (read once per frame)
        self._current_warp: Optional[np.ndarray] = None

        # AgentGuard integration
        self.agentguard_adapter = None
        ag_mode = getattr(args, 'agentguard_mode', 'off')
        if _AGENTGUARD_AVAILABLE and ag_mode in ('capture', 'iwg-rg-cma'):
            from agentguard.runtime.manager import AgentGuardRuntime

            event_sink = getattr(args, 'event_sink', None)

            runtime_config = build_runtime_config(args)
            device = getattr(args, 'agentguard_device', 'cpu')
            checkpoint_reid_dim = None
            checkpoint_norm_stats = None
            rg_cma_model = None

            if ag_mode == 'iwg-rg-cma':
                combined_ckpt = getattr(args, 'agentguard_checkpoint', None)
                if combined_ckpt is None:
                    raise RuntimeError(
                        "Combined AgentGuard checkpoint required for iwg-rg-cma mode"
                    )
                from agentguard.features.normalization import NormalizationStats
                from agentguard.training.train_iwg_rg_cma import (
                    load_iwg_rg_cma_checkpoint,
                )

                rg_cma_model, checkpoint = load_iwg_rg_cma_checkpoint(
                    combined_ckpt, map_location='cpu'
                )
                checkpoint_reid_dim = int(checkpoint['reid_dim'])
                checkpoint_norm_stats = NormalizationStats()
                checkpoint_norm_stats.mean = np.asarray(
                    checkpoint['normalization_mean'], dtype=np.float64
                )
                checkpoint_norm_stats.std = np.asarray(
                    checkpoint['normalization_std'], dtype=np.float64
                )
                runtime_config['rg_cma_max_gap'] = int(checkpoint['max_frame_gap'])
                runtime_config['iwg_context_size'] = int(checkpoint['context_size'])

            runtime = AgentGuardRuntime(
                runtime_config,
                rg_cma_model=rg_cma_model,
                device=device,
            )
            runtime.event_sink = event_sink

            if checkpoint_reid_dim is not None:
                runtime.init_feature_builder(
                    reid_dim=checkpoint_reid_dim,
                    norm_stats=checkpoint_norm_stats,
                )

            self.agentguard_adapter = AgentGuardTrackerAdapter(args, vid_name, agentguard_runtime=runtime)

    def init_tracks(self, dets):
        # Get alive tracks, iou_similarity, and scores
        tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])

        # Run track aware NMS
        allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)

        for idx, flag in enumerate(allow_indices):
            if flag:
                dets[idx].initiate(self.frame_id, self.counter)
                self.tracks.append(dets[idx])

    def update(self, dets, dets_95):
        # ==============================================================================================================
        # Update frame id
        self.frame_id += 1

        # Get deleted detections & Encode
        target_detection_indices = getattr(
            self.args,
            "agentguard_target_detection_indices",
            None,
        )
        source_detection_indices = getattr(
            self.args,
            "agentguard_source_detection_indices",
            None,
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
        dets = [Track(self.args, d) for d in dets]
        dets_del = [Track(self.args, d) for d in dets_del]
        if target_detection_indices is not None:
            for det, det_idx in zip(dets, target_detection_indices):
                det.frame_detection_index = int(det_idx)
        else:
            for det_idx, det in enumerate(dets):
                det.frame_detection_index = int(det_idx)
        if dets_del_indices is not None:
            for det, det_idx in zip(dets_del, dets_del_indices):
                det.frame_detection_index = int(det_idx)

        # Divide detections
        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        # Capture-only mode still needs a real FeatureBuilder for scalar and
        # feature shape handling. Initialize it once from the first detection.
        if (
            self.agentguard_adapter
            and self.agentguard_adapter.capture_only
            and self.agentguard_adapter.runtime is not None
            and self.agentguard_adapter.runtime.feature_builder is None
        ):
            first_det = next(
                iter(dets_high + dets_low + dets_del_high),
                None,
            )
            if first_det is not None and getattr(first_det, "feat", None) is not None:
                reid_dim = int(np.asarray(first_det.feat).reshape(-1).shape[0])
                self.agentguard_adapter.runtime.init_feature_builder(reid_dim=reid_dim)
                if self.agentguard_adapter.event_sink is not None:
                    self.agentguard_adapter.event_sink._reid_dim = reid_dim

        # AgentGuard: begin frame (after detections divided, for stable indexing)
        if self.agentguard_adapter:
            self.agentguard_adapter.begin_frame(
                self.frame_id,
                getattr(self.args, 'img_w', 1920),
                getattr(self.args, 'img_h', 1080),
                detection_pool=dets_high + dets_low + dets_del_high,
                detection_sources=[0] * len(dets_high)
                + [1] * len(dets_low)
                + [2] * len(dets_del_high),
            )

        # Split tracks
        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        # AgentGuard: save frame_start_state for mature tracks (pre-CMC)
        frame_start_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in tracked_lost:
                if rt and rt.is_mature_track(t):
                    frame_start_snapshots[t.track_id] = t.snapshot_state(compact_history=True)

        # Camera motion compensation
        if self.disable_gmc:
            effective_warp = np.eye(2, 3, dtype=np.float64)
        else:
            effective_warp = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, effective_warp)
            apply_cmc(new, effective_warp)
        self._current_warp = effective_warp.copy()
        if self.agentguard_adapter:
            self.agentguard_adapter.set_frame_warp(effective_warp)

        # Predict the current location with KF
        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]

        # AgentGuard: save pre_update_state for mature tracks (post-CMC, post-predict)
        pre_update_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in tracked_lost:
                if rt and rt.is_mature_track(t):
                    pre_update_snapshots[t.track_id] = t.snapshot_state(compact_history=True)

        # ==============================================================================================================
        # Association between (tracked and lost tracks) & (high confidence detections)
        dets_all = dets_high + dets_low + dets_del_high
        use_meta = self.agentguard_adapter is not None and self.agentguard_adapter.runtime is not None
        result = iterative_assignment(tracked_lost, dets_high, dets_low, dets_del_high,
                                       self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                                       self.args.reduce_step, self.frame_id,
                                       no_reid=getattr(self.args, 'no_reid', False),
                                       return_meta=use_meta)

        if use_meta:
            matches, u_tracks, u_dets, association_meta = result
        else:
            matches, u_tracks, u_dets = result
            association_meta = None
        if self.agentguard_adapter and association_meta is not None:
            self.agentguard_adapter.set_association_record(
                track_ids=[t.track_id for t in tracked_lost],
                detection_pool=dets_all,
                association_meta=association_meta,
                no_reid=getattr(self.args, 'no_reid', False),
            )

        # Process matched tracks
        agentguard_matched_batch = []
        for t_idx, d_idx in matches:
            track = tracked_lost[t_idx]
            detection = dets_all[d_idx]

            if self.agentguard_adapter and self.agentguard_adapter.runtime and \
               self.agentguard_adapter.runtime.is_mature_track(track):
                fs_snap = frame_start_snapshots.get(track.track_id)
                pu_snap = pre_update_snapshots.get(track.track_id)
                event = self.agentguard_adapter.build_matched_event(
                    track, detection, t_idx, d_idx,
                    association_meta, effective_warp, fs_snap, pu_snap,
                    num_tracks=len(tracked_lost),
                    num_detections=len(dets_all),
                    no_reid=getattr(self.args, 'no_reid', False),
                )
                if self.agentguard_adapter.capture_only:
                    # Capture-only: use baseline update but still record event
                    fb = self.agentguard_adapter.runtime.feature_builder
                    if fb is not None:
                        event.track_feature = np.asarray(
                            pu_snap.feature if pu_snap is not None else track.feat,
                            dtype=np.float64,
                        ).reshape(-1)
                        event.detection_feature = np.asarray(
                            detection.feat,
                            dtype=np.float64,
                        ).reshape(-1)
                        event.scalar_features = fb.compute_scalar(event)
                    track.update(self.frame_id, detection)
                    gate_decision = GateDecision(1.0, 1.0, np.ones(5, dtype=np.float64) / 5.0)
                    self.agentguard_adapter.record_event(track.track_id, event, gate_decision)
                else:
                    agentguard_matched_batch.append((track, detection, event))
            else:
                track.update(self.frame_id, detection)

        if agentguard_matched_batch:
            decisions = self.agentguard_adapter.get_iwg_decisions_batch(
                [(track.track_id, event) for track, _detection, event in agentguard_matched_batch]
            )
            for (track, detection, event), gate_decision in zip(agentguard_matched_batch, decisions):
                self.agentguard_adapter.apply_gate(track, detection, gate_decision)
                self.agentguard_adapter.record_event(track.track_id, event, gate_decision)

        # Mark unmatched tracks
        for t_idx in u_tracks:
            track = tracked_lost[t_idx]

            if self.agentguard_adapter and self.agentguard_adapter.runtime and \
               self.agentguard_adapter.runtime.is_mature_track(track):
                fs_snap = frame_start_snapshots.get(track.track_id)
                pu_snap = pre_update_snapshots.get(track.track_id)
                event = self.agentguard_adapter.build_unmatched_event(
                    track, effective_warp, fs_snap, pu_snap
                )
                # Populate unmatched features
                fb = self.agentguard_adapter.runtime.feature_builder
                if fb is not None and pu_snap is not None:
                    event.track_feature = np.asarray(
                        pu_snap.feature, dtype=np.float64
                    ).reshape(-1)
                    event.detection_feature = np.zeros(
                        fb.reid_dim, dtype=np.float64
                    )
                    event.scalar_features = fb.compute_scalar(event)
                if self.agentguard_adapter.capture_only:
                    track.mark_lost()
                    self.agentguard_adapter.record_unmatched_event(track.track_id, event)
                else:
                    self.agentguard_adapter.record_unmatched_event(track.track_id, event)
                    track.mark_lost()
            else:
                track.mark_lost()

        # AgentGuard: flush the current frame's compact events
        if self.agentguard_adapter:
            self.agentguard_adapter.end_frame()

        # ==============================================================================================================
        # Get remained high confidence detections
        dets_high_left = [dets_all[i] for i in u_dets if i < len(dets_high)]

        # Association between (new tracks) & (left high confidence detections)
        matches, u_tracks, u_dets = iterative_assignment(new, dets_high_left, [], [], self.args.match_thr,
                                                         self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id,
                                                         no_reid=getattr(self.args, 'no_reid', False))

        # Update matched tracks
        for t, d in matches:
            new[t].update(self.frame_id, dets_high_left[d])

        # Mark "remove" to unmatched tracks
        for t in u_tracks:
            new[t].mark_removed()

        # ==============================================================================================================
        # Mark "remove" lost tracks which are too old and add to finished
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                if self.agentguard_adapter:
                    self.agentguard_adapter.remove_track(track.track_id)
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Init new tracks
        self.init_tracks([dets_high_left[udx] for udx in u_dets])

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        self.frame_id += 1

        # AgentGuard: begin frame
        if self.agentguard_adapter:
            self.agentguard_adapter.begin_frame(
                self.frame_id,
                getattr(self.args, 'img_w', 1920),
                getattr(self.args, 'img_h', 1080)
            )

        dropped_new = [t for t in self.tracks if t.state == TrackState.New]
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # AgentGuard: save frame_start snapshots for mature tracks (pre-CMC)
        frame_start_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in self.tracks:
                if rt and rt.is_mature_track(t):
                    frame_start_snapshots[t.track_id] = t.snapshot_state(compact_history=True)

        # Camera motion compensation — read exactly once
        if self.disable_gmc:
            effective_warp = np.eye(2, 3, dtype=np.float64)
        else:
            effective_warp = self.cmc.get_warp_matrix()
            apply_cmc(self.tracks, effective_warp)
        self._current_warp = effective_warp.copy()
        if self.agentguard_adapter:
            self.agentguard_adapter.set_frame_warp(effective_warp)

        [t.predict() for t in self.tracks]

        if (
            self.agentguard_adapter
            and self.agentguard_adapter.capture_only
            and self.agentguard_adapter.runtime is not None
            and self.agentguard_adapter.runtime.feature_builder is None
        ):
            for t in self.tracks:
                feat = getattr(t, "feat", None)
                if feat is not None and np.asarray(feat).size > 0:
                    reid_dim = int(np.asarray(feat).reshape(-1).shape[0])
                    self.agentguard_adapter.runtime.init_feature_builder(reid_dim=reid_dim)
                    if self.agentguard_adapter.event_sink is not None:
                        self.agentguard_adapter.event_sink._reid_dim = reid_dim
                    break

        # AgentGuard: pre_update snapshots (post-CMC, post-predict)
        pre_update_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in self.tracks:
                if rt and rt.is_mature_track(t):
                    pre_update_snapshots[t.track_id] = t.snapshot_state(compact_history=True)

        # Mark all as lost — mature unmatched events enter runtime
        for t in self.tracks:
            if self.agentguard_adapter and self.agentguard_adapter.runtime and \
               self.agentguard_adapter.runtime.is_mature_track(t):
                fs_snap = frame_start_snapshots.get(t.track_id)
                pu_snap = pre_update_snapshots.get(t.track_id)
                event = self.agentguard_adapter.build_unmatched_event(
                    t, effective_warp, fs_snap, pu_snap
                )
                # Populate unmatched features: detection_feature = zero,
                # track_feature from pre_update_state, scalar via no-detection protocol
                fb = self.agentguard_adapter.runtime.feature_builder
                if fb is not None and pu_snap is not None:
                    event.track_feature = np.asarray(
                        pu_snap.feature, dtype=np.float64
                    ).reshape(-1)
                    event.detection_feature = np.zeros(
                        fb.reid_dim, dtype=np.float64
                    )
                    event.scalar_features = fb.compute_scalar(event)
                self.agentguard_adapter.record_unmatched_event(t.track_id, event)
                t.mark_lost()
            else:
                t.mark_lost()

        # AgentGuard: flush the current frame's compact events
        if self.agentguard_adapter:
            self.agentguard_adapter.end_frame()

        # Remove too-old tracks
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                if self.agentguard_adapter:
                    self.agentguard_adapter.remove_track(track.track_id)
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        return []
