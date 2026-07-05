from typing import Optional

from trackers.cmc import *
from trackers.utils import *
from trackers.track import *

# AgentGuard integration (optional, only loaded when enabled)
try:
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter
    from integrations.agentguard.config_bridge import build_runtime_config
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
        if _AGENTGUARD_AVAILABLE and ag_mode != 'off':
            import torch
            from agentguard.runtime.manager import AgentGuardRuntime
            from agentguard.models.iwg import IWG
            from agentguard.models.tgr import TGR

            runtime_config = build_runtime_config(args)
            iwg_ckpt = getattr(args, 'iwg_checkpoint', None)
            tgr_ckpt = getattr(args, 'tgr_checkpoint', None)
            device = getattr(args, 'agentguard_device', 'cpu')

            def _load_checkpoint(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                # State dict load order: model_state_dict > state_dict > entire checkpoint
                if 'model_state_dict' in checkpoint:
                    sd = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    sd = checkpoint['state_dict']
                else:
                    sd = checkpoint

                # Formal fields validation
                ckpt_reid_dim = checkpoint.get('reid_dim')
                scalar_dim = checkpoint.get('scalar_dim')
                event_dim = checkpoint.get('event_dim')
                policy_prototypes = checkpoint.get('policy_prototypes')
                norm_mean = checkpoint.get('normalization_mean')
                norm_std = checkpoint.get('normalization_std')
                feature_schema_sha256 = checkpoint.get('feature_schema_sha256')

                if ckpt_reid_dim is None:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing required field 'reid_dim'"
                    )
                if scalar_dim is None:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing required field 'scalar_dim'"
                    )
                if scalar_dim != 63:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} scalar_dim={scalar_dim}, expected 63"
                    )
                if event_dim is None:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing required field 'event_dim'"
                    )
                if event_dim != 128:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} event_dim={event_dim}, expected 128"
                    )
                if policy_prototypes is None:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing required field 'policy_prototypes'"
                    )
                if norm_mean is None or norm_std is None:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing normalization_mean/std"
                    )
                if feature_schema_sha256 is None:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing required field 'feature_schema_sha256'"
                    )
                pp_arr = np.asarray(policy_prototypes)
                if pp_arr.shape != (5, 2):
                    raise ValueError(
                        f"Checkpoint {ckpt_path} policy_prototypes shape {pp_arr.shape}, expected (5, 2)"
                    )
                if np.asarray(norm_mean).shape != (63,):
                    raise ValueError(
                        f"Checkpoint {ckpt_path} normalization_mean shape "
                        f"{np.asarray(norm_mean).shape}, expected (63,)"
                    )
                if np.asarray(norm_std).shape != (63,):
                    raise ValueError(
                        f"Checkpoint {ckpt_path} normalization_std shape "
                        f"{np.asarray(norm_std).shape}, expected (63,)"
                    )

                # ReID dimension consistency: checkpoint reid_dim must match
                # any inference of reid_dim from weight shapes
                reid_proj_weight = sd.get('reid_proj.weight')
                if reid_proj_weight is None:
                    reid_proj_weight = sd.get('encoder.reid_proj.weight')
                if reid_proj_weight is not None:
                    inferred_dim = reid_proj_weight.shape[1]
                    if inferred_dim != ckpt_reid_dim:
                        raise ValueError(
                            f"Checkpoint {ckpt_path} reid_dim={ckpt_reid_dim} "
                            f"inconsistent with weight shape dim={inferred_dim}"
                        )

                # Normalization stats
                from agentguard.features.normalization import NormalizationStats
                ns = NormalizationStats()
                ns.mean = np.asarray(norm_mean, dtype=np.float64)
                ns.std = np.asarray(norm_std, dtype=np.float64)
                norm_stats = ns

                return sd, int(ckpt_reid_dim), norm_stats

            iwg_model = None
            tgr_model = None
            checkpoint_reid_dim = None
            checkpoint_norm_stats = None

            if ag_mode in ('iwg', 'full'):
                if iwg_ckpt is None:
                    raise RuntimeError(
                        f"IWG checkpoint required for mode={ag_mode} but none provided"
                    )
                sd_iwg, checkpoint_reid_dim, checkpoint_norm_stats = _load_checkpoint(iwg_ckpt)
                iwg_model = IWG(reid_dim=checkpoint_reid_dim)
                iwg_model.load_state_dict(sd_iwg)
                iwg_model.eval()

            if ag_mode == 'full':
                if tgr_ckpt is None:
                    raise RuntimeError(
                        "TGR checkpoint required for full mode but none provided"
                    )
                sd_tgr, tgr_reid_dim, tgr_norm_stats = _load_checkpoint(tgr_ckpt)
                if tgr_reid_dim != checkpoint_reid_dim:
                    raise ValueError(
                        f"TGR reid_dim={tgr_reid_dim} does not match IWG reid_dim={checkpoint_reid_dim}"
                    )
                tgr_model = TGR(reid_dim=checkpoint_reid_dim)
                tgr_model.load_state_dict(sd_tgr)
                tgr_model.eval()

            runtime = AgentGuardRuntime(runtime_config, iwg_model, tgr_model, device)

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

        # AgentGuard: begin frame
        if self.agentguard_adapter:
            self.agentguard_adapter.begin_frame(
                self.frame_id,
                getattr(self.args, 'img_w', 1920),
                getattr(self.args, 'img_h', 1080)
            )

        # Get deleted detections &  Encode
        dets_del = find_deleted_detections(dets, dets_95)
        dets = [Track(self.args, d) for d in dets]
        dets_del = [Track(self.args, d) for d in dets_del]

        # Divide detections
        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        # Split tracks
        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        # AgentGuard: save frame_start_state for mature tracks (pre-CMC)
        frame_start_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in tracked_lost:
                if rt and rt.is_mature_track(t):
                    frame_start_snapshots[t.track_id] = t.snapshot_state()

        # Camera motion compensation
        if self.disable_gmc:
            effective_warp = np.eye(2, 3, dtype=np.float64)
        else:
            effective_warp = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, effective_warp)
            apply_cmc(new, effective_warp)
        self._current_warp = effective_warp.copy()

        # Predict the current location with KF
        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]

        # AgentGuard: save pre_update_state for mature tracks (post-CMC, post-predict)
        pre_update_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in tracked_lost:
                if rt and rt.is_mature_track(t):
                    pre_update_snapshots[t.track_id] = t.snapshot_state()

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

        # Process matched tracks
        for t_idx, d_idx in matches:
            track = tracked_lost[t_idx]
            detection = dets_all[d_idx]

            if self.agentguard_adapter and self.agentguard_adapter.runtime and \
               self.agentguard_adapter.runtime.is_mature_track(track):
                fs_snap = frame_start_snapshots.get(track.track_id)
                pu_snap = pre_update_snapshots.get(track.track_id)
                event = self.agentguard_adapter.build_matched_event(
                    track, detection, t_idx, d_idx,
                    association_meta, effective_warp, fs_snap, pu_snap
                )
                gate_decision = self.agentguard_adapter.get_iwg_decision(track.track_id, event)
                self.agentguard_adapter.apply_gate(track, detection, gate_decision)
                self.agentguard_adapter.record_event(track.track_id, event, gate_decision)
            else:
                track.update(self.frame_id, detection)

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
                self.agentguard_adapter.record_unmatched_event(track.track_id, event)
                track.mark_lost()
            else:
                track.mark_lost()

        # AgentGuard: finalize first stage (TGR)
        if self.agentguard_adapter:
            self.agentguard_adapter.finalize_first_stage(tracked_lost)

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

        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # AgentGuard: save frame_start snapshots for mature tracks
        frame_start_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in self.tracks:
                if rt and rt.is_mature_track(t):
                    frame_start_snapshots[t.track_id] = t.snapshot_state()

        if self.disable_gmc:
            effective_warp = np.eye(2, 3, dtype=np.float64)
        else:
            effective_warp = self.cmc.get_warp_matrix()
            apply_cmc(self.tracks, effective_warp)
        self._current_warp = effective_warp.copy()

        [t.predict() for t in self.tracks]

        # AgentGuard: pre_update snapshots
        pre_update_snapshots = {}
        if self.agentguard_adapter:
            rt = self.agentguard_adapter.runtime
            for t in self.tracks:
                if rt and rt.is_mature_track(t):
                    pre_update_snapshots[t.track_id] = t.snapshot_state()

        # Mark all as lost
        for t in self.tracks:
            if self.agentguard_adapter and self.agentguard_adapter.runtime and \
               self.agentguard_adapter.runtime.is_mature_track(t):
                fs_snap = frame_start_snapshots.get(t.track_id)
                pu_snap = pre_update_snapshots.get(t.track_id)
                event = self.agentguard_adapter.build_unmatched_event(
                    t, effective_warp, fs_snap, pu_snap
                )
                self.agentguard_adapter.record_unmatched_event(t.track_id, event)
                t.mark_lost()
            else:
                t.mark_lost()

        # AgentGuard: finalize first stage (TGR)
        if self.agentguard_adapter:
            self.agentguard_adapter.finalize_first_stage(self.tracks)

        # Remove too-old tracks
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                if self.agentguard_adapter:
                    self.agentguard_adapter.remove_track(track.track_id)
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        return []
