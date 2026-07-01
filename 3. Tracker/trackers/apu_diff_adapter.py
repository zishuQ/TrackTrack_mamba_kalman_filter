import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


class APUDiffAdapter:
    _shared_instance = None

    @classmethod
    def get_shared_instance(
        cls,
        model_path,
        repo_dir="/home/shang/workspace/diffusion_appearance",
        device="cuda",
        sample_steps=1,
        stochastic=False,
        history_update_mode="observed",
    ):
        sample_steps = max(int(sample_steps), 1)
        stochastic = bool(stochastic)
        key = (
            os.path.abspath(model_path),
            os.path.abspath(repo_dir),
            device,
            sample_steps,
            stochastic,
            history_update_mode,
        )
        if cls._shared_instance is None or cls._shared_instance._key != key:
            cls._shared_instance = cls(
                model_path=model_path,
                repo_dir=repo_dir,
                device=device,
                sample_steps=sample_steps,
                stochastic=stochastic,
                history_update_mode=history_update_mode,
                key=key,
            )
        return cls._shared_instance

    def __init__(
        self,
        model_path,
        repo_dir,
        device="cuda",
        sample_steps=1,
        stochastic=False,
        history_update_mode="observed",
        key=None,
    ):
        self._key = key
        self.repo_dir = os.path.abspath(repo_dir)
        self.history_update_mode = history_update_mode
        if self.history_update_mode not in {"observed", "predicted"}:
            raise ValueError(f"Unknown APUDiff history update mode: {self.history_update_mode!r}")
        self.sample_steps = max(int(sample_steps), 1)
        self.stochastic = bool(stochastic)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        from apu_diff.config import APUDiffConfig
        from apu_diff.models import APUDiff

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location="cpu")
        cfg = APUDiffConfig.from_dict(checkpoint.get("config", {}))
        if cfg.reid_dim == "auto":
            state = checkpoint.get("model_state_dict", checkpoint)
            cfg.reid_dim = state["predictor.context_encoder.history_proj.weight"].shape[1]
            cfg.latent_dim = int(cfg.reid_dim)
        self.history_len = int(cfg.history_len)
        self.latent_dim = int(cfg.latent_dim)
        self.reid_dim = int(cfg.reid_dim)
        self.model = APUDiff(
            reid_dim=cfg.reid_dim,
            latent_dim=cfg.latent_dim,
            time_dim=cfg.time_dim,
            num_diffusion_steps=cfg.num_diffusion_steps,
            denoiser_hidden_dim=cfg.denoiser_hidden_dim,
        ).to(self.device)
        load_result = self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "APUDiff checkpoint loaded with non-critical key mismatch: "
                f"missing={list(load_result.missing_keys)}, "
                f"unexpected={list(load_result.unexpected_keys)}"
            )
        self.model.eval()
        self.reset_stats()
        print(
            f"Loaded APUDiff from {model_path} on {self.device}, "
            f"sample_steps={self.sample_steps}, "
            f"stochastic={self.stochastic}, "
            f"history_update_mode={self.history_update_mode}"
        )

    def reset_stats(self):
        self.stats = {
            "num_cost_calls": 0,
            "num_cost_pairs": 0,
            "c_pred_sum": 0.0,
            "cost_sum": 0.0,
        }

    def project_np(self, feat):
        arr = np.asarray(feat, dtype=np.float32)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        if arr.shape[-1] == self.latent_dim:
            norm = np.linalg.norm(arr, axis=-1, keepdims=True)
            z = arr / np.maximum(norm, 1e-12)
            return z[0] if squeeze else z
        with torch.no_grad():
            z = self.model.project(torch.from_numpy(arr).to(self.device)).cpu().numpy()
        return z[0] if squeeze else z

    def _project_detections(self, dets):
        missing = [d for d in dets if getattr(d, "apu_det_z", None) is None]
        if missing:
            det_feats = np.concatenate([d.raw_feat for d in missing], axis=0).astype(np.float32)
            det_z = self.project_np(det_feats)
            for det, z in zip(missing, det_z):
                det.apu_det_z = z.astype(np.float32)
        return np.stack([d.apu_det_z for d in dets], axis=0).astype(np.float32)

    def init_track_state(self, det_feat):
        z = self.project_np(det_feat)
        local_queue = np.repeat(z[None, :], self.history_len, axis=0).astype(np.float32)
        history_mask = np.zeros((self.history_len,), dtype=np.float32)
        history_mask[-1] = 1.0
        return local_queue, history_mask

    def predict_tracks(self, tracks):
        ready = [t for t in tracks if getattr(t, "apu_local_queue", None) is not None]
        if not ready:
            return
        local_queue = np.stack([t.apu_local_queue for t in ready], axis=0).astype(np.float32)
        history_mask = np.stack([t.apu_history_mask for t in ready], axis=0).astype(np.float32)
        with torch.no_grad():
            pred = self.model.predict(
                torch.from_numpy(local_queue).to(self.device),
                torch.from_numpy(history_mask).to(self.device),
                deterministic=not self.stochastic,
                sample_steps=self.sample_steps,
            ).cpu().numpy()
        for track, pred_feat in zip(ready, pred):
            track.apu_pred_feat = pred_feat.astype(np.float32)

    def appearance_cost(self, tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return np.ones((len(tracks), len(dets)), dtype=np.float64)
        if any(getattr(t, "apu_pred_feat", None) is None for t in tracks):
            self.predict_tracks(tracks)
        det_z = self._project_detections(dets)
        track_pred = []
        for t in tracks:
            pred = getattr(t, "apu_pred_feat", None)
            if pred is None:
                pred = t.apu_local_queue[-1]
            track_pred.append(pred)
        pred_t = torch.from_numpy(np.stack(track_pred, axis=0)).float().to(self.device)
        det_t = torch.from_numpy(det_z).float().to(self.device)
        pred_t = F.normalize(pred_t, dim=-1)
        det_t = F.normalize(det_t, dim=-1)
        with torch.no_grad():
            c_pred = 1.0 - pred_t @ det_t.T
            cost = c_pred
            self.stats["num_cost_calls"] += 1
            self.stats["num_cost_pairs"] += int(cost.numel())
            self.stats["c_pred_sum"] += float(c_pred.sum().item())
            self.stats["cost_sum"] += float(cost.sum().item())
        return cost.cpu().numpy().astype(np.float64)

    def report_stats(self):
        pairs = max(self.stats["num_cost_pairs"], 1)
        print(
            "APUDiff cost stats: "
            f"calls={self.stats['num_cost_calls']} "
            f"pairs={self.stats['num_cost_pairs']} "
            f"c_pred={self.stats['c_pred_sum'] / pairs:.6f} "
            f"cost={self.stats['cost_sum'] / pairs:.6f} "
            f"sample_steps={self.sample_steps} "
            f"stochastic={self.stochastic} "
            f"history_update_mode={self.history_update_mode}",
            flush=True,
        )

    def update_track(self, track, detection):
        det_z = getattr(detection, "apu_det_z", None)
        if det_z is None:
            det_z = self.project_np(detection.raw_feat.squeeze(0))
        pred = getattr(track, "apu_pred_feat", None)
        if pred is None:
            pred = track.apu_local_queue[-1]

        if self.history_update_mode == "observed":
            next_feat = np.asarray(det_z, dtype=np.float32)
        else:
            next_feat = np.asarray(pred, dtype=np.float32)
        norm = np.linalg.norm(next_feat)
        if norm > 0:
            next_feat = next_feat / norm

        track.apu_local_queue = np.concatenate([track.apu_local_queue[1:], next_feat[None, :]], axis=0)
        track.apu_history_mask = np.concatenate([track.apu_history_mask[1:], np.ones((1,), dtype=np.float32)], axis=0)
        track.apu_pred_feat = None
