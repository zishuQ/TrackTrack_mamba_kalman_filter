#!/usr/bin/env python3
"""Train the supervision groups on the paired SportsMOT trainval data.

Groups (same data, schedule, optimizer budget, e200 init):
  control          original motion+appearance soft labels
  motion_diou      motion loss replaced by next-frame GT-box DIoU
  appearance_score appearance label replaced by the template-discrimination
                   score (fallback to the original soft label when the
                   discrimination diagnostic is invalid)
  motion_diou_appearance_score  both of the above
  appearance_rank  motion DIoU + identity ranking on the gate-updated track
                   template (fallback to the original soft label)
  motion_diou_speed  motion DIoU + KF centre-velocity L1 + discrimination
                   appearance soft labels
  motion_diou_speed4 motion_diou_speed plus KF size-velocity (vw/vh) L1

Schedule: 40 shard-epochs = 4 shards x 10 cycles; per shard each sequence
contributes its contiguous fraction of samples (frame order).  Weights are
saved at epoch 20 and 40; last.pt is written every epoch for resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap as mmap_module
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentguard.data.detection_cache import SequenceDetectionCache
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.training.checkpoint import (
    build_checkpoint_payload,
    load_shutrack_checkpoint,
    save_checkpoint_atomic,
)
from shutrack.training.loss import (
    appearance_rank_loss,
    appearance_column_loss,
    appearance_soft_loss,
    gate_loss,
    motion_box_diou_loss,
    motion_speed4_l1_loss,
    motion_speed_l1_loss,
)
from shutrack.training.train import _load_init_state_with_extension, _seed_everything
from shutrack.training.defaults import (
    LEGACY_APPEARANCE_PROJ_DIM, LEGACY_ARCHITECTURE_VARIANT,
    MAINLINE_APPEARANCE_PROJ_DIM, MAINLINE_ARCHITECTURE_VARIANT,
    MAINLINE_STAGE_CHUNK_SAMPLES, MAINLINE_STAGE_FEATURES,
)
from shutrack.training.paired_pipeline import DeferredMetrics, move_batch, npz_array_view, prepared_batches
from shutrack.training.feature_staging import (
    aligned_stage_chunk_samples, feature_chunks, staged_sequence_rows,
)
from paired_training_input import fit_input_statistics, official_sequences
from shutrack.schema import EVIDENCE_SCALAR_INDICES, COMPETITION_SCALAR_INDICES

ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
DATA = ROOT / "data"
CACHE = MAIN / "outputs/agentguard/detection_cache"
INIT_CKPT = MAIN / "outputs/agentguard/experiments/shutrack_a_submit_20261117/SportsMOT/checkpoints/last.pt"
SHARDS = 4
CYCLES = 10
EPOCHS = SHARDS * CYCLES


def tau_of(values: list[np.ndarray]) -> float:
    if not values:
        return 0.1
    array = np.concatenate(values)
    array = array[np.isfinite(array)]
    nonzero = np.abs(array[np.abs(array) > 1e-12])
    if nonzero.size == 0:
        return 0.1
    return float(np.clip(float(np.median(nonzero)), 1e-3, 0.1))


def load_global_taus(sequences: list[str], data_root: Path = DATA) -> dict:
    motion, appearance, delta = [], [], []
    for sequence in sequences:
        samples = np.load(data_root / sequence / "samples.npz", allow_pickle=False)
        motion.append(samples["motion_benefit"][samples["valid_motion"]])
        appearance.append(samples["appearance_benefit"][samples["valid_appearance"]])
        delta.append(samples["appearance_delta"][samples["appearance_diag_valid"]])
    return {
        "tau_motion": tau_of(motion),
        "tau_appearance": tau_of(appearance),
        "tau_delta": tau_of(delta),
    }


def derive_targets(samples, taus):
    motion_benefit = samples["motion_benefit"].astype(np.float64)
    appearance_benefit = samples["appearance_benefit"].astype(np.float64)
    valid_motion = samples["valid_motion"].astype(bool)
    valid_appearance = samples["valid_appearance"].astype(bool)
    coverage_gt = samples["gt_coverage"].astype(np.float64)
    coverage_oracle = samples["oracle_detection_coverage"].astype(np.float64)

    def safe(benefit, valid, coverage, tau):
        soft = 1.0 / (1.0 + np.exp(-benefit / tau))
        confidence = np.clip(
            coverage * np.minimum(np.abs(benefit) / (2.0 * tau), 1.0), 0.0, 1.0
        )
        value = confidence * soft + (1.0 - confidence) * 1.0
        value = np.where(valid, value, 1.0)
        return value.astype(np.float32)

    motion_safe = safe(motion_benefit, valid_motion, coverage_gt, taus["tau_motion"])
    appearance_safe = safe(
        appearance_benefit, valid_appearance, coverage_oracle, taus["tau_appearance"]
    )
    delta = samples["appearance_delta"].astype(np.float64)
    diag = samples["appearance_diag_valid"].astype(bool)
    appearance_new = np.where(
        diag,
        1.0 / (1.0 + np.exp(-delta / taus["tau_delta"])),
        appearance_safe,
    ).astype(np.float32)
    return {
        "motion_safe": motion_safe,
        "appearance_safe": appearance_safe,
        "appearance_new": appearance_new,
        "valid_motion": valid_motion,
        "valid_appearance": valid_appearance,
    }


class SequenceData:
    def __init__(
        self, sequence: str, split: str, data_root: Path = DATA, dataset: str = "SportsMOT"
    ):
        self.sequence = sequence
        self.dataset = dataset
        self.dir = data_root / sequence
        self.samples = np.load(self.dir / "samples.npz", allow_pickle=False)
        # NpzFile.__getitem__ re-reads the whole zip member on every access.
        # batch_tensors touches these small members for every batch. Even
        # page-cache hits allocate full temporary arrays. Preload them once;
        # large uncompressed rank features use read-only mapped views below.
        self.small = {
            key: np.asarray(self.samples[key])
            for key in (
                "window_indices",
                "current_event_index",
                "current_competition",
                "pred_box_g0",
                "pred_box_g1",
                "gt_box_t1",
                "motion_gt_valid",
                "column_det",
                "column_offset",
                "column_rival",
                "column_score",
                "gt_box_t1",
                "motion_gt_valid",
            )
            if key in self.samples.files
        }
        self.event_evidence = np.load(self.dir / "event_evidence.npy", mmap_mode="r")
        self.event_track = np.load(self.dir / "event_track.npy", mmap_mode="r")
        self.event_det = np.load(self.dir / "event_detection_index.npy", mmap_mode="r")
        self.event_has = np.load(self.dir / "event_has_detection.npy", mmap_mode="r")
        for name in ("event_state_mean", "event_state_cov", "event_det_measurement",
                     "event_det_score", "event_warp_next", "event_gap", "event_frame",
                     "event_target_row", "gt_box_t0", "gt_valid_t0"):
            path = self.dir / f"{name}.npy"
            if path.is_file():
                setattr(self, name, np.load(path, mmap_mode="r"))
        self.split = split
        self.n = int(self.samples["frame"].shape[0])
        self.cache: SequenceDetectionCache | None = None
        self.rank_pos: np.ndarray | None = None
        if "rank_rows" in self.samples.files:
            rows = self.samples["rank_rows"]
            self.rank_pos = np.full(self.n, -1, dtype=np.int64)
            self.rank_pos[rows] = np.arange(rows.size, dtype=np.int64)

    def open_cache(self) -> None:
        if self.cache is None:
            cache_split = "all" if self.dataset in ("MOT17", "MOT20") else self.split
            self.cache = SequenceDetectionCache(
                CACHE / self.dataset / cache_split / self.sequence
            )

    def close_cache(self) -> None:
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    def load_rank_arrays(self) -> None:
        if getattr(self, "_rank_arrays", None) is not None:
            return
        if self.rank_pos is None:
            raise RuntimeError(f"{self.sequence}: rank fields not present in data")
        s = self.samples
        self._rank_arrays = (
            np.asarray(npz_array_view(s, "rank_template0"), dtype=np.float16),
            np.asarray(npz_array_view(s, "rank_detection"), dtype=np.float16),
            np.asarray(npz_array_view(s, "rank_positive"), dtype=np.float16),
            np.asarray(npz_array_view(s, "rank_competitor"), dtype=np.float16),
            np.asarray(npz_array_view(s, "rank_score"), dtype=np.float32),
        )

    def preload(self) -> None:
        """Copy mapped event and rank arrays into resident RAM."""
        for attr in ("event_evidence", "event_track", "event_det", "event_has"):
            setattr(self, attr, np.array(getattr(self, attr)))
        self.load_rank_arrays()
        self._rank_arrays = tuple(np.array(array) for array in self._rank_arrays)


def batch_tensors(
    seq: SequenceData, rows: np.ndarray, targets, group: str, *, compact_reid: bool = False
) -> dict:
    samples = seq.samples
    # Only transport half precision when the source already is float16.
    # GPU inputs are restored to float32 before model/loss evaluation.
    reid_dtype = (
        np.float16 if compact_reid and seq.event_track.dtype == np.float16
        and seq.cache.features.dtype == np.float16 else np.float32
    )
    indices = seq.small["window_indices"][rows]
    mask = indices < 0
    safe_indices = np.where(mask, 0, indices)
    history_evidence = np.asarray(seq.event_evidence[safe_indices], dtype=np.float32)
    history_track = np.asarray(seq.event_track[safe_indices], dtype=reid_dtype)
    history_evidence[mask] = 0.0
    history_track[mask] = 0.0
    history_det = np.asarray(seq.event_det[safe_indices], dtype=np.int64)
    history_det[mask] = -1
    history_has = np.asarray(seq.event_has[safe_indices], dtype=np.bool_)
    history_has[mask] = False
    current = seq.small["current_event_index"][rows]
    current_evidence = np.asarray(seq.event_evidence[current], dtype=np.float32)
    current_track = np.asarray(seq.event_track[current], dtype=reid_dtype)
    current_det = np.asarray(seq.event_det[current], dtype=np.int64)
    current_has = np.asarray(seq.event_has[current], dtype=np.bool_)
    competition = seq.small["current_competition"][rows].astype(np.float32)
    if getattr(seq, "input_statistics", None) is not None:
        mean, std = seq.input_statistics
        cols = list(EVIDENCE_SCALAR_INDICES)
        history_evidence[:, :, :-1] = (history_evidence[:, :, :-1] - mean[cols]) / std[cols]
        history_evidence[mask] = 0.0
        current_evidence[:, :-1] = (current_evidence[:, :-1] - mean[cols]) / std[cols]
        cols = list(COMPETITION_SCALAR_INDICES)
        competition = ((competition - mean[cols]) / std[cols]).astype(np.float32)
    all_indices = np.concatenate([history_det, current_det[:, None]], axis=1)
    features = np.zeros(
        (all_indices.shape[0], 6, seq.event_track.shape[1]), dtype=reid_dtype
    )
    valid = all_indices >= 0
    features[valid] = np.asarray(seq.cache.features[all_indices[valid]], dtype=reid_dtype)
    batch = {
        "history_evidence": history_evidence,
        "history_track": history_track,
        "history_detection": features[:, :5],
        "history_mask": mask,
        "history_has_detection": history_has,
        "history_reset": np.zeros(mask.shape, dtype=np.bool_),
        "current_evidence": current_evidence,
        "current_competition": competition,
        "current_track": current_track,
        "current_detection": features[:, 5],
        "current_has_detection": current_has,
        "safe_gate_target": np.stack(
            [
                targets["motion_safe"][rows],
                targets["appearance_new"][rows]
                if group
                in (
                    "appearance_score",
                    "motion_diou_appearance_score",
                    "motion_gap3",
                    "motion_diou_speed",
                    "motion_diou_speed4",
                )
                else targets["appearance_safe"][rows],
            ],
            axis=1,
        ).astype(np.float32),
        "valid_channels": np.stack(
            [targets["valid_motion"][rows], targets["valid_appearance"][rows]], axis=1
        ).astype(np.bool_),
        "sample_weight": np.ones(len(rows), dtype=np.float32),
        "appearance_target": (
            targets["appearance_new"][rows]
            if group
            in (
                "appearance_score",
                "motion_diou_appearance_score",
                "motion_gap3",
                "motion_diou_speed",
                "motion_diou_speed4",
            )
            else targets["appearance_safe"][rows]
        ).astype(np.float32),
        "appearance_valid": targets["valid_appearance"][rows].astype(np.bool_),
        "pred_box_g0": seq.small["pred_box_g0"][rows].astype(np.float32),
        "pred_box_g1": seq.small["pred_box_g1"][rows].astype(np.float32),
        "gt_box_t1": seq.small["gt_box_t1"][rows].astype(np.float32),
        "motion_gt_valid": seq.small["motion_gt_valid"][rows].astype(np.bool_),
    }
    if group in ("appearance_rank", "appearance_row_column", "appearance_rank_speed4"):
        seq.load_rank_arrays()
        rank_position = seq.rank_pos[rows]
        hit = rank_position >= 0
        width = seq._rank_arrays[0].shape[1]

        def gather_rank(array: np.ndarray) -> np.ndarray:
            dtype = np.float16 if compact_reid else np.float32
            out = np.zeros((len(rows), width), dtype=dtype)
            if hit.any():
                out[hit] = array[rank_position[hit]].astype(dtype, copy=False)
            return out

        batch["rank_template0"] = gather_rank(seq._rank_arrays[0])
        batch["rank_detection"] = gather_rank(seq._rank_arrays[1])
        batch["rank_positive"] = gather_rank(seq._rank_arrays[2])
        batch["rank_competitor"] = gather_rank(seq._rank_arrays[3])
        scores = np.zeros(len(rows), dtype=np.float32)
        if hit.any():
            scores[hit] = seq._rank_arrays[4][rank_position[hit]]
        batch["rank_score"] = scores
        batch["rank_valid"] = hit
    if group == "motion_gap3":
        batch["gap3_boxes"] = samples["gap3_boxes"][rows]
        batch["gap3_valid"] = samples["gap3_valid"][rows]
    if group == "appearance_row_column":
        indices = seq.small["column_det"][rows]
        valid_col = indices >= 0
        positive = np.zeros_like(current_track)
        positive[valid_col] = seq.cache.features[indices[valid_col]]
        batch["column_positive"] = positive
        batch["column_valid"] = valid_col
        for key in ("column_offset", "column_rival", "column_score"):
            batch[key] = seq.small[key][rows].astype(np.float32)
    if group == "motion_diou_speed":
        batch["kf_velocity_g0"] = samples["kf_velocity_g0"][rows].astype(np.float32)
        batch["kf_velocity_g1"] = samples["kf_velocity_g1"][rows].astype(np.float32)
        batch["gt_velocity"] = samples["gt_velocity"][rows].astype(np.float32)
        batch["speed_valid"] = samples["speed_valid"][rows].astype(np.bool_)
    if group in ("motion_diou_speed4", "appearance_rank_speed4"):
        batch["kf_velocity4_g0"] = samples["kf_velocity4_g0"][rows].astype(np.float32)
        batch["kf_velocity4_g1"] = samples["kf_velocity4_g1"][rows].astype(np.float32)
        batch["gt_velocity4"] = samples["gt_velocity4"][rows].astype(np.float32)
        batch["size_adj_valid"] = samples["size_adj_valid"][rows].astype(np.bool_)
    return batch


def model_inputs(batch: dict) -> dict:
    keys = (
        "history_evidence",
        "history_track",
        "history_detection",
        "history_mask",
        "history_has_detection",
        "history_reset",
        "current_evidence",
        "current_competition",
        "current_track",
        "current_detection",
        "current_has_detection",
    )
    return {key: batch[key] for key in keys}


PAIRED_TENSOR_KEYS = {
    "history_evidence",
    "history_track",
    "history_detection",
    "history_mask",
    "history_has_detection",
    "history_reset",
    "current_evidence",
    "current_competition",
    "current_track",
    "current_detection",
    "current_has_detection",
    "safe_gate_target",
    "valid_channels",
    "sample_weight",
    "appearance_target",
    "appearance_valid",
    "pred_box_g0",
    "pred_box_g1",
    "gt_box_t1",
    "motion_gt_valid", "gap3_boxes", "gap3_valid",
    "rank_template0",
    "rank_detection",
    "rank_positive",
    "rank_competitor",
    "rank_score",
    "column_score", "column_positive", "column_valid", "column_rival", "column_offset",
    "rank_valid",
    "kf_velocity_g0",
    "kf_velocity_g1",
    "gt_velocity",
    "speed_valid",
    "kf_velocity4_g0",
    "kf_velocity4_g1",
    "gt_velocity4",
    "size_adj_valid",
}


PAIRED_REID_KEYS = frozenset((
    "history_track", "history_detection", "current_track", "current_detection",
    "rank_template0", "rank_detection", "rank_positive", "rank_competitor",
    "column_positive",
))


def to_device(batch: dict, device: torch.device) -> dict:
    return move_batch(batch, device, PAIRED_TENSOR_KEYS, float32_keys=PAIRED_REID_KEYS)


def scheduler_lambda(total_steps: int, warmup_steps: int):
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return multiplier


SHARED_PARAMETER_PREFIXES = (
    "evidence_encoder.", "appearance_encoder.", "event_fusion.", "hce.",
)


def shared_parameters(model):
    """Parameters of the trunk both gate losses backprop through."""
    return [(name, param) for name, param in model.named_parameters()
            if name.startswith(SHARED_PARAMETER_PREFIXES) and param.requires_grad]


def loss_grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, [param for _, param in parameters],
                                retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum())
    return math.sqrt(total)



def _shard_bounds(n: int, shard: int, shards: int) -> tuple[int, int]:
    return n * shard // shards, n * (shard + 1) // shards


def sequence_sampling_weights(counts: dict[str, int], power: float) -> dict[str, float]:
    active = [sequence for sequence, count in counts.items() if int(count) > 0]
    if not active:
        raise ValueError("sequence sampling requires at least one non-empty sequence")
    raw = {sequence: float(counts[sequence]) ** float(power) for sequence in active}
    total = sum(raw.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("sequence sampling weights must have a positive finite sum")
    return {sequence: value / total for sequence, value in raw.items()}


def allocate_sequence_quota(
    *,
    num_samples: int,
    sequence_counts: dict[str, int],
    active_sequences: list[str],
    power: float,
) -> dict[str, int]:
    """Exact sqrt-size quota allocation used by the IWG trainer."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if not active_sequences:
        if num_samples:
            raise ValueError("cannot allocate samples without active sequences")
        return {}
    weights = sequence_sampling_weights(
        {sequence: sequence_counts[sequence] for sequence in active_sequences},
        power=power,
    )
    raw = {sequence: float(num_samples) * weights[sequence] for sequence in active_sequences}
    allocated = {sequence: math.floor(value) for sequence, value in raw.items()}
    remaining = int(num_samples - sum(allocated.values()))
    remainder_order = sorted(
        active_sequences,
        key=lambda sequence: (-(raw[sequence] - allocated[sequence]), active_sequences.index(sequence)),
    )
    for sequence in remainder_order[:remaining]:
        allocated[sequence] += 1
    if sum(allocated.values()) != int(num_samples):
        raise AssertionError("sequence sample allocation changed the phase budget")
    return allocated


def phase_sampling(
    datasets: dict[str, "SequenceData"],
    shard: int,
    shards: int,
    sequence_sampling: str,
    sampling_power: float,
):
    shard_counts = {}
    for sequence, data in datasets.items():
        start, end = _shard_bounds(data.n, shard, shards)
        shard_counts[sequence] = end - start
    active = sorted(sequence for sequence, count in shard_counts.items() if count > 0)
    phase_samples = sum(shard_counts[sequence] for sequence in active)
    if sequence_sampling == "sqrt-size":
        quotas = allocate_sequence_quota(
            num_samples=phase_samples,
            sequence_counts={sequence: data.n for sequence, data in datasets.items()},
            active_sequences=active,
            power=sampling_power,
        )
    else:
        quotas = {sequence: shard_counts[sequence] for sequence in active}
    return active, quotas, shard_counts


def plan_phase(
    datasets: dict[str, "SequenceData"],
    shard: int,
    shards: int,
    sequence_sampling: str,
    sampling_power: float,
    seed: int,
    epoch: int,
):
    """Deterministic per-phase order and row selection.

    sample-proportional keeps the historical per-shard permutation and
    epoch+sha1(sequence) RNG. sqrt-size reuses the IWG quota allocation and
    samples each sequence's shard slice with replacement; batches stay
    sequence-contiguous (IWG shuffled the flat index list globally).
    """
    active, quotas, shard_counts = phase_sampling(
        datasets, shard, shards, sequence_sampling, sampling_power
    )
    rows_by_sequence = {}
    if sequence_sampling == "sqrt-size":
        sampler_rng = random.Random(int(seed) + int(epoch) * 1_000_003)
        for sequence in active:
            start, end = _shard_bounds(datasets[sequence].n, shard, shards)
            count = int(quotas[sequence])
            local = np.fromiter(
                (sampler_rng.randrange(end - start) for _ in range(count)),
                dtype=np.int64,
                count=count,
            )
            rows_by_sequence[sequence] = np.arange(start, end, dtype=np.int64)[local]
        order = list(active)
        sampler_rng.shuffle(order)
    else:
        rng = np.random.RandomState(int(seed) + int(epoch))
        order = list(datasets)
        rng.shuffle(order)
        for sequence in order:
            rows = shard_rows(datasets[sequence], shard, shards)
            local_seed = int(
                hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:8], 16
            ) % 100000
            local_rng = np.random.RandomState(
                int(seed) + int(epoch) * 1000 + local_seed
            )
            rows_by_sequence[sequence] = rows[local_rng.permutation(rows.size)]
    return order, rows_by_sequence, quotas, shard_counts


def build_schedule(
    datasets: dict[str, "SequenceData"],
    batch_size: int,
    epochs: int = EPOCHS,
    shards: int = SHARDS,
    sequence_sampling: str = "sample-proportional",
    sampling_power: float = 1.0,
):
    if shards < 1:
        raise ValueError("shards must be positive")
    epoch_steps = []
    for epoch in range(1, epochs + 1):
        shard = (epoch - 1) % shards
        _, quotas, _ = phase_sampling(
            datasets, shard, shards, sequence_sampling, sampling_power
        )
        steps = sum(int(math.ceil(count / batch_size)) for count in quotas.values())
        epoch_steps.append(steps)
    return epoch_steps


def shard_rows(data: "SequenceData", shard: int, shards: int = SHARDS) -> np.ndarray:
    start, end = _shard_bounds(data.n, shard, shards)
    return np.arange(start, end, dtype=np.int64)


def _process_rss_gb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _drop_memmap(obj) -> None:
    # Rank arrays are stored as np.asarray(memmap) views: the view itself has
    # no _mmap, the mapping lives on its .base chain. Walk the chain so the
    # guard can actually evict rank pages instead of only event_track.
    seen: set[int] = set()
    current = obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        mapping = getattr(current, "_mmap", None)
        if mapping is not None:
            try:
                mapping.madvise(mmap_module.MADV_DONTNEED)
            except (BufferError, ValueError, OSError):
                pass
            return
        current = getattr(current, "base", None)


def _guard_trainer_memory(datasets: dict[str, "SequenceData"]) -> None:
    """Evict mapped shard caches before the guest or Windows host is starved.

    The paired trainer maps event tracks and (uncompressed) rank matrices; on
    the large MOT20 pool this page cache can exceed the safe budget even when
    process RSS looks moderate. Evict on pressure, and fail the run cleanly if
    RSS still exceeds the hard budget instead of letting the VM die.
    """
    limit = float(os.environ.get("SHUTRACK_MAX_RSS_GB", "14"))
    rss = _process_rss_gb()
    if rss <= limit:
        return
    for data in datasets.values():
        _drop_memmap(data.event_track)
        rank_arrays = getattr(data, "_rank_arrays", None)
        if rank_arrays:
            for array in rank_arrays:
                _drop_memmap(array)
    rss_after = _process_rss_gb()
    print(
        f"[memory-guard] rss {rss:.1f} -> {rss_after:.1f} GB "
        f"(limit {limit:.1f})",
        flush=True,
    )
    if rss_after > limit + 1.0:
        raise MemoryError(
            f"memory guard: trainer RSS {rss_after:.1f} GB exceeds {limit:.1f} GB; "
            "stopping before the host is exhausted"
        )


REQUIRED_REPLAY_SCHEMA = 3


def validate_replay_summary(
    summary: dict,
    group: str,
    from_scratch: bool,
    expected_split: str,
    sequence: str,
    dataset: str = "SportsMOT",
) -> dict:
    """Gate formal training on the current label semantics (schema 3)."""
    protocol = summary.get("protocol") or {}
    if from_scratch and not (
        summary.get("complete")
        and protocol.get("full_write")
        and protocol.get("schema") == REQUIRED_REPLAY_SCHEMA
        and summary.get("split") == expected_split
    ):
        raise ValueError(
            f"{sequence}: scratch training requires split-matched raw full-write "
            f"replay with label semantics schema {REQUIRED_REPLAY_SCHEMA}"
        )
    if dataset != "SportsMOT" and protocol.get("dataset", "SportsMOT") != dataset:
        raise ValueError(f"{sequence}: replay dataset does not match {dataset}")
    if protocol.get("full_write") and not from_scratch and not protocol.get("schema"):
        raise ValueError(
            f"{sequence}: full-write replay requires weights-only initialization "
            "with label semantics schema 3"
        )
    if (
        protocol
        and group in ("appearance_rank", "appearance_row_column", "appearance_rank_speed4")
        and not protocol.get("rank_arrays")
    ):
        raise ValueError(f"{sequence}: rank features missing")
    if group == "appearance_row_column" and not protocol.get("column"):
        raise ValueError(f"{sequence}: column candidates missing")
    if group == "motion_gap3" and not protocol.get("gap3"):
        raise ValueError(f"{sequence}: third-frame targets missing")
    return protocol


def resolve_model_options(args) -> tuple[str, int]:
    """Resolve the shared mainline default; legacy runs opt out explicitly."""
    variant = getattr(args, "architecture_variant", MAINLINE_ARCHITECTURE_VARIANT)
    if variant not in (LEGACY_ARCHITECTURE_VARIANT, MAINLINE_ARCHITECTURE_VARIANT):
        raise ValueError(f"Unknown architecture_variant: {variant!r}")
    width = getattr(args, "appearance_proj_dim", None)
    if width is None:
        width = (MAINLINE_APPEARANCE_PROJ_DIM
                 if variant == MAINLINE_ARCHITECTURE_VARIANT
                 else LEGACY_APPEARANCE_PROJ_DIM)
    width = int(width)
    if width <= 0:
        raise ValueError("appearance_proj_dim must be positive")
    return variant, width


def resolve_dropout(args, init_config: dict | None = None) -> float:
    """An explicit --dropout wins; otherwise an init checkpoint's value carries over."""
    requested = getattr(args, "dropout", None)
    if requested is not None:
        return float(requested)
    if init_config is not None:
        return float(init_config.get("dropout", 0.0))
    return 0.0


def train(group: str, args) -> None:
    architecture_variant, appearance_proj_dim = resolve_model_options(args)
    motion_harm_weight = float(getattr(args, "motion_harm_weight", 1.0))
    run_started = time.time()
    dataset = args.dataset
    data_root = Path(args.data_root)
    groups_root = Path(args.groups_root)
    group_dir = groups_root / group
    checkpoint_dir = group_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    metrics_path = group_dir / "metrics.jsonl"
    log_path = group_dir / "training.log"
    if last_path.exists() and not args.resume:
        raise SystemExit(f"{last_path} exists; pass --resume or use a new group dir")

    sequence_dirs = sorted(
        path.name for path in data_root.iterdir() if (path / "samples.npz").is_file()
    )
    raw_sequence_dirs = set(sequence_dirs)
    if dataset == "DanceTrack":
        split_name = args.split or "train"
        if split_name not in ("train", "trainval"):
            raise ValueError(f"DanceTrack supports train/trainval, got {split_name}")
        expected = official_sequences("train", dataset)
        if split_name == "trainval":
            expected |= official_sequences("val", dataset)
        val_names = official_sequences("val", dataset)
        expected_split_of = lambda sequence: "val" if sequence in val_names else "train"
    elif dataset == "SportsMOT":
        split_name = args.split or ("train" if args.from_scratch else "trainval")
        if split_name not in ("train", "trainval"):
            raise ValueError(f"SportsMOT supports train/trainval, got {split_name}")
        expected = official_sequences("train", dataset)
        if split_name == "trainval":
            expected |= official_sequences("val", dataset)
        val_names = official_sequences("val", dataset)
        expected_split_of = lambda sequence: "val" if sequence in val_names else "train"
    else:
        if args.split not in (None, "train"):
            raise ValueError(f"{dataset} only supports --split train")
        split_name = "train"
        expected = official_sequences("train", dataset)
        val_names = set()
        expected_split_of = lambda sequence: "all"
    if args.sequences:
        if args.limit_sequences:
            raise ValueError("Use either --sequences or --limit-sequences, not both")
        requested = list(dict.fromkeys(args.sequences))
        unknown = sorted(set(requested) - expected)
        if unknown:
            raise ValueError(
                f"Requested sequences are not in the official {split_name} split: {unknown}"
            )
        expected = set(requested)
        missing = sorted(expected - raw_sequence_dirs)
        extras = sorted(raw_sequence_dirs - expected)
        if missing or extras:
            raise ValueError(
                f"Generated shards do not match the requested list: missing {missing}, extras {extras}"
            )
        sequence_dirs = sorted(expected)
    else:
        sequence_dirs = sorted(raw_sequence_dirs & expected)
        if args.limit_sequences <= 0 and set(sequence_dirs) != expected:
            raise ValueError(f"Missing training shards: {sorted(expected - set(sequence_dirs))}")
        if args.limit_sequences > 0:
            sequence_dirs = sequence_dirs[: args.limit_sequences]
    if not sequence_dirs:
        raise SystemExit("no generated sequences")
    # Allow max_epochs to override EPOCHS for long training runs.
    epochs_total = args.max_epochs if args.max_epochs > 0 else EPOCHS
    shards = int(getattr(args, "shards", SHARDS))
    sequence_sampling = str(
        getattr(args, "sequence_sampling", "sample-proportional")
    )
    if shards < 1:
        raise ValueError("--shards must be positive")
    if sequence_sampling not in ("sample-proportional", "sqrt-size"):
        raise ValueError(f"Unknown sequence sampling: {sequence_sampling}")
    if epochs_total % shards and not getattr(args, "allow_partial_cycles", False):
        raise ValueError(
            f"--max-epochs ({epochs_total}) must be divisible by --shards ({shards})"
        )
    sampling_power = 0.5 if sequence_sampling == "sqrt-size" else 1.0
    taus = load_global_taus(sequence_dirs, data_root)
    datasets: dict[str, SequenceData] = {}
    targets = {}
    protocols = set()
    summaries: dict[str, tuple[dict, str]] = {}
    for sequence in sequence_dirs:
        summary = json.loads((data_root / sequence / "summary.json").read_text())
        expected_split = expected_split_of(sequence)
        protocol = validate_replay_summary(
            summary, group, args.from_scratch, expected_split, sequence, dataset
        )
        protocols.add(json.dumps(protocol, sort_keys=True))
        if len(protocols) != 1:
            raise ValueError("Incompatible replay protocols in the selected sequences")
        summaries[sequence] = (summary, expected_split)
    for sequence in sequence_dirs:
        summary, expected_split = summaries[sequence]
        split = summary.get("split")
        if split not in ("train", "val", "all"):
            split = expected_split
        data = SequenceData(sequence, split, data_root, dataset)
        datasets[sequence] = data
        targets[sequence] = derive_targets(data.samples, taus)
    preload_features = bool(getattr(args, "preload_features", False))
    load_seconds = 0.0
    preload_rss_gb = None
    if preload_features:
        preload_started = time.time()
        for data in datasets.values():
            data.preload()
        load_seconds = time.time() - preload_started
        preload_rss_gb = round(_process_rss_gb(), 3)
    batch_size = int(args.batch_size)
    epoch_steps = build_schedule(
        datasets, batch_size, epochs_total, shards, sequence_sampling, sampling_power
    )
    total_steps = int(sum(epoch_steps))
    warmup_override = int(getattr(args, "warmup_steps", -1))
    warmup_steps = int(epoch_steps[0]) if warmup_override < 0 else warmup_override
    warmup_steps = max(0, min(warmup_steps, total_steps))
    stage_features = bool(getattr(args, "stage_features", MAINLINE_STAGE_FEATURES))
    if preload_features:
        stage_features = False
    stage_chunk_samples = aligned_stage_chunk_samples(
        int(getattr(args, "stage_chunk_samples", MAINLINE_STAGE_CHUNK_SAMPLES)),
        batch_size,
    )
    loading_mode = ("preload" if preload_features else
                    ("stage" if stage_features else "mmap"))

    full_write = bool(
        protocols and json.loads(next(iter(protocols))).get("full_write")
    )
    dropout = getattr(args, "dropout", None)
    if args.from_scratch:
        dropout = 0.0 if dropout is None else float(dropout)
        _seed_everything(int(args.seed))
        dims = {seq.event_track.shape[1] for seq in datasets.values()}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent ReID dimensions: {dims}")
        model = ShutrackV2Model(
            ShutrackV2Config(
                reid_dim=int(dims.pop()),
                appearance_proj_dim=appearance_proj_dim,
                architecture_variant=architecture_variant,
                dropout=dropout,
            )
        )
        norm_mean, norm_std = fit_input_statistics(datasets)
        for data in datasets.values():
            data.input_statistics = (norm_mean, norm_std)
        init_mode = "scratch"
        normalization_source = "fit_target"
    else:
        _seed_everything(int(args.seed))
        init_model, init_checkpoint = load_shutrack_checkpoint(args.init, map_location="cpu")
        init_config = init_model.config.as_dict()
        dropout = resolve_dropout(args, init_config)
        init_proj = int(init_config.get("appearance_proj_dim", 64))
        if init_config.get("architecture_variant", "legacy") != architecture_variant:
            raise ValueError("Init checkpoint architecture_variant does not match the requested model; use --from-scratch for a new architecture")
        if init_proj != appearance_proj_dim:
            raise ValueError(
                f"Init checkpoint appearance_proj_dim={init_proj} does not match "
                f"--appearance-proj-dim={appearance_proj_dim}"
            )
        model = ShutrackV2Model(ShutrackV2Config(**{**init_config, "dropout": dropout}))
        _load_init_state_with_extension(model, init_model)
        if full_write:
            # Weights-only initialization on a full-write replay: the source
            # checkpoint's scalar statistics belong to a different dataset, so
            # refit them on this target replay and use them at inference.
            norm_mean, norm_std = fit_input_statistics(datasets)
            for data in datasets.values():
                data.input_statistics = (norm_mean, norm_std)
            init_mode = "weights_only_refit"
            normalization_source = "fit_target"
        else:
            norm_mean = init_checkpoint["normalization_mean"]
            norm_std = init_checkpoint["normalization_std"]
            init_mode = "weights_only"
            normalization_source = "checkpoint"
    device = torch.device(args.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, scheduler_lambda(total_steps, warmup_steps)
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if (payload.get("model_config") or {}).get("architecture_variant", "legacy") != architecture_variant:
            raise ValueError("Resume checkpoint architecture_variant does not match the requested model")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        rng_state = payload.get("rng_state")
        if rng_state:
            torch.set_rng_state(rng_state["torch"])
            cuda_state = rng_state.get("cuda") or []
            if cuda_state and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(cuda_state)
                except (RuntimeError, ValueError):
                    pass
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            print("resume rng state: restored", flush=True)
        else:
            print("resume rng state: unavailable in checkpoint", flush=True)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        print(f"resume from epoch {start_epoch} step {global_step}", flush=True)

    config = {
        "group": group,
        "dataset": dataset,
        "data_root": str(data_root),
        "sequences": len(sequence_dirs),
        "batch_size": batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "shards": shards,
        "cycles": epochs_total // shards,
        "full_passes": epochs_total / shards,
        "sequence_sampling": sequence_sampling,
        "sequence_sampling_power": sampling_power,
        "sampling_replacement": sequence_sampling == "sqrt-size",
        "sequence_counts": {sequence: data.n for sequence, data in datasets.items()},
        "epochs": epochs_total,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "init": None if args.from_scratch else str(args.init),
        "from_scratch": args.from_scratch,
        "init_mode": init_mode,
        "normalization_source": normalization_source,
        "split": split_name,
        "sequence_names": sequence_dirs,
        "replay_protocol": json.loads(next(iter(protocols))),
        "taus": taus,
        "rank_tau": args.rank_tau,
        "rank_alpha": args.rank_alpha,
        "column_weight": args.column_weight,
        "column_hinge": args.column_hinge,
        "appearance_proj_dim": appearance_proj_dim,
        "architecture_variant": architecture_variant,
        "rank_weight": args.rank_weight,
        "motion_harm_weight": motion_harm_weight,
        "speed_weight": args.speed_weight,
        "size_weight": args.size_weight,
        "gap3_weight": args.gap3_weight,
        "snapshot_every": args.snapshot_every,
        "dropout": dropout,
        "preload_features": preload_features,
        "stage_features": stage_features,
        "stage_chunk_samples": stage_chunk_samples,
        "loading_mode": loading_mode,
        "warmup_steps_override": warmup_override,
        "grad_norm_every": int(getattr(args, "grad_norm_every", 0)),
    }
    if args.resume:
        previous_config = dict(payload.get("training_config") or {})
        previous_config.setdefault("column_hinge", False)
        previous_config.setdefault("motion_harm_weight", 1.0)
        previous_config.setdefault("appearance_proj_dim", 64)
        previous_config.setdefault("architecture_variant", "legacy")
        previous_sampling = previous_config.get(
            "sequence_sampling", "sample-proportional"
        )
        previous_config.setdefault(
            "sequence_sampling_power",
            1.0 if previous_sampling == "sample-proportional" else 0.5,
        )
        previous_config.setdefault(
            "sampling_replacement", previous_sampling == "sqrt-size"
        )
        previous_config.setdefault("dropout", 0.0)
        previous_config.setdefault("preload_features", False)
        previous_config.setdefault("warmup_steps_override", -1)
        previous_config.setdefault("grad_norm_every", 0)
        previous_config.setdefault("sequence_counts", config["sequence_counts"])
        loader_keys = ("preload_features", "stage_features", "stage_chunk_samples",
                       "loading_mode")
        current_compare = {key: value for key, value in config.items()
                           if key not in loader_keys}
        previous_compare = {key: value for key, value in previous_config.items()
                            if key not in loader_keys}
        if previous_compare != current_compare:
            raise ValueError("Resume config does not match the original run")
        payload_proj = int((payload.get("model_config") or {}).get("appearance_proj_dim", 64))
        if payload_proj != appearance_proj_dim:
            raise ValueError(
                f"Resume checkpoint appearance_proj_dim={payload_proj} does not match "
                f"--appearance-proj-dim={appearance_proj_dim}"
            )
    (group_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    log_handle = log_path.open("a" if args.resume else "w")
    metrics_handle = metrics_path.open("a" if args.resume else "w")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    model_param_numel = sum(p.numel() for p in model.parameters())
    optimizer_param_numel = sum(
        p.numel() for group_ in optimizer.param_groups for p in group_["params"]
    )
    if model_param_numel != optimizer_param_numel:
        raise RuntimeError("not all model parameters entered the optimizer")
    dropout_modules = [
        (name, float(module.p)) for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.p > 0
    ]
    parameter_digest = hashlib.sha256()
    for parameter_name, parameter in sorted(model.named_parameters()):
        parameter_digest.update(parameter_name.encode("utf-8"))
        parameter_digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    log("verification: " + json.dumps({
        "model_param_numel": model_param_numel,
        "optimizer_param_numel": optimizer_param_numel,
        "dropout_modules": dropout_modules,
        "model_training": model.training,
        "model_config": model.config.as_dict(),
        "initial_parameter_sha256": parameter_digest.hexdigest(),
    }, sort_keys=True))
    if preload_features:
        log("preload: " + json.dumps({
            "seconds": round(load_seconds, 2),
            "rss_gb": preload_rss_gb,
        }, sort_keys=True))

    prefetch = int(getattr(args, "prefetch", 1))
    pin_memory = device.type == "cuda" and not getattr(args, "no_pin_memory", False)
    compact_reid = not getattr(args, "no_compact_reid", False)
    for sequence in sequence_dirs:
        datasets[sequence].open_cache()
    # Small pools retain their warm mmap cache. Stage large pools whose file
    # working set competes with model/batch memory; no dataset-specific switch.
    feature_bytes = sum(
        data.event_track.nbytes + data.cache.features.nbytes
        + (data.dir / "samples.npz").stat().st_size
        for data in datasets.values()
    )
    log(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), **config}, sort_keys=True))
    # Loader options affect execution only, not the resume/training protocol.
    log("pipeline: " + json.dumps({"prefetch": prefetch, "pin_memory": pin_memory,
                                  "compact_reid": compact_reid,
                                  "loading_mode": loading_mode,
                                  "stage_features": stage_features,
                                  "stage_chunk_samples": stage_chunk_samples,
                                  "feature_working_set_gib": feature_bytes / 2**30}))
    seed_rng = np.random.RandomState(int(args.seed))
    grad_norm_every = int(getattr(args, "grad_norm_every", 0))
    for epoch in range(start_epoch + 1, epochs_total + 1):
        epoch_started = time.time()
        probe: dict[str, float] = {}
        _guard_trainer_memory(datasets)
        shard = (epoch - 1) % shards
        order, rows_by_sequence, quotas, shard_counts = plan_phase(
            datasets,
            shard,
            shards,
            sequence_sampling,
            sampling_power,
            int(args.seed),
            epoch,
        )
        log("epoch order: " + json.dumps({"epoch": epoch, "sequences": order}))
        totals = {
            "loss": 0.0,
            "motion_loss": 0.0,
            "appearance_loss": 0.0,
            "samples": 0,
            "batches": 0,
        }
        gate_sum = 0.0
        appearance_gate_sum = 0.0
        gate_count = 0
        column_valid_total = 0
        column_active_total = 0
        accumulator = DeferredMetrics(device)
        epoch_staging_seconds = 0.0
        epoch_staging_chunks = 0
        epoch_staging_peak_bytes = 0
        for sequence in order:
            data = datasets[sequence]
            try:
                rows = rows_by_sequence[sequence]
                if rows.size == 0:
                    continue
                sequence_staging_seconds = 0.0
                sequence_staging_chunks = 0
                sequence_staging_peak_bytes = 0
                for chunk_rows in feature_chunks(rows, stage_chunk_samples, enabled=stage_features):
                    with staged_sequence_rows(data, chunk_rows, group, enabled=stage_features) as staged, prepared_batches(
                        chunk_rows, batch_size,
                        lambda selected: batch_tensors(data, selected, targets[sequence], group,
                                                      compact_reid=compact_reid),
                        PAIRED_TENSOR_KEYS,
                        prefetch=prefetch, pin_memory=pin_memory,
                    ) as batches:
                        if stage_features:
                            sequence_staging_seconds += staged["seconds"]
                            sequence_staging_chunks += 1
                            sequence_staging_peak_bytes = max(
                                sequence_staging_peak_bytes, staged["bytes"]
                            )
                        for batch_rows, cpu_batch in batches:
                            batch = to_device(cpu_batch, device)
                            outputs = model(**model_inputs(batch))
                            extra: dict[str, torch.Tensor | float] = {}
                            if group in ("appearance_rank", "appearance_row_column"):
                                motion_loss, motion_components = motion_box_diou_loss(
                                    outputs, batch, harm_weight=motion_harm_weight
                                )
                                if motion_harm_weight != 1.0:
                                    extra["motion_unweighted_diou"] = motion_components["unweighted_diou_loss"]
                                    extra["motion_harm_fraction"] = motion_components["harm_fraction"]
                                    extra["motion_harm_weight_share"] = motion_components["harm_weight_share"]
                                appearance_loss, appearance_components = appearance_rank_loss(
                                    outputs,
                                    batch,
                                    alpha=args.rank_alpha,
                                    tau=args.rank_tau,
                                    rank_weight=args.rank_weight,
                                )
                                if group == "appearance_row_column":
                                    column_components: dict[str, object] = {}
                                    column = appearance_column_loss(
                                        outputs, batch, alpha=args.rank_alpha,
                                        tau=args.rank_tau, hinge=args.column_hinge,
                                        components=column_components, tensor_statistics=True,
                                    )
                                    accumulator.add("column_valid", column_components["valid"])
                                    accumulator.add("column_active", column_components["active"])
                                    appearance_loss = appearance_loss + args.column_weight * column
                                    extra["column_loss"] = column.detach()
                                loss = motion_loss + appearance_loss
                                first_batch = totals["batches"] == 0
                                if (first_batch and grad_norm_every > 0
                                        and (epoch == 1 or epoch % grad_norm_every == 0)):
                                    trunk = shared_parameters(model)
                                    probe["grad_norm_motion_shared"] = loss_grad_norm(motion_loss, trunk)
                                    probe["grad_norm_appearance_shared"] = loss_grad_norm(appearance_loss, trunk)
                                if first_batch and dropout > 0 and epoch == 1:
                                    cpu_state = torch.get_rng_state()
                                    cuda_state = (torch.cuda.get_rng_state_all()
                                                  if device.type == "cuda" else None)
                                    with torch.no_grad():
                                        first = model(**model_inputs(batch))["gate_logits"]
                                        second = model(**model_inputs(batch))["gate_logits"]
                                    torch.set_rng_state(cpu_state)
                                    if cuda_state is not None:
                                        torch.cuda.set_rng_state_all(cuda_state)
                                    probe["dropout_probe_max_abs_diff"] = float(
                                        (first - second).abs().max())
                                extra["rank_mean"] = appearance_components["rank_mean"]
                                extra["fallback_mean"] = appearance_components["fallback_mean"]
                            elif group == "appearance_rank_speed4":
                                box_loss, motion_components = motion_box_diou_loss(outputs, batch)
                                speed_loss, speed_components = motion_speed4_l1_loss(
                                    outputs,
                                    batch,
                                    center_weight=args.speed_weight,
                                    size_weight=args.size_weight,
                                )
                                motion_loss = box_loss + speed_loss
                                appearance_loss, appearance_components = appearance_rank_loss(
                                    outputs,
                                    batch,
                                    alpha=args.rank_alpha,
                                    tau=args.rank_tau,
                                    rank_weight=args.rank_weight,
                                )
                                loss = motion_loss + appearance_loss
                                extra["speed_center_l1"] = speed_components["speed_center_l1"]
                                extra["speed_size_l1"] = speed_components["speed_size_l1"]
                                extra["rank_mean"] = appearance_components["rank_mean"]
                                extra["fallback_mean"] = appearance_components["fallback_mean"]
                            elif group == "motion_diou_speed":
                                box_loss, motion_components = motion_box_diou_loss(outputs, batch)
                                speed_loss, speed_components = motion_speed_l1_loss(
                                    outputs, batch, weight=args.speed_weight
                                )
                                motion_loss = box_loss + speed_loss
                                appearance_loss, appearance_components = appearance_soft_loss(outputs, batch)
                                loss = motion_loss + appearance_loss
                                extra["speed_l1"] = speed_components["speed_l1"]
                            elif group == "motion_diou_speed4":
                                box_loss, motion_components = motion_box_diou_loss(outputs, batch)
                                speed_loss, speed_components = motion_speed4_l1_loss(
                                    outputs,
                                    batch,
                                    center_weight=args.speed_weight,
                                    size_weight=args.size_weight,
                                )
                                motion_loss = box_loss + speed_loss
                                appearance_loss, appearance_components = appearance_soft_loss(outputs, batch)
                                loss = motion_loss + appearance_loss
                                extra["speed_center_l1"] = speed_components["speed_center_l1"]
                                extra["speed_size_l1"] = speed_components["speed_size_l1"]
                            elif group in ("motion_diou", "motion_diou_appearance_score", "motion_gap3"):
                                motion_loss, motion_components = motion_box_diou_loss(outputs, batch)
                                if group == "motion_gap3":
                                    boxes = batch["gap3_boxes"]
                                    horizon = {"pred_box_g0": boxes[:, 0], "pred_box_g1": boxes[:, 1],
                                               "gt_box_t1": boxes[:, 2], "motion_gt_valid": batch["gap3_valid"]}
                                    extra_loss, _ = motion_box_diou_loss(outputs, horizon)
                                    motion_loss = motion_loss + args.gap3_weight * extra_loss
                                    extra["gap3_loss"] = extra_loss.detach()
                                appearance_loss, appearance_components = appearance_soft_loss(outputs, batch)
                                loss = motion_loss + appearance_loss
                            else:
                                loss, components = gate_loss(outputs, batch)
                                motion_loss = loss
                                appearance_loss = loss
                                motion_components = components
                                appearance_components = components
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            global_step += 1
                            if global_step % 10 == 0:
                                _guard_trainer_memory(datasets)
                            count = int(batch_rows.size)
                            accumulator.add("loss", loss, count)
                            accumulator.add("motion_loss", motion_loss, count)
                            accumulator.add("appearance_loss", appearance_loss, count)
                            totals["samples"] += count
                            totals["batches"] += 1
                            for key, value in extra.items():
                                accumulator.add(key, value, count)
                            with torch.no_grad():
                                gate_means = torch.sigmoid(outputs["gate_logits"]).mean(dim=0)
                            accumulator.add("motion_gate", gate_means[0], count)
                            accumulator.add("appearance_gate", gate_means[1], count)
                            gate_count += count
                if stage_features:
                    epoch_staging_seconds += sequence_staging_seconds
                    epoch_staging_chunks += sequence_staging_chunks
                    epoch_staging_peak_bytes = max(
                        epoch_staging_peak_bytes, sequence_staging_peak_bytes
                    )
                    log("feature stage: " + json.dumps({
                        "sequence": sequence, "chunks": sequence_staging_chunks,
                        "seconds": sequence_staging_seconds,
                        "peak_chunk_bytes": sequence_staging_peak_bytes,
                    }))
            except Exception:
                raise
        if totals["batches"] != epoch_steps[epoch - 1] or totals["samples"] != sum(quotas.values()):
            raise RuntimeError(
                f"Epoch {epoch} differs from the sample plan: "
                f"{totals['batches']}/{epoch_steps[epoch - 1]} batches, "
                f"{totals['samples']}/{sum(quotas.values())} samples"
            )
        sums = accumulator.summary()
        for name in ("loss", "motion_loss", "appearance_loss"):
            totals[name] = sums.get(name, 0.0)
        for name in ("rank_mean", "fallback_mean", "speed_l1", "speed_center_l1",
                     "speed_size_l1", "gap3_loss", "column_loss",
                     "motion_unweighted_diou", "motion_harm_fraction", "motion_harm_weight_share"):
            if name in sums:
                totals[name] = sums[name]
        gate_sum = sums.get("motion_gate", 0.0)
        appearance_gate_sum = sums.get("appearance_gate", 0.0)
        column_valid_total = int(sums.get("column_valid", 0))
        column_active_total = int(sums.get("column_active", 0))
        epoch_seconds = time.time() - epoch_started
        metrics = {
            "epoch": epoch,
            "shard": shard + 1,
            "cycle": (epoch - 1) // shards + 1,
            "global_step": global_step,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": epoch_seconds,
            "samples": totals["samples"],
            "batches": totals["batches"],
            "loss": totals["loss"] / max(totals["samples"], 1),
            "motion_loss": totals["motion_loss"] / max(totals["samples"], 1),
            "appearance_loss": totals["appearance_loss"] / max(totals["samples"], 1),
            "motion_gate_mean": gate_sum / max(gate_count, 1) if gate_count else None,
            "appearance_gate_mean": (
                appearance_gate_sum / max(gate_count, 1) if gate_count else None
            ),
        }
        metrics["staging_seconds"] = epoch_staging_seconds
        metrics["staging_chunks"] = epoch_staging_chunks
        metrics["staging_peak_chunk_bytes"] = epoch_staging_peak_bytes
        metrics["sampling_policy"] = sequence_sampling
        metrics["sampling_quota"] = {
            sequence: int(quotas[sequence]) for sequence in quotas
        }
        metrics.update(probe)
        metrics["rss_gb"] = round(_process_rss_gb(), 3)
        metrics["rss_peak_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 3)
        if device.type == "cuda":
            metrics["cuda_peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 20, 2)
            metrics["cuda_peak_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 2 ** 20, 2)
        for key in (
            "rank_mean",
            "fallback_mean",
            "speed_l1",
            "speed_center_l1",
            "speed_size_l1",
            "gap3_loss",
            "column_loss",
            "motion_unweighted_diou",
            "motion_harm_fraction",
            "motion_harm_weight_share",
        ):
            if key in totals:
                metrics[key] = totals[key] / max(totals["samples"], 1)
        if group == "appearance_row_column":
            metrics["column_valid_count"] = column_valid_total
            metrics["column_active_fraction"] = (
                column_active_total / column_valid_total if column_valid_total else None
            )
        log("epoch summary: " + json.dumps(metrics, sort_keys=True))
        metrics_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        metrics_handle.flush()

        payload = build_checkpoint_payload(
            model=model,
            training_config=config,
            dataset_descriptor={
                "dataset": dataset,
                "split": split_name,
                "dataset_dir": str(data_root),
                "competition_enabled": True,
                "label_source": "paired_supervision_pilot",
            },
            epoch=epoch,
            global_step=global_step,
            best_metric=metrics["loss"],
            optimizer=optimizer,
            scheduler=scheduler,
            normalization_mean=norm_mean,
            normalization_std=norm_std,
            include_optimizer=True,
        )
        save_checkpoint_atomic(last_path, payload)
        snapshot_every = max(1, int(args.snapshot_every))
        if epoch % snapshot_every == 0 or epoch == epochs_total:
            save_checkpoint_atomic(checkpoint_dir / f"epoch_{epoch:03d}.pt", payload)
            log(f"saved epoch_{epoch:03d}.pt at step {global_step}")
    log_handle.close()
    metrics_handle.close()
    for data in datasets.values():
        data.close_cache()
    (group_dir / "complete.json").write_text(
        json.dumps(
            {
                "group": group,
                "dataset": dataset,
                "epochs": epochs_total,
                "global_step": global_step,
                "seconds": time.time() - run_started,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        required=True,
        choices=[
            "control",
            "motion_diou",
            "appearance_score",
            "motion_diou_appearance_score",
            "appearance_rank",
            "appearance_row_column",
            "appearance_rank_speed4",
            "motion_diou_speed",
            "motion_diou_speed4",
            "motion_gap3",
        ],
    )
    parser.add_argument("--dataset", choices=("SportsMOT", "MOT17", "MOT20", "DanceTrack"), default="SportsMOT")
    init_choice = parser.add_mutually_exclusive_group()
    init_choice.add_argument("--init", default=str(INIT_CKPT))
    init_choice.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--split", choices=("train", "trainval"))
    parser.add_argument("--snapshot-every", type=int, default=20)
    parser.add_argument("--shards", type=int, default=SHARDS)
    parser.add_argument(
        "--sequence-sampling",
        choices=("sample-proportional", "sqrt-size"),
        default="sample-proportional",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefetch", type=int, default=1,
                        help="Ordered CPU batch prefetch depth; 0 disables the producer.")
    parser.add_argument("--no-compact-reid", action="store_true",
                        help="Transport source ReID values as float32 instead of exact float16.")
    staging = parser.add_mutually_exclusive_group()
    staging.add_argument("--stage-features", dest="stage_features", action="store_true",
                         default=MAINLINE_STAGE_FEATURES,
                         help="Stage selected feature rows in bounded RAM chunks (mainline default).")
    staging.add_argument("--no-stage-features", dest="stage_features", action="store_false",
                         help="Read mmap features per batch; use only for performance comparisons.")
    parser.add_argument("--stage-chunk-samples", type=int, default=MAINLINE_STAGE_CHUNK_SAMPLES,
                        help="Target samples per staged chunk, aligned to complete batches.")
    parser.add_argument("--no-pin-memory", action="store_true",
                        help="Disable pinned host batches for comparison/debugging.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-sequences", type=int, default=0)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Explicit training sequence list (must match the generated shards exactly).",
    )
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--data-root", default=str(DATA))
    parser.add_argument("--groups-root", default=str(ROOT / "groups"))
    parser.add_argument("--rank-tau", type=float, default=0.1)
    parser.add_argument("--rank-alpha", type=float, default=0.95)
    parser.add_argument("--column-weight", type=float, default=1.0)
    parser.add_argument("--column-hinge", action="store_true")
    parser.add_argument(
        "--appearance-proj-dim",
        type=int,
        default=None,
        help="Appearance projection width; default 256 for query_film_bilinear, 64 for legacy.",
    )
    parser.add_argument("--architecture-variant", choices=("legacy", "query_film_bilinear"),
                        default=MAINLINE_ARCHITECTURE_VARIANT,
                        help="Model architecture; legacy reproduction runs must pass 'legacy'.")
    parser.add_argument("--gap3-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--motion-harm-weight", type=float, default=1.0,
                        help="Reweight GT-harmful motion events in B/D training; 1.0 keeps the original loss.")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout inside the query HCE stack (query_film_bilinear); "
                             "default 0.0 from scratch, or the init checkpoint's value.")
    parser.add_argument("--preload-features", action="store_true",
                        help="Copy mapped event/rank features into resident RAM before training.")
    parser.add_argument("--warmup-steps", type=int, default=-1,
                        help="Scheduler warmup steps; -1 uses the first epoch's step count.")
    parser.add_argument("--grad-norm-every", type=int, default=0,
                        help="Log motion/appearance trunk gradient norms on the first batch "
                             "every N epochs (0 disables).")
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial-cycles", action="store_true",
                        help="Allow --max-epochs not divisible by --shards (last cycle partial).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(4)
    train(args.group, args)


if __name__ == "__main__":
    main()
