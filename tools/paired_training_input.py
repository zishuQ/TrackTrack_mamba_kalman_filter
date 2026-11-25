"""Inputs for a fresh SHUTrack run with full-write replay and train-only statistics."""

from pathlib import Path

import numpy as np
import torch

from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.schema import COMPETITION_SCALAR_INDICES, EVIDENCE_SCALAR_INDICES, LEGACY_SCALAR_DIM
from shutrack.training.checkpoint import build_checkpoint_payload, save_checkpoint_atomic


def official_sequences(split, dataset="SportsMOT"):
    if dataset in ("MOT17", "MOT20"):
        if split == "val":
            return set()
        if split not in ("train", "trainval"):
            raise ValueError(f"Unsupported training split: {split}")
        # These are the complete official training videos, with a single
        # private detector stream per MOT17 video (no DPM/SDP duplication).
        if dataset == "MOT17":
            return {f"MOT17-{i:02d}-FRCNN" for i in (2, 4, 5, 9, 10, 11, 13)}
        return {f"MOT20-{i:02d}" for i in (1, 2, 3, 5)}
    seqmap = "dancetrack" if dataset == "DanceTrack" else "sportsmot"
    base = Path(__file__).resolve().parents[1] / "3. Tracker/trackeval/seqmap" / seqmap
    path = base / f"{split}.txt"
    if not path.is_file():
        raise ValueError(f"Unsupported split {split!r} for {dataset}")
    return {name.strip() for name in path.read_text().splitlines()
            if name.strip() and name.strip() != "name"}


def replay_split(sequence, dataset="SportsMOT"):
    if dataset == "DanceTrack":
        return "val" if sequence in official_sequences("val", dataset) else "train"
    if dataset != "SportsMOT":
        return "all"
    return "val" if sequence in official_sequences("val") else "train"


def full_write_checkpoint(path, reid_dim):
    # The normal runtime captures windows. Both write gates are forced to one.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = ShutrackV2Model(ShutrackV2Config(reid_dim=int(reid_dim)))
    payload = build_checkpoint_payload(
        model=model, training_config={}, dataset_descriptor={"competition_enabled": True},
        epoch=0, global_step=0, best_metric=0.0, optimizer=None, scheduler=None,
        normalization_mean=np.zeros(LEGACY_SCALAR_DIM), normalization_std=np.ones(LEGACY_SCALAR_DIM),
        include_optimizer=False,
    )
    save_checkpoint_atomic(path, payload)


def fit_input_statistics(datasets):
    mean = np.zeros(LEGACY_SCALAR_DIM)
    std = np.ones(LEGACY_SCALAR_DIM)
    for columns, competition in ((EVIDENCE_SCALAR_INDICES, False), (COMPETITION_SCALAR_INDICES, True)):
        total = np.zeros(len(columns))
        squares = np.zeros(len(columns))
        count = 0
        for seq in datasets.values():
            values = seq.samples["current_competition"] if competition else seq.event_evidence[:, :-1]
            values = np.asarray(values, dtype=np.float64)
            total += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
            count += len(values)
        if not count:
            raise ValueError("Cannot fit normalization without training events")
        cols = list(columns)
        mean[cols] = total / count
        sigma = np.maximum(squares / count - np.square(mean[cols]), 0.0) ** 0.5
        std[cols] = np.where(sigma > 1e-8, sigma, 1.0)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("Non-finite training statistics")
    return mean, std
