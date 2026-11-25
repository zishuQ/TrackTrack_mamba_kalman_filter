"""MoTTrack observation corruption shared by window trainers (pixel cxcywh).

Stateless per-anchor streams make the same sample independent of batching and
prefetch.  Distribution and missing-state transitions follow MoTTrack; boundary
tests explicitly convert center boxes to edges.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservationNoise:
    position_std: float = 1.0
    size_std: float = 1.0
    missing: float = 0.004
    burst: float = 0.0
    stay_missing: float = 0.4
    small_area: float = 2500.0
    boundary_margin: float = 20.0
    burst_min: int = 3
    burst_max: int = 6

    @classmethod
    def for_dataset(cls, dataset):
        return {
            "SportsMOT": cls(),
            "MOT17": cls(missing=0.02, burst=0.02),
            "MOT20": cls(position_std=3.0, size_std=2.0, missing=0.02, burst=0.02),
        }[dataset]

def _uniform(rows, seed, epoch, length, stream):
    # SplitMix64 counters; integer wraparound is intentional.
    with np.errstate(over="ignore"):
        x = (np.asarray(rows, np.uint64)[:, None] * np.uint64(0x9E3779B97F4A7C15)
             + np.arange(length, dtype=np.uint64)[None] * np.uint64(0xD1B54A32D192ED03)
             + np.uint64(seed) + np.uint64(epoch) * np.uint64(0x94D049BB133111EB)
             + np.uint64(stream) * np.uint64(0xBF58476D1CE4E5B9))
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / 2**53

def corrupt_observations(measurements, scores, present, valid, rows, *,
                         seed, epoch, width, height, config):
    """Return corrupted boxes/scores/mask; never create an originally absent detection."""
    boxes = np.asarray(measurements, np.float32).copy()
    original = np.asarray(present, bool) & np.asarray(valid, bool)
    valid = np.asarray(valid, bool)
    n, length = original.shape
    probability = np.full((n, length), config.missing, np.float64)
    probability *= np.where(boxes[..., 2] * boxes[..., 3] < config.small_area, 2., 1.)
    margin = config.boundary_margin
    boundary = ((boxes[..., 0] - boxes[..., 2]/2 < margin)
                | (boxes[..., 1] - boxes[..., 3]/2 < margin)
                | (boxes[..., 0] + boxes[..., 2]/2 > width-margin)
                | (boxes[..., 1] + boxes[..., 3]/2 > height-margin))
    probability *= np.where(boundary, 1.5, 1.)
    probability = np.minimum(probability, 0.5)
    burst_uniform = _uniform(rows, seed, epoch, length, 1)
    burst_length = config.burst_min + np.floor(
        _uniform(rows, seed, epoch, length, 2) * (config.burst_max-config.burst_min+1)
    ).astype(np.int64)
    burst_mask = np.zeros_like(original)
    remaining = np.zeros(n, np.int64)
    for t in range(length):
        start = valid[:, t] & (remaining == 0) & (burst_uniform[:, t] < config.burst)
        remaining = np.where(start, burst_length[:, t], remaining)
        burst_mask[:, t] = valid[:, t] & (remaining > 0)
        remaining = np.maximum(remaining - valid[:, t], 0)
    markov = np.zeros(n, bool)
    keep = original.copy()
    random_missing = _uniform(rows, seed, epoch, length, 3)
    for t in range(length):
        eligible = valid[:, t] & ~burst_mask[:, t]
        sampled = random_missing[:, t] < np.where(markov, config.stay_missing,
                                                probability[:, t])
        markov = np.where(eligible, sampled, markov)
        keep[:, t] &= ~(burst_mask[:, t] | (eligible & sampled))
    # MoTTrack retains the first observation in each trajectory segment.
    first = valid & (np.cumsum(valid, axis=1) == 1)
    keep = np.where(first, original, keep)
    for axis in range(4):
        u = _uniform(rows, seed, epoch, length, 10+2*axis)
        v = _uniform(rows, seed, epoch, length, 11+2*axis)
        normal = np.sqrt(-2*np.log(u)) * np.cos(2*np.pi*v)
        boxes[..., axis] += (normal * (config.position_std if axis < 2
                                      else config.size_std)).astype(np.float32)
    boxes[..., 2:] = np.maximum(boxes[..., 2:], 1.)
    boxes[~keep] = 0.
    new_scores = np.where(keep, np.asarray(scores, np.float32), 0.).astype(np.float32)
    return boxes, new_scores, keep
