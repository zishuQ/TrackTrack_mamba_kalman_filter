# SHUTrack v3 — HCE + RSC

SHUTrack = **Suppress Harmful Updates**.  After a match is accepted, the motion
state and the appearance template should not always be written with full
strength.  This package implements the isolated v3 architecture:

* **HCE — Historical Context Encoder** reads at most the five events before the
  current one.  Each history event is encoded from two paths — motion+observation
  evidence and ReID appearance — and fused into one history token.  A shared
  temporal Transformer (d_model=128, 2 layers, 4 heads) encodes the tokens; two
  learned queries read the motion/observation and appearance history contexts.
* **RSC — Reliability-aware Suppression Controller** uses those two readouts as
  queries and reads the current motion+observation token, the ReID appearance
  token, and (optionally) the association-competition token as key/value.  One
  cross-modal attention layer plus FFN produces two representation deltas, which
  are fused residually in representation space and decoded by two independent
  sigmoid heads into the motion and appearance gates.

There is no policy head, cue/risk head, five-strategy prototype, base/refined
gate, correction/bound, or detached gradient path.  The gates do not use softmax
and are not required to sum to one.  Online writes keep TrackTrack's existing
`Track.update_with_gates`: the motion gate scales the original NSA measurement
correction and the appearance gate scales the original EMA write.

## Input schema (`shutrack_v3_evidence13_competition3`)

Six events = five past events + current event.  Every current input is captured
before the current write.

### Evidence branch — 13 dims (motion + observation fused)

| position | value | legacy 63-dim column |
|---|---|---|
| 1–2 | box residual center offset | 27, 28 |
| 3–4 | box log w/h ratio | 29, 30 |
| 5–8 | normalized KF velocity | 39–42 |
| 9–10 | center prediction variance (log var cx, log var cy) | 43, 44 |
| 11 | detection confidence | 9 |
| 12 | observation gap | 59 |
| 13 | has-detection flag (raw 0/1) | derived from the detection mask |

Removed from the old context vector: duplicated width/height variances, the
constant snapshot history length, and the unused selected-margin values and
validity bits.  A separate has-detection mask is still passed for appearance
masking, together with the history padding/reset mask.

### Competition branch — 3 dims (optional)

Normalized legacy columns **15, 21, 25**:

1. row top-2 cost gap (best-other vs accepted in the track's final-cost row),
2. column top-2 cost gap (best-other vs accepted in the detection's final-cost
   column),
3. maximum IoU between the accepted detection and the other detections in the
   same candidate pool (the accepted detection itself excluded; a single
   candidate yields 0, following the legacy feature definition).

Online inference passes the same recorded candidate pool through as
`track_cost_row`, `detection_cost_col` and `max_iou_with_other`, using the
pre-iteration `final_cost` matrix and the legacy 63-dim scalar definition, then
applies the same 63-dim normalization.  The packed training tensors already
contain these normalized columns, so both sides use identical values.

Training can disable the branch (`--competition off`) for the lean
dual-evidence variant.  The checkpoint records the effective input schema and
`competition_enabled`; the loader rejects a model/checkpoint branch mismatch.

### Appearance branch

Per-event pre-update track ReID + the event's detection ReID; both projected to
64 dims, then `concat(track, det, |diff|, product)` → 128.  Missing detections
use a learned placeholder and an explicit mask.

Packed scalars are already z-scored by the legacy train-split statistics and
are **not** normalized again.  Online inference applies the same 63-dim mean/std
stored in the checkpoint.

## History window and reset semantics

* HCE reads only the past five events; history never crosses track, segment,
  sequence or reset boundaries.
* A single event advances the per-track clock exactly once.  The evidence
  segment id and the buffered events' segment id are always consistent.
* Acceptance: with events at frames 1, 50, 51 and `max_frame_gap=30`, frame 50
  resets once and sees an empty history; frame 51 sees exactly the frame-50
  event in the model-visible history mask (`[T,T,T,T,F]`).
* Unmatched frames are recorded as real history events with `has_detection=False`
  and zero applied gates.  No gate is computed without a match; the gate is
  computed again on a re-match.

## Enablement modes

* **Default** (`--shutrack-min-observations 6`): a track enters the model in the
  tracked/lost path once it has six observations, matching the legacy event
  cache timeline.  Both competition-branch variants use the same default.
* **Early enable** (`--shutrack-early-enable`): independent option.  Match 1
  initializes the track; match 2 uses the original `Track.update` and records
  its pre-update evidence; match 3 first applies the gate inside the New
  association path.  The competition token comes from the New path's own
  candidate pool.  TrackTrack's confirmation rule (`observation_count >=
  min_len`) is unchanged.  Removing an unmatched candidate track clears its
  runtime buffer.
* The old packed training data has no supervised windows with 1–2 history
  events.  Truncated-history training is a separate future option; early-path
  gains on real early tracks must be measured, not assumed.

## Training

The only objective is the existing weighted gate loss:

```text
BCEWithLogits + 0.1 * MSE, normalized by valid_channels * sample_weight
```

No policy KL, cue/risk BCE, base/final duplication or correction terms.

Training reads the existing packed `outputs/agentguard/train_data/*` in place
(`data.pt`, `reid_features.npy`, `safe_gate_target`, `valid_channels`,
`sample_weight`); no cache, index, label or dataset is generated.  Batches are
organized per sequence (sequence order shuffled once per epoch, local samples
shuffled inside each sequence), so each `data.pt` file loads at most once per
epoch; `metrics.jsonl` records `samples_per_second`, `sequence_load_count` and
`rss_mb`.

### Two formal commands (not started by this round)

Lean dual-evidence variant:

```bash
./.venv/bin/python -m shutrack.cli train \
  --dataset-dir outputs/agentguard/train_data/MOT17 \
  --checkpoint-dir outputs/agentguard/experiments/shutrack_v3_mot17_lean_bs1024_100e/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 0 \
  --lr 1e-4 --weight-decay 1e-4 --warmup-epochs 1 --grad-clip 1.0 \
  --seed 42 --competition off
```

With competition token:

```bash
./.venv/bin/python -m shutrack.cli train \
  --dataset-dir outputs/agentguard/train_data/MOT17 \
  --checkpoint-dir outputs/agentguard/experiments/shutrack_v3_mot17_competition_bs1024_100e/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 0 \
  --lr 1e-4 --weight-decay 1e-4 --warmup-epochs 1 --grad-clip 1.0 \
  --seed 42 --competition on
```

`--epochs` counts **full dataset passes**.  MOT20/SportsMOT formal runs use 20
full passes to match the old `5×1×20` / `10×1×20` effective budgets.

Training writes only `last.pt`, `best.pt`, `metrics.jsonl`, `training.log` and
`training_summary.json`.  A run refuses to reuse an existing checkpoint
directory unless `--resume <last.pt>` is given.  Resume appends logs, validates
model config, normalization, competition branch and dataset contract before
loading optimizer state.  `best.pt` tracks the lowest average **training loss**
(`best_metric_name = "training_loss"`), not tracker HOTA.

### Inspect / validate

```bash
./.venv/bin/python -m shutrack.cli inspect-data \
  --dataset-dir outputs/agentguard/train_data/MOT17 --competition on

./.venv/bin/python -m shutrack.cli validate-checkpoint \
  --checkpoint outputs/agentguard/experiments/shutrack_v3_mot17_competition_bs1024_100e/checkpoints/last.pt \
  --dataset-dir outputs/agentguard/train_data/MOT17 --max-batches 4 --device cpu
```

## Online tracking

```bash
cd "3. Tracker"

"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode all \
  --sequences MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN \
              MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
  --agentguard-mode shutrack-v2 \
  --shutrack-checkpoint \
    "../outputs/agentguard/experiments/shutrack_v3_mot17_competition_bs1024_100e/checkpoints/best.pt" \
  --agentguard-device cpu \
  --tracker-suffix shutrack_v3_mot17_e100 \
  --print-per-sequence-metrics
```

`--agentguard-mode shutrack-v2` is kept as the TrackTrack entry name for
continuity; the checkpoint's architecture version identifies v3.  The mode
rejects `--iwg-rg-cma-output base`, `--rg-cma-alpha != 1`,
`--agentguard-fallback-threshold > 0`, legacy normalization overrides and
attention diagnostics.  `--shutrack-early-enable` enables the optional New-path
behavior described above.  `--agentguard-disable-kf-gate` /
`--agentguard-disable-ema-gate` remain channel=1 diagnostics.

Statistics semantics: `avg_motion_gate` / `avg_appearance_gate` and their
min/max are computed **only over matched events whose gates were actually
applied**, each channel over its own valid-update count.  Unmatched
(no-detection) events are counted separately in `unmatched_events` /
`no_detection_events`; their forced `[0,0]` does not bias the gate diagnostics.
When no matched gate exists, all diagnostic values are finite zeros.

Detection dropout in the mmap entry applies one keep mask to both the detection
rows and `agentguard_target_detection_indices`; source candidates keep their
rows and `agentguard_source_detection_indices` aligned for the existing recovery
rule.  `validate-checkpoint` compares the checkpoint's 63-dim normalization
mean/std with the packed metadata before computing loss and names both the
checkpoint and data directory on mismatch.

## Tests

```bash
./.venv/bin/python -m pytest shutrack/tests -q
```

The suite covers evidence/competition column selection, online competition
values vs the legacy scalar definition (including the single-candidate
boundary), packed↔online feature/mask/competition/gate parity, window
causality, reset-once and model-visible masks at frames 1/50/51, unmatched
history recording, real-`Track` early-enable match 2/3 behavior, branch on/off
training, checkpoint branch validation, sequence-organized sampling/load
counts, resume append/contract checks, read-only guards and a tiny end-to-end
training run.

The real TrackTrack entry smoke is opt-in:

```bash
SHUTRACK_RUN_TRACKER_SMOKE=1 \
SHUTRACK_SMOKE_CHECKPOINT=outputs/agentguard/experiments/shutrack_v3_mot17_competition_smoke/checkpoints/last.pt \
./.venv/bin/python -m pytest shutrack/tests/test_tracktrack_entry_smoke.py -q -s
```

## Known limitations

1. The packed data still has no source event caches; competition columns are
   read directly from the existing normalized packed scalars (they are
   complete), so no margin-style degradation exists anymore.
2. Early enablement is optional and not represented in the legacy supervised
   windows; its real early-track benefit must be validated separately.
3. This round does not launch the formal long training and does not regenerate
   caches, labels, trajectories or datasets.
