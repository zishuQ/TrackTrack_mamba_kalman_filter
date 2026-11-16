# SHUTrack v2 — HCE + RSC

SHUTrack = **Suppress Harmful Updates**.  After a match is accepted, the motion
state and the appearance template should not always be written with full
strength.  This package implements the isolated v2 architecture:

* **HCE — Historical Context Encoder** encodes only the five events *before*
  the current one with a shared temporal Transformer (d_model=128, 2 layers,
  4 heads) and two learned queries that read the motion and appearance history
  contexts `h_m`, `h_a`.
* **RSC — Reliability-aware Suppression Controller** uses `h_m`, `h_a` as
  queries and the current motion / appearance / match-context tokens as
  key/value.  One cross-modal attention layer + FFN produces two representation
  deltas, which are fused residually in representation space and decoded by two
  independent sigmoid heads into the motion and appearance gates.

There is no policy head, cue/risk head, five-strategy prototype, base/refined
gate, correction/bound, or detached gradient path.  The two gates do not use
softmax and are not required to sum to one.

The online path keeps TrackTrack's existing `Track.update_with_gates`; only the
gate source changes.  Association, Kalman filters, EMA, thresholds and track
management are untouched.

## Isolation and read-only data

The package is independent from the legacy `agentguard` model/trainer/runtime.
The only AgentGuard imports live in `shutrack/data/legacy_adapter.py` and reuse
the stable read-only pieces (63-dim scalar definition, compact cache reader,
compact label loader).

Training reads the existing packed `outputs/agentguard/train_data/*` in place:

* `metadata.json` — dataset/split, context=6, ReID dim, 63-dim normalization;
* `<sequence>/data.pt` — `event_indices [N,6]`, `track_ids`, `segment_ids`,
  `safe_gate_target [N,2]`, `valid_channels [N,2]`, `sample_weight [N]`,
  normalized `timeline_scalar_feats [T,63]`, `timeline_has_detection [T]`;
* `<sequence>/reid_features.npy` — `[T,2,2048]` pre-update track template and
  detection ReID, memory-mapped read-only.

No cache, index, label, rollout, window array or dataset is generated.  The
source directory is never written.  At most two `data.pt` sequences are held in
an LRU; ReID files are always memory-mapped.

### Margin data limitation (explicit)

The packed format does **not** contain association cost matrices,
`accepted_detection_index` or `association_track_row`; the source compact event
caches (`events_*.pt` / `associations_*.pt` / `states_*.pt`) and compact label
directories are not present in this checkout.  Therefore the real packed path
trains with:

* row/column margins = 0;
* `row_margin_valid = col_margin_valid = 0` for every event;
* `margin_source = "unavailable_packed_train_data"` in dataset metadata,
  checkpoints and logs (`margin_valid_rate = 0.0`).

This is a reported degradation, not an assertion that no competitor existed.
The full margin definition (`min(other feasible raw cost) - accepted raw cost`,
excluding self, non-finite values and `IoU <= 0.10` infeasible entries) is
implemented and tested against synthetic recorded candidate pools.  Use
`--margin-mode required` to make missing margin data a hard startup error.

## Input schema (`shutrack_v2_event18`)

Six events = five past events + current event.  Every current input is captured
before the current write.

* motion (12): legacy scalar indices `27:31`, `39:43`, `43:47` — center offset,
  log w/h ratio, normalized KF velocity, normalized log covariance diagonal
  (legacy image-scale definition).
* context (8), fixed order: `det_score` (index 9), `row_margin`, `col_margin`,
  `history_length` (index 58), `observation_gap` (index 59),
  `has_detection` (index 26), `row_margin_valid`, `col_margin_valid`.
* appearance: per-event pre-update track ReID + detection ReID; projected to 64
  dims, then `concat(track, det, |diff|, product)` → 128.
* masks: padding, detection availability, and reset; cold start uses a learned
  null-history token so an all-masked history never produces NaN.

Packed scalars are already z-scored by the legacy train-split statistics and are
**not** normalized again.  Online inference applies the same 63-dim mean/std
stored in the checkpoint before selecting the same indices.

## Commands

All commands run from the repository root unless noted.

### Inspect the read-only source

```bash
./.venv/bin/python -m shutrack.cli inspect-data \
  --dataset-dir outputs/agentguard/train_data/MOT17
```

### Short real training smoke (20 steps)

```bash
./.venv/bin/python -m shutrack.cli train \
  --dataset-dir outputs/agentguard/train_data/MOT17 \
  --checkpoint-dir outputs/agentguard/experiments/shutrack_v2_smoke/checkpoints \
  --device cuda --batch-size 32 --num-workers 0 \
  --max-steps 20 --warmup-steps 2 --log-interval 5 --seed 42
```

### Formal training (not started by this round)

MOT17:

```bash
./.venv/bin/python -m shutrack.cli train \
  --dataset-dir outputs/agentguard/train_data/MOT17 \
  --checkpoint-dir outputs/agentguard/experiments/shutrack_v2_mot17_seed42_bs1024_100e/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 0 \
  --lr 1e-4 --weight-decay 1e-4 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 --margin-mode auto
```

MOT20:

```bash
./.venv/bin/python -m shutrack.cli train \
  --dataset-dir outputs/agentguard/train_data/MOT20 \
  --checkpoint-dir outputs/agentguard/experiments/shutrack_v2_mot20_seed42_bs1024_20e_full/checkpoints \
  --device cuda --epochs 20 --batch-size 1024 --num-workers 0 \
  --lr 1e-4 --weight-decay 1e-4 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 --margin-mode auto
```

SportsMOT train+val:

```bash
./.venv/bin/python -m shutrack.cli train \
  --dataset-dir outputs/agentguard/train_data/SportsMOT \
  --checkpoint-dir outputs/agentguard/experiments/shutrack_v2_sportsmot_trainval_seed42_bs1024_20e_full/checkpoints \
  --device cuda --epochs 20 --batch-size 1024 --num-workers 0 \
  --lr 1e-4 --weight-decay 1e-4 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 --margin-mode auto
```

`--epochs` here counts **full dataset passes**.  The old nominal shard
schedules `5×1×20` (MOT20) and `10×1×20` (SportsMOT train+val) equal 20
effective full passes, so `--epochs 20` matches their data budget.  MOT17's
legacy packed-data schedule was 100 full passes, hence `--epochs 100`.  The new
trainer has no memory-shard phases; do not map the old nominal epoch numbers
directly.

Training writes only `last.pt`, `best.pt`, `metrics.jsonl`, `training.log` and
`training_summary.json` inside the new run directory.  A run refuses to reuse an
existing checkpoint directory unless `--resume <last.pt>` is given.

### Checkpoint contract check

```bash
./.venv/bin/python -m shutrack.cli validate-checkpoint \
  --checkpoint outputs/agentguard/experiments/shutrack_v2_smoke/checkpoints/last.pt \
  --dataset-dir outputs/agentguard/train_data/MOT17 \
  --max-batches 4 --device cpu
```

### Online tracking

```bash
cd "3. Tracker"

"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode all \
  --sequences MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN \
              MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
  --agentguard-mode shutrack-v2 \
  --shutrack-checkpoint \
    "../outputs/agentguard/experiments/shutrack_v2_mot17_seed42_bs1024_100e/checkpoints/best.pt" \
  --agentguard-device cpu \
  --tracker-suffix shutrack_v2_mot17_e100 \
  --print-per-sequence-metrics
```

Test submission (no local GT):

```bash
cd "3. Tracker"

"../.venv/bin/python" -u run.py \
  --dataset MOT17 --mode test \
  --agentguard-mode shutrack-v2 \
  --shutrack-checkpoint \
    "../outputs/agentguard/experiments/shutrack_v2_mot17_seed42_bs1024_100e/checkpoints/best.pt" \
  --tracker-suffix shutrack_v2_mot17_e100 \
  --use_post --skip-eval
```

`--agentguard-mode shutrack-v2` rejects `--iwg-rg-cma-output base`,
`--rg-cma-alpha != 1`, `--agentguard-fallback-threshold > 0`, legacy
normalization overrides and attention-diagnostics flags: those concepts do not
exist in the new architecture.  `--agentguard-disable-kf-gate` /
`--agentguard-disable-ema-gate` remain available as channel=1 diagnostics.
`--shutrack-min-observations` defaults to 1 so cold starts enter the model
rather than being bypassed by the old `observation_count >= 6` rule.

## Tests

```bash
./.venv/bin/python -m pytest shutrack/tests -q
```

The suite covers margin semantics, feature order/normalization, cold start and
missing measurements, offline-vs-online equivalence for sequences longer than
six events, future-input causality, cross-track/segment/reset isolation, gate
range/independence, HCE/RSC gradients, checkpoint roundtrip and legacy
checkpoint rejection, read-only guards (builders stubbed to fail), and a tiny
end-to-end training run.

## Artifacts

* `last.pt` — model + optimizer + scheduler + progress, for resume.
* `best.pt` — model + metadata only.
* Both embed architecture/version, feature schema and hash, context=6,
  HCE/RSC hyperparameters, 63-dim normalization, margin source/statistics,
  dataset/split/source path and training config.
* The loader is strict (`strict=True`) and rejects legacy IWG/RG-CMA
  checkpoints with an explicit message.

## Known limitations

1. Real packed training currently has no margin data (see above).
2. Online inference starts applying the model once a track is `Tracked` or
   `Lost` (and has at least `--shutrack-min-observations` observations), which
   is earlier than the legacy cache timeline start (≥6 observations).  Short
   histories are represented by padding + the null token, but the distribution
   difference needs validation after a formal training run.
3. This round does not launch the three-dataset formal training and does not
   regenerate any cache, label or dataset.
