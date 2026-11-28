# SHUTrack v3 — HCE + RSC

SHUTrack = **Suppress Harmful Updates**. After a match, its two gates scale
the NSA Kalman correction and the weight of the new appearance embedding in
the EMA. The maintained model variant is `query_film_bilinear`.

Each event combines motion and observation evidence with ReID appearance.
The current event produces two queries, modulated by association competition.
A temporal Transformer encodes the previous five events. The queries read this
history and the two bilinear heads produce independent sigmoid gates.

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
applies the same 63-dim normalization. Paired training fits the statistics on
the selected training sequences and stores them in the checkpoint. The runtime
and trainer both use the same feature columns.

### Appearance branch

Per-event pre-update track ReID + the event's detection ReID; both projected to
256 dims by default, then `concat(track, det, |diff|, product)` → 128.  Missing detections
use a learned placeholder and an explicit mask.

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

## Training

Use `tools/train_supervision_groups.py` with the `appearance_row_column` group,
the paired replay data root, and the dataset's documented recipe. The
[SHUTrack pipeline guide](../.claude/skills/shutrack-pipeline/SKILL.md) links
to the verified MOT17, MOT20, and SportsMOT commands. For fine-tuning, use
`--init CHECKPOINT` with a current SHUTrack checkpoint. For fresh training,
use `--from-scratch`. To resume, repeat the recipe with `--resume`.

Strict-load a checkpoint and inspect its recorded configuration:

```bash
./.venv/bin/python -m shutrack validate-checkpoint \
  --checkpoint outputs/agentguard/keep/MOT17_68.023_1068665/weights/epoch_050.pt
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
    "../outputs/agentguard/keep/MOT17_68.023_1068665/weights/epoch_050.pt" \
  --agentguard-device cpu \
  --tracker-suffix shutrack_mot17 \
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
rule.

## Tests

```bash
./.venv/bin/python -m pytest shutrack/tests -q
```

The suite covers feature parity, replay collection, the two gate heads,
history resets, early enablement, checkpoint compatibility, and paired
training feature loading.

The real TrackTrack entry smoke is opt-in:

```bash
SHUTRACK_RUN_TRACKER_SMOKE=1 \
SHUTRACK_SMOKE_CHECKPOINT=outputs/agentguard/keep/MOT17_68.023_1068665/weights/epoch_050.pt \
./.venv/bin/python -m pytest shutrack/tests/test_tracktrack_entry_smoke.py -q -s
```
