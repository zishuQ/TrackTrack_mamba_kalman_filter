# Paired-data collection fixes — 2026-11-21

Scope: replay/data generation, packing, collection completeness checks, and data
validation. No model architecture, loss, sampling, training budget, inference
thresholds, or detector-cache changes were made by this work.

## Implemented fixes

| Problem | Implementation |
| --- | --- |
| Retained EventEvidence held two 2048-D features per event | adapter_events is now a membership set; original events returned unchanged |
| summary samples was overwritten by the number of events | Separate sample_count and event count; header-only count repair with a backup |
| Exceptions left hooks/cache/mmap resources open | collect_pilot finally-close; partial initialization and repeated close supported |
| Summary existence was treated as completion | Exact replay protocol, complete flag, required files, NPY/NPZ lengths and row counts checked |
| Validator attributed earlier index errors to later sequences | Compare each sequence's counter deltas |

## Memory and runtime improvements

- GT prototypes: retain detection indices, materialize one identity at a time, use the original prototype builder.
- GT binding: vectorized IoU, preserving thresholds, first-tie choice and coordinate convention.
- Future context: bounded eight-frame cache of best detection indices, not features.
- Column probe: reuse the original scene's cost matrix, replace only the changed appearance row, run the same association algorithm. All three gate probes and mismatch checks remain.
- Rank materialization and event-track copying: byte-bounded blocks, flush/drop mmap pages after blocks; scratch maps explicitly closed.
- NPZ packing: streaming ZIP_STORED members, atomic publication. A completed summary is not published before arrays finish.
- Completed-shard detection: headers only, no loading full rank features.
- Full data validator: scan each of four rank arrays in bounded chunks instead of retaining all four; small metadata read once, archive closed. All feature rows are checked for NaN/Inf; detection/frame checks retain their existing sampling.
- Per-sequence locks, atomic state/progress JSON, phase/RSS reporting and memory guards.
- Collection-only controller has an independent state/log, one queue, and waits for active project training/tracking jobs. It never stops those jobs.

The memory changes bound the large feature buffers, not all Python metadata. The
densest remaining sequence MOT20-05 still needs its complete production run; the
small replay checks below are not a peak-memory certification or a controlled
performance benchmark.

## Verification

```bash
.venv/bin/python -m pytest shutrack/tests -q
```

Result: **189 passed, 2 skipped**. Regression coverage includes interrupted
atomic writes, resource cleanup after exceptions, count repair, duplicate locks,
vectorized tie/threshold semantics, float32/float64 association parity, C/F layout
streaming NaN/Inf checks, and the actual validator entry rejecting bad rank data
without materializing rank members.

Real old/new full-write, lean, column replay comparisons:

| Sequence | Frames | Samples | Events | Rank valid | Column valid | Actual mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SportsMOT v_9MHDmAMxO5I_c003 | 120 | 1089 | 1091 | 327 | 366 | 0 |
| MOT20-01 | 80 | 2667 | 2667 | 1358 | 221 | 0 |

For each comparison: **all 46 NPZ members and all four event arrays are bitwise
equal**, including labels, masks, indices and feature dtype. Count metadata is
intentionally corrected. Evidence: outputs/agentguard/experiments/shutrack_collection_fix_20261121/parity/.

The updated full validator passed the real MOT20 smoke shard: 2667 keys,
383 detection-index/frame checks, 202 column-frame checks, zero mismatch,
zero nonfinite rows, data_ok=true. This is a real small shard, not full MOT20.

Existing MOT20-01 summary repaired samples 19377 -> 19109, events 19377;
original summary backed up, numerical arrays unchanged. MOT20-02/03 counts
already agreed with headers.

## Pending production work and recovery

Plan at handoff:
- MOT20: 3/4 complete; MOT20-05 pending.
- MOT17: 5/7 complete; MOT17-09-FRCNN and MOT17-10-FRCNN pending.
- SportsMOT: 45/90 complete; official val 45 pending.
- All existing completed shards match the intended schema-3 full-write protocol.

Collection queue launched in tmux session collection_fix_20261121. Verified
COLLECTION_STATUS.json = waiting_for_existing_jobs. The existing
arch_night_driver.py --stage-a chain was active and had reached MOT20 tracking
evaluation. No existing job was stopped or restarted.

After that chain exits, the controller collects missing shards serially, then
runs the bounded full data validator for each dataset. It reuses complete
compatible shards, stops and records failure on error, and does not run model
training/tracking/submission. It writes no state into the architecture driver's
TASK_STATE.json.

```bash
# Read-only current plan:
.venv/bin/python tools/collect_pending_data.py

# Resume only if this controller is no longer running:
.venv/bin/python tools/collect_pending_data.py --run --wait-idle
```

State/logs:
outputs/agentguard/experiments/shutrack_collection_fix_20261121/
- COLLECTION_STATUS.json
- COLLECTION_TASK_STATE.json
- COLLECTION_ONLY_RESULT.json (created as groups complete)
- controller.log
- logs/
- reports/

If full validation is interrupted, completed collected shards remain reusable;
validation can be rerun without recollecting them. If collection is interrupted,
previous completed sequences remain reusable; the currently incomplete sequence
is replayed from its beginning. Frame-level resume is not implemented.

New collectors use per-sequence locks. Older already-loaded processes cannot be
made to honor new locks retroactively. No such collector was active when this
queue was launched; the observed active chain was training/tracking.
