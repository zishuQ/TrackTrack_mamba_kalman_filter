# Paired B/D pipeline optimization — 2026-11-20

The paired-supervision trainer now overlaps bounded CPU batch preparation with
GPU training, avoids per-step statistic synchronization, maps large stored rank
arrays, and transports existing float16 ReID values without an extra CPU expansion.
Model inputs are restored to float32 before model/loss evaluation. This is not AMP.

## Preserved training semantics

- Same model, loss formulas, weights, masks, fallback targets and normalization.
- Same sequence order, per-sequence row permutation, tail batches and optimizer steps.
- Current MOT20 schedule remains four shards × 25 cycles = 100 shard-epochs,
  25 complete data passes and 11,100 updates.
- Current MOT17 schedule remains four shards × 50 cycles = 200 shard-epochs.
- This does not silently reproduce the older IWG five-shard × 20-cycle schedule.
  Old IWG also divided each sequence into fractions and combined the corresponding
  fractions from all sequences in each phase; it did not train one whole video
  100 times before moving to the next.
- Runtime pipeline switches are logged separately from the semantic resume config.

## Code

- `tools/train_supervision_groups.py`: bounded prefetch, compact batch construction,
  pinned transfer, deferred statistics and read-only rank views.
- `shutrack/training/paired_pipeline.py`: adapters for the existing
  `BatchIndexLoader`, host preparation/transfer, NPZ stored-member mapping and metrics.
- `shutrack/training/loss.py`: optional tensor-valued column statistics.
  Existing callers still receive Python counts unless explicitly opting in.
- `tools/run_bd_mot_20261120.sh`: optional separate experiment/data roots.
- `tools/verify_paired_pipeline.py`: isolated real-data equivalence/throughput check.
- `shutrack/tests/test_paired_pipeline.py`: ordering, tail, failure, shutdown,
  mapping, dtype restoration, statistics and CUDA pinning coverage.

The producer has one thread and a queue depth of one. It shares existing source
arrays and is joined on normal completion, early exit and exceptions.
Existing float32 ReID sources are not rounded to float16.
Uncompressed rank members are mapped read-only; compressed archives retain the
normal NumPy loading path. No dataset regeneration is required.

Debug/comparison flags:
`--prefetch 0`, `--no-pin-memory`, `--no-compact-reid`.

## State preserved when pausing the production chain

The old driver was stopped before its active training child.
MOT17-B was terminated after a newly completed checkpoint was verified:
epoch 159, global_step 3816, scheduler last_epoch 3816,
102 model tensors and 102 optimizer states.
Existing MOT20 B/D results and all datasets remain intact.
No long production training was automatically launched after the optimization.

## Validation

- Full CPU regression: 138 passed, 3 skipped.
- Full regression with CUDA available: 139 passed, 2 skipped.
- Pipeline tests with CUDA available: 15 passed.
- MOT20-03: three fixed real batches per group match exactly after dtype restoration.
  After 6 warmup + 24 measured updates per case, both B and D have parameter,
  optimizer-state and final-loss maximum differences of 0.0.
- MOT17-04-FRCNN: same real-batch checks; after 2 warmup + 8 measured updates,
  both B and D have parameter, optimizer-state and final-loss differences of 0.0.
- Normal MOT20 training entry: one complete shard-epoch for B and D, each
  113,233 samples / 111 batches, finite loss and valid final checkpoints.
  First-epoch loss/gates match the original recorded first epoch.
- Resume-entry smoke restored epoch 1 / step 111 with the exact saved config;
  no additional updates were requested in that finished one-epoch smoke run.
- Python compile and shell syntax checks pass.
- The existing target-label sigmoid emits an overflow warning for extreme
  negative benefits on MOT20; that label calculation was not changed here.

## Isolated timings (not a full-run runtime guarantee)

Batch size 1024, RTX 4070 Laptop GPU, no concurrent production training.
Reference is the saved trainer before this optimization, already including the
small-field NPZ cache fix.

| Dataset/sequence | Group | Reference ms/step | Optimized ms/step | Notes |
|---|---|---:|---:|---|
| MOT20-03 | B | 1167.1 | 106.5 | Reference hit a long stall; 10.96x is not a stable speedup claim |
| MOT20-03 | D | 127.2 | 72.4 | 1.76x in the short check |
| MOT17-04-FRCNN | B | 175.2 | 64.2 | 2.73x in an 8-step measured check |
| MOT17-04-FRCNN | D | 117.0 | 94.5 | 1.24x in an 8-step measured check |

Normal-entry MOT20 first shard-epoch: B 28.42 s including cold first-use work;
D 9.73 s after caches were warm. These are different cache conditions and are
not a B-vs-D compute-cost comparison.
Full-run estimates should use several complete shard-epochs after startup.

Evidence root:
`outputs/agentguard/experiments/shutrack_pipeline_optimization_20261120/`

- `reference_train.py`
- `mot20_pipeline_compact.json`
- `mot17_pipeline_compact.json`
- `smoke_groups/{appearance_rank,appearance_row_column}/`

## Restart without overwriting old experiments

To start the four production training jobs afresh while reusing existing data:

```bash
cd /home/shang/workspace/TrackTrack
EXPERIMENT_ROOT="$PWD/outputs/agentguard/experiments/shutrack_bd_mot_fast_20261120" \
TRAIN_DATA_ROOT="$PWD/outputs/agentguard/experiments/shutrack_bd_mot_20261120" \
bash tools/run_bd_mot_20261120.sh
```

This creates new MOT20 B/D weights from scratch, then initializes both MOT17
groups from that same new MOT20-B final checkpoint. Old experiment files are retained.

To continue the already completed/paused original chain instead:

```bash
cd /home/shang/workspace/TrackTrack
bash tools/run_bd_mot_20261120.sh
```

The default driver skips complete groups and resumes partial ones. Do not run
both commands simultaneously. Evaluation scripts must point to the experiment
root actually selected; a new root does not change detection caches or sequence splits.
