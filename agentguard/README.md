# AgentGuard

AgentGuard is the current IWG-RG-CMA guard for online multi-object tracking.
The maintained path uses a six-event context, a Safe-Direct IWG gate, and a
bounded RG-CMA correction. Historical experiments under
`outputs/agentguard/experiments/` are retained as reproducibility records.

## Pipeline

```text
detection cache
      |
      v
compact event cache
      |
      v
rollout labels -> streaming IWG-RG-CMA dataset -> training checkpoint
                                                        |
                                                        v
                                              raw / post evaluation
```

The shared cache roots are:

```text
outputs/agentguard/detection_cache
outputs/agentguard/event_cache_v3_iwg_v2
```

## Quick Start

The standard experiment scripts prepare data, train, validate the checkpoint,
and record provenance in one run:

```bash
bash scripts/agentguard/run_iwg_rg_cma_100e.sh
bash scripts/agentguard/run_iwg_rg_cma_mot20_100e.sh
bash scripts/agentguard/run_iwg_rg_cma_sportsmot_trainval_200e.sh
```

For an existing dataset, use the CLI directly:

```bash
python -m agentguard.cli build_iwg_rg_cma_data \
  --dataset MOT20 --mode all \
  --event-cache-root outputs/agentguard/event_cache_v3_iwg_v2 \
  --detection-cache-root outputs/agentguard/detection_cache \
  --label-dir outputs/agentguard/labels/iwg_rg_cma/MOT20/nsa_all_v3_compact \
  --output-dir outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact

python -m agentguard.cli train_iwg_rg_cma \
  --dataset-dir outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact \
  --checkpoint-dir outputs/agentguard/experiments/<run>/checkpoints \
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 \
  --grad-clip 1.0 --seed 42 --memory-shards 4 \
  --epochs-per-shard 1 --shard-cycles 25 \
  --sequence-sampling sqrt-size
```

The CLI supports these maintained commands:

```text
cache_events
validate_cache
rollout_smoke
build_rollout_labels
build_iwg_rg_cma_data
train_iwg_rg_cma
validate_iwg_rg_cma_checkpoint
```

## Official Baselines

The authoritative checkpoint paths, hashes, dataset metadata, and metrics are
in `agentguard/configs/official_baselines.json`. All three baselines use
`correction_bound=0.05`, `context_size=6`, and `batch_size=1024`.

| Dataset | Checkpoint epoch | Raw HOTA | Test final + post HOTA |
| --- | ---: | ---: | ---: |
| MOT17 | 50 | 77.7277% | 67.79 |
| MOT20 | 100 | 79.1448% | 66.40 |
| SportsMOT | 200 | 83.2326% | 76.47 |

The recorded MOT20 schedule is `4 x 1 x 25` with `sqrt-size` sampling, and the
recorded SportsMOT schedule is `4 x 1 x 50` with `sqrt-size` sampling. The
MOT17 baseline is a warm start from the historical MOT20 interleaved epoch100
checkpoint followed by 50 full-data fine-tuning epochs using the default
sample-proportional (`full_shuffle`) policy at learning rate `1e-5`.

## Runtime

Run the tracker with a combined checkpoint:

```bash
cd "3. Tracker"
../.venv/bin/python run.py --dataset MOT20 --mode all \
  --agentguard-mode iwg-rg-cma \
  --agentguard-checkpoint \
  ../outputs/agentguard/experiments/iwg_rg_cma_sqrt_size_mot20_bound005_context6_seed42_bs1024_shard4x1_100e/checkpoints/iwg_rg_cma_epoch100.pt \
  --iwg-rg-cma-output final
```

Use raw evaluation without `--use_post`. Add `--use_post` only for the final
test output. `base` evaluates the Safe-Direct IWG gate; `final` applies the
bounded RG-CMA correction.

## Outputs and Configuration

| Path | Purpose |
| --- | --- |
| `outputs/agentguard/detection_cache/` | Shared detection mmap cache |
| `outputs/agentguard/event_cache_v3_iwg_v2/` | Compact event cache |
| `outputs/agentguard/labels/iwg_rg_cma/` | Current rollout labels |
| `outputs/agentguard/datasets/iwg_rg_cma/` | Streaming training datasets |
| `outputs/agentguard/experiments/` | Checkpoints, logs, evaluation, and provenance |
| `agentguard/configs/official_baselines.json` | Official baseline manifest |

Configuration references are in `agentguard/configs/runtime.yaml`,
`agentguard/configs/data.yaml`, and `agentguard/configs/training.yaml`.
The detailed training and evaluation guide is
`agentguard/IWG_RG_CMA_COMMAND_GUIDE.md`.
