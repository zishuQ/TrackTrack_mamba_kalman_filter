# AgentGuard

AgentGuard is a modular guard system for online multi-object tracking that combines
an Interaction-Wise Gating (IWG) module for per-frame gate prediction with a
Temporal Gate Revision (TGR) module for temporal consistency refinement.

## Quick Start: Cache and Validate Events

```bash
# 0. Verify environment
DATASET=MOT17 bash scripts/agentguard/00_verify_environment.sh

# 1. Run baseline tracking
DATASET=MOT17 MODE=val_custom bash scripts/agentguard/01_run_baseline.sh

# 2. Cache events for training
DATASET=MOT17 bash scripts/agentguard/02_cache_events.sh

# 3. Validate cache
DATASET=MOT17 bash scripts/agentguard/03_validate_cache.sh
```

## Quick Start: Generate Labels and Run Teacher

```bash
# 4. Generate GT labels
DATASET=MOT17 bash scripts/agentguard/04_generate_labels.sh

# 5. Run teacher annotation (requires API key)
export BAILIAN_API_KEY="your-api-key"
DATASET=MOT17 bash scripts/agentguard/05_run_teacher.sh

# 6. Build identity prototypes
DATASET=MOT17 bash scripts/agentguard/06_build_prototypes.sh

# 7. Run oracle matching
DATASET=MOT17 bash scripts/agentguard/07_run_oracle.sh

# 8. Fuse labels
DATASET=MOT17 bash scripts/agentguard/08_fuse_labels.sh

# 9. Generate rollout labels
DATASET=MOT17 bash scripts/agentguard/09_generate_rollout.sh
```

## Quick Start: Train Models

```bash
# 10. Train IWG
DATASET=MOT17 DEVICE=cuda bash scripts/agentguard/10_train_iwg.sh

# 11. Validate IWG
DATASET=MOT17 DEVICE=cuda bash scripts/agentguard/11_validate_iwg.sh

# 12. Train TGR (requires trained IWG)
DATASET=MOT17 DEVICE=cuda bash scripts/agentguard/12_train_tgr.sh

# 13. Train Student V1 (requires trained IWG)
DATASET=MOT17 DEVICE=cuda bash scripts/agentguard/13_train_student_v1.sh
```

## Quick Start: Run AgentGuard Inference

After training, run the tracker with AgentGuard enabled:

```bash
# IWG-only mode
cd "3. Tracker"
../.venv/bin/python run.py --dataset MOT17 --mode val_custom \
  --agentguard-mode iwg \
  --iwg-checkpoint outputs/agentguard/models/iwg/iwg_best.pt

# Full mode (IWG + TGR)
../.venv/bin/python run.py --dataset MOT17 --mode val_custom \
  --agentguard-mode full \
  --iwg-checkpoint outputs/agentguard/models/iwg/iwg_best.pt \
  --tgr-checkpoint outputs/agentguard/models/tgr/tgr_best.pt
```

## Output Directory Reference

| Directory | Contents |
|---|---|
| `outputs/agentguard/cache/` | Cached detection events and GT matches (sharded) |
| `outputs/agentguard/labels/` | Generated GT labels per event |
| `outputs/agentguard/teacher/` | Teacher annotation responses and evidence packets |
| `outputs/agentguard/prototypes/` | Identity prototype embeddings |
| `outputs/agentguard/oracle/` | Oracle future-frame matching results |
| `outputs/agentguard/fused/` | Fused labels combining teacher + oracle |
| `outputs/agentguard/rollout/` | Rollout training labels for IWG/TGR |
| `outputs/agentguard/models/iwg/` | IWG model checkpoints |
| `outputs/agentguard/models/tgr/` | TGR model checkpoints |
| `outputs/agentguard/models/student_v1/` | Student V1 model checkpoints |

## Configuration

All configuration is in `agentguard/configs/`:

| File | Purpose |
|---|---|
| `runtime.yaml` | Runtime mode, buffer sizes, replay settings |
| `data.yaml` | Cache sharding, detection thresholds, GT matching |
| `training.yaml` | Hyperparameters for IWG, TGR, and Student V1 |
| `bailian.yaml.example` | Bailian API configuration (copy to `bailian.yaml`) |
| `splits/MOT17.yaml` | MOT17 train/val split |
| `splits/MOT20.yaml` | MOT20 train/val split |
| `splits/SportsMOT.yaml` | SportsMOT train/val split |

## Pipeline Overview

```
Baseline Tracking ──> Cache Events ──> Generate Labels
                          │
                          ├──> Teacher Annotation ──┐
                          │                         │
                          └──> Oracle Matching ──────┤
                                                     │
                                              Fuse Labels
                                                     │
                                           Generate Rollout
                                                     │
                                            ┌────────┴────────┐
                                      Train IWG          Train TGR
                                            │                │
                                     Validate IWG     Run Full Mode
                                            │
                                     Train Student V1
```

## Project Structure

```
agentguard/
├── configs/                  # YAML configuration files
├── src/agentguard/
│   ├── cli/                  # CLI entry points
│   ├── contracts/            # Data contracts (events, outputs, enums)
│   ├── data/                 # Data loading, caching, GT matching
│   ├── datasets/             # PyTorch Datasets for IWG/TGR training
│   ├── evaluation/           # Evaluation metrics
│   ├── features/             # Feature building and normalization
│   ├── hard_events/          # Hard event mining
│   ├── labels/               # Label generation
│   ├── models/               # IWG, TGR, and event encoder models
│   ├── motion/               # Motion models (NSA Kalman filter)
│   ├── oracle/               # Oracle future-frame matching
│   ├── rollout/              # Rollout label generation
│   ├── runtime/              # Online inference runtime
│   ├── teacher/              # Teacher annotation pipeline
│   ├── training/             # Training loops and utilities
│   └── verifier/             # Verification and label fusion
├── tests/                    # Unit tests
└── README.md
```
