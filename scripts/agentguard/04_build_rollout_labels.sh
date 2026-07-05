#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-train_custom}"
echo "=== Building Rollout Labels: $DATASET $MODE ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli build_rollout_labels --dataset "$DATASET" --mode "$MODE"
echo "=== Rollout labels complete ==="
