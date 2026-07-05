#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-val_custom}"
ORACLE_TYPE="${ORACLE_TYPE:-full}"
echo "=== Running Oracle Evaluation: $DATASET $MODE ($ORACLE_TYPE) ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli evaluate_oracle --dataset "$DATASET" --mode "$MODE" --oracle-type "$ORACLE_TYPE"
echo "=== Oracle evaluation complete ==="
