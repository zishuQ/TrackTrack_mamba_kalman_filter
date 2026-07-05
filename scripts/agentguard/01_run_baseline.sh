#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MOT17}"
MODE="${MODE:-val_custom}"

echo "Running baseline tracking: DATASET=$DATASET MODE=$MODE"
cd "3. Tracker"
../.venv/bin/python run.py --dataset "$DATASET" --mode "$MODE"
echo "Baseline complete."
