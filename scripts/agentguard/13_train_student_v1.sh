#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
DEVICE="${DEVICE:-cuda}"
echo "=== Training Student-V1: $DATASET on $DEVICE ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli train_student_v1 --dataset "$DATASET" --device "$DEVICE"
echo "=== Student-V1 training complete ==="
