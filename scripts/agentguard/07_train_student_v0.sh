#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
DEVICE="${DEVICE:-cuda}"
echo "=== Training Student-V0: $DATASET on $DEVICE ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli train_student_v0 --dataset "$DATASET" --device "$DEVICE"
echo "=== Student-V0 training complete ==="
