#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Building Student-V1 Dataset: $DATASET ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli build_student_v1_data --dataset "$DATASET"
echo "=== Student-V1 data complete ==="
