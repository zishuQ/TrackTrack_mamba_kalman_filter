#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Building Student-V0 Dataset: $DATASET ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli build_student_v0_data --dataset "$DATASET"
echo "=== Student-V0 data complete ==="
