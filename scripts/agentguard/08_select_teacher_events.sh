#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Selecting Teacher Events: $DATASET ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli select_teacher_events --dataset "$DATASET"
echo "=== Teacher event selection complete ==="
