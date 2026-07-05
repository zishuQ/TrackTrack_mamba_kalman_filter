#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Running Bailian Teacher: $DATASET ==="
if [ -z "${BAILIAN_API_KEY:-}" ]; then
    echo "WARNING: BAILIAN_API_KEY not set. Teacher will not run."
    exit 1
fi
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli run_bailian_teacher --dataset "$DATASET"
echo "=== Bailian Teacher complete ==="
