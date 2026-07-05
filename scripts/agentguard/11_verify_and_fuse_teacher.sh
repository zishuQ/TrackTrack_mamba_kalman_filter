#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Verifying and Fusing Teacher Labels: $DATASET ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli verify_and_fuse --dataset "$DATASET"
echo "=== Verifier and label fusion complete ==="
