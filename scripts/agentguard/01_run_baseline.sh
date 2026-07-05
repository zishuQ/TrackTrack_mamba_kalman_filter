#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MOT17}"
MODE="${MODE:-val_custom}"

echo "Running baseline tracking: DATASET=$DATASET MODE=$MODE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}/3. Tracker"
"${PYTHON_BIN}" run.py --dataset "$DATASET" --mode "$MODE"
echo "Baseline complete."
