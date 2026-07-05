#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-train_custom}"
echo "=== Building Rollout Labels: $DATASET $MODE ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}/3. Tracker"
"${PYTHON_BIN}" -m agentguard.cli build_rollout_labels --dataset "$DATASET" --mode "$MODE"
echo "=== Rollout labels complete ==="
