#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-all}"
SEQUENCE="${SEQUENCE:-}"
MAX_EVENTS="${MAX_EVENTS:-0}"
CANDIDATE_TYPES="${CANDIDATE_TYPES:-A}"
echo "=== Building Rollout Labels: $DATASET $MODE ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3. Tracker:${REPO_ROOT}/agentguard/src:${PYTHONPATH:-}"
CMD=("${PYTHON_BIN}" -m agentguard.cli build_rollout_labels --dataset "$DATASET" --mode "$MODE" --max-events "$MAX_EVENTS" --candidate-types "$CANDIDATE_TYPES")
if [[ -n "${SEQUENCE}" ]]; then
  CMD+=(--sequence "${SEQUENCE}")
fi
"${CMD[@]}"
echo "=== Rollout labels complete ==="
