#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-MOT17}"
MODE="${MODE:-train_custom}"
EVENT_CACHE_ROOT="${EVENT_CACHE_ROOT:-${REPO_ROOT}/outputs/agentguard/event_cache}"
DETECTION_CACHE_ROOT="${DETECTION_CACHE_ROOT:-${REPO_ROOT}/outputs/agentguard/detection_cache}"

echo "=== AgentGuard: Validate Cache ==="
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Event cache root: $EVENT_CACHE_ROOT"
echo "Detection cache root: $DETECTION_CACHE_ROOT"
echo "Repo root: $REPO_ROOT"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m agentguard.cli validate_cache \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}"

echo "Validate cache complete."
