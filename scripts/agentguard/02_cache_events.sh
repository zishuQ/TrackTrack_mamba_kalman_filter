#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-MOT17}"
MODE="${MODE:-train_custom}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DETECTION_CACHE_ROOT="${DETECTION_CACHE_ROOT:-${REPO_ROOT}/outputs/agentguard/detection_cache}"
EVENT_CACHE_ROOT="${EVENT_CACHE_ROOT:-${REPO_ROOT}/outputs/agentguard/event_cache}"
SEQUENCE="${SEQUENCE:-}"

echo "=== AgentGuard: Cache Events ==="
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Max frames: $MAX_FRAMES"
echo "Detection cache root: $DETECTION_CACHE_ROOT"
echo "Event cache root: $EVENT_CACHE_ROOT"
echo "Repo root: $REPO_ROOT"

cd "${REPO_ROOT}"
CMD=(
    "${PYTHON_BIN}" -m agentguard.cli cache_events
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --max-frames "${MAX_FRAMES}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --event-cache-root "${EVENT_CACHE_ROOT}"
)
if [[ -n "${SEQUENCE}" ]]; then
    CMD+=(--sequence "${SEQUENCE}")
fi
"${CMD[@]}"

echo "Cache events complete."
