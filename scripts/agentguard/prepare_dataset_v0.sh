#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

DATASET="${DATASET:-MOT20}"
MODE="${MODE:-all}"
SOURCE_PICKLE="${SOURCE_PICKLE:?set SOURCE_PICKLE to the 0.95/source detection pickle}"
TARGET_PICKLE="${TARGET_PICKLE:?set TARGET_PICKLE to the 0.80/target detection pickle}"
SEQUENCES="${SEQUENCES:-}"
DETECTOR="${DETECTOR:-FRCNN}"
MAX_FRAMES="${MAX_FRAMES:-0}"
MAX_EVENTS="${MAX_EVENTS:-0}"
RUN_SWEEP_SMOKE="${RUN_SWEEP_SMOKE:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3. Tracker:${REPO_ROOT}/agentguard/src:${PYTHONPATH:-}"

split_cmd=(
  "${PYTHON_BIN}" scripts/agentguard/00_split_detection_cache.py
  --dataset "${DATASET}"
  --split "${MODE}"
  --source-pickle "${SOURCE_PICKLE}"
  --target-pickle "${TARGET_PICKLE}"
)
if [[ -n "${SEQUENCES}" ]]; then
  split_cmd+=(--sequences "${SEQUENCES}")
fi

echo "=== split detection cache: ${DATASET}/${MODE} ==="
"${split_cmd[@]}"

echo "=== cache events: ${DATASET}/${MODE} ==="
DATASET="${DATASET}" MODE="${MODE}" MAX_FRAMES="${MAX_FRAMES}" DETECTOR="${DETECTOR}" \
  "${PYTHON_BIN}" -m agentguard.cli cache_events \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --max-frames "${MAX_FRAMES}" \
    --detector "${DETECTOR}"

echo "=== validate event cache: ${DATASET}/${MODE} ==="
"${PYTHON_BIN}" -m agentguard.cli validate_cache --dataset "${DATASET}" --mode "${MODE}"

echo "=== build rollout labels: ${DATASET}/${MODE} ==="
"${PYTHON_BIN}" -m agentguard.cli build_rollout_labels \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --max-events "${MAX_EVENTS}"

if [[ "${RUN_SWEEP_SMOKE}" == "1" ]]; then
  echo "=== Student-V0 smoke sweep: ${DATASET}/${MODE} ==="
  "${PYTHON_BIN}" scripts/agentguard/run_bc_weight_sweep.py \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --experiments a_only \
    --epochs 1 \
    --batch-size 16 \
    --val-max-samples 64 \
    --full-val-every 1 \
    --num-workers 0 \
    --skip-existing
fi

echo "=== prepare_dataset_v0 complete: ${DATASET}/${MODE} ==="
