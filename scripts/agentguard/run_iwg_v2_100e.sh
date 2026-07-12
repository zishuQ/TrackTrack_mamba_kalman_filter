#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

DATASET="MOT17"
MODE="all"
SEED="42"
RUN_NAME="iwg_v2_a_only_100e_seed42"
RUN_ROOT="${REPO_ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
EVENT_CACHE_ROOT="${REPO_ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${REPO_ROOT}/outputs/agentguard/detection_cache"
LABEL_DIR="${RUN_ROOT}/labels"
DATASET_DIR="${RUN_ROOT}/dataset"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${LABEL_DIR}" "${DATASET_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}"
echo "$$" > "${RUN_ROOT}/pipeline.pid"
trap 'status=$?; echo "${status}" > "${RUN_ROOT}/pipeline.exit_code"' EXIT
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3. Tracker:${REPO_ROOT}/agentguard/src:${PYTHONPATH:-}"

git rev-parse HEAD > "${RUN_ROOT}/git_commit.txt"
{
  echo "dataset=${DATASET}"
  echo "mode=${MODE}"
  echo "candidate_types=A"
  echo "split_policy=train_all"
  echo "epochs=100"
  echo "batch_size=1024"
  echo "learning_rate=0.0001"
  echo "num_workers=2"
  echo "seed=${SEED}"
  echo "val_max_samples=20000"
  echo "full_val_every=5"
  echo "skip_tgr_training=true"
  echo "event_cache_root=${EVENT_CACHE_ROOT}"
  echo "detection_cache_root=${DETECTION_CACHE_ROOT}"
} > "${RUN_ROOT}/run_config.txt"

echo "[$(date --iso-8601=seconds)] Building full schema-v3 event cache"
"${PYTHON_BIN}" -m agentguard.cli cache_events \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --detector FRCNN \
  --max-frames 0 \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --event-cache-root "${EVENT_CACHE_ROOT}"

echo "[$(date --iso-8601=seconds)] Validating event cache"
"${PYTHON_BIN}" -m agentguard.cli validate_cache \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --event-cache-root "${EVENT_CACHE_ROOT}"
cp "${REPO_ROOT}/outputs/agentguard/reports/cache_validation/${DATASET}/all/summary.json" \
  "${RUN_ROOT}/cache_validation_summary.json"

echo "[$(date --iso-8601=seconds)] Running rollout materialization smoke"
"${PYTHON_BIN}" -m agentguard.cli rollout_smoke \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --max-events 1000 \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --event-cache-root "${EVENT_CACHE_ROOT}"

echo "[$(date --iso-8601=seconds)] Building A-only safe rollout labels"
"${PYTHON_BIN}" -m agentguard.cli build_rollout_labels \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --max-events 0 \
  --future-frames 5 \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --label-dir "${LABEL_DIR}" \
  --candidate-types A

echo "[$(date --iso-8601=seconds)] Building train-all Student-V0 dataset"
"${PYTHON_BIN}" -m agentguard.cli build_student_v0_data \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --max-samples 0 \
  --split-policy train_all \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --label-dir "${LABEL_DIR}" \
  --output-dir "${DATASET_DIR}" \
  --candidate-types A \
  --candidate-weights A:1 \
  --max-per-candidate-type 0

echo "[$(date --iso-8601=seconds)] Training IWG-v2"
"${PYTHON_BIN}" -m agentguard.cli train_student_v0 \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --device cuda \
  --epochs 100 \
  --batch-size 1024 \
  --lr 0.0001 \
  --tgr-lr 0.0001 \
  --num-workers 2 \
  --seed "${SEED}" \
  --tgr-window-stride 1 \
  --val-max-samples 20000 \
  --full-val-every 5 \
  --dataset-dir "${DATASET_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --skip-tgr-training

echo "[$(date --iso-8601=seconds)] IWG-v2 pipeline complete"
