#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-all}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0003}"
TGR_LR="${TGR_LR:-$LR}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
TGR_WINDOW_STRIDE="${TGR_WINDOW_STRIDE:-1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-0}"
FULL_VAL_EVERY="${FULL_VAL_EVERY:-0}"
DATASET_DIR="${DATASET_DIR:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
echo "=== Training Student-V0: $DATASET on $DEVICE ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3. Tracker:${REPO_ROOT}/agentguard/src:${PYTHONPATH:-}"
cmd=(
  "${PYTHON_BIN}" -m agentguard.cli train_student_v0
  --dataset "$DATASET" \
  --mode "$MODE" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --tgr-lr "$TGR_LR" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --tgr-window-stride "$TGR_WINDOW_STRIDE" \
  --val-max-samples "$VAL_MAX_SAMPLES" \
  --full-val-every "$FULL_VAL_EVERY"
)
if [[ -n "$DATASET_DIR" ]]; then
  cmd+=(--dataset-dir "$DATASET_DIR")
fi
if [[ -n "$CHECKPOINT_DIR" ]]; then
  cmd+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
"${cmd[@]}"
echo "=== Student-V0 training complete ==="
