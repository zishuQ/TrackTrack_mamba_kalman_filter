#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CANDIDATE_TYPES="${CANDIDATE_TYPES:-A}"
CANDIDATE_WEIGHTS="${CANDIDATE_WEIGHTS:-A:1}"
MAX_PER_CANDIDATE_TYPE="${MAX_PER_CANDIDATE_TYPE:-0}"
SPLIT_POLICY="${SPLIT_POLICY:-train_all}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
echo "=== Building Student-V0 Dataset: $DATASET ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3. Tracker:${REPO_ROOT}/agentguard/src:${PYTHONPATH:-}"
cmd=(
  "${PYTHON_BIN}" -m agentguard.cli build_student_v0_data
  --dataset "$DATASET"
  --mode "$MODE"
  --max-samples "$MAX_SAMPLES"
  --split-policy "$SPLIT_POLICY"
  --candidate-types "$CANDIDATE_TYPES"
  --candidate-weights "$CANDIDATE_WEIGHTS"
  --max-per-candidate-type "$MAX_PER_CANDIDATE_TYPE"
)
if [[ -n "$OUTPUT_DIR" ]]; then
  cmd+=(--output-dir "$OUTPUT_DIR")
fi
"${cmd[@]}"
echo "=== Student-V0 data complete ==="
