#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRAINING_MODE="${1:-joint}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/dataset}"
if [[ "${TRAINING_MODE}" == "joint" ]]; then
  RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed${SEED}}"
  SELECTION="final_gate_loss"
else
  RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_base_v5_balanced_holdout_seed${SEED}}"
  SELECTION="base_gate_loss"
fi
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

mkdir -p "${CHECKPOINT_DIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
if [[ -e "${RUN_ROOT}/completed" ]]; then
  echo "Refusing to overwrite completed run: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/provenance/dataset_metadata.sha256" ]]; then
  sha256sum --check --status "${RUN_ROOT}/provenance/dataset_metadata.sha256" || {
    echo "Refusing to reuse run with incompatible dataset metadata: ${RUN_ROOT}" >&2
    exit 2
  }
fi
rm -f "${RUN_ROOT}/failed"
echo "$$" > "${RUN_ROOT}/pipeline.pid"
touch "${RUN_ROOT}/running"
finish() {
  local code=$?
  rm -f "${RUN_ROOT}/running"
  echo "${code}" > "${RUN_ROOT}/pipeline.exit_code"
  [[ ${code} -eq 0 ]] && touch "${RUN_ROOT}/completed" || touch "${RUN_ROOT}/failed"
  exit "${code}"
}
trap finish EXIT

git -C "${ROOT}" rev-parse HEAD > "${RUN_ROOT}/provenance/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${RUN_ROOT}/provenance/git_status_porcelain.txt"
"${PY}" -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" > "${RUN_ROOT}/provenance/environment.txt"
sha256sum "${DATASET_DIR}/metadata.json" > "${RUN_ROOT}/provenance/dataset_metadata.sha256"
cp "${DATASET_DIR}/metadata.json" "${RUN_ROOT}/provenance/dataset_metadata.json"

COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_tsrm
  --training-mode "${TRAINING_MODE}"
  --dataset-dir "${DATASET_DIR}" --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda --amp --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}" --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 --grad-clip 5.0
  --seed "${SEED}" --window-size 16 --delta-max 0.2
  --lambda-final 1.0 --lambda-residual 0.5
  --lambda-dynamics 0.0 --lambda-revision 0.01
  --sampling-policy sequence_balanced
  --selection-metric "${SELECTION}" --early-stop-patience 15
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  COMMAND+=(--resume-from "${RESUME_FROM}")
fi
printf '%q ' "${COMMAND[@]}" > "${RUN_ROOT}/provenance/command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/command.txt"
"${COMMAND[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"

if [[ "${TRAINING_MODE}" == "joint" ]]; then
  BEST_CHECKPOINT="${CHECKPOINT_DIR}/iwg_tsrm_best.pt"
else
  BEST_CHECKPOINT="${CHECKPOINT_DIR}/iwg_base_best.pt"
fi
"${PY}" -m agentguard.cli validate_iwg_tsrm_checkpoint \
  --checkpoint "${BEST_CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" --split val --device cuda \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${RUN_ROOT}/logs/validate.log"
sha256sum "${BEST_CHECKPOINT}" > "${RUN_ROOT}/provenance/checkpoint.sha256"
