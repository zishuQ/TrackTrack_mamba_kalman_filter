#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRAINING_MODE="${1:-joint}"
SEED="${SEED:-42}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v1_holdout_seed42/dataset}"
if [[ "${TRAINING_MODE}" == "joint" ]]; then
  RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v1_holdout_seed${SEED}}"
  SELECTION="final_gate_loss"
else
  RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_base_v3_holdout_seed${SEED}}"
  SELECTION="base_gate_loss"
fi
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

mkdir -p "${CHECKPOINT_DIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
if [[ -e "${RUN_ROOT}/completed" ]]; then
  echo "Refusing to overwrite completed run: ${RUN_ROOT}" >&2
  exit 2
fi
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
  --device cuda --amp --epochs 100 --batch-size 64 --num-workers 2
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1 --grad-clip 5.0
  --seed "${SEED}" --window-size 16 --delta-max 0.2
  --lambda-final 1.0 --lambda-residual 0.5
  --lambda-dynamics 0.1 --lambda-revision 0.01
  --selection-metric "${SELECTION}" --early-stop-patience 15
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  COMMAND+=(--resume-from "${RESUME_FROM}")
fi
printf '%q ' "${COMMAND[@]}" > "${RUN_ROOT}/provenance/command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/command.txt"
"${COMMAND[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
