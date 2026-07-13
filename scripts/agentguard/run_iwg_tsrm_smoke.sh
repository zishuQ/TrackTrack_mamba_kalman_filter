#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_smoke_seed42}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/dataset}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if [[ -e "${RUN_ROOT}/running" ]]; then
  existing_pid="$(cat "${RUN_ROOT}/pipeline.pid" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Refusing to reuse active smoke directory: ${RUN_ROOT}" >&2
    exit 2
  fi
  rm -f "${RUN_ROOT}/running"
fi
if [[ -e "${RUN_ROOT}/completed" ]]; then
  echo "Refusing to reuse active/completed smoke directory: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
rm -f "${RUN_ROOT}/failed"
echo "$$" > "${RUN_ROOT}/pipeline.pid"
touch "${RUN_ROOT}/running"

finish() {
  local code=$?
  rm -f "${RUN_ROOT}/running"
  echo "${code}" > "${RUN_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 ]]; then
    touch "${RUN_ROOT}/completed"
  else
    touch "${RUN_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"
"${PY}" -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" > "${PROVENANCE_DIR}/environment.txt"
sha256sum "${DATASET_DIR}/metadata.json" > "${PROVENANCE_DIR}/dataset_metadata.sha256"
cp "${DATASET_DIR}/metadata.json" "${PROVENANCE_DIR}/dataset_metadata.json"

COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_tsrm
  --training-mode joint
  --dataset-dir "${DATASET_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda --amp
  --epochs 2 --batch-size 8 --num-workers 0
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 0 --grad-clip 5.0
  --seed 42 --window-size 16 --delta-max 0.2
  --lambda-final 1.0 --lambda-residual 0.5
  --lambda-dynamics 0.0 --lambda-revision 0.01
  --sampling-policy sequence_balanced
  --selection-metric final_gate_loss
  --max-train-windows 512 --max-val-windows 128
)
printf '%q ' "${COMMAND[@]}" > "${PROVENANCE_DIR}/command.txt"
printf '\n' >> "${PROVENANCE_DIR}/command.txt"
"${COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -m agentguard.cli validate_iwg_tsrm_checkpoint \
  --checkpoint "${CHECKPOINT_DIR}/iwg_tsrm_last.pt" \
  --dataset-dir "${DATASET_DIR}" --split val --device cuda --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/validate.log"
