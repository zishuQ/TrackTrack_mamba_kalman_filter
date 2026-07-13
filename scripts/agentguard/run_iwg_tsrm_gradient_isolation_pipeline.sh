#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_ROOT="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_pipeline_seed42"
DATASET_DIR="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/dataset"
SMOKE_ROOT="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_smoke_seed42"
JOINT_ROOT="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42"
EVAL_ROOT="${JOINT_ROOT}/evaluation_strict_raw"
BASE_CKPT="${ROOT}/outputs/agentguard/experiments/iwg_base_v5_balanced_holdout_seed42/checkpoints/iwg_base_best.pt"
JOINT_CKPT="${JOINT_ROOT}/checkpoints/iwg_tsrm_best.pt"

mkdir -p "${PIPELINE_ROOT}/logs" "${PIPELINE_ROOT}/provenance"
if [[ -e "${PIPELINE_ROOT}/running" ]]; then
  existing_pid="$(cat "${PIPELINE_ROOT}/pipeline.pid" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Refusing active gradient-isolation pipeline: ${PIPELINE_ROOT}" >&2
    exit 2
  fi
  rm -f "${PIPELINE_ROOT}/running"
fi
[[ ! -e "${PIPELINE_ROOT}/completed" ]] || {
  echo "Refusing completed gradient-isolation pipeline: ${PIPELINE_ROOT}" >&2
  exit 2
}
rm -f "${PIPELINE_ROOT}/failed"
echo "$$" > "${PIPELINE_ROOT}/pipeline.pid"
touch "${PIPELINE_ROOT}/running"
finish() {
  local code=$?
  rm -f "${PIPELINE_ROOT}/running"
  echo "${code}" > "${PIPELINE_ROOT}/pipeline.exit_code"
  [[ ${code} -eq 0 ]] && touch "${PIPELINE_ROOT}/completed" || touch "${PIPELINE_ROOT}/failed"
  exit "${code}"
}
trap finish EXIT

git -C "${ROOT}" rev-parse HEAD > "${PIPELINE_ROOT}/provenance/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PIPELINE_ROOT}/provenance/git_status_porcelain.txt"
sha256sum "${DATASET_DIR}/metadata.json" "${BASE_CKPT}" \
  > "${PIPELINE_ROOT}/provenance/inputs.sha256"

if [[ -e "${SMOKE_ROOT}/completed" ]]; then
  echo "Skipping completed gradient-isolation smoke: ${SMOKE_ROOT}"
else
  RUN_ROOT="${SMOKE_ROOT}" DATASET_DIR="${DATASET_DIR}" \
    TEMPORAL_IWG_GRADIENT_SCALE=0.0 \
    "${ROOT}/scripts/agentguard/run_iwg_tsrm_smoke.sh" \
    2>&1 | tee "${PIPELINE_ROOT}/logs/smoke.log"
fi

if [[ -e "${JOINT_ROOT}/completed" ]]; then
  echo "Skipping completed gradient-isolation joint training: ${JOINT_ROOT}"
else
  RUN_ROOT="${JOINT_ROOT}" DATASET_DIR="${DATASET_DIR}" \
    TEMPORAL_IWG_GRADIENT_SCALE=0.0 \
    "${ROOT}/scripts/agentguard/run_iwg_tsrm_holdout.sh" joint \
    2>&1 | tee "${PIPELINE_ROOT}/logs/joint.log"
fi

if [[ -e "${EVAL_ROOT}/completed" ]]; then
  echo "Skipping completed gradient-isolation raw evaluation: ${EVAL_ROOT}"
else
  RUN_ROOT="${EVAL_ROOT}" BASE_CKPT="${BASE_CKPT}" JOINT_CKPT="${JOINT_CKPT}" \
    TRACKER_PREFIX="gradiso_endpoint" \
    "${ROOT}/scripts/agentguard/eval_iwg_tsrm_holdout_raw.sh" \
    2>&1 | tee "${PIPELINE_ROOT}/logs/eval_raw.log"
fi
