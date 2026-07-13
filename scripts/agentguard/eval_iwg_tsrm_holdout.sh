#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRACKER_DIR="${ROOT}/3. Tracker"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/evaluation_all7}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/outputs/agentguard/experiments/iwg_base_v4_holdout_seed42/checkpoints/iwg_base_best.pt}"
JOINT_CKPT="${JOINT_CKPT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/checkpoints/iwg_tsrm_best.pt}"
DETECTION_CACHE_ROOT="${DETECTION_CACHE_ROOT:-${ROOT}/outputs/agentguard/detection_cache}"
TRACKER_SEED="${TRACKER_SEED:-10000}"
SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN
  MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

export PYTHONPATH="${ROOT}:${TRACKER_DIR}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

[[ -f "${BASE_CKPT}" ]] || { echo "Missing base checkpoint: ${BASE_CKPT}" >&2; exit 2; }
[[ -f "${JOINT_CKPT}" ]] || { echo "Missing joint checkpoint: ${JOINT_CKPT}" >&2; exit 2; }
if [[ -e "${RUN_ROOT}/completed" ]]; then
  echo "Refusing to overwrite completed evaluation: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/resources" "${RUN_ROOT}/provenance"
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
sha256sum "${BASE_CKPT}" "${JOINT_CKPT}" > "${RUN_ROOT}/provenance/checkpoints.sha256"
printf '%s\n' "${SEQUENCES[@]}" > "${RUN_ROOT}/provenance/sequences.txt"

run_case() {
  local name="$1"
  local post="$2"
  shift 2
  local status="${RUN_ROOT}/${name}.status"
  local command=(
    "${PY}" run.py --dataset MOT17 --mode all
    --sequences "${SEQUENCES[@]}" --seed "${TRACKER_SEED}"
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "holdout_${name}"
    --print-per-sequence-metrics
    --resource-log "${RUN_ROOT}/resources/${name}.jsonl"
    --profile-every 500
    "$@"
  )
  [[ "${post}" == "post" ]] && command+=(--use_post)
  printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/${name}.command.txt"
  printf '\n' >> "${RUN_ROOT}/provenance/${name}.command.txt"
  echo running > "${status}"
  (cd "${TRACKER_DIR}" && "${command[@]}") 2>&1 | tee "${RUN_ROOT}/logs/${name}.log"
  echo completed > "${status}"
}

run_case baseline_raw raw --agentguard-mode off
run_case base_only_raw raw --agentguard-mode iwg --iwg-checkpoint "${BASE_CKPT}" --agentguard-device cuda
run_case joint_base_raw raw --agentguard-mode joint --agentguard-checkpoint "${JOINT_CKPT}" --joint-output base --agentguard-device cuda
run_case joint_final_raw raw --agentguard-mode joint --agentguard-checkpoint "${JOINT_CKPT}" --joint-output final --agentguard-device cuda
run_case baseline_post post --agentguard-mode off
run_case base_only_post post --agentguard-mode iwg --iwg-checkpoint "${BASE_CKPT}" --agentguard-device cuda
run_case joint_base_post post --agentguard-mode joint --agentguard-checkpoint "${JOINT_CKPT}" --joint-output base --agentguard-device cuda
run_case joint_final_post post --agentguard-mode joint --agentguard-checkpoint "${JOINT_CKPT}" --joint-output final --agentguard-device cuda
