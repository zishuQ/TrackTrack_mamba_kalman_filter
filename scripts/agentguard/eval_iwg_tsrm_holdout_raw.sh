#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRACKER_DIR="${ROOT}/3. Tracker"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/evaluation_strict_raw}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/outputs/agentguard/experiments/iwg_base_v5_balanced_holdout_seed42/checkpoints/iwg_base_best.pt}"
JOINT_CKPT="${JOINT_CKPT:-${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/checkpoints/iwg_tsrm_best.pt}"
DETECTION_CACHE_ROOT="${DETECTION_CACHE_ROOT:-${ROOT}/outputs/agentguard/detection_cache}"
SEQUENCES=(MOT17-02-FRCNN MOT17-11-FRCNN)
TRACKER_SEED="${TRACKER_SEED:-10000}"

export PYTHONPATH="${ROOT}:${TRACKER_DIR}:${ROOT}/agentguard/src:${PYTHONPATH:-}"
[[ -f "${BASE_CKPT}" ]] || { echo "Missing base checkpoint: ${BASE_CKPT}" >&2; exit 2; }
[[ -f "${JOINT_CKPT}" ]] || { echo "Missing joint checkpoint: ${JOINT_CKPT}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}/completed" && ! -e "${RUN_ROOT}/running" ]] || {
  echo "Refusing to reuse active/completed evaluation: ${RUN_ROOT}" >&2
  exit 2
}
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
sha256sum "${BASE_CKPT}" "${JOINT_CKPT}" > "${RUN_ROOT}/provenance/checkpoints.sha256"
printf '%s\n' "${SEQUENCES[@]}" > "${RUN_ROOT}/provenance/sequences.txt"

run_case() {
  local name="$1"
  shift
  local command=(
    "${PY}" run.py --dataset MOT17 --mode all
    --sequences "${SEQUENCES[@]}" --seed "${TRACKER_SEED}"
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "endpoint_${name}"
    --print-per-sequence-metrics
    --resource-log "${RUN_ROOT}/resources/${name}.jsonl"
    --profile-every 500
    "$@"
  )
  printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/${name}.command.txt"
  printf '\n' >> "${RUN_ROOT}/provenance/${name}.command.txt"
  (cd "${TRACKER_DIR}" && "${command[@]}") 2>&1 | tee "${RUN_ROOT}/logs/${name}.log"
}

run_case base_only_raw --agentguard-mode iwg --iwg-checkpoint "${BASE_CKPT}" --agentguard-device cuda
run_case joint_base_raw --agentguard-mode joint --agentguard-checkpoint "${JOINT_CKPT}" --joint-output base --agentguard-device cuda
run_case joint_final_raw --agentguard-mode joint --agentguard-checkpoint "${JOINT_CKPT}" --joint-output final --agentguard-device cuda

"${PY}" "${ROOT}/scripts/agentguard/aggregate_iwg_tsrm_holdout.py" \
  --output "${RUN_ROOT}/strict_holdout_raw_manifest.json" \
  --case baseline_raw=mot17_all_0.80_holdout_baseline_raw \
  --case base_only_raw=mot17_all_0.80_endpoint_base_only_raw_agentguard_iwg \
  --case joint_base_raw=mot17_all_0.80_endpoint_joint_base_raw_agentguard_joint_base \
  --case joint_final_raw=mot17_all_0.80_endpoint_joint_final_raw_agentguard_joint_final \
  2>&1 | tee "${RUN_ROOT}/logs/aggregate.log"
