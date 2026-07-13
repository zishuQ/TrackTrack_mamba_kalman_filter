#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TEST_PY="${TEST_PYTHON_BIN:-/home/shang/miniconda3/bin/python}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_pipeline_seed42"
DATASET_DIR="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/dataset"
export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
[[ ! -e "${RUN_ROOT}/running" && ! -e "${RUN_ROOT}/completed" ]] || {
  echo "Refusing to reuse active/completed endpoint pipeline: ${RUN_ROOT}" >&2
  exit 2
}
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

{
  git -C "${ROOT}" diff --check
  bash -n "${ROOT}/scripts/agentguard/run_iwg_tsrm_smoke.sh"
  bash -n "${ROOT}/scripts/agentguard/run_iwg_tsrm_holdout.sh"
  bash -n "${ROOT}/scripts/agentguard/eval_iwg_tsrm_holdout_raw.sh"
  bash -n "${ROOT}/scripts/agentguard/run_iwg_tsrm_endpoint_pipeline.sh"
  "${TEST_PY}" -m compileall -q \
    "${ROOT}/agentguard/src" "${ROOT}/agentguard/tests" \
    "${ROOT}/scripts/agentguard" \
    "${ROOT}/3. Tracker/integrations/agentguard" \
    "${ROOT}/3. Tracker/trackers/tracker.py"
  "${TEST_PY}" -m pytest "${ROOT}/agentguard/tests" -q
} 2>&1 | tee "${RUN_ROOT}/logs/tests.log"

"${PY}" "${ROOT}/scripts/agentguard/aggregate_iwg_tsrm_holdout.py" \
  --output "${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/strict_holdout_eight_manifest.json" \
  2>&1 | tee "${RUN_ROOT}/logs/aggregate_existing.log"

"${PY}" -m agentguard.cli build_iwg_tsrm_data \
  --dataset MOT17 --mode all --candidate-types A \
  --event-cache-root "${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2" \
  --detection-cache-root "${ROOT}/outputs/agentguard/detection_cache" \
  --label-dir "${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v1_phase_a/labels" \
  --output-dir "${DATASET_DIR}" \
  --split-policy explicit_sequence_holdout \
  --val-sequences MOT17-02-FRCNN MOT17-11-FRCNN \
  --window-size 16 --window-stride 4 --max-frame-gap 30 \
  2>&1 | tee "${RUN_ROOT}/logs/build_dataset.log"

"${ROOT}/scripts/agentguard/run_iwg_tsrm_smoke.sh"
"${ROOT}/scripts/agentguard/run_iwg_tsrm_holdout.sh" base
"${ROOT}/scripts/agentguard/run_iwg_tsrm_holdout.sh" joint
"${ROOT}/scripts/agentguard/eval_iwg_tsrm_holdout_raw.sh"
