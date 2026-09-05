#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_mot17_packed_seed42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/outputs/agentguard/train_data/MOT17}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/outputs/3. track"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"
SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN
  MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

if [[ "${TRAIN_BATCH_SIZE}" != "1024" && "${TRAIN_BATCH_SIZE}" != "2048" ]]; then
  echo "TRAIN_BATCH_SIZE must be 1024 or 2048, got ${TRAIN_BATCH_SIZE}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
if [[ -e "${RUN_ROOT}/running" ]]; then
  echo "Refusing an already-running RG-CMA experiment: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/completed" || -e "${CHECKPOINT}" ]]; then
  echo "Refusing to overwrite an existing formal RG-CMA run: ${RUN_ROOT}" >&2
  exit 2
fi
echo "$$" > "${RUN_ROOT}/pipeline.pid"
touch "${RUN_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
  if [[ ${pipeline_complete} -eq 0 && ${code} -eq 0 ]]; then
    code=130
  fi
  rm -f "${RUN_ROOT}/running"
  echo "${code}" > "${RUN_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${pipeline_complete} -eq 1 ]]; then
    touch "${RUN_ROOT}/completed"
  else
    touch "${RUN_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"
"${PY}" -c "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" \
  > "${PROVENANCE_DIR}/environment.txt"

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  echo "Packed MOT17 dataset not found: ${DATASET_DIR}" >&2
  echo "Build it before running this training script." >&2
  exit 2
fi
"${PY}" -c "import json; x=json.load(open('${DATASET_DIR}/metadata.json')); assert x.get('format') == 'agentguard_train_data_v1', x.get('format'); print('packed dataset:', x['num_train_samples'], 'samples')" \
  2>&1 | tee "${LOG_DIR}/dataset_check.log"

TRAIN_COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_rg_cma
  --dataset-dir "${DATASET_DIR}" --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda --epochs 100 --batch-size "${TRAIN_BATCH_SIZE}" --num-workers 4
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1
  --grad-clip 1.0 --seed 42
  --memory-shards 1 --epochs-per-shard 100 --shard-cycles 1
  --sequence-sampling sample-proportional
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -m agentguard.cli validate_iwg_rg_cma_checkpoint \
  --checkpoint "${CHECKPOINT}" --dataset-dir "${DATASET_DIR}" \
  --device cuda --max-batches 8 --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

run_case() {
  local output="$1"
  local suffix="$2"
  local use_post="$3"
  local command=(
    "${PY}" run.py --dataset MOT17 --mode all --sequences "${SEQUENCES[@]}"
    --seed 10000 --agentguard-mode iwg-rg-cma
    --legacy-output-naming
    --agentguard-checkpoint "${CHECKPOINT}" --iwg-rg-cma-output "${output}"
    --agentguard-device cuda --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}" --print-per-sequence-metrics
    --resource-log "${RUN_ROOT}/resource_${suffix}.jsonl" --profile-every 500
  )
  if [[ "${use_post}" == "true" ]]; then
    command+=(--use_post)
  fi
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/eval_${suffix}.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/eval_${suffix}.command.txt"
  (cd "${ROOT}/3. Tracker" && "${command[@]}") \
    2>&1 | tee "${LOG_DIR}/eval_${suffix}.log"
}

run_case base "${RUN_NAME}_base_raw" false
run_case final "${RUN_NAME}_final_raw" false
run_case base "${RUN_NAME}_base_post" true
run_case final "${RUN_NAME}_final_post" true

BASE_RAW="mot17_all_0.80_${RUN_NAME}_base_raw_agentguard_iwg_rg_cma_base"
FINAL_RAW="mot17_all_0.80_${RUN_NAME}_final_raw_agentguard_iwg_rg_cma_final"
BASE_POST="mot17_all_0.80_${RUN_NAME}_base_post_agentguard_iwg_rg_cma_base_post"
FINAL_POST="mot17_all_0.80_${RUN_NAME}_final_post_agentguard_iwg_rg_cma_final_post"
"${PY}" "${ROOT}/scripts/agentguard/evaluate_iwg_rg_cma.py" \
  --case "base_raw=${BASE_RAW}" --case "final_raw=${FINAL_RAW}" \
  --case "base_post=${BASE_POST}" --case "final_post=${FINAL_POST}" \
  --output "${RUN_ROOT}/promotion.json" 2>&1 | tee "${LOG_DIR}/promotion.log"

PROMOTED="$("${PY}" -c "import json; print(str(json.load(open('${RUN_ROOT}/promotion.json'))['promoted']).lower())")"
if [[ "${PROMOTED}" != "true" ]]; then
  touch "${RUN_ROOT}/not_promoted"
  pipeline_complete=1
  exit 0
fi
touch "${RUN_ROOT}/promoted"

TEST_SUFFIX="${RUN_NAME}_test_final_post"
TEST_COMMAND=(
  "${PY}" run.py --dataset MOT17 --mode test --seed 10000
  --agentguard-mode iwg-rg-cma --legacy-output-naming --agentguard-checkpoint "${CHECKPOINT}"
  --iwg-rg-cma-output final --agentguard-device cuda
  --detection-cache-root "${DETECTION_CACHE_ROOT}"
  --tracker-suffix "${TEST_SUFFIX}" --use_post --skip-eval
  --resource-log "${RUN_ROOT}/resource_test.jsonl" --profile-every 500
)
printf '%q ' "${TEST_COMMAND[@]}" > "${PROVENANCE_DIR}/test.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/test.command.txt"
(cd "${ROOT}/3. Tracker" && "${TEST_COMMAND[@]}") \
  2>&1 | tee "${LOG_DIR}/test.log"

TEST_FOLDER="mot17_test_0.80_${TEST_SUFFIX}_agentguard_iwg_rg_cma_final_post"
"${PY}" "${ROOT}/scripts/agentguard/package_mot17_submission.py" \
  --source-dir "${TRACKER_ROOT}/${TEST_FOLDER}" \
  --output-zip "${RUN_ROOT}/MOT17_test_RG_CMA_final_post.zip" \
  --manifest "${RUN_ROOT}/submission_manifest.json" \
  2>&1 | tee "${LOG_DIR}/submission.log"

pipeline_complete=1
