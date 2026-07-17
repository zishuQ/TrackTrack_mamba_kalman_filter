#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2}"
DATA_RUN_NAME="${DATA_RUN_NAME:-iwg_rg_cma_v1_mot20_trainall_seed42_bs1024}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
DATA_RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${DATA_RUN_NAME}"
LABEL_DIR="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/MOT20/nsa_v3_compact"
DATASET_DIR="${DATA_RUN_ROOT}/dataset"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/outputs/3. track"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"
SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
if [[ -e "${RUN_ROOT}/running" ]]; then
  echo "Refusing an already-running MOT20 RG-CMA experiment: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/completed" || -e "${CHECKPOINT}" ]]; then
  echo "Refusing to overwrite an existing formal MOT20 RG-CMA run: ${RUN_ROOT}" >&2
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
printf '%s\n' "${DATA_RUN_NAME}" > "${PROVENANCE_DIR}/data_run_name.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/models/event_encoder.py" \
  "${ROOT}/agentguard/src/agentguard/models/iwg.py" \
  "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/data/cache_reader.py" \
  "${ROOT}/agentguard/src/agentguard/data/compact_iwg_labels.py" \
  "${ROOT}/agentguard/src/agentguard/datasets/iwg_attn_dataset.py" \
  "${ROOT}/agentguard/src/agentguard/rollout_labels.py" \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/v0_pipeline.py" \
  > "${PROVENANCE_DIR}/training_sources.sha256"
"${PY}" -c "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" \
  > "${PROVENANCE_DIR}/environment.txt"
"${PY}" -u -m agentguard.cli cache_events \
  --dataset MOT20 --mode all \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --data-dir /home/shang/datasets \
  2>&1 | tee "${LOG_DIR}/cache_events.log"

LABELS_READY=false
if [[ -f "${LABEL_DIR}/summary.json" ]]; then
  LABELS_READY="$("${PY}" -c "import json; print(str(bool(json.load(open('${LABEL_DIR}/summary.json')).get('complete', False))).lower())")"
fi
if [[ "${LABELS_READY}" != "true" ]]; then
  "${PY}" -u "${ROOT}/scripts/agentguard/prepare_mot20_iwg_rg_cma_labels.py" \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --gt-root /home/shang/datasets/MOT20/train \
    --output-dir "${LABEL_DIR}" \
    2>&1 | tee "${LOG_DIR}/prepare_labels.log"
else
  "${PY}" -c "import json; x=json.load(open('${LABEL_DIR}/summary.json')); assert x['complete'] and x['retained_labels'] > 0; print(json.dumps(x, indent=2, sort_keys=True))" \
    2>&1 | tee "${LOG_DIR}/prepare_labels.log"
fi

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  "${PY}" -m agentguard.cli build_iwg_attn_data \
    --dataset MOT20 --mode all \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --label-dir "${LABEL_DIR}" --output-dir "${DATASET_DIR}" \
    --max-frame-gap 30 2>&1 | tee "${LOG_DIR}/build.log"
else
  "${PY}" -c "from agentguard.datasets.iwg_attn_dataset import StreamingIWGAttnDataset; d=StreamingIWGAttnDataset('${DATASET_DIR}', max_samples=1); print(d.metadata['dataset_sha256']); d.close()" \
    2>&1 | tee "${LOG_DIR}/build.log"
fi

TRAIN_COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_attn
  --dataset-dir "${DATASET_DIR}" --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1
  --grad-clip 1.0 --seed 42
  --memory-shards 5 --epochs-per-shard 10 --shard-cycles 2
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -m agentguard.cli validate_iwg_attn_checkpoint \
  --checkpoint "${CHECKPOINT}" --dataset-dir "${DATASET_DIR}" \
  --device cpu --max-batches 8 --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"
sha256sum "${DATASET_DIR}/metadata.json" "${CHECKPOINT}" \
  > "${PROVENANCE_DIR}/formal_artifacts.sha256"

run_raw_case() {
  local output="$1"
  local suffix="$2"
  local command=(
    "${PY}" run.py --dataset MOT20 --mode all --sequences "${SEQUENCES[@]}"
    --seed 10000 --agentguard-mode iwg-attn --legacy-output-naming
    --agentguard-checkpoint "${CHECKPOINT}" --iwg-attn-output "${output}"
    --agentguard-device cpu --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}" --print-per-sequence-metrics
    --resource-log "${RUN_ROOT}/resource_${suffix}.jsonl" --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/eval_${suffix}.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/eval_${suffix}.command.txt"
  (cd "${ROOT}/3. Tracker" && "${command[@]}") \
    2>&1 | tee "${LOG_DIR}/eval_${suffix}.log"
}

run_raw_case base "${RUN_NAME}_base_raw"
run_raw_case final "${RUN_NAME}_final_raw"

BASE_RAW="mot20_all_0.80_${RUN_NAME}_base_raw_agentguard_iwg_attn_base"
FINAL_RAW="mot20_all_0.80_${RUN_NAME}_final_raw_agentguard_iwg_attn_final"
BASE_POST="${BASE_RAW}_post"
FINAL_POST="${FINAL_RAW}_post"
"${PY}" "${ROOT}/scripts/agentguard/postprocess_mot_tracker_folder.py" \
  --source-dir "${TRACKER_ROOT}/${BASE_RAW}" \
  --output-dir "${TRACKER_ROOT}/${BASE_POST}" --interval 30 --tau 12 \
  2>&1 | tee "${LOG_DIR}/post_base.log"
"${PY}" "${ROOT}/scripts/agentguard/postprocess_mot_tracker_folder.py" \
  --source-dir "${TRACKER_ROOT}/${FINAL_RAW}" \
  --output-dir "${TRACKER_ROOT}/${FINAL_POST}" --interval 30 --tau 12 \
  2>&1 | tee "${LOG_DIR}/post_final.log"

"${PY}" "${ROOT}/scripts/agentguard/evaluate_iwg_rg_cma_mot20.py" \
  --case "base_raw=${BASE_RAW}" --case "final_raw=${FINAL_RAW}" \
  --case "base_post=${BASE_POST}" --case "final_post=${FINAL_POST}" \
  --output "${RUN_ROOT}/evaluation.json" \
  2>&1 | tee "${LOG_DIR}/evaluation.log"

pipeline_complete=1
