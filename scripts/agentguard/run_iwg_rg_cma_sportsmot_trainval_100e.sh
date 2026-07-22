#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024_shard10x1}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
DATASET_ROOT="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/SportsMOT"
LABEL_DIR="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_trainval_v3_compact"
DATASET_DIR="${DATASET_DIR:-${DATASET_ROOT}/nsa_trainval_v3_compact}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
EXISTING_TRAIN_LABELS="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_train_v3_compact"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"
EPOCHS="${EPOCHS:-100}"
MEMORY_SHARDS="${MEMORY_SHARDS:-10}"
EPOCHS_PER_SHARD="${EPOCHS_PER_SHARD:-10}"
SHARD_CYCLES="${SHARD_CYCLES:-1}"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if (( MEMORY_SHARDS * EPOCHS_PER_SHARD * SHARD_CYCLES != EPOCHS )); then
  echo "Invalid schedule: ${MEMORY_SHARDS} shards * ${EPOCHS_PER_SHARD} epochs/shard * ${SHARD_CYCLES} cycles != ${EPOCHS} epochs" >&2
  exit 2
fi
echo "Reusable SportsMOT trainval data: ${DATASET_DIR}"
echo "Training schedule: ${MEMORY_SHARDS} shards x ${EPOCHS_PER_SHARD} epochs/shard x ${SHARD_CYCLES} cycles = ${EPOCHS} nominal epochs ($(( EPOCHS / MEMORY_SHARDS )) effective full-data epochs)"
if [[ -e "${RUN_ROOT}/running" || -e "${LAST_CHECKPOINT}" ]]; then
  echo "Refusing to overwrite or duplicate run: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${DATASET_ROOT}" "$(dirname "${LABEL_DIR}")" \
  "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
touch "${RUN_ROOT}/running"
complete=0
finish() {
  local code=$?
  rm -f "${RUN_ROOT}/running"
  echo "${code}" > "${RUN_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${complete} -eq 1 ]]; then
    touch "${RUN_ROOT}/completed"
  else
    touch "${RUN_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"

for split in train val; do
  manifest="${DETECTION_CACHE_ROOT}/SportsMOT/${split}/manifest.json"
  if [[ ! -f "${manifest}" ]]; then
    echo "SportsMOT ${split} detection cache is missing: ${manifest}" >&2
    exit 2
  fi
  "${PY}" -c "import json; p='${manifest}'; x=json.load(open(p)); assert x.get('complete') and len(x.get('sequences', {})) == 45 and int(x.get('reid_dim', 0)) == 2048, p; print(p, 'OK')" \
    2>&1 | tee "${LOG_DIR}/detection_${split}.log"
done

"${PY}" -u -m agentguard.cli cache_events \
  --dataset SportsMOT --mode train_custom \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --data-dir /home/shang/datasets/ \
  2>&1 | tee "${LOG_DIR}/cache_events_train.log"

"${PY}" -u -m agentguard.cli cache_events \
  --dataset SportsMOT --mode val \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --data-dir /home/shang/datasets/ \
  2>&1 | tee "${LOG_DIR}/cache_events_val.log"

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  if [[ ! -d "${LABEL_DIR}" && -f "${EXISTING_TRAIN_LABELS}/summary.json" ]]; then
    cp -a "${EXISTING_TRAIN_LABELS}" "${LABEL_DIR}"
  fi

  "${PY}" -u "${ROOT}/scripts/agentguard/prepare_sportsmot_iwg_rg_cma_labels.py" \
    --split trainval \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --dataset-root /home/shang/datasets/SportsMOT/dataset \
    --output-dir "${LABEL_DIR}" \
    2>&1 | tee "${LOG_DIR}/prepare_labels.log"

  "${PY}" -u -m agentguard.cli build_iwg_rg_cma_data \
    --dataset SportsMOT --mode trainval \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --label-dir "${LABEL_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --max-frame-gap 30 \
    2>&1 | tee "${LOG_DIR}/build_dataset.log"
else
  "${PY}" -c "from agentguard.datasets.iwg_rg_cma_dataset import StreamingIWGRGCMADataset; d=StreamingIWGRGCMADataset('${DATASET_DIR}', max_samples=1); print(d.metadata['dataset_sha256'], d.metadata['num_train_samples']); d.close()" \
    2>&1 | tee "${LOG_DIR}/build_dataset.log"
fi

sha256sum "${DATASET_DIR}/metadata.json" > "${PROVENANCE_DIR}/dataset_metadata.sha256"
TRAIN_COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_rg_cma
  --dataset-dir "${DATASET_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda
  --epochs "${EPOCHS}"
  --batch-size 1024
  --num-workers 4
  --lr 0.0001
  --weight-decay 0.0001
  --warmup-epochs 1
  --grad-clip 1.0
  --seed 42
  --memory-shards "${MEMORY_SHARDS}"
  --epochs-per-shard "${EPOCHS_PER_SHARD}"
  --shard-cycles "${SHARD_CYCLES}"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -m agentguard.cli validate_iwg_rg_cma_checkpoint \
  --checkpoint "${LAST_CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  --device cuda \
  --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

sha256sum "${LAST_CHECKPOINT}" > "${PROVENANCE_DIR}/last_checkpoint.sha256"
complete=1
