#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_mot20e050_to_mot17_finetune25_cachefix_seed42_bs1024}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
DATASET_DIR="${MOT17_DATASET_DIR:-${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_jsonl}"
INIT_CHECKPOINT="${MOT20_INIT_CHECKPOINT:-${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt}"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  echo "MOT17 dataset not found: ${DATASET_DIR}" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "MOT20 initial checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/running" || -e "${LAST_CHECKPOINT}" ]]; then
  echo "Refusing to overwrite or duplicate run: ${RUN_ROOT}" >&2
  exit 2
fi

"${PY}" -c \
  "from agentguard.datasets.iwg_attn_dataset import StreamingIWGAttnDataset; d=StreamingIWGAttnDataset(r'${DATASET_DIR}', max_samples=1); assert d.index_format == 'jsonl_v1', d.index_format; r=d._reader(d.metadata['train_sequences'][0]); assert r.max_cached_shards is None, r.max_cached_shards; print('MOT17 event-shard cache: unbounded resident (cachefix active)'); d.close()"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
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
sha256sum "${INIT_CHECKPOINT}" > "${PROVENANCE_DIR}/initial_checkpoint.sha256"
sha256sum "${DATASET_DIR}/metadata.json" > "${PROVENANCE_DIR}/dataset_metadata.sha256"

TRAIN_COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_attn
  --dataset-dir "${DATASET_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda
  --epochs 25
  --batch-size 1024
  --num-workers 4
  --lr 0.00001
  --weight-decay 0.0001
  --warmup-epochs 1
  --grad-clip 1.0
  --seed 42
  --memory-shards 1
  --epochs-per-shard 25
  --shard-cycles 1
  --init-checkpoint "${INIT_CHECKPOINT}"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -m agentguard.cli validate_iwg_attn_checkpoint \
  --checkpoint "${LAST_CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  --device cuda \
  --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

sha256sum "${LAST_CHECKPOINT}" > "${PROVENANCE_DIR}/last_checkpoint.sha256"
complete=1
