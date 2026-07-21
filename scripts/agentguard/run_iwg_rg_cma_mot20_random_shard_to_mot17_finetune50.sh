#!/usr/bin/env bash
set -euo pipefail

# Warm-start MOT17 from the MOT20 random-shard-order epoch100 checkpoint.
# MOT17 is trained as one full-data phase; no memory-shard arguments are used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"

RUN_NAME="${RUN_NAME:-iwg_rg_cma_mot20_random_shard_to_mot17_bound010_finetune50_seed42_bs1024}"
RUN_ROOT="${EXPERIMENT_ROOT}/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
MOT17_DATASET="${MOT17_DATASET_DIR:-${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_jsonl}"
MOT20_INIT_CHECKPOINT="${MOT20_INIT_CHECKPOINT:-${EXPERIMENT_ROOT}/iwg_rg_cma_v2_mot20_random_shard_order_bound010_seed42_bs1024_shard4x1_100e/checkpoints/iwg_rg_cma_epoch100.pt}"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_epoch050.pt"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"

MOT17_SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_mot20_random_shard_to_mot17_finetune50.sh

The script expects the MOT20 random-shard-order epoch100 checkpoint, trains
MOT17 for 50 epochs at lr=1e-5, validates the checkpoint, and runs final raw
MOT17 all-sequence evaluation. No post-processing is run.

Environment overrides:
  RUN_NAME              Target experiment name.
  MOT17_DATASET_DIR     MOT17 public dataset directory.
  MOT20_INIT_CHECKPOINT MOT20 epoch100 source checkpoint.
  PYTHON_BIN            Python executable.
EOF
}

DRY_RUN=0
while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((DRY_RUN)); then
  cat <<EOF
RUN_ROOT=${RUN_ROOT}
MOT17_DATASET=${MOT17_DATASET}
MOT20_INIT_CHECKPOINT=${MOT20_INIT_CHECKPOINT}
FINAL_CHECKPOINT=${FINAL_CHECKPOINT}
TRAINING=dataset:MOT17,epochs:50,batch:1024,lr:1e-5,bound:0.10,context:6,full_data_phase:true
SHARD_ARGUMENTS=none
EVALUATION=MOT17/all,final,raw,no_post
EOF
  exit 0
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

require_file "${MOT17_DATASET}/metadata.json"
require_file "${MOT20_INIT_CHECKPOINT}"

if [[ -e "${RUN_ROOT}/running" || -e "${LAST_CHECKPOINT}" || -e "${FINAL_CHECKPOINT}" ]]; then
  echo "Refusing to overwrite or duplicate run: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" -c 'import sys, torch; print(f"python={sys.version}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)' \
  > "${PROVENANCE_DIR}/environment.txt" 2>&1

git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"
sha256sum "${MOT20_INIT_CHECKPOINT}" > "${PROVENANCE_DIR}/initial_checkpoint.sha256"
sha256sum "${MOT17_DATASET}/metadata.json" > "${PROVENANCE_DIR}/dataset_metadata.sha256"

echo "$$" > "${RUN_ROOT}/pipeline.pid"
touch "${RUN_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
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

TRAIN_COMMAND=(
  "${PY}" -u -m agentguard.cli train_iwg_attn
  --dataset-dir "${MOT17_DATASET}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda
  --epochs 50
  --batch-size 1024
  --num-workers 4
  --lr 0.00001
  --weight-decay 0.0001
  --warmup-epochs 1
  --grad-clip 1.0
  --seed 42
  --correction-bound 0.10
  --context-size 6
  --init-checkpoint "${MOT20_INIT_CHECKPOINT}"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
require_file "${FINAL_CHECKPOINT}"
touch "${RUN_ROOT}/training_completed"

"${PY}" -u -m agentguard.cli validate_iwg_attn_checkpoint \
  --checkpoint "${FINAL_CHECKPOINT}" \
  --dataset-dir "${MOT17_DATASET}" \
  --device cpu --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

EVAL_SUFFIX="${RUN_NAME}_final_all_raw"
EVAL_LOG_NAME="mot17_final_all_raw"
EVAL_COMMAND=(
  "${PY}" -u run.py
  --dataset MOT17 --mode all
  --sequences "${MOT17_SEQUENCES[@]}"
  --seed 10000 --kf-type nsa
  --agentguard-mode iwg-attn
  --agentguard-checkpoint "${FINAL_CHECKPOINT}"
  --iwg-attn-output final
  --agentguard-device cpu
  --detection-cache-root "${DETECTION_CACHE_ROOT}"
  --tracker-suffix "${EVAL_SUFFIX}"
  --print-per-sequence-metrics
  --resource-log "${RUN_ROOT}/resource_${EVAL_LOG_NAME}.jsonl"
  --profile-every 500
)
printf '%q ' "${EVAL_COMMAND[@]}" > "${PROVENANCE_DIR}/${EVAL_LOG_NAME}.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/${EVAL_LOG_NAME}.command.txt"
(
  cd "${TRACKER_ROOT}"
  "${EVAL_COMMAND[@]}"
) 2>&1 | tee "${LOG_DIR}/${EVAL_LOG_NAME}.log"
touch "${RUN_ROOT}/evaluation_${EVAL_LOG_NAME}.completed"

sha256sum "${FINAL_CHECKPOINT}" > "${PROVENANCE_DIR}/final_checkpoint.sha256"
pipeline_complete=1
