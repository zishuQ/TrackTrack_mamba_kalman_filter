#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate one SportsMOT train+val horizon setting.
# Validation is intentionally the official val split, even though trainval
# labels include it, because this is the user's requested trainval experiment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
HORIZON="${HORIZON:-8}"
CONTEXT_SIZE="${CONTEXT_SIZE:-6}"
CORRECTION_BOUND="${CORRECTION_BOUND:-0.10}"
EPOCHS="${EPOCHS:-200}"
MEMORY_SHARDS="${MEMORY_SHARDS:-4}"
EPOCHS_PER_SHARD="${EPOCHS_PER_SHARD:-1}"
SHARD_CYCLES="${SHARD_CYCLES:-50}"
LABEL_WORKERS="${LABEL_WORKERS:-4}"
BOUND_TAG="${CORRECTION_BOUND//./}"
if [[ "${CONTEXT_SIZE}" == "6" && "${CORRECTION_BOUND}" == "0.05" ]]; then
  MODEL_TAG="v1"
elif [[ "${CONTEXT_SIZE}" == "6" && "${CORRECTION_BOUND}" == "0.10" ]]; then
  MODEL_TAG="v2"
elif [[ "${CONTEXT_SIZE}" == "8" && "${CORRECTION_BOUND}" == "0.05" ]]; then
  MODEL_TAG="v3"
else
  MODEL_TAG="v4"
fi
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${CONTEXT_SIZE}" == "6" && "${CORRECTION_BOUND}" == "0.10" ]]; then
    RUN_NAME="iwg_rg_cma_v2_sportsmot_trainval_h${HORIZON}_bound010_seed42_bs1024_shard${MEMORY_SHARDS}x${EPOCHS_PER_SHARD}_${EPOCHS}e"
  else
    RUN_NAME="iwg_rg_cma_${MODEL_TAG}_sportsmot_trainval_future${HORIZON}_context${CONTEXT_SIZE}_bound${BOUND_TAG}_seed42_bs1024_shard${MEMORY_SHARDS}x${EPOCHS_PER_SHARD}_${EPOCHS}e"
  fi
fi

EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
RUN_ROOT="${EXPERIMENT_ROOT}/${RUN_NAME}"
LABEL_DIR="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_trainval_h${HORIZON}_v3_compact"
# Reusable training data is separate from per-experiment artifacts.
DATASET_ROOT="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/SportsMOT"
if [[ -n "${DATASET_DIR:-}" ]]; then
  DATASET_DIR="${DATASET_DIR}"
elif [[ "${CONTEXT_SIZE}" == "6" ]]; then
  DATASET_DIR="${DATASET_ROOT}/nsa_trainval_h${HORIZON}_v3_compact"
else
  DATASET_DIR="${DATASET_ROOT}/nsa_trainval_future${HORIZON}_context${CONTEXT_SIZE}_v3_compact"
fi
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_epoch$(printf '%03d' "${EPOCHS}").pt"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"

if ! [[ "${HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HORIZON must be a positive integer: ${HORIZON}" >&2
  exit 2
fi
if [[ "${CONTEXT_SIZE}" != "6" && "${CONTEXT_SIZE}" != "8" ]]; then
  echo "CONTEXT_SIZE must be 6 or 8: ${CONTEXT_SIZE}" >&2
  exit 2
fi
if [[ "${CORRECTION_BOUND}" != "0.05" && "${CORRECTION_BOUND}" != "0.10" ]]; then
  echo "CORRECTION_BOUND must be 0.05 or 0.10: ${CORRECTION_BOUND}" >&2
  exit 2
fi
if ! (( MEMORY_SHARDS >= 1 && EPOCHS_PER_SHARD >= 1 && SHARD_CYCLES >= 1 )); then
  echo "Shard schedule values must be positive" >&2
  exit 2
fi
if (( MEMORY_SHARDS * EPOCHS_PER_SHARD * SHARD_CYCLES != EPOCHS )); then
  echo "Invalid schedule: ${MEMORY_SHARDS} * ${EPOCHS_PER_SHARD} * ${SHARD_CYCLES} != ${EPOCHS}" >&2
  exit 2
fi
if (( LABEL_WORKERS < 1 )); then
  echo "LABEL_WORKERS must be positive" >&2
  exit 2
fi

mapfile -t SPORTS_VAL_SEQUENCES < <(
  sed '/^name$/d;/^$/d' "${ROOT}/3. Tracker/trackeval/seqmap/sportsmot/val.txt"
)
if ((${#SPORTS_VAL_SEQUENCES[@]} == 0)); then
  echo "SportsMOT val sequence map is empty" >&2
  exit 2
fi

usage() {
  cat <<EOF
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_sportsmot_trainval_horizon.sh

Defaults:
  HORIZON=${HORIZON}
  CONTEXT_SIZE=${CONTEXT_SIZE}
  CORRECTION_BOUND=${CORRECTION_BOUND}
  EPOCHS=${EPOCHS}
  MEMORY_SHARDS=${MEMORY_SHARDS}
  EPOCHS_PER_SHARD=${EPOCHS_PER_SHARD}
  SHARD_CYCLES=${SHARD_CYCLES}
  LABEL_WORKERS=${LABEL_WORKERS}
  RUN_NAME=${RUN_NAME}
  DATASET_DIR=${DATASET_DIR}

The run builds fresh trainval labels and data, trains from scratch, then
evaluates both base and final gates on SportsMOT val using raw metrics only.
It never runs post-processing and never overwrites an existing run.
EOF
}

DRY_RUN=0
RESUME=0
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
    --resume)
      RESUME=1
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
ROOT=${ROOT}
PY=${PY}
HORIZON=${HORIZON}
CONTEXT_SIZE=${CONTEXT_SIZE}
CORRECTION_BOUND=${CORRECTION_BOUND}
LABEL_WORKERS=${LABEL_WORKERS}
RUN_ROOT=${RUN_ROOT}
DATASET_ROOT=${DATASET_ROOT}
LABEL_DIR=${LABEL_DIR}
DATASET_DIR=${DATASET_DIR}
SCHEDULE=epochs:${EPOCHS},memory_shards:${MEMORY_SHARDS},epochs_per_shard:${EPOCHS_PER_SHARD},shard_cycles:${SHARD_CYCLES}
EFFECTIVE_FULL_PASSES=$((EPOCHS_PER_SHARD * SHARD_CYCLES))
EVALUATION=SportsMOT/val,base+final,raw,no_post
EOF
  exit 0
fi

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if [[ -e "${RUN_ROOT}/completed" ]]; then
  echo "Refusing a completed experiment: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/running" ]]; then
  pipeline_pid=""
  if [[ -f "${RUN_ROOT}/pipeline.pid" ]]; then
    pipeline_pid="$(<"${RUN_ROOT}/pipeline.pid")"
  fi
  if [[ -n "${pipeline_pid}" ]] && kill -0 "${pipeline_pid}" 2>/dev/null; then
    echo "An active experiment already owns ${RUN_ROOT} (pid=${pipeline_pid})" >&2
    exit 2
  fi
  if ((RESUME)); then
    rm -f "${RUN_ROOT}/running"
  else
    echo "Found a stale running marker; rerun with --resume: ${RUN_ROOT}" >&2
    exit 2
  fi
fi
if [[ -e "${LAST_CHECKPOINT}" ]]; then
  echo "Existing checkpoint requires manual audit; choose a fresh RUN_NAME: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
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

git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"
printf '%s\n' "${HORIZON}" > "${PROVENANCE_DIR}/future_frames.txt"
printf '%s\n' "${CONTEXT_SIZE}" > "${PROVENANCE_DIR}/context_size.txt"
printf '%s\n' "${CORRECTION_BOUND}" > "${PROVENANCE_DIR}/correction_bound.txt"
printf '%s\n' "${EPOCHS_PER_SHARD} * ${SHARD_CYCLES} = $((EPOCHS_PER_SHARD * SHARD_CYCLES)) effective full passes" \
  > "${PROVENANCE_DIR}/training_budget.txt"

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

"${PY}" -u "${ROOT}/scripts/agentguard/prepare_sportsmot_iwg_rg_cma_labels.py" \
  --split trainval \
  --future-frames "${HORIZON}" \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --dataset-root /home/shang/datasets/SportsMOT/dataset \
  --workers "${LABEL_WORKERS}" \
  --output-dir "${LABEL_DIR}" \
  2>&1 | tee "${LOG_DIR}/prepare_labels.log"

"${PY}" -c 'import json, sys; from pathlib import Path; p = Path(sys.argv[1]); expected_horizon = int(sys.argv[2]); expected_sequences = int(sys.argv[3]); summary = json.loads(p.read_text()); actual_sequences = len(summary.get("sequences", [])); assert summary.get("complete") and int(summary.get("future_frames", -1)) == expected_horizon and actual_sequences == expected_sequences, (p, summary.get("complete"), summary.get("future_frames"), actual_sequences); print(f"{p}: complete trainval labels ({actual_sequences} sequences, horizon={expected_horizon})")' \
  "${LABEL_DIR}/summary.json" "${HORIZON}" 90 \
  2>&1 | tee "${LOG_DIR}/label_contract.log"

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  "${PY}" -u -m agentguard.cli build_iwg_rg_cma_data \
    --dataset SportsMOT --mode trainval \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --label-dir "${LABEL_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --max-frame-gap 30 \
    --context-size "${CONTEXT_SIZE}" \
    2>&1 | tee "${LOG_DIR}/build_dataset.log"
else
  echo "Reusing existing dataset: ${DATASET_DIR}" | tee "${LOG_DIR}/build_dataset.log"
fi

sha256sum "${DATASET_DIR}/metadata.json" > "${PROVENANCE_DIR}/dataset_metadata.sha256"

TRAIN_COMMAND=(
  "${PY}" -u -m agentguard.cli train_iwg_rg_cma
  --dataset-dir "${DATASET_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda --epochs "${EPOCHS}" --batch-size 1024 --num-workers 4
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1
  --grad-clip 1.0 --seed 42
  --correction-bound "${CORRECTION_BOUND}"
  --context-size "${CONTEXT_SIZE}"
  --memory-shards "${MEMORY_SHARDS}"
  --epochs-per-shard "${EPOCHS_PER_SHARD}"
  --shard-cycles "${SHARD_CYCLES}"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
touch "${RUN_ROOT}/training_completed"

"${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
  --checkpoint "${FINAL_CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  --device cpu --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

run_raw_case() {
  local gate="$1"
  local checkpoint="${FINAL_CHECKPOINT}"
  local suffix="${RUN_NAME}_${gate}_val_raw"
  local log_name="sportsmot_val_${gate}_raw"
  local command=(
    "${PY}" -u run.py
    --dataset SportsMOT --mode val
    --sequences "${SPORTS_VAL_SEQUENCES[@]}"
    --seed 10000 --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output "${gate}" --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}" --print-per-sequence-metrics
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/eval_${gate}.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/eval_${gate}.command.txt"
  (
    cd "${ROOT}/3. Tracker"
    "${command[@]}"
  ) 2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

run_raw_case base
run_raw_case final
sha256sum "${FINAL_CHECKPOINT}" > "${PROVENANCE_DIR}/final_checkpoint.sha256"
pipeline_complete=1
