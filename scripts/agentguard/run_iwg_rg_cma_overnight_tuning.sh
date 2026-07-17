#!/usr/bin/env bash
set -euo pipefail

# Serial overnight experiments. This script reuses prepared datasets and caches.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MASTER_NAME="${MASTER_NAME:-iwg_rg_cma_overnight_tuning_20260717}"
MASTER_ROOT="${ROOT}/outputs/agentguard/overnight/${MASTER_NAME}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_overnight_tuning.sh

The pipeline runs serially in this order:
  1. MOT20 epoch-50 warm-start -> MOT17, 50 epochs, then MOT17 all/raw eval
  2. SportsMOT train+val, 5-shard and 4-shard schedules, 200 epochs each,
     then val/raw eval for each
  3. MOT20 NSA data, 10-shard and 4-shard schedules, 100 epochs each,
     then all/raw eval for each

Options:
  -h, --help       Show this message and exit without starting work.
      --dry-run    Print resolved paths/configuration and exit.

Environment:
  MASTER_NAME     Output directory name (default: iwg_rg_cma_overnight_tuning_20260717)
  PYTHON_BIN      Python executable (default: <repo>/.venv/bin/python)

Training requires a CUDA-capable PyTorch installation. Evaluation is CPU-only.
Use a fresh MASTER_NAME for a retry after a failed run; existing checkpoints are
never overwritten automatically.
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

MOT17_DATASET="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_trainall_seed42_bs1024_native_log/dataset"
MOT20_DATASET="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/dataset"
SPORTS_DATASET="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024/dataset"
MOT20_INIT="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024_shard20x2/checkpoints/iwg_rg_cma_epoch050.pt"

MOT17_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_mot20e050_to_mot17_finetune50_lr1e5_seed42_bs1024"
SPORTS_5_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_sportsmot_trainval_interleaved_shard5x1_seed42_bs1024_200e"
SPORTS_4_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_sportsmot_trainval_interleaved_shard4x1_seed42_bs1024_200e"
MOT20_10_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_mot20_interleaved_shard10x1_seed42_bs1024_100e"
MOT20_4_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_mot20_interleaved_shard4x1_seed42_bs1024_100e"

MOT17_SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)
MOT20_SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
mapfile -t SPORTS_VAL_SEQUENCES < <(
  sed '/^name$/d;/^$/d' "${ROOT}/3. Tracker/trackeval/seqmap/sportsmot/val.txt"
)

if ((DRY_RUN)); then
  cat <<EOF
ROOT=${ROOT}
PY=${PY}
MASTER_ROOT=${MASTER_ROOT}
MOT17_DATASET=${MOT17_DATASET}
MOT20_DATASET=${MOT20_DATASET}
SPORTS_DATASET=${SPORTS_DATASET}
MOT20_INIT=${MOT20_INIT}
DETECTION_CACHE_ROOT=${DETECTION_CACHE_ROOT}
EOF
  exit 0
fi

mkdir -p "${MASTER_ROOT}"
if [[ -e "${MASTER_ROOT}/running" ]]; then
  echo "Refusing an already-running tuning pipeline: ${MASTER_ROOT}" >&2
  exit 2
fi
if [[ -e "${MASTER_ROOT}/completed" ]]; then
  echo "Tuning pipeline is already complete: ${MASTER_ROOT}" >&2
  exit 2
fi

stage() {
  local message="$1"
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" \
    | tee -a "${MASTER_ROOT}/stages.log"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

git -C "${ROOT}" rev-parse HEAD > "${MASTER_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${MASTER_ROOT}/git_status_porcelain.txt"

require_file "${MOT17_DATASET}/metadata.json"
require_file "${MOT20_DATASET}/metadata.json"
require_file "${SPORTS_DATASET}/metadata.json"
require_file "${MOT20_INIT}"

# Fail before creating the running marker when this host cannot execute the
# requested CUDA training. This avoids a misleading partial pipeline state.
if ! "${PY}" -c 'import sys, torch; print(f"python={sys.version}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)' \
    > "${MASTER_ROOT}/environment.txt" 2>&1; then
  echo "CUDA preflight failed: this PyTorch environment cannot see a CUDA device." >&2
  echo "Run this script on the GPU host, or inspect ${MASTER_ROOT}/environment.txt." >&2
  exit 2
fi

echo "$$" > "${MASTER_ROOT}/pipeline.pid"
touch "${MASTER_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
  if [[ ${pipeline_complete} -eq 0 && ${code} -eq 0 ]]; then
    code=130
  fi
  rm -f "${MASTER_ROOT}/running"
  echo "${code}" > "${MASTER_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${pipeline_complete} -eq 1 ]]; then
    touch "${MASTER_ROOT}/completed"
  else
    touch "${MASTER_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

train_run() {
  local run_root="$1"
  local dataset_dir="$2"
  local epochs="$3"
  local lr="$4"
  local memory_shards="$5"
  local epochs_per_shard="$6"
  local shard_cycles="$7"
  local init_checkpoint="${8:-}"
  local checkpoint_dir="${run_root}/checkpoints"
  local last_checkpoint="${checkpoint_dir}/iwg_rg_cma_last.pt"

  if (( memory_shards * epochs_per_shard * shard_cycles != epochs )); then
    echo "Invalid schedule for ${run_root}: ${memory_shards} * ${epochs_per_shard} * ${shard_cycles} != ${epochs}" >&2
    return 2
  fi
  mkdir -p "${checkpoint_dir}" "${run_root}/logs" "${run_root}/provenance"
  if [[ ! -e "${run_root}/training_completed" ]]; then
    if [[ -e "${last_checkpoint}" ]]; then
      echo "Incomplete existing run requires manual audit: ${run_root}" >&2
      return 2
    fi
    stage "TRAIN ${run_root##*/}: epochs=${epochs}, shards=${memory_shards}, epochs_per_shard=${epochs_per_shard}, cycles=${shard_cycles}"
    local command=(
      "${PY}" -u -m agentguard.cli train_iwg_attn
      --dataset-dir "${dataset_dir}"
      --checkpoint-dir "${checkpoint_dir}"
      --device cuda --epochs "${epochs}" --batch-size 1024 --num-workers 4
      --lr "${lr}" --weight-decay 0.0001 --warmup-epochs 1
      --grad-clip 1.0 --seed 42
      --memory-shards "${memory_shards}"
      --epochs-per-shard "${epochs_per_shard}"
      --shard-cycles "${shard_cycles}"
    )
    if [[ -n "${init_checkpoint}" ]]; then
      command+=(--init-checkpoint "${init_checkpoint}")
    fi
    printf '%q ' "${command[@]}" > "${run_root}/provenance/train.command.txt"
    printf '\n' >> "${run_root}/provenance/train.command.txt"
    "${command[@]}" 2>&1 | tee "${run_root}/logs/train.log"
    touch "${run_root}/training_completed"
  else
    stage "TRAIN ${run_root##*/}: already complete, reuse checkpoint"
  fi

  "${PY}" -u -m agentguard.cli validate_iwg_attn_checkpoint \
    --checkpoint "${last_checkpoint}" --dataset-dir "${dataset_dir}" \
    --device cpu --max-batches 8 \
    --output "${run_root}/checkpoint_validation.json" \
    2>&1 | tee "${run_root}/logs/checkpoint_validation.log"
  sha256sum "${dataset_dir}/metadata.json" "${last_checkpoint}" \
    > "${run_root}/provenance/formal_artifacts.sha256"
}

eval_tracker() {
  local run_root="$1"
  local dataset="$2"
  local mode="$3"
  local checkpoint_epoch="$4"
  local suffix="$5"
  local log_name="$6"
  shift 6
  local checkpoint="${run_root}/checkpoints/iwg_rg_cma_epoch${checkpoint_epoch}.pt"
  local marker="${run_root}/evaluation_${log_name}.completed"
  require_file "${checkpoint}"
  if [[ -e "${marker}" ]]; then
    stage "EVAL ${run_root##*/}/${log_name}: already complete, skip"
    return
  fi
  local sequences=("$@")
  local command=(
    "${PY}" -u run.py
    --dataset "${dataset}" --mode "${mode}"
    --agentguard-mode iwg-attn
    --agentguard-checkpoint "${checkpoint}"
    --iwg-attn-output final --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
  )
  if ((${#sequences[@]} > 0)); then
    command+=(--sequences "${sequences[@]}")
  fi
  stage "EVAL ${run_root##*/}/${log_name}: ${dataset} ${mode} final raw"
  printf '%q ' "${command[@]}" > "${run_root}/provenance/${log_name}.command.txt"
  printf '\n' >> "${run_root}/provenance/${log_name}.command.txt"
  (
    cd "${ROOT}/3. Tracker"
    "${command[@]}"
  ) 2>&1 | tee "${run_root}/logs/${log_name}.log"
  touch "${marker}"
}

stage "START serial tuning pipeline"

# 1. MOT20 old e050 warm-start -> MOT17, controlled 50e extension at 1e-5.
train_run "${MOT17_RUN}" "${MOT17_DATASET}" 50 0.00001 1 50 1 "${MOT20_INIT}"
eval_tracker "${MOT17_RUN}" MOT17 all 050 mot17_mot20e050_finetune50_final_all mot17_all_final \
  "${MOT17_SEQUENCES[@]}"

# 2. SportsMOT train+val, new interleaved schedule with 5 and 4 fractions.
train_run "${SPORTS_5_RUN}" "${SPORTS_DATASET}" 200 0.0001 5 1 40
eval_tracker "${SPORTS_5_RUN}" SportsMOT val 200 sportsmot_trainval_s5_e200_final_val sportsmot_val_final \
  "${SPORTS_VAL_SEQUENCES[@]}"

train_run "${SPORTS_4_RUN}" "${SPORTS_DATASET}" 200 0.0001 4 1 50
eval_tracker "${SPORTS_4_RUN}" SportsMOT val 200 sportsmot_trainval_s4_e200_final_val sportsmot_val_final \
  "${SPORTS_VAL_SEQUENCES[@]}"

# 3. MOT20, new interleaved schedule with 10 and 4 fractions.
train_run "${MOT20_10_RUN}" "${MOT20_DATASET}" 100 0.0001 10 1 10
eval_tracker "${MOT20_10_RUN}" MOT20 all 100 mot20_interleaved_s10_e100_final_all mot20_all_final \
  "${MOT20_SEQUENCES[@]}"

train_run "${MOT20_4_RUN}" "${MOT20_DATASET}" 100 0.0001 4 1 25
eval_tracker "${MOT20_4_RUN}" MOT20 all 100 mot20_interleaved_s4_e100_final_all mot20_all_final \
  "${MOT20_SEQUENCES[@]}"

stage "DONE serial tuning pipeline"
pipeline_complete=1
