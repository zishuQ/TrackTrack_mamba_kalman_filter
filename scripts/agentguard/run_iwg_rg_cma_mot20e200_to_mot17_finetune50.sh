#!/usr/bin/env bash
set -euo pipefail

# Serial MOT20 e200 training -> MOT17 warm-start fine-tuning pipeline.
# Both evaluations are final-gate, all-split, raw tracker evaluations.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
PIPELINE_NAME="${PIPELINE_NAME:-iwg_rg_cma_mot20e200_to_mot17_finetune50_20260717}"
PIPELINE_ROOT="${ROOT}/outputs/agentguard/overnight/${PIPELINE_NAME}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"

MOT20_DATASET="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_mot20_trainall_seed42_bs1024/dataset"
MOT17_DATASET="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_trainall_seed42_bs1024_native_log/dataset"

MOT20_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v1_mot20_interleaved_shard4x1_seed42_bs1024_200e"
MOT17_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_mot20s4e200_to_mot17_finetune50_lr1e5_bs1024"

MOT20_CHECKPOINT="${MOT20_RUN}/checkpoints/iwg_rg_cma_epoch200.pt"
MOT17_CHECKPOINT="${MOT17_RUN}/checkpoints/iwg_rg_cma_epoch050.pt"

MOT20_SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
MOT17_SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_mot20e200_to_mot17_finetune50.sh

Serial stages:
  1. MOT20 NSA-event data, new 4-shard schedule, 200 epochs
     (memory_shards=4, epochs_per_shard=1, shard_cycles=50)
  2. MOT20 epoch200 checkpoint -> MOT17 full FRCNN data, 50 epochs at lr=1e-5
  3. After each stage, evaluate final gate on all train sequences, raw only

The script never runs base-gate evaluation and never passes --use_post.
Training uses CUDA; tracker evaluation uses CPU.

Environment overrides:
  PIPELINE_NAME   Pipeline state directory name.
  PYTHON_BIN      Python executable (default: <repo>/.venv/bin/python).

Use a fresh PIPELINE_NAME for a separately audited experiment. Existing
checkpoints are never overwritten automatically.
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
ROOT=${ROOT}
PY=${PY}
PIPELINE_ROOT=${PIPELINE_ROOT}
MOT20_DATASET=${MOT20_DATASET}
MOT20_RUN=${MOT20_RUN}
MOT20_CHECKPOINT=${MOT20_CHECKPOINT}
MOT17_DATASET=${MOT17_DATASET}
MOT17_RUN=${MOT17_RUN}
MOT17_CHECKPOINT=${MOT17_CHECKPOINT}
DETECTION_CACHE_ROOT=${DETECTION_CACHE_ROOT}
MOT20_SCHEDULE=epochs:200,memory_shards:4,epochs_per_shard:1,shard_cycles:50
MOT17_SCHEDULE=epochs:50,memory_shards:1,epochs_per_shard:50,shard_cycles:1,lr:1e-5
EVALUATION=final,all,raw,cpu,no_post
EOF
  exit 0
fi

mkdir -p "${PIPELINE_ROOT}"
if [[ -e "${PIPELINE_ROOT}/running" ]]; then
  echo "Refusing an already-running pipeline: ${PIPELINE_ROOT}" >&2
  exit 2
fi
if [[ -e "${PIPELINE_ROOT}/completed" ]]; then
  echo "Pipeline is already complete: ${PIPELINE_ROOT}" >&2
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

stage() {
  local message="$1"
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" \
    | tee -a "${PIPELINE_ROOT}/stages.log"
}

require_file "${MOT20_DATASET}/metadata.json"
require_file "${MOT17_DATASET}/metadata.json"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if ! "${PY}" -c 'import sys, torch; print(f"python={sys.version}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)' \
    > "${PIPELINE_ROOT}/environment.txt" 2>&1; then
  echo "CUDA preflight failed. See ${PIPELINE_ROOT}/environment.txt" >&2
  exit 2
fi

git -C "${ROOT}" rev-parse HEAD > "${PIPELINE_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PIPELINE_ROOT}/git_status_porcelain.txt"
sha256sum "${MOT20_DATASET}/metadata.json" "${MOT17_DATASET}/metadata.json" \
  > "${PIPELINE_ROOT}/dataset_metadata.sha256"

echo "$$" > "${PIPELINE_ROOT}/pipeline.pid"
touch "${PIPELINE_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
  rm -f "${PIPELINE_ROOT}/running"
  echo "${code}" > "${PIPELINE_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${pipeline_complete} -eq 1 ]]; then
    touch "${PIPELINE_ROOT}/completed"
  else
    touch "${PIPELINE_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

train_stage() {
  local stage_name="$1"
  local run_root="$2"
  local dataset_dir="$3"
  local epochs="$4"
  local lr="$5"
  local memory_shards="$6"
  local epochs_per_shard="$7"
  local shard_cycles="$8"
  local init_checkpoint="${9:-}"
  local expected_checkpoint="${10}"
  local checkpoint_dir="${run_root}/checkpoints"
  local completed_marker="${run_root}/training_completed"

  if ((memory_shards * epochs_per_shard * shard_cycles != epochs)); then
    echo "Invalid schedule for ${run_root}: ${memory_shards} * ${epochs_per_shard} * ${shard_cycles} != ${epochs}" >&2
    exit 2
  fi
  mkdir -p "${checkpoint_dir}" "${run_root}/logs" "${run_root}/provenance"

  if [[ -e "${completed_marker}" ]]; then
    require_file "${expected_checkpoint}"
    stage "TRAIN ${stage_name}: already complete, reuse checkpoint"
    return
  fi
  if [[ -e "${checkpoint_dir}/iwg_rg_cma_last.pt" || -e "${expected_checkpoint}" ]]; then
    echo "Incomplete existing training run requires manual audit: ${run_root}" >&2
    exit 2
  fi

  local command=(
    "${PY}" -u -m agentguard.cli train_iwg_rg_cma
    --dataset-dir "${dataset_dir}"
    --checkpoint-dir "${checkpoint_dir}"
    --device cuda
    --epochs "${epochs}"
    --batch-size 1024
    --num-workers 4
    --lr "${lr}"
    --weight-decay 0.0001
    --warmup-epochs 1
    --grad-clip 1.0
    --seed 42
    --memory-shards "${memory_shards}"
    --epochs-per-shard "${epochs_per_shard}"
    --shard-cycles "${shard_cycles}"
  )
  if [[ -n "${init_checkpoint}" ]]; then
    require_file "${init_checkpoint}"
    command+=(--init-checkpoint "${init_checkpoint}")
    sha256sum "${init_checkpoint}" > "${run_root}/provenance/initial_checkpoint.sha256"
  fi

  printf '%q ' "${command[@]}" > "${run_root}/provenance/train.command.txt"
  printf '\n' >> "${run_root}/provenance/train.command.txt"
  stage "TRAIN ${stage_name}: epochs=${epochs}, shards=${memory_shards}, epochs_per_shard=${epochs_per_shard}, cycles=${shard_cycles}, lr=${lr}"
  "${command[@]}" 2>&1 | tee "${run_root}/logs/train.log"
  require_file "${expected_checkpoint}"
  touch "${completed_marker}"
  sha256sum "${dataset_dir}/metadata.json" "${expected_checkpoint}" \
    > "${run_root}/provenance/formal_artifacts.sha256"
}

eval_final_raw() {
  local stage_name="$1"
  local run_root="$2"
  local dataset="$3"
  local checkpoint="$4"
  local suffix="$5"
  local log_name="$6"
  shift 6
  local marker="${run_root}/evaluation_${log_name}.completed"
  local log_path="${run_root}/logs/${log_name}.log"
  local sequences=("$@")

  require_file "${checkpoint}"
  if [[ -e "${marker}" ]]; then
    stage "EVAL ${stage_name}: already complete, reuse raw metrics"
    return
  fi

  local command=(
    "${PY}" -u run.py
    --dataset "${dataset}"
    --mode all
    --sequences "${sequences[@]}"
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output final
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${run_root}/resource_${log_name}.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${run_root}/provenance/${log_name}.command.txt"
  printf '\n' >> "${run_root}/provenance/${log_name}.command.txt"
  stage "EVAL ${stage_name}: ${dataset} all final raw (no post, no base)"
  (
    cd "${TRACKER_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
  require_file "${log_path}"
  touch "${marker}"

  local metric_line
  metric_line="$(awk '$1 == "HOTA" && $2 == "MOTA" { getline; line=$0 } END { print line }' "${log_path}")"
  if [[ -n "${metric_line}" ]]; then
    printf '%s\t%s\n' "${stage_name}" "${metric_line}" \
      | tee -a "${PIPELINE_ROOT}/metrics_summary.txt"
  else
    echo "${stage_name}: HOTA line was not found; inspect ${log_path}" \
      | tee -a "${PIPELINE_ROOT}/metrics_summary.txt"
  fi
}

stage "START MOT20 e200 -> MOT17 finetune50 serial pipeline"

# Stage 1: the only new variable versus MOT20 4-shard e100 is total training
# duration. This gives 50 effective full passes without loading two fractions.
train_stage \
  "MOT20 new 4-shard e200" \
  "${MOT20_RUN}" \
  "${MOT20_DATASET}" \
  200 0.0001 4 1 50 "" "${MOT20_CHECKPOINT}"

eval_final_raw \
  "MOT20 e200" \
  "${MOT20_RUN}" \
  MOT20 \
  "${MOT20_CHECKPOINT}" \
  mot20_s4_e200_final_all_raw \
  mot20_e200_final_all_raw \
  "${MOT20_SEQUENCES[@]}"

# Stage 2: strict model-weight warm start; optimizer and scheduler are reset by
# train_iwg_rg_cma, as required for cross-dataset fine-tuning.
train_stage \
  "MOT17 from MOT20 e200, 50e" \
  "${MOT17_RUN}" \
  "${MOT17_DATASET}" \
  50 0.00001 1 50 1 "${MOT20_CHECKPOINT}" "${MOT17_CHECKPOINT}"

eval_final_raw \
  "MOT17 finetune e050" \
  "${MOT17_RUN}" \
  MOT17 \
  "${MOT17_CHECKPOINT}" \
  mot20s4e200_mot17_finetune_e050_final_all_raw \
  mot17_finetune_e050_final_all_raw \
  "${MOT17_SEQUENCES[@]}"

stage "DONE MOT20 e200 -> MOT17 finetune50 serial pipeline"
pipeline_complete=1
