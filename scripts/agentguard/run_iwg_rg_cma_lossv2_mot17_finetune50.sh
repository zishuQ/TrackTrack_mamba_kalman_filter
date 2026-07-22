#!/usr/bin/env bash
set -euo pipefail

# Controlled MOT17 warm-start experiment for the RG-CMA residual-loss v2.
# The model contract stays at bound=0.05/context=6 so this run isolates loss
# changes from the previously rejected bound=0.10 change.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_mot20e100_to_mot17_lossv2_finetune50_bound005_seed42_bs1024}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_epoch050.pt"
DATASET_DIR="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_jsonl_native_log"
INIT_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_v1_mot20_interleaved_shard4x1_seed42_bs1024_100e/checkpoints/iwg_rg_cma_epoch100.pt"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/outputs/3. track"

SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_lossv2_mot17_finetune50.sh

Stages:
  1. Warm-start MOT17 for 50 epochs from the best MOT20 epoch100 weights.
  2. Validate the checkpoint on the MOT17 training dataset.
  3. Evaluate MOT17 train with base/final raw gates, then postprocess both.
  4. Run MOT17 test with final gate + postprocessing and package the submission.

The loss-v2 settings are residual_beta=0.01, residual_weight=1.0,
revision_weight=0.0, hard_example_gain=2.0. The model contract remains
correction_bound=0.05 and context_size=6 for a controlled comparison.

Environment overrides:
  RUN_NAME    Experiment directory name. Use a fresh name for a retry.
  PYTHON_BIN  Python executable (default: <repo>/.venv/bin/python).
  --dry-run   Print paths and the training configuration without running.
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
RUN_ROOT=${RUN_ROOT}
DATASET_DIR=${DATASET_DIR}
INIT_CHECKPOINT=${INIT_CHECKPOINT}
CHECKPOINT=${CHECKPOINT}
SCHEDULE=epochs:50,memory_shards:1,epochs_per_shard:50,shard_cycles:1
MODEL=correction_bound:0.05,context_size:6
LOSS=residual_beta:0.01,residual_weight:1.0,revision_weight:0.0,hard_example_gain:2.0
EVALUATION=train:base+final,raw+post;test:final,post
EOF
  exit 0
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
if [[ -e "${RUN_ROOT}/completed" ]]; then
  echo "Experiment is already complete: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/running" ]]; then
  echo "Experiment is already running: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${CHECKPOINT_DIR}/iwg_rg_cma_last.pt" || -e "${CHECKPOINT}" ]]; then
  echo "Refusing to reuse an incomplete checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

require_file "${DATASET_DIR}/metadata.json"
require_file "${INIT_CHECKPOINT}"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if ! "${PY}" -c 'import sys, torch; print(f"python={sys.version}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)' \
    > "${RUN_ROOT}/environment.txt" 2>&1; then
  echo "CUDA preflight failed. See ${RUN_ROOT}/environment.txt" >&2
  exit 2
fi

git -C "${ROOT}" rev-parse HEAD > "${RUN_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${RUN_ROOT}/git_status_porcelain.txt"
sha256sum "${DATASET_DIR}/metadata.json" "${INIT_CHECKPOINT}" \
  > "${RUN_ROOT}/provenance/input_artifacts.sha256"

echo "$$" > "${RUN_ROOT}/pipeline.pid"
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

TRAIN_COMMAND=(
  "${PY}" -u -m agentguard.cli train_iwg_rg_cma
  --dataset-dir "${DATASET_DIR}"
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
  --correction-bound 0.05
  --context-size 6
  --residual-beta 0.01
  --residual-weight 1.0
  --revision-weight 0.0
  --hard-example-gain 2.0
  --init-checkpoint "${INIT_CHECKPOINT}"
  --memory-shards 1
  --epochs-per-shard 50
  --shard-cycles 1
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${RUN_ROOT}/provenance/train.command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/train.command.txt"
echo "$(date '+%Y-%m-%d %H:%M:%S') | TRAIN MOT17 loss v2" | tee "${RUN_ROOT}/stages.log"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"
require_file "${CHECKPOINT}"

"${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
  --checkpoint "${CHECKPOINT}" --dataset-dir "${DATASET_DIR}" \
  --device cpu --max-batches 8 --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${RUN_ROOT}/logs/checkpoint_validation.log"
sha256sum "${DATASET_DIR}/metadata.json" "${CHECKPOINT}" \
  > "${RUN_ROOT}/provenance/formal_artifacts.sha256"

run_train_case() {
  local output="$1"
  local suffix="$2"
  local log_name="$3"
  local command=(
    "${PY}" -u run.py
    --dataset MOT17
    --mode all
    --sequences "${SEQUENCES[@]}"
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --legacy-output-naming
    --agentguard-checkpoint "${CHECKPOINT}"
    --iwg-rg-cma-output "${output}"
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${RUN_ROOT}/resource_${log_name}.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/${log_name}.command.txt"
  printf '\n' >> "${RUN_ROOT}/provenance/${log_name}.command.txt"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${RUN_ROOT}/logs/${log_name}.log"
}

BASE_RAW="mot17_all_0.80_${RUN_NAME}_base_raw_agentguard_iwg_rg_cma_base"
FINAL_RAW="mot17_all_0.80_${RUN_NAME}_final_raw_agentguard_iwg_rg_cma_final"
run_train_case base "${RUN_NAME}_base_raw" mot17_base_raw
run_train_case final "${RUN_NAME}_final_raw" mot17_final_raw

BASE_POST="${BASE_RAW}_post"
FINAL_POST="${FINAL_RAW}_post"
"${PY}" "${ROOT}/scripts/agentguard/postprocess_mot_tracker_folder.py" \
  --source-dir "${TRACKER_ROOT}/${BASE_RAW}" \
  --output-dir "${TRACKER_ROOT}/${BASE_POST}" \
  2>&1 | tee "${RUN_ROOT}/logs/post_base.log"
"${PY}" "${ROOT}/scripts/agentguard/postprocess_mot_tracker_folder.py" \
  --source-dir "${TRACKER_ROOT}/${FINAL_RAW}" \
  --output-dir "${TRACKER_ROOT}/${FINAL_POST}" \
  2>&1 | tee "${RUN_ROOT}/logs/post_final.log"

"${PY}" "${ROOT}/scripts/agentguard/evaluate_iwg_rg_cma.py" \
  --case "base_raw=${BASE_RAW}" \
  --case "final_raw=${FINAL_RAW}" \
  --case "base_post=${BASE_POST}" \
  --case "final_post=${FINAL_POST}" \
  --output "${RUN_ROOT}/train_evaluation.json" \
  2>&1 | tee "${RUN_ROOT}/logs/train_evaluation.log"

TEST_SUFFIX="${RUN_NAME}_test_final_post"
TEST_FOLDER="mot17_test_0.80_${TEST_SUFFIX}_agentguard_iwg_rg_cma_final_post"
TEST_COMMAND=(
  "${PY}" -u run.py
  --dataset MOT17
  --mode test
  --seed 10000
  --agentguard-mode iwg-rg-cma
  --legacy-output-naming
  --agentguard-checkpoint "${CHECKPOINT}"
  --iwg-rg-cma-output final
  --agentguard-device cpu
  --detection-cache-root "${DETECTION_CACHE_ROOT}"
  --tracker-suffix "${TEST_SUFFIX}"
  --use_post
  --skip-eval
  --resource-log "${RUN_ROOT}/resource_test.jsonl"
  --profile-every 500
)
printf '%q ' "${TEST_COMMAND[@]}" > "${RUN_ROOT}/provenance/test.command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/test.command.txt"
(cd "${TRACKER_ROOT}" && "${TEST_COMMAND[@]}") \
  2>&1 | tee "${RUN_ROOT}/logs/test.log"
require_file "${TRACKER_ROOT}/${TEST_FOLDER}/MOT17-01-FRCNN.txt"

"${PY}" "${ROOT}/scripts/agentguard/package_mot17_submission.py" \
  --source-dir "${TRACKER_ROOT}/${TEST_FOLDER}" \
  --output-zip "${RUN_ROOT}/MOT17_test_lossv2_final_post.zip" \
  --manifest "${RUN_ROOT}/submission_manifest.json" \
  2>&1 | tee "${RUN_ROOT}/logs/submission.log"

touch "${RUN_ROOT}/training_completed"
complete=1
echo "$(date '+%Y-%m-%d %H:%M:%S') | DONE MOT17 loss v2" | tee -a "${RUN_ROOT}/stages.log"
