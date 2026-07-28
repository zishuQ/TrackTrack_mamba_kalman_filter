#!/usr/bin/env bash
set -euo pipefail

# MOT20 from-scratch loss-v2 experiment using the schedule of the best
# checkpoint family: 4 shards x 1 epoch x 25 cycles (25 effective full
# passes). Labels and the public compact dataset are reused as-is.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_mot20_lossv2_best_schedule_bound005_seed42_bs1024_100e}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
DATASET_DIR="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_epoch100.pt"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_mot20_lossv2_best_schedule.sh

Configuration:
  MOT20 from scratch, bound=0.05, context=6, batch=1024, lr=1e-4,
  memory_shards=4, epochs_per_shard=1, shard_cycles=25.
  Loss-v2: residual_beta=0.01, residual_weight=1.0,
  revision_weight=0.0, hard_example_gain=2.0.

Stages:
  train -> checkpoint validation -> all base/final raw -> all base/final post
  -> test final+post.

Options:
  --dry-run   Print paths and settings without training or evaluation.
  --preflight Validate public dataset, model contract, and CUDA only.
EOF
}

DRY_RUN=0
PREFLIGHT=0
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
    --preflight)
      PREFLIGHT=1
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
CHECKPOINT=${CHECKPOINT}
SCHEDULE=epochs:100,memory_shards:4,epochs_per_shard:1,shard_cycles:25,effective_full_epochs:25
MODEL=correction_bound:0.05,context_size:6
LOSS=residual_beta:0.01,residual_weight:1.0,revision_weight:0.0,hard_example_gain:2.0
EVALUATION=all:base+final,raw+post;test:final,post
EOF
  exit 0
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

require_file "${DATASET_DIR}/metadata.json"
mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
if [[ -e "${RUN_ROOT}/running" || -e "${RUN_ROOT}/completed" ]]; then
  echo "Refusing an active or completed experiment: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${CHECKPOINT_DIR}/iwg_rg_cma_last.pt" || -e "${CHECKPOINT}" ]]; then
  echo "Refusing to reuse an existing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" - "${DATASET_DIR}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(f"{path}/metadata.json") as handle:
    metadata = json.load(handle)
expected = {
    "dataset": "MOT20",
    "split": "all",
    "index_format": "compact_memmap_v1",
    "context_size": 6,
}
for key, value in expected.items():
    actual = metadata.get(key, "jsonl_v1" if key == "index_format" else None)
    if actual != value:
        raise SystemExit(f"unexpected metadata {key}: {actual!r} != {value!r}")
if int(metadata.get("num_train_samples", 0)) <= 0:
    raise SystemExit("public MOT20 dataset has no training samples")
print(
    "dataset=",
    metadata["dataset"],
    "split=",
    metadata["split"],
    "samples=",
    metadata["num_train_samples"],
    "context_size=",
    metadata["context_size"],
    "index_format=",
    metadata["index_format"],
)
PY

"${PY}" - <<'PY'
import sys
import torch
from agentguard.models.iwg_rg_cma import model_contract

contract = model_contract(correction_bound=0.05, context_size=6)
print(
    "python=",
    sys.version.split()[0],
    "torch=",
    torch.__version__,
    "cuda=",
    torch.cuda.is_available(),
    "contract=",
    contract["model_schema"],
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this formal MOT20 training run")
PY

git -C "${ROOT}" rev-parse HEAD > "${RUN_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${RUN_ROOT}/git_status_porcelain.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/cli/__init__.py" \
  "${DATASET_DIR}/metadata.json" \
  > "${PROVENANCE_DIR}/formal_inputs.sha256"

if ((PREFLIGHT)); then
  echo "Preflight passed; no training or evaluation was started."
  exit 0
fi

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
  --epochs 100
  --batch-size 1024
  --num-workers 4
  --lr 0.0001
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
  --memory-shards 4
  --epochs-per-shard 1
  --shard-cycles 25
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
echo "$(date '+%Y-%m-%d %H:%M:%S') | TRAIN MOT20 loss-v2 best schedule" \
  | tee "${RUN_ROOT}/stages.log"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
require_file "${CHECKPOINT}"

"${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  --device cpu \
  --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

sha256sum "${DATASET_DIR}/metadata.json" "${CHECKPOINT}" \
  > "${PROVENANCE_DIR}/formal_artifacts.sha256"

run_raw_case() {
  local output="$1"
  local suffix="$2"
  local log_name="$3"
  local command=(
    "${PY}" -u run.py
    --dataset MOT20
    --mode all
    --sequences "${SEQUENCES[@]}"
    --seed 10000
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
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/${log_name}.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/${log_name}.command.txt"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

BASE_RAW="mot20_all_0.80_${RUN_NAME}_base_raw_agentguard_iwg_rg_cma_base"
FINAL_RAW="mot20_all_0.80_${RUN_NAME}_final_raw_agentguard_iwg_rg_cma_final"
run_raw_case base "${RUN_NAME}_base_raw" mot20_base_raw
run_raw_case final "${RUN_NAME}_final_raw" mot20_final_raw

BASE_POST="${BASE_RAW}_post"
FINAL_POST="${FINAL_RAW}_post"
for kind in base final; do
  source_dir="${TRACKER_OUTPUT_ROOT}/${BASE_RAW}"
  output_dir="${TRACKER_OUTPUT_ROOT}/${BASE_POST}"
  log_name="post_base"
  if [[ "${kind}" == "final" ]]; then
    source_dir="${TRACKER_OUTPUT_ROOT}/${FINAL_RAW}"
    output_dir="${TRACKER_OUTPUT_ROOT}/${FINAL_POST}"
    log_name="post_final"
  fi
  "${PY}" "${ROOT}/scripts/agentguard/postprocess_mot_tracker_folder.py" \
    --source-dir "${source_dir}" \
    --output-dir "${output_dir}" \
    --interval 30 \
    --tau 12 \
    2>&1 | tee "${LOG_DIR}/${log_name}.log"
done

"${PY}" "${ROOT}/scripts/agentguard/evaluate_iwg_rg_cma_mot20.py" \
  --case "base_raw=${BASE_RAW}" \
  --case "final_raw=${FINAL_RAW}" \
  --case "base_post=${BASE_POST}" \
  --case "final_post=${FINAL_POST}" \
  --output "${RUN_ROOT}/evaluation.json" \
  2>&1 | tee "${LOG_DIR}/evaluation.log"

TEST_SUFFIX="${RUN_NAME}_test_final_post"
TEST_FOLDER="mot20_test_0.80_${TEST_SUFFIX}_agentguard_iwg_rg_cma_final_post"
TEST_COMMAND=(
  "${PY}" -u run.py
  --dataset MOT20
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
  --resource-log "${RUN_ROOT}/resource_test_final.jsonl"
  --profile-every 500
)
printf '%q ' "${TEST_COMMAND[@]}" > "${PROVENANCE_DIR}/test_final.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/test_final.command.txt"
(cd "${TRACKER_ROOT}" && "${TEST_COMMAND[@]}") \
  2>&1 | tee "${LOG_DIR}/test_final_post.log"
require_file "${TRACKER_OUTPUT_ROOT}/${TEST_FOLDER}/MOT20-04.txt"

touch "${RUN_ROOT}/training_completed"
complete=1
echo "$(date '+%Y-%m-%d %H:%M:%S') | DONE MOT20 loss-v2 best schedule" \
  | tee -a "${RUN_ROOT}/stages.log"
