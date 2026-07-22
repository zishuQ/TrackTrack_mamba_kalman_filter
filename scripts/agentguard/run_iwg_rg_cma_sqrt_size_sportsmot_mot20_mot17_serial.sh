#!/usr/bin/env bash
set -euo pipefail

# Serial sqrt-size sequence-sampling experiment:
#   1. SportsMOT trainval, baseline context=6/bound=0.05, 200 epochs
#   2. MOT20 all, baseline context=6/bound=0.05, 100 epochs
#   3. MOT20 epoch100 -> MOT17, baseline context=6/bound=0.05, 50 epochs
#
# The first two stages use the fixed 4-shard schedule. MOT17 intentionally
# omits shard arguments and runs as one full-data phase. Every stage uses the
# sqrt-size sampler, then validates final raw output and runs test with post.
# Labels and datasets are reused from the public outputs/agentguard assets.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
PIPELINE_NAME="${PIPELINE_NAME:-iwg_rg_cma_sqrt_size_sportsmot_mot20_mot17_serial_20260719}"
PIPELINE_ROOT="${ROOT}/outputs/agentguard/overnight/${PIPELINE_NAME}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DATASET_ROOT="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"

SPORTS_DATASET="${DATASET_ROOT}/SportsMOT/nsa_trainval_v3_compact"
MOT20_DATASET="${DATASET_ROOT}/MOT20/nsa_all_v3_compact"
MOT17_DATASET="${DATASET_ROOT}/MOT17/nsa_all_v3_jsonl"

SPORTS_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_sqrt_size_sportsmot_trainval_bound005_context6_seed42_bs1024_shard4x1_200e"
MOT20_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_sqrt_size_mot20_bound005_context6_seed42_bs1024_shard4x1_100e"
MOT17_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_sqrt_size_mot20e100_to_mot17_bound005_context6_seed42_bs1024_finetune50"

SPORTS_CHECKPOINT="${SPORTS_RUN}/checkpoints/iwg_rg_cma_epoch200.pt"
MOT20_CHECKPOINT="${MOT20_RUN}/checkpoints/iwg_rg_cma_epoch100.pt"
MOT17_CHECKPOINT="${MOT17_RUN}/checkpoints/iwg_rg_cma_epoch050.pt"

SPORTS_SEQUENCES=()
mapfile -t SPORTS_SEQUENCES < <(
  sed '/^name$/d;/^$/d' "${ROOT}/3. Tracker/trackeval/seqmap/sportsmot/val.txt"
)
MOT20_SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
MOT17_SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_sqrt_size_sportsmot_mot20_mot17_serial.sh

Serial stages:
  1. SportsMOT trainval, sqrt-size sampling, 200 epochs,
     4 shards x 1 epoch/shard x 50 cycles.
  2. MOT20 all, sqrt-size sampling, 100 epochs,
     4 shards x 1 epoch/shard x 25 cycles.
  3. MOT20 epoch100 warm-start -> MOT17, sqrt-size sampling, 50 epochs.

Each stage performs:
  train -> checkpoint validation -> final raw validation -> test with post.

The main experiment variables are fixed at bound=0.05, context=6, seed=42,
batch=1024, lr=1e-4 for fresh training, lr=1e-5 for MOT17 warm-start, and
fixed shard order. The sqrt-size sampler preserves the number of samples and
optimizer steps per phase but samples within sequences with replacement.

Options:
  -h, --help       Show this message and exit.
      --dry-run    Print paths and schedules without training or evaluation.
      --preflight  Validate datasets, CUDA, and model contract only.

Environment overrides:
  PIPELINE_NAME    Pipeline state directory name.
  PYTHON_BIN       Python executable.
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
PIPELINE_ROOT=${PIPELINE_ROOT}
SAMPLING=sequence_sampling:sqrt-size,replacement:true
CONTROL=correction_bound:0.05,context_size:6,seed:42,batch_size:1024
SPORTS_DATASET=${SPORTS_DATASET}
SPORTS_RUN=${SPORTS_RUN}
SPORTS_SCHEDULE=epochs:200,memory_shards:4,epochs_per_shard:1,shard_cycles:50
SPORTS_CHECKPOINT=${SPORTS_CHECKPOINT}
MOT20_DATASET=${MOT20_DATASET}
MOT20_RUN=${MOT20_RUN}
MOT20_SCHEDULE=epochs:100,memory_shards:4,epochs_per_shard:1,shard_cycles:25
MOT20_CHECKPOINT=${MOT20_CHECKPOINT}
MOT17_DATASET=${MOT17_DATASET}
MOT17_RUN=${MOT17_RUN}
MOT17_INIT_CHECKPOINT=${MOT20_CHECKPOINT}
MOT17_SCHEDULE=epochs:50,memory_shards:none,epochs_per_shard:none,shard_cycles:none,lr:1e-5
MOT17_CHECKPOINT=${MOT17_CHECKPOINT}
EVALUATION=SportsMOT:val,final,raw,no_post+test,final,post;MOT20:all,final,raw,no_post+test,final,post;MOT17:all,final,raw,no_post+test,final,post
EOF
  exit 0
fi

if [[ -e "${PIPELINE_ROOT}/running" ]]; then
  echo "Refusing an already-running pipeline: ${PIPELINE_ROOT}" >&2
  exit 2
fi
if [[ -e "${PIPELINE_ROOT}/completed" ]]; then
  echo "Pipeline is already complete: ${PIPELINE_ROOT}" >&2
  exit 2
fi

mkdir -p "${PIPELINE_ROOT}/logs"

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

require_file "${SPORTS_DATASET}/metadata.json"
require_file "${MOT20_DATASET}/metadata.json"
require_file "${MOT17_DATASET}/metadata.json"
require_file "${ROOT}/3. Tracker/trackeval/seqmap/sportsmot/val.txt"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" - "${SPORTS_DATASET}" "${MOT20_DATASET}" "${MOT17_DATASET}" <<'PY'
import json
import sys

expected = {
    sys.argv[1]: ("SportsMOT", "compact_memmap_v1"),
    sys.argv[2]: ("MOT20", "compact_memmap_v1"),
    sys.argv[3]: ("MOT17", "jsonl_v1"),
}
for path, (dataset, index_format) in expected.items():
    with open(f"{path}/metadata.json") as handle:
        metadata = json.load(handle)
    if metadata.get("dataset") != dataset:
        raise SystemExit(
            f"unexpected dataset in {path}: {metadata.get('dataset')!r} != {dataset!r}"
        )
    if int(metadata.get("context_size", -1)) != 6:
        raise SystemExit(f"sqrt-size run requires context_size=6: {path}")
    if metadata.get("index_format", "jsonl_v1") != index_format:
        raise SystemExit(
            f"unexpected index format in {path}: "
            f"{metadata.get('index_format')!r} != {index_format!r}"
        )
    if int(metadata.get("num_train_samples", 0)) <= 0:
        raise SystemExit(f"dataset has no training samples: {path}")
    print(
        path,
        "dataset=",
        metadata["dataset"],
        "samples=",
        metadata["num_train_samples"],
        "context_size=",
        metadata["context_size"],
        "index_format=",
        metadata.get("index_format", "jsonl_v1"),
    )
PY

"${PY}" - <<'PY'
import sys
import torch

from agentguard.models.iwg_rg_cma import model_contract

contract = model_contract(correction_bound=0.05, context_size=6)
if contract["model_schema"] != "agentguard_iwg_rg_cma_v1":
    raise SystemExit(f"unexpected baseline model contract: {contract['model_schema']!r}")
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
    raise SystemExit("CUDA is required for formal IWG RG-CMA training")
PY

git -C "${ROOT}" rev-parse HEAD > "${PIPELINE_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PIPELINE_ROOT}/git_status_porcelain.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/cli/__init__.py" \
  "${SPORTS_DATASET}/metadata.json" \
  "${MOT20_DATASET}/metadata.json" \
  "${MOT17_DATASET}/metadata.json" \
  > "${PIPELINE_ROOT}/formal_artifacts.sha256"

if ((PREFLIGHT)); then
  echo "Preflight passed; no training or evaluation was started."
  exit 0
fi

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

train_fresh_stage() {
  local stage_name="$1"
  local run_root="$2"
  local dataset_dir="$3"
  local epochs="$4"
  local shard_cycles="$5"
  local expected_checkpoint="$6"
  local checkpoint_dir="${run_root}/checkpoints"
  local log_dir="${run_root}/logs"
  local provenance_dir="${run_root}/provenance"
  local completed_marker="${run_root}/training_completed"

  mkdir -p "${checkpoint_dir}" "${log_dir}" "${provenance_dir}"
  if [[ -e "${completed_marker}" ]]; then
    require_file "${expected_checkpoint}"
    stage "TRAIN ${stage_name}: already complete, reusing checkpoint"
  else
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
      --lr 0.0001
      --weight-decay 0.0001
      --warmup-epochs 1
      --grad-clip 1.0
      --seed 42
      --correction-bound 0.05
      --context-size 6
      --sequence-sampling sqrt-size
      --memory-shards 4
      --epochs-per-shard 1
      --shard-cycles "${shard_cycles}"
    )
    printf '%q ' "${command[@]}" > "${provenance_dir}/train.command.txt"
    printf '\n' >> "${provenance_dir}/train.command.txt"
    printf '%s\n' \
      "correction_bound=0.05" \
      "context_size=6" \
      "sequence_sampling=sqrt-size" \
      "sampling_with_replacement=true" \
      "epochs=${epochs}" \
      "memory_shards=4" \
      "epochs_per_shard=1" \
      "shard_cycles=${shard_cycles}" \
      "fixed_shard_order=1,2,3,4" \
      > "${provenance_dir}/schedule.txt"
    stage "TRAIN ${stage_name}: sqrt-size, epochs=${epochs}, 4 shards x 1 x ${shard_cycles}"
    "${command[@]}" 2>&1 | tee "${log_dir}/train.log"
    require_file "${expected_checkpoint}"
    touch "${completed_marker}"
  fi

  "${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
    --checkpoint "${expected_checkpoint}" \
    --dataset-dir "${dataset_dir}" \
    --device cpu --max-batches 8 \
    --output "${run_root}/checkpoint_validation.json" \
    2>&1 | tee "${log_dir}/checkpoint_validation.log"
  sha256sum "${dataset_dir}/metadata.json" "${expected_checkpoint}" \
    > "${provenance_dir}/formal_artifacts.sha256"
}

train_mot17_stage() {
  local stage_name="MOT17 from MOT20 e100"
  local run_root="${MOT17_RUN}"
  local checkpoint_dir="${run_root}/checkpoints"
  local log_dir="${run_root}/logs"
  local provenance_dir="${run_root}/provenance"
  local completed_marker="${run_root}/training_completed"

  mkdir -p "${checkpoint_dir}" "${log_dir}" "${provenance_dir}"
  if [[ -e "${completed_marker}" ]]; then
    require_file "${MOT17_CHECKPOINT}"
    stage "TRAIN ${stage_name}: already complete, reusing checkpoint"
  else
    if [[ -e "${checkpoint_dir}/iwg_rg_cma_last.pt" || -e "${MOT17_CHECKPOINT}" ]]; then
      echo "Incomplete existing training run requires manual audit: ${run_root}" >&2
      exit 2
    fi
    require_file "${MOT20_CHECKPOINT}"
    local command=(
      "${PY}" -u -m agentguard.cli train_iwg_rg_cma
      --dataset-dir "${MOT17_DATASET}"
      --checkpoint-dir "${checkpoint_dir}"
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
      --sequence-sampling sqrt-size
      --init-checkpoint "${MOT20_CHECKPOINT}"
    )
    printf '%q ' "${command[@]}" > "${provenance_dir}/train.command.txt"
    printf '\n' >> "${provenance_dir}/train.command.txt"
    printf '%s\n' \
      "correction_bound=0.05" \
      "context_size=6" \
      "sequence_sampling=sqrt-size" \
      "sampling_with_replacement=true" \
      "epochs=50" \
      "memory_shards=none" \
      "init_checkpoint=${MOT20_CHECKPOINT}" \
      > "${provenance_dir}/schedule.txt"
    sha256sum "${MOT20_CHECKPOINT}" > "${provenance_dir}/initial_checkpoint.sha256"
    stage "TRAIN ${stage_name}: sqrt-size, 50 epochs, no shard arguments"
    "${command[@]}" 2>&1 | tee "${log_dir}/train.log"
    require_file "${MOT17_CHECKPOINT}"
    touch "${completed_marker}"
  fi

  "${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
    --checkpoint "${MOT17_CHECKPOINT}" \
    --dataset-dir "${MOT17_DATASET}" \
    --device cpu --max-batches 8 \
    --output "${run_root}/checkpoint_validation.json" \
    2>&1 | tee "${log_dir}/checkpoint_validation.log"
  sha256sum "${MOT17_DATASET}/metadata.json" "${MOT17_CHECKPOINT}" \
    > "${provenance_dir}/formal_artifacts.sha256"
}

run_validation() {
  local stage_name="$1"
  local run_root="$2"
  local dataset="$3"
  local mode="$4"
  local checkpoint="$5"
  local log_name="$6"
  local suffix="$7"
  shift 7
  local sequences=("$@")
  local marker="${run_root}/validation_${log_name}.completed"
  local log_path="${run_root}/logs/${log_name}.log"

  require_file "${checkpoint}"
  if [[ -e "${marker}" ]]; then
    stage "VALIDATE ${stage_name}: already complete"
    return
  fi
  local command=(
    "${PY}" -u run.py
    --dataset "${dataset}"
    --mode "${mode}"
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
  stage "VALIDATE ${stage_name}: ${dataset} ${mode}, final raw, no post"
  (
    cd "${TRACKER_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
  require_file "${log_path}"
  touch "${marker}"
  awk '$1 == "HOTA" && $2 == "MOTA" { getline; line=$0 } END { if (line != "") print "'"${stage_name}"'\t" line }' "${log_path}" \
    | tee -a "${PIPELINE_ROOT}/metrics_summary.txt"
}

run_test_post() {
  local stage_name="$1"
  local run_root="$2"
  local dataset="$3"
  local checkpoint="$4"
  local log_name="$5"
  local suffix="$6"
  local marker="${run_root}/test_${log_name}.completed"
  local log_path="${run_root}/logs/${log_name}.log"

  require_file "${checkpoint}"
  if [[ -e "${marker}" ]]; then
    stage "TEST+POST ${stage_name}: already complete"
    return
  fi
  local command=(
    "${PY}" -u run.py
    --dataset "${dataset}"
    --mode test
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output final
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --use_post
    --skip-eval
  )
  printf '%q ' "${command[@]}" > "${run_root}/provenance/${log_name}.command.txt"
  printf '\n' >> "${run_root}/provenance/${log_name}.command.txt"
  stage "TEST+POST ${stage_name}: ${dataset} test, final output"
  (
    cd "${TRACKER_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
  require_file "${log_path}"
  touch "${marker}"
}

stage "START sqrt-size serial pipeline"

train_fresh_stage \
  "SportsMOT trainval" \
  "${SPORTS_RUN}" \
  "${SPORTS_DATASET}" \
  200 50 "${SPORTS_CHECKPOINT}"
run_validation \
  "SportsMOT trainval" "${SPORTS_RUN}" SportsMOT val \
  "${SPORTS_CHECKPOINT}" sportsmot_sqrt_size_e200_final_val \
  iwg_rg_cma_sqrt_size_sportsmot_e200_final_val \
  "${SPORTS_SEQUENCES[@]}"
run_test_post \
  "SportsMOT" "${SPORTS_RUN}" SportsMOT \
  "${SPORTS_CHECKPOINT}" sportsmot_sqrt_size_e200_final_test_post \
  iwg_rg_cma_sqrt_size_sportsmot_e200_final_test_post

train_fresh_stage \
  "MOT20" \
  "${MOT20_RUN}" \
  "${MOT20_DATASET}" \
  100 25 "${MOT20_CHECKPOINT}"
run_validation \
  "MOT20" "${MOT20_RUN}" MOT20 all \
  "${MOT20_CHECKPOINT}" mot20_sqrt_size_e100_final_all \
  iwg_rg_cma_sqrt_size_mot20_e100_final_all \
  "${MOT20_SEQUENCES[@]}"
run_test_post \
  "MOT20" "${MOT20_RUN}" MOT20 \
  "${MOT20_CHECKPOINT}" mot20_sqrt_size_e100_final_test_post \
  iwg_rg_cma_sqrt_size_mot20_e100_final_test_post

train_mot17_stage
run_validation \
  "MOT17 from MOT20 e100" "${MOT17_RUN}" MOT17 all \
  "${MOT17_CHECKPOINT}" mot17_sqrt_size_e050_final_all \
  iwg_rg_cma_sqrt_size_mot20e100_to_mot17_e050_final_all \
  "${MOT17_SEQUENCES[@]}"
run_test_post \
  "MOT17 from MOT20 e100" "${MOT17_RUN}" MOT17 \
  "${MOT17_CHECKPOINT}" mot17_sqrt_size_e050_final_test_post \
  iwg_rg_cma_sqrt_size_mot20e100_to_mot17_e050_final_test_post

stage "DONE sqrt-size serial pipeline"
pipeline_complete=1
