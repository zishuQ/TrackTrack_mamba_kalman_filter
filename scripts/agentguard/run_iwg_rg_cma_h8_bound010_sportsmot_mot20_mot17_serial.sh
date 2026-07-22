#!/usr/bin/env bash
set -euo pipefail

# Serial bound=0.10 pipeline:
#   1. SportsMOT trainval, future horizon=8 + context=8, 200e, 4-shard schedule
#   2. MOT20 all, context=6, 100e, 4-shard schedule
#   3. MOT20 e100 weights -> MOT17 context=6, 50e warm-start
#
# The formal trainer currently requires warm-start runs to use one full-data
# phase, so the MOT17 stage intentionally follows the existing overnight
# script with memory_shards=1. Changing that would require changing the
# trainer contract rather than just composing a launch script.
#
# Every tracker evaluation uses final gates, raw output, and no post-processing.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
PIPELINE_NAME="${PIPELINE_NAME:-iwg_rg_cma_bound010_sportsmot_future8_context8_mot20e100_mot17e050_serial_20260718}"
PIPELINE_ROOT="${ROOT}/outputs/agentguard/overnight/${PIPELINE_NAME}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DATASET_ROOT="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"

SPORTS_DATASET="${DATASET_ROOT}/SportsMOT/nsa_trainval_future8_context8_v3_compact"
MOT20_DATASET="${DATASET_ROOT}/MOT20/nsa_all_v3_compact"
MOT17_DATASET="${DATASET_ROOT}/MOT17/nsa_all_v3_jsonl"

SPORTS_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v4_sportsmot_trainval_future8_context8_bound010_seed42_bs1024_shard4x1_200e_serial_20260718"
MOT20_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v2_mot20_all_bound010_seed42_bs1024_shard4x1_100e_serial_20260718"
MOT17_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_v2_mot20e100_to_mot17_bound010_seed42_bs1024_finetune50_serial_20260718"

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
  bash scripts/agentguard/run_iwg_rg_cma_h8_bound010_sportsmot_mot20_mot17_serial.sh

Serial stages:
  1. SportsMOT trainval future horizon=8/context=8, 200 epochs,
     4 shards x 1 epoch x 50 cycles.
  2. MOT20 all context=6, 100 epochs, 4 shards x 1 epoch x 25 cycles.
  3. MOT20 epoch100 -> MOT17 context=6, 50 epochs, strict model-weight
     warm-start.

All tracker evaluations are final-gate, raw, and no-post. The MOT17
warm-start uses the existing formal one-phase schedule required by the trainer.

Options:
  -h, --help       Show this message and exit.
      --dry-run    Print paths and schedules without training.
      --preflight  Validate datasets and the model contract without training.

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
SPORTS_DATASET=${SPORTS_DATASET}
SPORTS_RUN=${SPORTS_RUN}
SPORTS_SCHEDULE=epochs:200,memory_shards:4,epochs_per_shard:1,shard_cycles:50
MOT20_DATASET=${MOT20_DATASET}
MOT20_RUN=${MOT20_RUN}
MOT20_SCHEDULE=epochs:100,memory_shards:4,epochs_per_shard:1,shard_cycles:25
MOT17_DATASET=${MOT17_DATASET}
MOT17_RUN=${MOT17_RUN}
MOT17_INIT_CHECKPOINT=${MOT20_CHECKPOINT}
MOT17_SCHEDULE=epochs:50,memory_shards:1,epochs_per_shard:50,shard_cycles:1,lr:1e-5
EVALUATION=SportsMOT:val,final,raw,no_post;MOT20:all,final,raw,no_post;MOT17:all,final,raw,no_post
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

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" - "${PIPELINE_ROOT}/model_contract.txt" <<'PY'
import sys

from agentguard.models.iwg_rg_cma import model_contract

expected = (
    ("SportsMOT", 0.10, 8, "agentguard_iwg_rg_cma_v4"),
    ("MOT20", 0.10, 6, "agentguard_iwg_rg_cma_v2"),
    ("MOT17", 0.10, 6, "agentguard_iwg_rg_cma_v2"),
)
lines = []
for dataset, bound, context_size, schema in expected:
    contract = model_contract(
        correction_bound=bound,
        context_size=context_size,
    )
    actual = contract["model_schema"]
    if actual != schema:
        raise SystemExit(
            f"{dataset} model contract mismatch: {actual!r} != {schema!r}"
        )
    lines.append(
        f"{dataset}: model_contract={actual} "
        f"correction_bound={bound:.2f} context_size={context_size}"
    )
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

"${PY}" - \
  "${SPORTS_DATASET}/metadata.json" 8 \
  "${MOT20_DATASET}/metadata.json" 6 \
  "${MOT17_DATASET}/metadata.json" 6 <<'PY'
import json
import sys

arguments = sys.argv[1:]
if len(arguments) % 2:
    raise SystemExit("metadata validation expects path/context pairs")
for path, expected_context in zip(arguments[::2], arguments[1::2]):
    metadata = json.loads(open(path).read())
    if path.endswith("MOT17/nsa_all_v3_jsonl/metadata.json"):
        expected = "jsonl_v1"
    else:
        expected = "compact_memmap_v1"
    actual = metadata.get("index_format", "jsonl_v1")
    if actual != expected:
        raise SystemExit(
            f"unexpected index format in {path}: "
            f"{actual!r} != {expected!r}"
        )
    context_size = int(metadata.get("context_size", -1))
    if context_size != int(expected_context):
        raise SystemExit(
            f"unexpected context_size in {path}: "
            f"{context_size!r} != {expected_context!r}"
        )
    print(path, "index_format=", actual, "context_size=", context_size)
PY

git -C "${ROOT}" rev-parse HEAD > "${PIPELINE_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PIPELINE_ROOT}/git_status_porcelain.txt"
sha256sum \
  "${SPORTS_DATASET}/metadata.json" \
  "${MOT20_DATASET}/metadata.json" \
  "${MOT17_DATASET}/metadata.json" \
  > "${PIPELINE_ROOT}/dataset_metadata.sha256"

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

train_stage() {
  local stage_name="$1"
  local run_root="$2"
  local dataset_dir="$3"
  local epochs="$4"
  local lr="$5"
  local correction_bound="$6"
  local context_size="$7"
  local memory_shards="$8"
  local epochs_per_shard="$9"
  local shard_cycles="${10}"
  local init_checkpoint="${11:-}"
  local expected_checkpoint="${12}"
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
      --lr "${lr}"
      --weight-decay 0.0001
      --warmup-epochs 1
      --grad-clip 1.0
      --seed 42
      --correction-bound "${correction_bound}"
      --context-size "${context_size}"
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
    printf '%s\n' "bound=${correction_bound}" "context_size=${context_size}" "epochs=${epochs}" \
      "memory_shards=${memory_shards}" \
      "epochs_per_shard=${epochs_per_shard}" \
      "shard_cycles=${shard_cycles}" \
      > "${run_root}/provenance/schedule.txt"
    stage "TRAIN ${stage_name}: epochs=${epochs}, shards=${memory_shards}, epochs_per_shard=${epochs_per_shard}, cycles=${shard_cycles}, lr=${lr}"
    "${command[@]}" 2>&1 | tee "${run_root}/logs/train.log"
    require_file "${expected_checkpoint}"
    touch "${completed_marker}"
  fi

  "${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
    --checkpoint "${expected_checkpoint}" \
    --dataset-dir "${dataset_dir}" \
    --device cpu --max-batches 8 \
    --output "${run_root}/checkpoint_validation.json" \
    2>&1 | tee "${run_root}/logs/checkpoint_validation.log"
  sha256sum "${dataset_dir}/metadata.json" "${expected_checkpoint}" \
    > "${run_root}/provenance/formal_artifacts.sha256"
}

eval_final_raw() {
  local stage_name="$1"
  local run_root="$2"
  local dataset="$3"
  local mode="$4"
  local checkpoint="$5"
  local suffix="$6"
  local log_name="$7"
  shift 7
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
  stage "EVAL ${stage_name}: ${dataset} ${mode} final raw (no post, no base)"
  (
    cd "${TRACKER_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
  require_file "${log_path}"
  touch "${marker}"
  awk '$1 == "HOTA" && $2 == "MOTA" { getline; line=$0 } END { if (line != "") print "'"${stage_name}"'\t" line }' "${log_path}" \
    | tee -a "${PIPELINE_ROOT}/metrics_summary.txt"
}

stage "START bound=0.10 serial pipeline"

train_stage \
  "SportsMOT h8 e200" \
  "${SPORTS_RUN}" \
  "${SPORTS_DATASET}" \
  200 0.0001 0.10 8 4 1 50 "" "${SPORTS_CHECKPOINT}"
eval_final_raw \
  "SportsMOT h8 e200" "${SPORTS_RUN}" SportsMOT val \
  "${SPORTS_CHECKPOINT}" \
  iwg_rg_cma_v4_sportsmot_future8_context8_e200_final_val \
  sportsmot_future8_context8_e200_final_val \
  "${SPORTS_SEQUENCES[@]}"

train_stage \
  "MOT20 e100" \
  "${MOT20_RUN}" \
  "${MOT20_DATASET}" \
  100 0.0001 0.10 6 4 1 25 "" "${MOT20_CHECKPOINT}"
eval_final_raw \
  "MOT20 e100" "${MOT20_RUN}" MOT20 all \
  "${MOT20_CHECKPOINT}" \
  iwg_rg_cma_bound010_mot20_e100_final_all \
  mot20_e100_final_all \
  "${MOT20_SEQUENCES[@]}"

train_stage \
  "MOT17 from MOT20 e100, finetune e050" \
  "${MOT17_RUN}" \
  "${MOT17_DATASET}" \
  50 0.00001 0.10 6 1 50 1 "${MOT20_CHECKPOINT}" "${MOT17_CHECKPOINT}"
eval_final_raw \
  "MOT17 from MOT20 e100" "${MOT17_RUN}" MOT17 all \
  "${MOT17_CHECKPOINT}" \
  iwg_rg_cma_bound010_mot20e100_mot17e050_final_all \
  mot17_mot20e100_finetune_e050_final_all \
  "${MOT17_SEQUENCES[@]}"

stage "DONE bound=0.10 serial pipeline"
pipeline_complete=1
