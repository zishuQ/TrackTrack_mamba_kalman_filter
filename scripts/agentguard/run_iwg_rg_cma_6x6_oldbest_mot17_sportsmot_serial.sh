#!/usr/bin/env bash
set -euo pipefail

# 6x6 architecture under the old best training contract:
#   1. MOT17 warm-start from the compatible 6x6 MOT20 checkpoint, 50 epochs
#   2. SportsMOT trainval from scratch, 200 epochs and 4 memory shards
#
# The labels and compact/jsonl datasets are reused from public outputs. This
# script does not rebuild labels or place data under an experiment directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
PIPELINE_NAME="${PIPELINE_NAME:-iwg_rg_cma_6x6_oldbest_mot17_sportsmot_serial_20260724}"
PIPELINE_ROOT="${ROOT}/outputs/agentguard/overnight/${PIPELINE_NAME}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DATASET_ROOT="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"

MOT17_DATASET="${DATASET_ROOT}/MOT17/nsa_all_v3_jsonl_native_log"
SPORTS_DATASET="${DATASET_ROOT}/SportsMOT/nsa_trainval_v3_compact"
MOT20_INIT_CHECKPOINT="${EXPERIMENT_ROOT}/iwg_rg_cma_mot20_clean_6x6_cma_sqrt_seed42_bs1024_shard4x1_100e_20260723/checkpoints/iwg_rg_cma_epoch100.pt"

MOT17_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_6x6_oldbest_mot20e100_to_mot17_bound005_sampleprop_seed42_bs1024_finetune50"
SPORTS_RUN="${EXPERIMENT_ROOT}/iwg_rg_cma_6x6_oldbest_sportsmot_trainval_bound005_sampleprop_seed42_bs1024_shard4x1_200e"
MOT17_CHECKPOINT="${MOT17_RUN}/checkpoints/iwg_rg_cma_epoch050.pt"
SPORTS_CHECKPOINT="${SPORTS_RUN}/checkpoints/iwg_rg_cma_epoch200.pt"

ARCHITECTURE_VARIANT="legacy-clean-6x6-cma"
MOT17_SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)
SPORTS_SEQMAP="${TRACKER_ROOT}/trackeval/seqmap/sportsmot/val.txt"

DRY_RUN=0
PREFLIGHT=0
while (($#)); do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_6x6_oldbest_mot17_sportsmot_serial.sh

Stages:
  1. MOT17 all/FRCNN, warm-start from the compatible 6x6 MOT20 epoch100,
     50 epochs, lr=1e-5, one full-data phase.
  2. SportsMOT trainval, fresh 6x6 model, 200 epochs,
     4 shards x 1 epoch x 50 cycles.

Both stages use bound=0.05, context=6, sample-proportional sampling,
batch=1024, seed=42, AdamW weight decay=1e-4, warmup=1, grad clip=1.0,
and the default residual/revision losses. After each stage the checkpoint is
validated and final raw tracker metrics are evaluated; no post-processing is
run.

Options:
  --dry-run    Print paths and the fixed experiment contract.
  --preflight  Check datasets, source checkpoint, model contracts, and CUDA.
EOF
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
      exit 2
      ;;
  esac
done

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

require_absent() {
  if [[ -e "$1" ]]; then
    echo "Refusing to overwrite existing path: $1" >&2
    exit 2
  fi
}

require_file "${MOT17_DATASET}/metadata.json"
require_file "${SPORTS_DATASET}/metadata.json"
require_file "${MOT20_INIT_CHECKPOINT}"
require_file "${TRACKER_ROOT}/run.py"
require_file "${SPORTS_SEQMAP}"

mapfile -t SPORTS_SEQUENCES < <(sed '/^name$/d;/^$/d' "${SPORTS_SEQMAP}")
if [[ "${#SPORTS_SEQUENCES[@]}" -ne 45 ]]; then
  echo "Expected 45 SportsMOT validation sequences, got ${#SPORTS_SEQUENCES[@]}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if ((DRY_RUN)); then
  cat <<EOF
PIPELINE_ROOT=${PIPELINE_ROOT}
MOT17_DATASET=${MOT17_DATASET}
MOT17_INIT_CHECKPOINT=${MOT20_INIT_CHECKPOINT}
MOT17_RUN=${MOT17_RUN}
MOT17_CHECKPOINT=${MOT17_CHECKPOINT}
MOT17_SCHEDULE=epochs:50,memory_shards:default-1,epochs_per_shard:default,shard_cycles:default,lr:0.00001
SPORTS_DATASET=${SPORTS_DATASET}
SPORTS_RUN=${SPORTS_RUN}
SPORTS_CHECKPOINT=${SPORTS_CHECKPOINT}
SPORTS_SCHEDULE=epochs:200,memory_shards:4,epochs_per_shard:1,shard_cycles:50,lr:0.0001
MODEL=architecture:${ARCHITECTURE_VARIANT},bound:0.05,context:6,reliability:full
COMMON=batch:1024,seed:42,weight_decay:0.0001,warmup:1,grad_clip:1.0,sequence_sampling:sample-proportional
LOSS=residual_beta:1.0,residual_weight:0.5,revision_weight:0.01,hard_example_gain:0.0,no_harm_weight:0.0
EVALUATION=MOT17:all/final/raw;SportsMOT:val/final/raw;post:none
EOF
  exit 0
fi

"${PY}" - "${MOT17_DATASET}" "${SPORTS_DATASET}" "${MOT20_INIT_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

from agentguard.models.iwg_rg_cma import model_contract
from agentguard.training.train_iwg_rg_cma import (
    _validate_formal_config,
    validate_iwg_rg_cma_checkpoint_contract,
)

mot17_dir, sports_dir, source_path = sys.argv[1:]

def check_dataset(path, dataset, split, index_format, samples):
    metadata = json.loads((Path(path) / "metadata.json").read_text())
    actual_format = metadata.get("index_format", "jsonl_v1")
    expected = {
        "dataset": dataset,
        "split": split,
        "context_size": 6,
        "index_format": index_format,
        "num_train_samples": samples,
    }
    actual = {
        "dataset": metadata.get("dataset"),
        "split": metadata.get("split"),
        "context_size": int(metadata.get("context_size", -1)),
        "index_format": actual_format,
        "num_train_samples": int(metadata.get("num_train_samples", -1)),
    }
    if actual != expected:
        raise SystemExit(f"dataset contract mismatch for {path}: {actual!r} != {expected!r}")
    print(path, "OK", actual, "sha256=", metadata["dataset_sha256"])

check_dataset(mot17_dir, "MOT17", "all", "jsonl_v1", 94052)
check_dataset(sports_dir, "SportsMOT", "trainval", "compact_memmap_v1", 577972)

variant = "legacy-clean-6x6-cma"
contract = model_contract(correction_bound=0.05, context_size=6, architecture_variant=variant)
if contract["model_schema"] != "agentguard_iwg_rg_cma_legacy_clean_6x6_cma_v1":
    raise SystemExit(f"unexpected 6x6 model schema: {contract['model_schema']!r}")

common = {
    "batch_size": 1024,
    "num_workers": 4,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "grad_clip": 1.0,
    "seed": 42,
    "correction_bound": 0.05,
    "context_size": 6,
    "reliability_mode": "full",
    "architecture_variant": variant,
    "residual_beta": 1.0,
    "residual_weight": 0.5,
    "revision_weight": 0.01,
    "hard_example_gain": 0.0,
    "no_harm_weight": 0.0,
    "sequence_sampling": "sample-proportional",
    "amp": False,
}
mot17_config = {
    **common,
    "epochs": 50,
    "lr": 1e-5,
    "base_lr": 1e-5,
    "cma_lr": 1e-5,
    "init_checkpoint": source_path,
    "memory_shards": 1,
    "epochs_per_shard": 0,
    "shard_cycles": 1,
}
sports_config = {
    **common,
    "epochs": 200,
    "lr": 1e-4,
    "base_lr": 1e-4,
    "cma_lr": 1e-4,
    "init_checkpoint": "",
    "memory_shards": 4,
    "epochs_per_shard": 1,
    "shard_cycles": 50,
}
_validate_formal_config(mot17_config)
_validate_formal_config(sports_config)

checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
validate_iwg_rg_cma_checkpoint_contract(checkpoint)
expected_checkpoint = {
    "dataset": "MOT20",
    "context_size": 6,
    "correction_bound": 0.05,
    "architecture_variant": variant,
    "model_schema": contract["model_schema"],
}
for key, expected in expected_checkpoint.items():
    actual = checkpoint.get(key)
    if key == "context_size":
        actual = int(actual)
    if key == "correction_bound":
        actual = float(actual)
    if actual != expected:
        raise SystemExit(
            f"source checkpoint contract mismatch {key}: {actual!r} != {expected!r}"
        )
print("source checkpoint OK:", source_path, "epoch=", checkpoint.get("epoch"))
PY

if ((PREFLIGHT)); then
  "${PY}" -c 'import sys, torch; print("python=", sys.version.split()[0], "torch=", torch.__version__, "cuda=", torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)'
  echo "Preflight passed; no training or evaluation was started."
  exit 0
fi

require_absent "${PIPELINE_ROOT}"
require_absent "${MOT17_RUN}"
require_absent "${SPORTS_RUN}"
mkdir -p "${PIPELINE_ROOT}/logs"

"${PY}" -c 'import sys, torch; print(f"python={sys.version}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); raise SystemExit(0 if torch.cuda.is_available() else 1)' \
  > "${PIPELINE_ROOT}/environment.txt" 2>&1
git -C "${ROOT}" rev-parse HEAD > "${PIPELINE_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PIPELINE_ROOT}/git_status_porcelain.txt"
sha256sum "${MOT17_DATASET}/metadata.json" "${SPORTS_DATASET}/metadata.json" \
  "${MOT20_INIT_CHECKPOINT}" "${BASH_SOURCE[0]}" \
  > "${PIPELINE_ROOT}/formal_inputs.sha256"
printf '%s\n' \
  "architecture_variant=${ARCHITECTURE_VARIANT}" \
  "correction_bound=0.05" \
  "context_size=6" \
  "sequence_sampling=sample-proportional" \
  "mot17_lr=1e-5" \
  "sportsmot_lr=1e-4" \
  "mot17_memory_shards=default-1" \
  "sportsmot_memory_shards=4" \
  > "${PIPELINE_ROOT}/contract.txt"

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

stage() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" \
    | tee -a "${PIPELINE_ROOT}/stages.log"
}

train_stage() {
  local stage_name="$1"
  local run_root="$2"
  local dataset_dir="$3"
  local epochs="$4"
  local lr="$5"
  local expected_checkpoint="$6"
  local init_checkpoint="${7:-}"
  local memory_shards="${8:-}"
  local epochs_per_shard="${9:-}"
  local shard_cycles="${10:-}"
  local checkpoint_dir="${run_root}/checkpoints"
  local log_dir="${run_root}/logs"
  local provenance_dir="${run_root}/provenance"

  mkdir -p "${checkpoint_dir}" "${log_dir}" "${provenance_dir}"
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
    --correction-bound 0.05
    --context-size 6
    --reliability-mode full
    --architecture-variant "${ARCHITECTURE_VARIANT}"
    --residual-beta 1.0
    --residual-weight 0.5
    --revision-weight 0.01
    --hard-example-gain 0.0
    --no-harm-weight 0.0
    --sequence-sampling sample-proportional
  )
  if [[ -n "${init_checkpoint}" ]]; then
    command+=(--init-checkpoint "${init_checkpoint}")
  else
    command+=(
      --memory-shards "${memory_shards}"
      --epochs-per-shard "${epochs_per_shard}"
      --shard-cycles "${shard_cycles}"
    )
  fi
  printf '%q ' "${command[@]}" > "${provenance_dir}/train.command.txt"
  printf '\n' >> "${provenance_dir}/train.command.txt"
  if [[ -n "${init_checkpoint}" ]]; then
    sha256sum "${init_checkpoint}" > "${provenance_dir}/initial_checkpoint.sha256"
    stage "TRAIN ${stage_name}: epochs=${epochs}, lr=${lr}, one full-data phase"
  else
    stage "TRAIN ${stage_name}: epochs=${epochs}, 4 shards x 1 x ${shard_cycles}"
  fi
  "${command[@]}" 2>&1 | tee "${log_dir}/train.log"
  require_file "${expected_checkpoint}"

  "${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
    --checkpoint "${expected_checkpoint}" \
    --dataset-dir "${dataset_dir}" \
    --device cpu \
    --max-batches 8 \
    --output "${run_root}/checkpoint_validation.json" \
    2>&1 | tee "${log_dir}/checkpoint_validation.log"
  sha256sum "${dataset_dir}/metadata.json" "${expected_checkpoint}" \
    > "${provenance_dir}/formal_artifacts.sha256"
  touch "${run_root}/training_completed"
}

eval_final_raw() {
  local stage_name="$1"
  local run_root="$2"
  local dataset="$3"
  local mode="$4"
  local checkpoint="$5"
  local log_name="$6"
  local suffix="$7"
  shift 7
  local sequences=("$@")
  local log_path="${run_root}/logs/${log_name}.log"
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
    --iwg-rg-cma-alpha 1.0
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --output_dir "${TRACKER_OUTPUT_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${run_root}/resource_${log_name}.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${run_root}/provenance/${log_name}.command.txt"
  printf '\n' >> "${run_root}/provenance/${log_name}.command.txt"
  stage "EVAL ${stage_name}: ${dataset} ${mode}, final raw, alpha=1"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${log_path}"
  local metric_line
  metric_line="$(awk '$1 == "HOTA" && $2 == "MOTA" { getline; line=$0 } END { print line }' "${log_path}")"
  if [[ -n "${metric_line}" ]]; then
    printf '%s\t%s\n' "${stage_name}" "${metric_line}" \
      | tee -a "${PIPELINE_ROOT}/metrics_summary.txt"
  else
    echo "${stage_name}: HOTA summary was not found; inspect ${log_path}" \
      | tee -a "${PIPELINE_ROOT}/metrics_summary.txt"
  fi
}

stage "START 6x6 old-best MOT17 migration + SportsMOT serial pipeline"

# MOT17 intentionally omits shard arguments. The trainer defaults to one
# full-data phase; the formal warm-start contract resolves it to 1 x 1 x 50.
train_stage \
  "MOT17 from compatible 6x6 MOT20 epoch100" \
  "${MOT17_RUN}" \
  "${MOT17_DATASET}" \
  50 0.00001 "${MOT17_CHECKPOINT}" \
  "${MOT20_INIT_CHECKPOINT}"
eval_final_raw \
  "MOT17 6x6 migration epoch050" \
  "${MOT17_RUN}" MOT17 all "${MOT17_CHECKPOINT}" \
  mot17_6x6_final_all_raw \
  iwg_rg_cma_6x6_oldbest_mot17_e050_final_all \
  "${MOT17_SEQUENCES[@]}"

train_stage \
  "SportsMOT 6x6 from scratch" \
  "${SPORTS_RUN}" \
  "${SPORTS_DATASET}" \
  200 0.0001 "${SPORTS_CHECKPOINT}" \
  "" 4 1 50
eval_final_raw \
  "SportsMOT 6x6 epoch200" \
  "${SPORTS_RUN}" SportsMOT val "${SPORTS_CHECKPOINT}" \
  sportsmot_6x6_final_val_raw \
  iwg_rg_cma_6x6_oldbest_sportsmot_e200_final_val \
  "${SPORTS_SEQUENCES[@]}"

stage "DONE 6x6 old-best MOT17 migration + SportsMOT serial pipeline"
pipeline_complete=1
