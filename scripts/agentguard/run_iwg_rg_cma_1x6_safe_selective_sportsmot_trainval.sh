#!/usr/bin/env bash
set -euo pipefail

# SportsMOT 1x6 safe-selective residual experiment.
# Reuses the public compact dataset; never rebuilds labels or dataset files.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_1x6_safe_selective_sportsmot_trainval_bound005_sampleprop_seed42_bs1024_shard4x1_200e}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
DATASET_DIR="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/SportsMOT/nsa_trainval_v3_compact"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_epoch200.pt"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
SEQMAP="${TRACKER_ROOT}/trackeval/seqmap/sportsmot/val.txt"

DRY_RUN=0
PREFLIGHT=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_1x6_safe_selective_sportsmot_trainval.sh

Runs 1x6 SportsMOT trainval training for 200 epochs, validates the checkpoint,
then evaluates base/final on SportsMOT val raw. It does not run test, post, or
AFLink, and does not rebuild labels or datasets.

Options:
  --dry-run   Print paths and settings without training or validation.
  --preflight Check the existing dataset and target derivation, then exit.
EOF
}

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
RUN_ROOT=${RUN_ROOT}
DATASET_DIR=${DATASET_DIR}
CHECKPOINT=${CHECKPOINT}
MODEL=legacy-clean-cross-modal,context_size:6,correction_bound:0.05,reliability:full
TARGET=safe-selective,confidence_ramp:0.50-0.75,decisiveness_ramp:0.20-0.50,pure_cma_residual:true
LOSS=residual_weight:0.5,revision_weight:0.05(exact-abstain-only),no_harm_weight:0.1(residual-space)
SCHEDULE=nominal_epochs:200,memory_shards:4,epochs_per_shard:1,shard_cycles:50,effective_full_epochs:50
VALIDATION=SportsMOT/val,base+final,raw only
LABEL_REBUILD=none
EOF
  exit 0
fi

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

require_file "${DATASET_DIR}/metadata.json"
require_file "${SEQMAP}"
require_file "${DETECTION_CACHE_ROOT}/SportsMOT/val/manifest.json"
require_absent "${RUN_ROOT}"

mapfile -t SEQUENCES < <(sed '/^name$/d;/^$/d' "${SEQMAP}")
if [[ "${#SEQUENCES[@]}" -ne 45 ]]; then
  echo "Expected 45 SportsMOT validation sequences, got ${#SEQUENCES[@]}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" - "${DATASET_DIR}" <<'PY'
import json
import sys

from agentguard.datasets.iwg_rg_cma_dataset import StreamingIWGRGCMADataset

dataset_dir = sys.argv[1]
metadata = json.load(open(f"{dataset_dir}/metadata.json"))
expected = {
    "dataset": "SportsMOT",
    "split": "trainval",
    "index_format": "compact_memmap_v1",
    "context_size": 6,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"unexpected metadata {key}: {metadata.get(key)!r} != {value!r}")
if not {"risk_target", "cue_target"}.issubset(
    set(metadata.get("compact_index_array_files", {}))
):
    raise SystemExit("existing compact dataset cannot derive oracle/confidence targets")

dataset = StreamingIWGRGCMADataset(dataset_dir, max_samples=1)
try:
    sample = dataset[0]
    for key in ("oracle_gate_target", "gate_confidence"):
        if key not in sample:
            raise SystemExit(f"missing derived sample field: {key}")
    print(
        "dataset=SportsMOT",
        "samples=", metadata["num_train_samples"],
        "context_size=", metadata["context_size"],
        "target_source=existing_safe+risk+cue_fields",
    )
finally:
    dataset.close()
PY

"${PY}" - <<'PY'
import sys
import torch
from agentguard.models.iwg_rg_cma import model_contract

print(
    "python=", sys.version.split()[0],
    "torch=", torch.__version__,
    "cuda=", torch.cuda.is_available(),
    "model_schema=", model_contract(
        correction_bound=0.05,
        context_size=6,
        architecture_variant="legacy-clean-cross-modal",
    )["model_schema"],
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this formal training run")
PY

if ((PREFLIGHT)); then
  echo "Preflight passed; no training or validation was started."
  exit 0
fi

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/cli/__init__.py" \
  "${DATASET_DIR}/metadata.json" \
  > "${PROVENANCE_DIR}/formal_inputs.sha256"

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
  --epochs 200
  --batch-size 1024
  --num-workers 4
  --lr 0.0001
  --weight-decay 0.0001
  --warmup-epochs 1
  --grad-clip 1.0
  --seed 42
  --correction-bound 0.05
  --context-size 6
  --reliability-mode full
  --architecture-variant legacy-clean-cross-modal
  --residual-beta 1.0
  --residual-weight 0.5
  --revision-weight 0.05
  --hard-example-gain 0.0
  --no-harm-weight 0.1
  --residual-target-mode safe-selective
  --sequence-sampling sample-proportional
  --memory-shards 4
  --epochs-per-shard 1
  --shard-cycles 50
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
echo "$(date '+%Y-%m-%d %H:%M:%S') | TRAIN SportsMOT safe-selective" \
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

run_val_case() {
  local output_mode="$1"
  local suffix="$2"
  local log_name="$3"
  local command=(
    "${PY}" -u run.py
    --dataset SportsMOT
    --mode val
    --sequences "${SEQUENCES[@]}"
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${CHECKPOINT}"
    --iwg-rg-cma-output "${output_mode}"
    --iwg-rg-cma-alpha 1.0
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --output_dir "${TRACKER_OUTPUT_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${RUN_ROOT}/resource_${log_name}.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/${log_name}.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/${log_name}.command.txt"
  echo "$(date '+%Y-%m-%d %H:%M:%S') | VAL raw ${output_mode}" \
    | tee -a "${RUN_ROOT}/stages.log"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${LOG_DIR}/${log_name}.log"
  metric_line="$(awk '$1 == "HOTA" && $2 == "MOTA" { getline; line=$0 } END { print line }' "${LOG_DIR}/${log_name}.log")"
  if [[ -z "${metric_line}" ]]; then
    echo "Could not find TrackEval summary in ${LOG_DIR}/${log_name}.log" >&2
    exit 2
  fi
  read -r hota mota idf1 deta assa <<< "${metric_line}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${output_mode}" "${hota}" "${mota}" "${idf1}" "${deta}" "${assa}" \
    | tee -a "${RUN_ROOT}/metrics_summary.tsv"
}

printf 'output\tHOTA\tMOTA\tIDF1\tDetA\tAssA\n' \
  > "${RUN_ROOT}/metrics_summary.tsv"
run_val_case base "${RUN_NAME}_e200_base_val_raw" sportsmot_base_val_raw
run_val_case final "${RUN_NAME}_e200_final_val_raw" sportsmot_final_val_raw

sha256sum "${RUN_ROOT}/metrics_summary.tsv" \
  > "${RUN_ROOT}/metrics_summary.sha256"
complete=1
echo "$(date '+%Y-%m-%d %H:%M:%S') | DONE; test/post/AFLink were not run" \
  | tee -a "${RUN_ROOT}/stages.log"
