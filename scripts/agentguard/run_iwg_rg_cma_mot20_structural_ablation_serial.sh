#!/usr/bin/env bash
set -euo pipefail

# Controlled MOT20 structural ablations for RG-CMA. All stages reuse the same
# public compact dataset and the best 4-shard x 1-epoch x 25-cycle schedule.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT}/scripts/agentguard/run_iwg_rg_cma_mot20_structural_ablation_serial.sh"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GROUP_NAME="${GROUP_NAME:-iwg_rg_cma_mot20_structural_ablation_20260722}"
GROUP_ROOT="${ROOT}/outputs/agentguard/experiments/${GROUP_NAME}"
DATASET_DIR="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_mot20_structural_ablation_serial.sh

Stages (serial):
  A1 direct-base:
     continuous Base-IWG gate; legacy scalar63 motion and 3-token RG-CMA.
  A2 clean-cross-modal:
     A1 + pure geometry/KF motion stream + 2-token motion/appearance attention.
  A3 selective-correction:
     A2 + separate two-channel reliability scale + no-harm loss weight 1.0.

Fixed controls:
  MOT20/all, context=6, bound=0.05, seed=42, batch=1024, lr=1e-4,
  4 shards x 1 epoch x 25 cycles = 100 nominal epochs = 25 full passes,
  sample-proportional sequence sampling, fixed shard order, legacy loss settings.

Evaluation:
  checkpoint validation -> MOT20/all base raw -> MOT20/all final raw.
  No post-processing and no test split.

Options:
  --dry-run    Print the fixed experiment matrix and output paths.
  --preflight  Validate data, model contracts, CLI arguments, and CUDA only.
  --resume     Reuse completed/validated stages in the existing failed group.
EOF
}

DRY_RUN=0
PREFLIGHT=0
RESUME=0
RESUME_STAMP=""
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

STAGE_IDS=(a1_direct_base a2_clean_cross_modal a3_selective_correction)
VARIANTS=(direct-base clean-cross-modal selective-correction)
NO_HARM_WEIGHTS=(0.0 0.0 1.0)

if ((DRY_RUN)); then
  cat <<EOF
GROUP_ROOT=${GROUP_ROOT}
DATASET_DIR=${DATASET_DIR}
SCHEDULE=epochs:100,memory_shards:4,epochs_per_shard:1,shard_cycles:25,effective_full_epochs:25
COMMON=bound:0.05,context:6,seed:42,batch:1024,lr:0.0001,sequence_sampling:sample-proportional,shard_order:fixed
A1=architecture:direct-base,no_harm_weight:0.0
A2=architecture:clean-cross-modal,no_harm_weight:0.0
A3=architecture:selective-correction,no_harm_weight:1.0
EVALUATION=all:base_raw+final_raw;post:none;test:none
RESUME=${RESUME}
EOF
  exit 0
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

require_dir_absent() {
  if [[ -e "$1" ]]; then
    echo "Refusing to overwrite existing path: $1" >&2
    exit 2
  fi
}

training_stage_is_complete() {
  local summary="$1"
  local checkpoint="$2"
  local variant="$3"
  [[ -f "${summary}" && -f "${checkpoint}" ]] || return 1
  "${PY}" - "${summary}" "${checkpoint}" "${variant}" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(sys.argv[2]).resolve()
variant = sys.argv[3]
if summary.get("status") != "completed":
    raise SystemExit(1)
if int(summary.get("selected_epoch", -1)) != 100:
    raise SystemExit(1)
if float(summary.get("effective_full_epochs", -1)) != 25.0:
    raise SystemExit(1)
if int(summary.get("optimizer_steps", -1)) != 25_700:
    raise SystemExit(1)
if summary.get("architecture_variant") != variant:
    raise SystemExit(1)
if Path(summary.get("checkpoint", "")).resolve() != checkpoint.parent / "iwg_rg_cma_last.pt":
    raise SystemExit(1)
PY
}

checkpoint_validation_is_valid() {
  local validation="$1"
  local checkpoint="$2"
  local variant="$3"
  [[ -f "${validation}" && -f "${checkpoint}" ]] || return 1
  "${PY}" - "${validation}" "${checkpoint}" "${variant}" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(sys.argv[2]).resolve()
variant = sys.argv[3]
if validation.get("status") != "valid":
    raise SystemExit(1)
if int(validation.get("epoch", -1)) != 100:
    raise SystemExit(1)
if validation.get("architecture_variant") != variant:
    raise SystemExit(1)
if Path(validation.get("checkpoint", "")).resolve() != checkpoint:
    raise SystemExit(1)
PY
}

require_file "${DATASET_DIR}/metadata.json"
require_file "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/cli/__init__.py"
require_file "${TRACKER_ROOT}/run.py"

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

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
    "num_train_samples": 1_049_088,
}
for key, value in expected.items():
    actual = metadata.get(key, "jsonl_v1" if key == "index_format" else None)
    if actual != value:
        raise SystemExit(f"unexpected metadata {key}: {actual!r} != {value!r}")
print(
    "dataset=", metadata["dataset"],
    "split=", metadata["split"],
    "samples=", metadata["num_train_samples"],
    "dataset_sha256=", metadata["dataset_sha256"],
)
PY

"${PY}" - <<'PY'
import sys
import torch
from agentguard.models.iwg_rg_cma import model_contract
from agentguard.training.train_iwg_rg_cma import _validate_formal_config

variants = ("direct-base", "clean-cross-modal", "selective-correction")
for variant in variants:
    contract = model_contract(
        correction_bound=0.05,
        context_size=6,
        architecture_variant=variant,
    )
    print(variant, contract["model_schema"], contract["model_schema_sha256"])
    _validate_formal_config(
        {
            "seed": 42,
            "epochs": 100,
            "batch_size": 1024,
            "num_workers": 4,
            "lr": 1e-4,
            "base_lr": 1e-4,
            "cma_lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "grad_clip": 1.0,
            "amp": False,
            "correction_bound": 0.05,
            "context_size": 6,
            "reliability_mode": "full",
            "architecture_variant": variant,
            "residual_beta": 1.0,
            "residual_weight": 0.5,
            "revision_weight": 0.01,
            "hard_example_gain": 0.0,
            "no_harm_weight": 1.0 if variant == "selective-correction" else 0.0,
            "memory_shards": 4,
            "epochs_per_shard": 1,
            "shard_cycles": 25,
            "sequence_sampling": "sample-proportional",
        }
    )
print(
    "python=", sys.version.split()[0],
    "torch=", torch.__version__,
    "cuda=", torch.cuda.is_available(),
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the formal structural ablations")
PY

if ((PREFLIGHT)); then
  echo "Preflight passed; no experiment directory was created."
  exit 0
fi

if ((RESUME)); then
  if [[ ! -d "${GROUP_ROOT}" ]]; then
    echo "Cannot resume missing experiment group: ${GROUP_ROOT}" >&2
    exit 2
  fi
  if [[ -e "${GROUP_ROOT}/running" ]]; then
    echo "Refusing to resume an active experiment group: ${GROUP_ROOT}" >&2
    exit 2
  fi
  if [[ ! -e "${GROUP_ROOT}/failed" ]]; then
    echo "Resume requires the existing failed marker: ${GROUP_ROOT}/failed" >&2
    exit 2
  fi
  RESUME_STAMP="$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "${GROUP_ROOT}/provenance"
  for artifact in failed pipeline.exit_code pipeline.pid; do
    if [[ -e "${GROUP_ROOT}/${artifact}" ]]; then
      mv "${GROUP_ROOT}/${artifact}" \
        "${GROUP_ROOT}/provenance/${artifact}.pre_resume_${RESUME_STAMP}"
    fi
  done
else
  require_dir_absent "${GROUP_ROOT}"
  mkdir -p "${GROUP_ROOT}/provenance"
fi
echo "$$" > "${GROUP_ROOT}/pipeline.pid"
touch "${GROUP_ROOT}/running"
provenance_suffix=""
if ((RESUME)); then
  provenance_suffix=".resume_${RESUME_STAMP}"
fi
git -C "${ROOT}" rev-parse HEAD \
  > "${GROUP_ROOT}/provenance/git_commit${provenance_suffix}.txt"
git -C "${ROOT}" status --porcelain \
  > "${GROUP_ROOT}/provenance/git_status_porcelain${provenance_suffix}.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/cli/__init__.py" \
  "${DATASET_DIR}/metadata.json" \
  "${SCRIPT_PATH}" \
  > "${GROUP_ROOT}/provenance/formal_inputs${provenance_suffix}.sha256"

complete=0
finish() {
  local code=$?
  rm -f "${GROUP_ROOT}/running"
  echo "${code}" > "${GROUP_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${complete} -eq 1 ]]; then
    touch "${GROUP_ROOT}/completed"
  else
    touch "${GROUP_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

run_raw_case() {
  local checkpoint="$1"
  local stage_root="$2"
  local stage_name="$3"
  local output="$4"
  local suffix="${GROUP_NAME}_${stage_name}_${output}_raw"
  local log_name="mot20_${output}_raw"
  RAW_FOLDER="mot20_all_0.80_${suffix}_iwg_rg_cma_${output}"
  require_dir_absent "${TRACKER_OUTPUT_ROOT}/${RAW_FOLDER}"
  if ((RESUME)) && [[ -f "${stage_root}/logs/${log_name}.log" ]]; then
    mv "${stage_root}/logs/${log_name}.log" \
      "${stage_root}/logs/${log_name}.pre_resume_${RESUME_STAMP}.log"
  fi
  local command=(
    "${PY}" -u run.py
    --dataset MOT20
    --mode all
    --sequences "${SEQUENCES[@]}"
    --seed 10000
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output "${output}"
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${stage_root}/resource_${log_name}.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${stage_root}/provenance/${log_name}.command.txt"
  printf '\n' >> "${stage_root}/provenance/${log_name}.command.txt"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${stage_root}/logs/${log_name}.log"
}

evaluate_raw_pair() {
  local stage_root="$1"
  local stage_name="$2"
  local variant="$3"
  local checkpoint="$4"
  local base_folder="$5"
  local final_folder="$6"
  "${PY}" - \
    "${ROOT}" "${stage_name}" "${variant}" "${checkpoint}" \
    "${base_folder}" "${final_folder}" "${stage_root}/evaluation.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
stage_name, variant = sys.argv[2], sys.argv[3]
checkpoint = Path(sys.argv[4]).resolve()
folders = {"base_raw": sys.argv[5], "final_raw": sys.argv[6]}
output = Path(sys.argv[7])
tracker_root = root / "outputs" / "3. track"
sys.path.insert(0, str(root / "3. Tracker"))
from utils.etc import evaluate_sequences

sequences = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
args = SimpleNamespace(
    mode="all",
    data_path="/home/shang/datasets/MOT20/train",
    output_dir=str(tracker_root),
    print_per_sequence_metrics=False,
)
results = {}
for name, folder in folders.items():
    hashes = {}
    for sequence in sequences:
        source = tracker_root / folder / f"{sequence}.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        hashes[sequence] = hashlib.sha256(source.read_bytes()).hexdigest()
    results[name] = {
        "tracker_folder": folder,
        "source_sha256": hashes,
        **evaluate_sequences(args, folder, "MOT20", sequences),
    }
base_hota = results["base_raw"]["combined"]["HOTA"]
final_hota = results["final_raw"]["combined"]["HOTA"]
manifest = {
    "schema_version": 1,
    "dataset": "MOT20",
    "mode": "all",
    "stage": stage_name,
    "architecture_variant": variant,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "comparisons": {"final_raw_minus_base_raw": final_hota - base_hota},
    "cases": results,
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
}

for index in "${!STAGE_IDS[@]}"; do
  stage_id="${STAGE_IDS[index]}"
  variant="${VARIANTS[index]}"
  no_harm_weight="${NO_HARM_WEIGHTS[index]}"
  stage_root="${GROUP_ROOT}/${stage_id}"
  checkpoint_dir="${stage_root}/checkpoints"
  checkpoint="${checkpoint_dir}/iwg_rg_cma_epoch100.pt"
  training_summary="${checkpoint_dir}/training_summary.json"
  checkpoint_validation="${stage_root}/checkpoint_validation.json"
  if [[ -f "${stage_root}/completed" && -f "${stage_root}/evaluation.json" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') | RESUME SKIP COMPLETED ${stage_id}" \
      | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
    continue
  fi
  mkdir -p "${checkpoint_dir}" "${stage_root}/logs" "${stage_root}/provenance"

  train_command=(
    "${PY}" -u -m agentguard.cli train_iwg_rg_cma
    --dataset-dir "${DATASET_DIR}"
    --checkpoint-dir "${checkpoint_dir}"
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
    --reliability-mode full
    --architecture-variant "${variant}"
    --residual-beta 1.0
    --residual-weight 0.5
    --revision-weight 0.01
    --hard-example-gain 0.0
    --no-harm-weight "${no_harm_weight}"
    --memory-shards 4
    --epochs-per-shard 1
    --shard-cycles 25
    --sequence-sampling sample-proportional
  )
  if ((RESUME)) && training_stage_is_complete \
    "${training_summary}" "${checkpoint}" "${variant}"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') | RESUME REUSE TRAINED ${stage_id}" \
      | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
  else
    if find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "Refusing to overwrite partial checkpoint directory: ${checkpoint_dir}" >&2
      exit 2
    fi
    printf '%q ' "${train_command[@]}" > "${stage_root}/provenance/train.command.txt"
    printf '\n' >> "${stage_root}/provenance/train.command.txt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | TRAIN ${stage_id} variant=${variant}" \
      | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
    "${train_command[@]}" 2>&1 | tee "${stage_root}/logs/train.log"
  fi
  require_file "${checkpoint}"

  if ((RESUME)) && checkpoint_validation_is_valid \
    "${checkpoint_validation}" "${checkpoint}" "${variant}"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') | RESUME REUSE VALIDATION ${stage_id}" \
      | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') | VALIDATE CHECKPOINT ${stage_id}" \
      | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
    "${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
      --checkpoint "${checkpoint}" \
      --dataset-dir "${DATASET_DIR}" \
      --device cpu \
      --max-batches 8 \
      --output "${checkpoint_validation}" \
      2>&1 | tee "${stage_root}/logs/checkpoint_validation.log"
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') | EVAL BASE RAW ${stage_id}" \
    | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
  run_raw_case "${checkpoint}" "${stage_root}" "${stage_id}" base
  base_folder="${RAW_FOLDER}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') | EVAL FINAL RAW ${stage_id}" \
    | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
  run_raw_case "${checkpoint}" "${stage_root}" "${stage_id}" final
  final_folder="${RAW_FOLDER}"

  evaluate_raw_pair \
    "${stage_root}" "${stage_id}" "${variant}" "${checkpoint}" \
    "${base_folder}" "${final_folder}" \
    2>&1 | tee "${stage_root}/logs/evaluation.log"
  sha256sum "${DATASET_DIR}/metadata.json" "${checkpoint}" \
    > "${stage_root}/provenance/formal_artifacts.sha256"
  touch "${stage_root}/completed"
  echo "$(date '+%Y-%m-%d %H:%M:%S') | DONE ${stage_id}" \
    | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
done

"${PY}" - "${GROUP_ROOT}" "${STAGE_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stages = sys.argv[2:]
results = {}
for stage in stages:
    results[stage] = json.loads((root / stage / "evaluation.json").read_text())
summary = {
    "schema_version": 1,
    "dataset": "MOT20",
    "mode": "all",
    "schedule": {
        "nominal_epochs": 100,
        "memory_shards": 4,
        "epochs_per_shard": 1,
        "shard_cycles": 25,
        "effective_full_epochs": 25,
    },
    "stages": results,
}
(root / "evaluation_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
PY

complete=1
echo "$(date '+%Y-%m-%d %H:%M:%S') | DONE ALL STRUCTURAL ABLATIONS" \
  | tee -a "${GROUP_ROOT}/stages.log"
