#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT}/scripts/agentguard/run_iwg_rg_cma_mot20_clean_bidirectional_cma_sqrt.sh"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-iwg_rg_cma_mot20_legacy_clean_bidirectional_cma_sqrt_seed42_100e}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_epoch100.pt"
DATASET_DIR="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
COUNTERFACTUAL_SCRIPT="${ROOT}/scripts/agentguard/analyze_iwg_rg_cma_cross_modal_counterfactual.py"
COUNTERFACTUAL_OUTPUT="${EXPERIMENT_ROOT}/counterfactual_bidirectional_cma_116172.json"
SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
ARCHITECTURE_VARIANT="legacy-clean-bidirectional-cma"
REFERENCE_CLEAN_SQRT_FINAL_HOTA="0.7934412923397672"
REFERENCE_CLEAN_SQRT_BASE_HOTA="0.7929834681462621"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_mot20_clean_bidirectional_cma_sqrt.sh

Experiment:
  Legacy Base-IWG + clean motion/appearance streams + symmetric bidirectional
  history-level CMA + existing two-token CMA, trained with sqrt-size sampling.

Fixed controls:
  MOT20/all, context=6, bound=0.05, seed=42, batch=1024, lr=1e-4,
  4 shards x 1 epoch x 25 cycles = 100 nominal epochs = 25 full passes,
  fixed shard order, no base conditioning, and no reliability gate.

Evaluation:
  checkpoint validation -> all/base raw -> all/final raw -> paired cross-modal
  counterfactual. Test and post-processing are intentionally deferred.

Options:
  --dry-run    Print the fixed experiment contract and output paths.
  --preflight  Validate data, model contract, CLI settings, and CUDA only.
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
EXPERIMENT_ROOT=${EXPERIMENT_ROOT}
DATASET_DIR=${DATASET_DIR}
ARCHITECTURE_VARIANT=${ARCHITECTURE_VARIANT}
SCHEDULE=epochs:100,memory_shards:4,epochs_per_shard:1,shard_cycles:25,effective_full_epochs:25
COMMON=bound:0.05,context:6,seed:42,batch:1024,lr:0.0001,sequence_sampling:sqrt-size,replacement:true,shard_order:fixed
CMA=appearance_history_to_motion:true,motion_history_to_appearance:true,heads:4,history:6,two_token_fusion:true
EXCLUDED=base_conditioning:false,reliability_gate:false,no_harm_weight:0.0
EVALUATION=all:base_raw+final_raw;counterfactual:paired_bidirectional;test:none;post:none
REFERENCE_CLEAN_SQRT_BASE_HOTA=${REFERENCE_CLEAN_SQRT_BASE_HOTA}
REFERENCE_CLEAN_SQRT_FINAL_HOTA=${REFERENCE_CLEAN_SQRT_FINAL_HOTA}
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

stage() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" \
    | tee -a "${EXPERIMENT_ROOT}/stages.log"
}

require_file "${DATASET_DIR}/metadata.json"
require_file "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/cli/__init__.py"
require_file "${TRACKER_ROOT}/run.py"
require_file "${COUNTERFACTUAL_SCRIPT}"

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" - "${DATASET_DIR}" "${ARCHITECTURE_VARIANT}" <<'PY'
import json
import sys

from agentguard.models.iwg_rg_cma import model_contract
from agentguard.training.train_iwg_rg_cma import _validate_formal_config

dataset_dir, variant = sys.argv[1], sys.argv[2]
with open(f"{dataset_dir}/metadata.json") as handle:
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
contract = model_contract(
    correction_bound=0.05,
    context_size=6,
    architecture_variant=variant,
)
if contract["model_schema"] != (
    "agentguard_iwg_rg_cma_legacy_clean_bidirectional_cma_v1"
):
    raise SystemExit(f"unexpected model schema: {contract['model_schema']!r}")
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
        "no_harm_weight": 0.0,
        "memory_shards": 4,
        "epochs_per_shard": 1,
        "shard_cycles": 25,
        "sequence_sampling": "sqrt-size",
    }
)
print(
    "dataset=", metadata["dataset"],
    "samples=", metadata["num_train_samples"],
    "dataset_sha256=", metadata["dataset_sha256"],
)
print(
    "architecture_variant=", variant,
    "model_schema=", contract["model_schema"],
    "model_schema_sha256=", contract["model_schema_sha256"],
)
PY

"${PY}" - <<'PY'
import sys
import torch

print(
    "python=", sys.version.split()[0],
    "torch=", torch.__version__,
    "cuda=", torch.cuda.is_available(),
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the formal experiment")
PY

if ((PREFLIGHT)); then
  echo "Preflight passed; no experiment directory was created."
  exit 0
fi

require_absent "${EXPERIMENT_ROOT}"
mkdir -p "${CHECKPOINT_DIR}" "${EXPERIMENT_ROOT}/logs" "${EXPERIMENT_ROOT}/provenance"
echo "$$" > "${EXPERIMENT_ROOT}/pipeline.pid"
touch "${EXPERIMENT_ROOT}/running"
git -C "${ROOT}" rev-parse HEAD > "${EXPERIMENT_ROOT}/provenance/git_commit.txt"
git -C "${ROOT}" status --porcelain \
  > "${EXPERIMENT_ROOT}/provenance/git_status_porcelain.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/cli/__init__.py" \
  "${COUNTERFACTUAL_SCRIPT}" \
  "${DATASET_DIR}/metadata.json" \
  "${SCRIPT_PATH}" \
  > "${EXPERIMENT_ROOT}/provenance/formal_inputs.sha256"

complete=0
finish() {
  local code=$?
  rm -f "${EXPERIMENT_ROOT}/running"
  echo "${code}" > "${EXPERIMENT_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${complete} -eq 1 ]]; then
    touch "${EXPERIMENT_ROOT}/completed"
  else
    touch "${EXPERIMENT_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

train_command=(
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
  --reliability-mode full
  --architecture-variant "${ARCHITECTURE_VARIANT}"
  --residual-beta 1.0
  --residual-weight 0.5
  --revision-weight 0.01
  --hard-example-gain 0.0
  --no-harm-weight 0.0
  --memory-shards 4
  --epochs-per-shard 1
  --shard-cycles 25
  --sequence-sampling sqrt-size
)
printf '%q ' "${train_command[@]}" \
  > "${EXPERIMENT_ROOT}/provenance/train.command.txt"
printf '\n' >> "${EXPERIMENT_ROOT}/provenance/train.command.txt"
stage "TRAIN ${ARCHITECTURE_VARIANT} sampling=sqrt-size"
"${train_command[@]}" 2>&1 | tee "${EXPERIMENT_ROOT}/logs/train.log"
require_file "${CHECKPOINT}"

"${PY}" - "${CHECKPOINT_DIR}/training_summary.json" "${ARCHITECTURE_VARIANT}" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
variant = sys.argv[2]
expected = {
    "status": "completed",
    "selected_epoch": 100,
    "optimizer_steps": 25_700,
    "architecture_variant": variant,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(
            f"unexpected training summary {key}: {summary.get(key)!r} != {value!r}"
        )
if not math.isclose(float(summary["effective_full_epochs"]), 25.0):
    raise SystemExit(
        f"unexpected effective_full_epochs: {summary['effective_full_epochs']!r}"
    )
PY

stage "VALIDATE CHECKPOINT"
"${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  --device cpu \
  --max-batches 8 \
  --output "${EXPERIMENT_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/checkpoint_validation.log"

run_raw_case() {
  local output="$1"
  local suffix="${EXPERIMENT_NAME}_${output}_raw"
  RAW_FOLDER="mot20_all_0.80_${suffix}_iwg_rg_cma_${output}"
  require_absent "${TRACKER_OUTPUT_ROOT}/${RAW_FOLDER}"
  local command=(
    "${PY}" -u run.py
    --dataset MOT20
    --mode all
    --sequences "${SEQUENCES[@]}"
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${CHECKPOINT}"
    --iwg-rg-cma-output "${output}"
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${EXPERIMENT_ROOT}/resource_mot20_${output}_raw.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" \
    > "${EXPERIMENT_ROOT}/provenance/mot20_${output}_raw.command.txt"
  printf '\n' >> "${EXPERIMENT_ROOT}/provenance/mot20_${output}_raw.command.txt"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${EXPERIMENT_ROOT}/logs/mot20_${output}_raw.log"
}

stage "EVAL BASE RAW"
run_raw_case base
base_folder="${RAW_FOLDER}"
stage "EVAL FINAL RAW"
run_raw_case final
final_folder="${RAW_FOLDER}"

"${PY}" - \
  "${ROOT}" "${CHECKPOINT}" "${ARCHITECTURE_VARIANT}" \
  "${base_folder}" "${final_folder}" \
  "${REFERENCE_CLEAN_SQRT_BASE_HOTA}" "${REFERENCE_CLEAN_SQRT_FINAL_HOTA}" \
  "${EXPERIMENT_ROOT}/evaluation.json" <<'PY' \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/evaluation.log"
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2]).resolve()
variant = sys.argv[3]
folders = {"base_raw": sys.argv[4], "final_raw": sys.argv[5]}
reference_base_hota = float(sys.argv[6])
reference_final_hota = float(sys.argv[7])
output = Path(sys.argv[8])
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
    "architecture_variant": variant,
    "sequence_sampling": "sqrt-size",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "references": {
        "clean_sqrt_base_raw_hota": reference_base_hota,
        "clean_sqrt_final_raw_hota": reference_final_hota,
    },
    "comparisons": {
        "final_raw_minus_base_raw": final_hota - base_hota,
        "base_raw_minus_clean_sqrt_base_raw": base_hota - reference_base_hota,
        "final_raw_minus_clean_sqrt_final_raw": final_hota - reference_final_hota,
    },
    "cases": results,
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

counterfactual_command=(
  "${PY}" -u "${COUNTERFACTUAL_SCRIPT}"
  --checkpoint "${CHECKPOINT}"
  --dataset-dir "${DATASET_DIR}"
  --output "${COUNTERFACTUAL_OUTPUT}"
  --device cuda
  --batch-size 1024
  --num-workers 4
  --samples-per-sequence 32768
  --bootstrap-resamples 5000
  --progress-every 20
)
printf '%q ' "${counterfactual_command[@]}" \
  > "${EXPERIMENT_ROOT}/provenance/counterfactual.command.txt"
printf '\n' >> "${EXPERIMENT_ROOT}/provenance/counterfactual.command.txt"
stage "COUNTERFACTUAL BIDIRECTIONAL CMA"
"${counterfactual_command[@]}" \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/counterfactual.log"
require_file "${COUNTERFACTUAL_OUTPUT}"

sha256sum "${DATASET_DIR}/metadata.json" "${CHECKPOINT}" \
  "${EXPERIMENT_ROOT}/evaluation.json" "${COUNTERFACTUAL_OUTPUT}" \
  > "${EXPERIMENT_ROOT}/provenance/formal_artifacts.sha256"
complete=1
stage "DONE CLEAN BIDIRECTIONAL CMA SQRT"
