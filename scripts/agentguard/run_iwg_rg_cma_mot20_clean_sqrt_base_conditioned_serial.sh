#!/usr/bin/env bash
set -euo pipefail

# Two controlled MOT20 experiments. Both reuse the public compact dataset and
# differ only in whether each RG-CMA correction head sees its detached base gate.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT}/scripts/agentguard/run_iwg_rg_cma_mot20_clean_sqrt_base_conditioned_serial.sh"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GROUP_NAME="${GROUP_NAME:-iwg_rg_cma_mot20_clean_sqrt_base_conditioned_20260723}"
GROUP_ROOT="${ROOT}/outputs/agentguard/experiments/${GROUP_NAME}"
DATASET_DIR="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
TEST_SEQUENCES=(MOT20-04 MOT20-06 MOT20-07 MOT20-08)
STAGE_IDS=(a_clean_sqrt b_base_conditioned_sqrt)
VARIANTS=(legacy-clean-cross-modal legacy-clean-cross-modal-base-conditioned)
CLEAN_SAMPLE_FINAL_HOTA="0.7930392819755346"
LEGACY_BEST_FINAL_HOTA="0.792449"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_mot20_clean_sqrt_base_conditioned_serial.sh

Serial stages:
  A. legacy-clean-cross-modal + sqrt-size sampling.
  B. A + detached Base-IWG gate as an input to each correction head.

Fixed controls:
  MOT20/all, context=6, bound=0.05, seed=42, batch=1024, lr=1e-4,
  4 shards x 1 epoch x 25 cycles = 100 nominal epochs = 25 full passes,
  fixed shard order and sqrt-size sequence sampling with replacement.

Each stage runs:
  training -> checkpoint validation -> all/base raw -> all/final raw
  -> test/final+post -> submission ZIP.

Options:
  --dry-run    Print the fixed experiment matrix and output paths.
  --preflight  Validate data, contracts, CLI settings, and CUDA only.
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
GROUP_ROOT=${GROUP_ROOT}
DATASET_DIR=${DATASET_DIR}
SCHEDULE=epochs:100,memory_shards:4,epochs_per_shard:1,shard_cycles:25,effective_full_epochs:25
COMMON=bound:0.05,context:6,seed:42,batch:1024,lr:0.0001,sequence_sampling:sqrt-size,replacement:true,shard_order:fixed
LOSS=residual_beta:1.0,residual_weight:0.5,revision_weight:0.01,hard_example_gain:0.0,no_harm_weight:0.0
A=architecture:legacy-clean-cross-modal
B=architecture:legacy-clean-cross-modal-base-conditioned
EVALUATION=all:base_raw+final_raw;test:final_post+zip
REFERENCE_CLEAN_SAMPLE_FINAL_HOTA=${CLEAN_SAMPLE_FINAL_HOTA}
REFERENCE_LEGACY_BEST_FINAL_HOTA=${LEGACY_BEST_FINAL_HOTA}
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
  local stage_root="$1"
  local message="$2"
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" \
    | tee -a "${GROUP_ROOT}/stages.log" "${stage_root}/stages.log"
}

require_file "${DATASET_DIR}/metadata.json"
require_file "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py"
require_file "${ROOT}/agentguard/src/agentguard/cli/__init__.py"
require_file "${TRACKER_ROOT}/run.py"

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" - "${DATASET_DIR}" "${VARIANTS[@]}" <<'PY'
import json
import sys

from agentguard.models.iwg_rg_cma import model_contract
from agentguard.training.train_iwg_rg_cma import _validate_formal_config

dataset_dir = sys.argv[1]
variants = sys.argv[2:]
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
for variant in variants:
    contract = model_contract(
        correction_bound=0.05,
        context_size=6,
        architecture_variant=variant,
    )
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
        "variant=", variant,
        "model_schema=", contract["model_schema"],
        "model_schema_sha256=", contract["model_schema_sha256"],
    )
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

print(
    "python=", sys.version.split()[0],
    "torch=", torch.__version__,
    "cuda=", torch.cuda.is_available(),
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the formal experiments")
PY

if ((PREFLIGHT)); then
  echo "Preflight passed; no experiment directory was created."
  exit 0
fi

require_absent "${GROUP_ROOT}"
mkdir -p "${GROUP_ROOT}/provenance"
echo "$$" > "${GROUP_ROOT}/pipeline.pid"
touch "${GROUP_ROOT}/running"
git -C "${ROOT}" rev-parse HEAD > "${GROUP_ROOT}/provenance/git_commit.txt"
git -C "${ROOT}" status --porcelain \
  > "${GROUP_ROOT}/provenance/git_status_porcelain.txt"
sha256sum \
  "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/cli/__init__.py" \
  "${DATASET_DIR}/metadata.json" \
  "${SCRIPT_PATH}" \
  > "${GROUP_ROOT}/provenance/formal_inputs.sha256"

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
  local stage_id="$3"
  local output="$4"
  local suffix="${GROUP_NAME}_${stage_id}_${output}_raw"
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
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output "${output}"
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${stage_root}/resource_mot20_${output}_raw.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" \
    > "${stage_root}/provenance/mot20_${output}_raw.command.txt"
  printf '\n' >> "${stage_root}/provenance/mot20_${output}_raw.command.txt"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${stage_root}/logs/mot20_${output}_raw.log"
}

evaluate_raw_pair() {
  local stage_root="$1"
  local stage_id="$2"
  local variant="$3"
  local checkpoint="$4"
  local base_folder="$5"
  local final_folder="$6"
  "${PY}" - \
    "${ROOT}" "${stage_id}" "${variant}" "${checkpoint}" \
    "${base_folder}" "${final_folder}" \
    "${CLEAN_SAMPLE_FINAL_HOTA}" "${LEGACY_BEST_FINAL_HOTA}" \
    "${stage_root}/evaluation.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
stage_id, variant = sys.argv[2], sys.argv[3]
checkpoint = Path(sys.argv[4]).resolve()
folders = {"base_raw": sys.argv[5], "final_raw": sys.argv[6]}
clean_sample_hota = float(sys.argv[7])
legacy_best_hota = float(sys.argv[8])
output = Path(sys.argv[9])
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
    "stage": stage_id,
    "architecture_variant": variant,
    "sequence_sampling": "sqrt-size",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "references": {
        "clean_sample_proportional_final_raw_hota": clean_sample_hota,
        "legacy_best_final_raw_hota": legacy_best_hota,
    },
    "comparisons": {
        "final_raw_minus_base_raw": final_hota - base_hota,
        "final_raw_minus_clean_sample_proportional": final_hota - clean_sample_hota,
        "final_raw_minus_legacy_best": final_hota - legacy_best_hota,
    },
    "cases": results,
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
}

run_test_post() {
  local stage_root="$1"
  local stage_id="$2"
  local checkpoint="$3"
  local suffix="${GROUP_NAME}_${stage_id}_final_test_post"
  local raw_folder="mot20_test_0.80_${suffix}_iwg_rg_cma_final"
  TEST_FOLDER="${raw_folder}_post"
  TEST_ZIP="${stage_root}/mot20_test_${stage_id}_final_post.zip"
  require_absent "${TRACKER_OUTPUT_ROOT}/${raw_folder}"
  require_absent "${TRACKER_OUTPUT_ROOT}/${TEST_FOLDER}"
  require_absent "${TEST_ZIP}"
  local command=(
    "${PY}" -u run.py
    --dataset MOT20
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
    --resource-log "${stage_root}/resource_mot20_final_test_post.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" \
    > "${stage_root}/provenance/mot20_final_test_post.command.txt"
  printf '\n' >> "${stage_root}/provenance/mot20_final_test_post.command.txt"
  (cd "${TRACKER_ROOT}" && "${command[@]}") \
    2>&1 | tee "${stage_root}/logs/mot20_final_test_post.log"
  for sequence in "${TEST_SEQUENCES[@]}"; do
    require_file "${TRACKER_OUTPUT_ROOT}/${TEST_FOLDER}/${sequence}.txt"
  done
  "${PY}" - \
    "${TRACKER_OUTPUT_ROOT}/${TEST_FOLDER}" "${TEST_ZIP}" \
    "${TEST_SEQUENCES[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

source_dir = Path(sys.argv[1])
output_zip = Path(sys.argv[2])
sequences = sys.argv[3:]
hashes = {}
with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as archive:
    for sequence in sequences:
        source = source_dir / f"{sequence}.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        archive.write(source, arcname=source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
manifest = {
    "tracker_folder": source_dir.name,
    "submission_zip": str(output_zip.resolve()),
    "files": hashes,
}
(output_zip.parent / "test_submission.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
}

for index in "${!STAGE_IDS[@]}"; do
  stage_id="${STAGE_IDS[index]}"
  variant="${VARIANTS[index]}"
  stage_root="${GROUP_ROOT}/${stage_id}"
  checkpoint_dir="${stage_root}/checkpoints"
  checkpoint="${checkpoint_dir}/iwg_rg_cma_epoch100.pt"
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
    --no-harm-weight 0.0
    --memory-shards 4
    --epochs-per-shard 1
    --shard-cycles 25
    --sequence-sampling sqrt-size
  )
  printf '%q ' "${train_command[@]}" \
    > "${stage_root}/provenance/train.command.txt"
  printf '\n' >> "${stage_root}/provenance/train.command.txt"
  stage "${stage_root}" "TRAIN ${stage_id} variant=${variant} sampling=sqrt-size"
  "${train_command[@]}" 2>&1 | tee "${stage_root}/logs/train.log"
  require_file "${checkpoint}"

  "${PY}" - "${checkpoint_dir}/training_summary.json" "${variant}" <<'PY'
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
        raise SystemExit(f"unexpected training summary {key}: {summary.get(key)!r} != {value!r}")
if not math.isclose(float(summary["effective_full_epochs"]), 25.0):
    raise SystemExit(
        f"unexpected effective_full_epochs: {summary['effective_full_epochs']!r}"
    )
PY

  stage "${stage_root}" "VALIDATE CHECKPOINT ${stage_id}"
  "${PY}" -u -m agentguard.cli validate_iwg_rg_cma_checkpoint \
    --checkpoint "${checkpoint}" \
    --dataset-dir "${DATASET_DIR}" \
    --device cpu \
    --max-batches 8 \
    --output "${stage_root}/checkpoint_validation.json" \
    2>&1 | tee "${stage_root}/logs/checkpoint_validation.log"

  stage "${stage_root}" "EVAL BASE RAW ${stage_id}"
  run_raw_case "${checkpoint}" "${stage_root}" "${stage_id}" base
  base_folder="${RAW_FOLDER}"
  stage "${stage_root}" "EVAL FINAL RAW ${stage_id}"
  run_raw_case "${checkpoint}" "${stage_root}" "${stage_id}" final
  final_folder="${RAW_FOLDER}"
  evaluate_raw_pair \
    "${stage_root}" "${stage_id}" "${variant}" "${checkpoint}" \
    "${base_folder}" "${final_folder}" \
    2>&1 | tee "${stage_root}/logs/evaluation.log"

  stage "${stage_root}" "TEST FINAL+POST ${stage_id}"
  run_test_post "${stage_root}" "${stage_id}" "${checkpoint}"
  printf '%s\n' "${TEST_FOLDER}" > "${stage_root}/test_tracker_folder.txt"
  printf '%s\n' "${TEST_ZIP}" > "${stage_root}/test_submission_zip.txt"
  sha256sum "${DATASET_DIR}/metadata.json" "${checkpoint}" "${TEST_ZIP}" \
    > "${stage_root}/provenance/formal_artifacts.sha256"
  touch "${stage_root}/completed"
  stage "${stage_root}" "DONE ${stage_id}"
done

"${PY}" - "${GROUP_ROOT}" "${STAGE_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stages = {}
for stage_id in sys.argv[2:]:
    stage_root = root / stage_id
    stages[stage_id] = {
        "evaluation": json.loads((stage_root / "evaluation.json").read_text()),
        "test_submission": json.loads((stage_root / "test_submission.json").read_text()),
    }
summary = {
    "schema_version": 1,
    "dataset": "MOT20",
    "schedule": {
        "nominal_epochs": 100,
        "memory_shards": 4,
        "epochs_per_shard": 1,
        "shard_cycles": 25,
        "effective_full_epochs": 25,
        "sequence_sampling": "sqrt-size",
        "replacement": True,
        "shard_order": "fixed",
    },
    "stages": stages,
}
(root / "evaluation_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
PY

complete=1
touch "${GROUP_ROOT}/completed"
printf '%s | DONE ALL CLEAN+SQRT EXPERIMENTS\n' "$(date '+%Y-%m-%d %H:%M:%S')" \
  | tee -a "${GROUP_ROOT}/stages.log"
