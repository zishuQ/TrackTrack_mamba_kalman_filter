#!/usr/bin/env bash
set -euo pipefail

# Serial CUDA test pipeline for the three main 1x6 checkpoints.
# MOT17/MOT20 use Tracker's built-in post; SportsMOT additionally uses AFLink.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
PIPELINE_NAME="${PIPELINE_NAME:-iwg_rg_cma_1x6_cuda_test_post_serial_20260725}"
PIPELINE_ROOT="${ROOT}/outputs/agentguard/overnight/${PIPELINE_NAME}"
LOG_DIR="${PIPELINE_ROOT}/logs"
PROVENANCE_DIR="${PIPELINE_ROOT}/provenance"

MOT17_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_1x6_oldbest_mot20e100_to_mot17_bound005_sampleprop_seed42_bs1024_finetune50/checkpoints/iwg_rg_cma_epoch050.pt"
MOT20_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_mot20_legacy_clean_cross_modal_seed42_100e/checkpoints/iwg_rg_cma_epoch100.pt"
SPORTS_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_1x6_oldbest_sportsmot_trainval_bound005_sampleprop_seed42_bs1024_shard4x1_200e/checkpoints/iwg_rg_cma_epoch200.pt"
AF_LINK_MODEL="${TRACKER_ROOT}/AFLink/AFLink_epoch20.pth"

MOT17_SEQUENCES=(
  MOT17-01-FRCNN MOT17-03-FRCNN MOT17-06-FRCNN MOT17-07-FRCNN
  MOT17-08-FRCNN MOT17-12-FRCNN MOT17-14-FRCNN
)
MOT20_SEQUENCES=(MOT20-04 MOT20-06 MOT20-07 MOT20-08)
mapfile -t SPORTS_SEQUENCES < <(
  sed '/^name$/d;/^$/d' "${TRACKER_ROOT}/trackeval/seqmap/sportsmot/test.txt"
)

DRY_RUN=0
PREFLIGHT=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_1x6_cuda_test_post_serial.sh

Stages:
  1. MOT17 1x6 epoch050: test + built-in post
  2. MOT20 1x6 epoch100: test + built-in post
  3. SportsMOT 1x6 epoch200: test + built-in post + AFLink

All tracker calls use --agentguard-device cuda and alpha=1.0.
The script does not train, rebuild labels, or run TrackEval on test data.

Options:
  --dry-run    Print paths and settings without running inference.
  --preflight  Check weights, AFLink model, test sequence lists, and CUDA.
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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

stage() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" \
    | tee -a "${PIPELINE_ROOT}/stages.log"
}

require_file "${MOT17_CHECKPOINT}"
require_file "${MOT20_CHECKPOINT}"
require_file "${SPORTS_CHECKPOINT}"
require_file "${AF_LINK_MODEL}"
require_file "${TRACKER_ROOT}/run.py"
require_file "${TRACKER_ROOT}/aflink_use.py"
require_file "${TRACKER_ROOT}/trackeval/seqmap/mot17/test.txt"
require_file "${TRACKER_ROOT}/trackeval/seqmap/mot20/test.txt"
require_file "${TRACKER_ROOT}/trackeval/seqmap/sportsmot/test.txt"

if [[ "${#SPORTS_SEQUENCES[@]}" -eq 0 ]]; then
  echo "SportsMOT test sequence list is empty" >&2
  exit 2
fi

if ((DRY_RUN)); then
  printf 'PIPELINE_ROOT=%s\n' "${PIPELINE_ROOT}"
  printf 'MOT17_CHECKPOINT=%s\n' "${MOT17_CHECKPOINT}"
  printf 'MOT20_CHECKPOINT=%s\n' "${MOT20_CHECKPOINT}"
  printf 'SPORTS_CHECKPOINT=%s\n' "${SPORTS_CHECKPOINT}"
  printf 'SPORTS_TEST_SEQUENCES=%d\n' "${#SPORTS_SEQUENCES[@]}"
  printf 'DEVICE=cuda\nALPHA=1.0\nPOST=tracker_builtin_then_sportsmot_aflink\n'
  exit 0
fi

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if ((PREFLIGHT)); then
  "${PY}" - <<'PY'
import sys
import torch

print("python=", sys.version.split()[0])
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this test pipeline")
PY
  echo "Preflight passed; no inference was started."
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

mkdir -p "${LOG_DIR}" "${PROVENANCE_DIR}"
rm -f "${PIPELINE_ROOT}/failed"
git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"
printf 'mot17=%s\nmot20=%s\nsportsmot=%s\n' \
  "${MOT17_CHECKPOINT}" "${MOT20_CHECKPOINT}" "${SPORTS_CHECKPOINT}" \
  > "${PROVENANCE_DIR}/checkpoints.txt"

echo "$$" > "${PIPELINE_ROOT}/pipeline.pid"
touch "${PIPELINE_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
  rm -f "${PIPELINE_ROOT}/running"
  printf '%s\n' "${code}" > "${PIPELINE_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${pipeline_complete} -eq 1 ]]; then
    touch "${PIPELINE_ROOT}/completed"
  else
    touch "${PIPELINE_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

run_tracker_test_post() {
  local stage_name="$1"
  local dataset="$2"
  local checkpoint="$3"
  local suffix="$4"
  local expected_file="$5"
  shift 5
  local -a sequences=("$@")
  local post_folder="${TRACKER_OUTPUT_ROOT}/mot17_test_0.80_${suffix}_iwg_rg_cma_final_post"
  if [[ "${dataset}" == "MOT20" ]]; then
    post_folder="${TRACKER_OUTPUT_ROOT}/mot20_test_0.80_${suffix}_iwg_rg_cma_final_post"
  elif [[ "${dataset}" == "SportsMOT" ]]; then
    post_folder="${TRACKER_OUTPUT_ROOT}/sportsmot_test_0.80_${suffix}_iwg_rg_cma_final_post"
  fi
  if [[ -f "${post_folder}/${expected_file}" ]]; then
    printf '%s\n' "${post_folder}" > "${PIPELINE_ROOT}/${stage_name}_post_folder.txt"
    stage "SKIP ${stage_name}: existing post output ${post_folder}"
    return 0
  fi
  local -a command=(
    "${PY}" -u run.py
    --dataset "${dataset}"
    --mode test
    --seed 10000
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output final
    --iwg-rg-cma-alpha 1.0
    --agentguard-device cuda
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}"
    --sequences "${sequences[@]}"
    --use_post
    --skip-eval
    --resource-log "${PIPELINE_ROOT}/resource_${stage_name}.jsonl"
    --profile-every 500
  )
  local command_file="${PROVENANCE_DIR}/${stage_name}.command.txt"
  printf '%q ' "${command[@]}" > "${command_file}"
  printf '\n' >> "${command_file}"
  stage "START ${stage_name}: ${dataset} CUDA test + post"
  (
    cd "${TRACKER_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${LOG_DIR}/${stage_name}.log"

  require_file "${post_folder}/${expected_file}"
  printf '%s\n' "${post_folder}" > "${PIPELINE_ROOT}/${stage_name}_post_folder.txt"
  stage "DONE ${stage_name}: ${post_folder}"
}

run_sports_aflink() {
  local input_folder="$1"
  local output_folder="${input_folder}_aflink_post"
  local -a command=(
    "${PY}" aflink_use.py
    --result-folder "${input_folder}"
    --output-folder "${output_folder}"
    --model-path "${AF_LINK_MODEL}"
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/sportsmot_aflink.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/sportsmot_aflink.command.txt"
  stage "START sportsmot AFLink: ${input_folder}"
  (
    cd "${TRACKER_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${LOG_DIR}/sportsmot_aflink.log"
  require_file "${output_folder}/${SPORTS_SEQUENCES[0]}.txt"
  printf '%s\n' "${output_folder}" > "${PIPELINE_ROOT}/sportsmot_aflink_post_folder.txt"
  stage "DONE sportsmot AFLink: ${output_folder}"
}

run_tracker_test_post \
  mot17_test_post MOT17 "${MOT17_CHECKPOINT}" \
  mot17_1x6_clean_cross_modal_e050_cuda_test MOT17-01-FRCNN.txt \
  "${MOT17_SEQUENCES[@]}"

run_tracker_test_post \
  mot20_test_post MOT20 "${MOT20_CHECKPOINT}" \
  mot20_1x6_clean_cross_modal_e100_cuda_test MOT20-04.txt \
  "${MOT20_SEQUENCES[@]}"

run_tracker_test_post \
  sportsmot_test_post SportsMOT "${SPORTS_CHECKPOINT}" \
  sportsmot_1x6_clean_cross_modal_e200_cuda_test \
  "${SPORTS_SEQUENCES[0]}.txt" \
  "${SPORTS_SEQUENCES[@]}"

SPORTS_POST_FOLDER="$(<"${PIPELINE_ROOT}/sportsmot_test_post_post_folder.txt")"
run_sports_aflink "${SPORTS_POST_FOLDER}"

pipeline_complete=1
stage "DONE full pipeline"
