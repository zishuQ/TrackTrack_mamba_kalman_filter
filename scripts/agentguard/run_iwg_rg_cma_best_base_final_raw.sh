#!/usr/bin/env bash
set -euo pipefail

# Evaluate the three selected checkpoints with the base and final gates.
# This script only runs raw TrackEval; it does not train or post-process.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRACKER_ROOT="${ROOT}/3. Tracker"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"

EVAL_NAME="${EVAL_NAME:-iwg_rg_cma_best_base_final_raw_20260725}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${EVAL_NAME}"
LOG_ROOT="${RUN_ROOT}/logs"
RESOURCE_ROOT="${RUN_ROOT}/resources"
PROVENANCE_ROOT="${RUN_ROOT}/provenance"
SUMMARY_TSV="${RUN_ROOT}/metrics.tsv"
DELTA_TSV="${RUN_ROOT}/base_final_delta.tsv"

MOT17_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_mot20s4e100_to_mot17_finetune50_lr1e5_bs1024/checkpoints/iwg_rg_cma_epoch050.pt"
MOT20_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_sqrt_size_mot20_bound005_context6_seed42_bs1024_shard4x1_100e/checkpoints/iwg_rg_cma_epoch100.pt"
SPORTSMOT_CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_v1_sportsmot_trainval_interleaved_shard4x1_seed42_bs1024_200e/checkpoints/iwg_rg_cma_epoch200.pt"

MOT17_SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN
  MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)
MOT20_SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)
SPORTSMOT_SEQMAP="${TRACKER_ROOT}/trackeval/seqmap/sportsmot/val.txt"

ONLY="all"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_best_base_final_raw.sh [options]

Runs raw all/val evaluation for both base and final gates:
  MOT17     epoch050, all
  MOT20     epoch100, all
  SportsMOT epoch200, val

Options:
  --only NAME       Run only mot17, mot20, sportsmot, or all (default: all).
  --dry-run         Print the selected paths without running them.
  -h, --help        Show this help.

Environment:
  EVAL_NAME         Output experiment directory name.
  AGENTGUARD_DEVICE cpu (default) or cuda.
  PYTHON_BIN        Python executable override.

Outputs:
  outputs/agentguard/experiments/$EVAL_NAME/logs/*.log
  outputs/agentguard/experiments/$EVAL_NAME/metrics.tsv
  outputs/agentguard/experiments/$EVAL_NAME/base_final_delta.tsv
EOF
}

while (( $# )); do
  case "$1" in
    --only)
      ONLY="${2:?--only requires mot17, mot20, sportsmot, or all}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${ONLY}" in
  all|mot17|mot20|sportsmot) ;;
  *)
    echo "Invalid --only value: ${ONLY}" >&2
    exit 2
    ;;
esac

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory is missing: $1" >&2
    exit 2
  fi
}

metric_line() {
  awk '
    $1 == "HOTA" && $2 == "MOTA" {
      getline
      if ($1 ~ /^0[.]/ && $2 ~ /^0[.]/ && $3 ~ /^0[.]/ && $4 ~ /^0[.]/ && $5 ~ /^0[.]/) {
        print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
        found = 1
      }
      exit
    }
    END { if (!found) exit 1 }
  ' "$1"
}

has_metrics() {
  metric_line "$1" >/dev/null 2>&1
}

write_summary() {
  local output="$1"
  local dataset="$2"
  local split="$3"
  local epoch="$4"
  local log_path="$5"
  local values

  values="$(metric_line "${log_path}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${dataset}" "${split}" "${epoch}" "${output}" ${values} "${log_path}" \
    >> "${SUMMARY_TSV}"
}

run_case() {
  local dataset="$1"
  local split="$2"
  local epoch="$3"
  local output="$4"
  local checkpoint="$5"
  local case_name="$6"
  shift 6
  local -a sequences=("$@")
  local log_path="${LOG_ROOT}/${case_name}.log"
  local resource_path="${RESOURCE_ROOT}/${case_name}.jsonl"
  local -a command=(
    "${PY}" -u run.py
    --dataset "${dataset}"
    --mode "${split}"
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${checkpoint}"
    --iwg-rg-cma-output "${output}"
    --agentguard-device "${AGENTGUARD_DEVICE}"
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${EVAL_NAME}_${case_name}"
    --print-per-sequence-metrics
    --resource-log "${resource_path}"
    --profile-every 500
  )

  if (( ${#sequences[@]} )); then
    command+=(--sequences "${sequences[@]}")
  fi

  if [[ -f "${log_path}" ]]; then
    if has_metrics "${log_path}"; then
      echo "SKIP completed ${case_name}: ${log_path}"
      write_summary "${output}" "${dataset}" "${split}" "${epoch}" "${log_path}"
      return 0
    fi
    echo "Existing log is incomplete; choose a new EVAL_NAME: ${log_path}" >&2
    exit 2
  fi

  printf '%q ' "${command[@]}" > "${PROVENANCE_ROOT}/${case_name}.command.txt"
  printf '\n' >> "${PROVENANCE_ROOT}/${case_name}.command.txt"
  echo "START ${case_name}"
  (cd "${TRACKER_ROOT}" && "${command[@]}") 2>&1 | tee "${log_path}"
  if ! has_metrics "${log_path}"; then
    echo "Evaluation finished without a complete metric block: ${log_path}" >&2
    exit 1
  fi
  write_summary "${output}" "${dataset}" "${split}" "${epoch}" "${log_path}"
  echo "DONE ${case_name}"
}

run_dataset() {
  local dataset="$1"
  local split="$2"
  local epoch="$3"
  local checkpoint="$4"
  shift 4
  local -a sequences=("$@")

  run_case "${dataset}" "${split}" "${epoch}" base "${checkpoint}" \
    "${dataset,,}_e${epoch}_base_${split}_raw" "${sequences[@]}"
  run_case "${dataset}" "${split}" "${epoch}" final "${checkpoint}" \
    "${dataset,,}_e${epoch}_final_${split}_raw" "${sequences[@]}"
}

require_file "${PY}"
require_file "${TRACKER_ROOT}/run.py"
require_dir "${DETECTION_CACHE_ROOT}/MOT17/all"
require_dir "${DETECTION_CACHE_ROOT}/MOT20/all"
require_dir "${DETECTION_CACHE_ROOT}/SportsMOT/val"
require_file "${MOT17_CHECKPOINT}"
require_file "${MOT20_CHECKPOINT}"
require_file "${SPORTSMOT_CHECKPOINT}"
require_file "${SPORTSMOT_SEQMAP}"

mapfile -t SPORTSMOT_SEQUENCES < <(sed '/^name$/d;/^$/d' "${SPORTSMOT_SEQMAP}")
if [[ "${#SPORTSMOT_SEQUENCES[@]}" -ne 45 ]]; then
  echo "Expected 45 SportsMOT val sequences, got ${#SPORTSMOT_SEQUENCES[@]}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"
AGENTGUARD_DEVICE="${AGENTGUARD_DEVICE:-cpu}"

if (( DRY_RUN )); then
  echo "EVAL_NAME=${EVAL_NAME}"
  echo "AGENTGUARD_DEVICE=${AGENTGUARD_DEVICE}"
  case "${ONLY}" in
    all|mot17) printf 'MOT17 checkpoint: %s\n' "${MOT17_CHECKPOINT}" ;;
  esac
  case "${ONLY}" in
    all|mot20) printf 'MOT20 checkpoint: %s\n' "${MOT20_CHECKPOINT}" ;;
  esac
  case "${ONLY}" in
    all|sportsmot) printf 'SportsMOT checkpoint: %s\n' "${SPORTSMOT_CHECKPOINT}" ;;
  esac
  echo "No tracker process was started."
  exit 0
fi

mkdir -p "${LOG_ROOT}" "${RESOURCE_ROOT}" "${PROVENANCE_ROOT}"
printf 'dataset\tsplit\tepoch\toutput\tHOTA\tMOTA\tIDF1\tDetA\tAssA\tlog\n' > "${SUMMARY_TSV}"
sha256sum "${MOT17_CHECKPOINT}" "${MOT20_CHECKPOINT}" "${SPORTSMOT_CHECKPOINT}" \
  > "${PROVENANCE_ROOT}/checkpoint_sha256.txt"
printf 'agentguard_device=%s\nonly=%s\neval_name=%s\n' \
  "${AGENTGUARD_DEVICE}" "${ONLY}" "${EVAL_NAME}" \
  > "${PROVENANCE_ROOT}/settings.txt"

case "${ONLY}" in
  all|mot17)
    run_dataset MOT17 all 050 "${MOT17_CHECKPOINT}" "${MOT17_SEQUENCES[@]}"
    ;;
esac
case "${ONLY}" in
  all|mot20)
    run_dataset MOT20 all 100 "${MOT20_CHECKPOINT}" "${MOT20_SEQUENCES[@]}"
    ;;
esac
case "${ONLY}" in
  all|sportsmot)
    run_dataset SportsMOT val 200 "${SPORTSMOT_CHECKPOINT}" "${SPORTSMOT_SEQUENCES[@]}"
    ;;
esac

awk -F '\t' '
  NR == 1 { next }
  {
    key = $1 "/" $2
    if ($4 == "base") {
      base_hota[key] = $5
      base_mota[key] = $6
      base_idf1[key] = $7
      base_deta[key] = $8
      base_assa[key] = $9
    } else if ($4 == "final") {
      final_hota[key] = $5
      final_mota[key] = $6
      final_idf1[key] = $7
      final_deta[key] = $8
      final_assa[key] = $9
    }
  }
  END {
    print "dataset/split\tbase_HOTA\tfinal_HOTA\tdelta_HOTA_pp\tbase_MOTA\tfinal_MOTA\tdelta_MOTA_pp\tbase_IDF1\tfinal_IDF1\tdelta_IDF1_pp\tbase_DetA\tfinal_DetA\tdelta_DetA_pp\tbase_AssA\tfinal_AssA\tdelta_AssA_pp"
    for (key in base_hota) {
      if (!(key in final_hota)) continue
      printf "%s\t%.4f\t%.4f\t%+.4f\t%.4f\t%.4f\t%+.4f\t%.4f\t%.4f\t%+.4f\t%.4f\t%.4f\t%+.4f\t%.4f\t%.4f\t%+.4f\n", \
        key, 100 * base_hota[key], 100 * final_hota[key], 100 * (final_hota[key] - base_hota[key]), \
        100 * base_mota[key], 100 * final_mota[key], 100 * (final_mota[key] - base_mota[key]), \
        100 * base_idf1[key], 100 * final_idf1[key], 100 * (final_idf1[key] - base_idf1[key]), \
        100 * base_deta[key], 100 * final_deta[key], 100 * (final_deta[key] - base_deta[key]), \
        100 * base_assa[key], 100 * final_assa[key], 100 * (final_assa[key] - base_assa[key])
    }
  }
' "${SUMMARY_TSV}" > "${DELTA_TSV}"

echo "All requested evaluations are complete."
echo "Metrics: ${SUMMARY_TSV}"
echo "Base/final deltas: ${DELTA_TSV}"
