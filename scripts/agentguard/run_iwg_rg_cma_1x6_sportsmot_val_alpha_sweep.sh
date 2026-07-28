#!/usr/bin/env bash
set -euo pipefail

# Validation-only alpha sweep for the existing 1x6 SportsMOT checkpoint.
# No test split, post-processing, AFLink, label rebuild, or training is run.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SWEEP_NAME="${SWEEP_NAME:-iwg_rg_cma_1x6_sportsmot_val_alpha_sweep_20260725}"
SWEEP_ROOT="${ROOT}/outputs/agentguard/overnight/${SWEEP_NAME}"
CHECKPOINT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_1x6_oldbest_sportsmot_trainval_bound005_sampleprop_seed42_bs1024_shard4x1_200e/checkpoints/iwg_rg_cma_epoch200.pt"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/3. Tracker"
TRACKER_OUTPUT_ROOT="${ROOT}/outputs/3. track"
SEQMAP="${TRACKER_ROOT}/trackeval/seqmap/sportsmot/val.txt"

ALPHAS=(0 0.25 0.5 0.75)
LABELS=(alpha0p00 alpha0p25 alpha0p50 alpha0p75)
REFERENCE_ALPHA1_HOTA="0.831551"

DRY_RUN=0
while (($#)); do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/agentguard/run_iwg_rg_cma_1x6_sportsmot_val_alpha_sweep.sh

Runs SportsMOT validation only for alpha=0, 0.25, 0.5, and 0.75.
The existing alpha=1 validation result (HOTA=0.831551) is kept as the
reference. No test output, post-processing, AFLink, or training is run.

Options:
  --dry-run    Print the fixed paths and alpha values.
EOF
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
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

require_file "${CHECKPOINT}"
require_file "${SEQMAP}"
mapfile -t SEQUENCES < <(sed '/^name$/d;/^$/d' "${SEQMAP}")
if [[ "${#SEQUENCES[@]}" -ne 45 ]]; then
  echo "Expected 45 SportsMOT validation sequences, got ${#SEQUENCES[@]}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${TRACKER_ROOT}:${ROOT}/agentguard/src:${PYTHONPATH:-}"

if ((DRY_RUN)); then
  cat <<EOF
SWEEP_ROOT=${SWEEP_ROOT}
CHECKPOINT=${CHECKPOINT}
DATASET=SportsMOT/val
ALPHAS=0,0.25,0.5,0.75
REFERENCE_ALPHA1_HOTA=${REFERENCE_ALPHA1_HOTA}
POST=none
AF_LINK=none
EOF
  exit 0
fi

require_absent "${SWEEP_ROOT}"
mkdir -p "${SWEEP_ROOT}/logs" "${SWEEP_ROOT}/provenance"
git -C "${ROOT}" rev-parse HEAD > "${SWEEP_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${SWEEP_ROOT}/git_status_porcelain.txt"
sha256sum "${CHECKPOINT}" "${BASH_SOURCE[0]}" > "${SWEEP_ROOT}/formal_inputs.sha256"

echo "$$" > "${SWEEP_ROOT}/pipeline.pid"
touch "${SWEEP_ROOT}/running"
complete=0
finish() {
  local code=$?
  rm -f "${SWEEP_ROOT}/running"
  echo "${code}" > "${SWEEP_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${complete} -eq 1 ]]; then
    touch "${SWEEP_ROOT}/completed"
  else
    touch "${SWEEP_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

printf 'alpha\tHOTA\tMOTA\tIDF1\tDetA\tAssA\n' \
  > "${SWEEP_ROOT}/metrics_summary.tsv"

for index in "${!ALPHAS[@]}"; do
  alpha="${ALPHAS[index]}"
  label="${LABELS[index]}"
  suffix="iwg_rg_cma_1x6_sportsmot_e200_${label}_val"
  folder="sportsmot_val_0.80_${suffix}_iwg_rg_cma_final"
  log_name="sportsmot_${label}_val_raw"
  log_path="${SWEEP_ROOT}/logs/${log_name}.log"

  require_absent "${TRACKER_OUTPUT_ROOT}/${folder}"
  require_absent "${TRACKER_OUTPUT_ROOT}/${folder}_post"

  command=(
    "${PY}" -u run.py
    --dataset SportsMOT
    --mode val
    --sequences "${SEQUENCES[@]}"
    --seed 10000
    --kf-type nsa
    --agentguard-mode iwg-rg-cma
    --agentguard-checkpoint "${CHECKPOINT}"
    --iwg-rg-cma-output final
    --iwg-rg-cma-alpha "${alpha}"
    --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --output_dir "${TRACKER_OUTPUT_ROOT}"
    --tracker-suffix "${suffix}"
    --print-per-sequence-metrics
    --resource-log "${SWEEP_ROOT}/resource_${log_name}.jsonl"
    --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${SWEEP_ROOT}/provenance/${log_name}.command.txt"
  printf '\n' >> "${SWEEP_ROOT}/provenance/${log_name}.command.txt"
  printf '%s | alpha=%s | SportsMOT val raw, no post\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${alpha}" \
    | tee -a "${SWEEP_ROOT}/stages.log"
  (cd "${TRACKER_ROOT}" && "${command[@]}") 2>&1 | tee "${log_path}"

  metric_line="$(awk '$1 == "HOTA" && $2 == "MOTA" { getline; line=$0 } END { print line }' "${log_path}")"
  if [[ -z "${metric_line}" ]]; then
    echo "Could not find TrackEval summary in ${log_path}" >&2
    exit 2
  fi
  read -r hota mota idf1 deta assa <<< "${metric_line}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${alpha}" "${hota}" "${mota}" "${idf1}" "${deta}" "${assa}" \
    | tee -a "${SWEEP_ROOT}/metrics_summary.tsv"
done

printf 'reference_alpha1\t%s\n' "${REFERENCE_ALPHA1_HOTA}" \
  | tee -a "${SWEEP_ROOT}/metrics_summary.tsv"
sha256sum "${SWEEP_ROOT}/metrics_summary.tsv" \
  > "${SWEEP_ROOT}/metrics_summary.sha256"
complete=1
printf '%s | DONE validation alpha sweep; test was not run\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  | tee -a "${SWEEP_ROOT}/stages.log"
