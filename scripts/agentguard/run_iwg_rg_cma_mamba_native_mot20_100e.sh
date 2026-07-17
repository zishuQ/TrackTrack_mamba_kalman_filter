#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/agentguard/run_iwg_rg_cma_mamba_native_mot20_100e.sh

Continue from the completed MOT20 Mamba-native event cache, build current-GT
safe-write labels and a compact six-event dataset, then train IWG+RG-CMA for
100 nominal epochs with the fixed 5-shard x 2-cycle schedule.
EOF
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "This fixed experiment script does not accept arguments." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_mamba_v1"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
DATA_RUN_ROOT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_mamba_native_mot20_train_data"
LABEL_DIR="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/MOT20/mamba_native_exp31_1_epoch55_v3_compact"
DATASET_DIR="${DATA_RUN_ROOT}/dataset"
DATA_LOG_DIR="${DATA_RUN_ROOT}/logs"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/iwg_rg_cma_mamba_native_mot20_seed42_bs1024_shard5x2"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"
mkdir -p "${DATA_LOG_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"

if [[ -e "${RUN_ROOT}/running" ]]; then
  recorded_pid="$(cat "${RUN_ROOT}/pipeline.pid" 2>/dev/null || true)"
  if [[ -n "${recorded_pid}" ]] && kill -0 "${recorded_pid}" 2>/dev/null; then
    echo "Refusing already-running experiment: ${RUN_ROOT}" >&2
    exit 2
  fi
  echo "Recovering stale running marker for PID ${recorded_pid:-unknown}."
  rm -f "${RUN_ROOT}/running"
fi
if [[ -e "${RUN_ROOT}/completed" || -e "${CHECKPOINT_DIR}/iwg_rg_cma_last.pt" ]]; then
  echo "Refusing to overwrite completed experiment: ${RUN_ROOT}" >&2
  exit 2
fi
rm -f "${RUN_ROOT}/failed" "${RUN_ROOT}/pipeline.exit_code"

echo "$$" > "${RUN_ROOT}/pipeline.pid"
touch "${RUN_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
  if [[ ${pipeline_complete} -eq 0 && ${code} -eq 0 ]]; then
    code=130
  fi
  rm -f "${RUN_ROOT}/running"
  echo "${code}" > "${RUN_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${pipeline_complete} -eq 1 ]]; then
    touch "${RUN_ROOT}/completed"
  else
    touch "${RUN_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

git -C "${ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${PROVENANCE_DIR}/git_status_porcelain.txt"

"${PY}" - "${EVENT_CACHE_ROOT}" <<'PY' \
  2>&1 | tee "${DATA_LOG_DIR}/cache_manifest_audit.log"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "MOT20" / "all"
expected = "3576c957e8e8041bfd216da52a14b0bbfb675566d234d1d298c1cb4fc51d92b9"
for sequence in ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"):
    path = root / sequence / "manifest.json"
    data = json.loads(path.read_text())
    assert data["complete"] and not data["truncated"], path
    assert data["event_source"] == "mamba_native", path
    assert data["mamba_checkpoint_sha256"] == expected, path
    assert data["processed_frames"] == data["total_sequence_frames"], path
    print(sequence, data["num_events"], data["num_matched_events"], data["num_unmatched_events"])
PY

labels_ready=false
if [[ -f "${LABEL_DIR}/summary.json" ]]; then
  labels_ready="$("${PY}" -c "import json; x=json.load(open('${LABEL_DIR}/summary.json')); print(str(bool(x.get('complete')) and x.get('motion_target_mode') == 'mamba_native').lower())")"
fi
if [[ "${labels_ready}" != "true" ]]; then
  "${PY}" -u "${ROOT}/scripts/agentguard/prepare_mot20_iwg_rg_cma_labels.py" \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --gt-root /home/shang/datasets/MOT20/train \
    --output-dir "${LABEL_DIR}" \
    --motion-label-mode mamba_native_current \
    2>&1 | tee "${DATA_LOG_DIR}/prepare_labels.log"
else
  "${PY}" -c "import json; print(json.dumps(json.load(open('${LABEL_DIR}/summary.json')), indent=2, sort_keys=True))" \
    2>&1 | tee "${DATA_LOG_DIR}/prepare_labels.log"
fi

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  "${PY}" -u -m agentguard.cli build_iwg_attn_data \
    --dataset MOT20 --mode all \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --label-dir "${LABEL_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --max-frame-gap 30 \
    2>&1 | tee "${DATA_LOG_DIR}/build_dataset.log"
else
  "${PY}" -c "import json; x=json.load(open('${DATASET_DIR}/metadata.json')); assert x['motion_target_mode'] == 'mamba_native'; print(json.dumps(x, indent=2, sort_keys=True))" \
    2>&1 | tee "${DATA_LOG_DIR}/build_dataset.log"
fi

TRAIN_COMMAND=(
  "${PY}" -u -m agentguard.cli train_iwg_attn
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
  --memory-shards 5
  --epochs-per-shard 10
  --shard-cycles 2
  --motion-target-mode mamba_native
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -u -m agentguard.cli validate_iwg_attn_checkpoint \
  --checkpoint "${CHECKPOINT_DIR}/iwg_rg_cma_last.pt" \
  --dataset-dir "${DATASET_DIR}" \
  --device cpu --max-batches 8 \
  --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"

sha256sum "${DATASET_DIR}/metadata.json" "${CHECKPOINT_DIR}/iwg_rg_cma_last.pt" \
  > "${PROVENANCE_DIR}/formal_artifacts.sha256"
pipeline_complete=1
