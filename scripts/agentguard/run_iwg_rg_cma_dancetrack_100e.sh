#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_rg_cma_v1_dancetrack_train_seed42_bs1024_shard20x2}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
LABEL_DIR="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/DanceTrack/nsa_train_v3_compact"
DATASET_DIR="${DATASET_DIR:-${ROOT}/outputs/agentguard/datasets/iwg_rg_cma/DanceTrack/nsa_train_v3_compact}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
PROVENANCE_DIR="${RUN_ROOT}/provenance"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
SOURCE_PICKLE="${ROOT}/outputs/2. det_feat/dance_train_0.95.pickle"
TARGET_PICKLE="${ROOT}/outputs/2. det_feat/dance_train_0.80.pickle"
CHECKPOINT="${CHECKPOINT_DIR}/iwg_rg_cma_last.pt"
VAL_SEQUENCES=(
  dancetrack0004 dancetrack0005 dancetrack0007 dancetrack0010
  dancetrack0014 dancetrack0018 dancetrack0019 dancetrack0025
  dancetrack0026 dancetrack0030 dancetrack0034 dancetrack0035
  dancetrack0041 dancetrack0043 dancetrack0047 dancetrack0058
  dancetrack0063 dancetrack0065 dancetrack0073 dancetrack0077
  dancetrack0079 dancetrack0081 dancetrack0090 dancetrack0094
  dancetrack0097
)

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"
if [[ -e "${RUN_ROOT}/running" ]]; then
  echo "Refusing an already-running DanceTrack RG-CMA experiment: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/completed" || -e "${CHECKPOINT}" ]]; then
  echo "Refusing to overwrite an existing formal DanceTrack run: ${RUN_ROOT}" >&2
  exit 2
fi
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
sha256sum \
  "${ROOT}/agentguard/src/agentguard/models/event_encoder.py" \
  "${ROOT}/agentguard/src/agentguard/models/iwg.py" \
  "${ROOT}/agentguard/src/agentguard/models/iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/data/cache_reader.py" \
  "${ROOT}/agentguard/src/agentguard/data/compact_iwg_labels.py" \
  "${ROOT}/agentguard/src/agentguard/datasets/iwg_attn_dataset.py" \
  "${ROOT}/agentguard/src/agentguard/training/loss_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/training/train_iwg_rg_cma.py" \
  "${ROOT}/agentguard/src/agentguard/v0_pipeline.py" \
  > "${PROVENANCE_DIR}/training_sources.sha256"
"${PY}" -c "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" \
  > "${PROVENANCE_DIR}/environment.txt"

DETECTION_READY=false
if [[ -f "${DETECTION_CACHE_ROOT}/DanceTrack/train/manifest.json" ]]; then
  DETECTION_READY="$("${PY}" -c "import json; x=json.load(open('${DETECTION_CACHE_ROOT}/DanceTrack/train/manifest.json')); print(str(bool(x.get('complete')) and len(x.get('sequences', {})) == 40).lower())")"
fi
if [[ "${DETECTION_READY}" != "true" ]]; then
  "${PY}" -u "${ROOT}/scripts/agentguard/build_detection_mmap_cache.py" \
    --dataset DanceTrack --split train \
    --source-pickle "${SOURCE_PICKLE}" --target-pickle "${TARGET_PICKLE}" \
    --output-root "${DETECTION_CACHE_ROOT}" \
    2>&1 | tee "${LOG_DIR}/detection_cache.log"
else
  "${PY}" -c "import json; x=json.load(open('${DETECTION_CACHE_ROOT}/DanceTrack/train/manifest.json')); print('complete sequences', len(x['sequences']), 'reid_dim', x['reid_dim'])" \
    2>&1 | tee "${LOG_DIR}/detection_cache.log"
fi

"${PY}" -u -m agentguard.cli cache_events \
  --dataset DanceTrack --mode train_custom \
  --event-cache-root "${EVENT_CACHE_ROOT}" \
  --detection-cache-root "${DETECTION_CACHE_ROOT}" \
  --data-dir /home/shang/datasets \
  2>&1 | tee "${LOG_DIR}/cache_events.log"

LABELS_READY=false
if [[ -f "${LABEL_DIR}/summary.json" ]]; then
  LABELS_READY="$("${PY}" -c "import json; x=json.load(open('${LABEL_DIR}/summary.json')); print(str(bool(x.get('complete')) and len(x.get('sequences', [])) == 40).lower())")"
fi
if [[ "${LABELS_READY}" != "true" ]]; then
  "${PY}" -u "${ROOT}/scripts/agentguard/prepare_dancetrack_iwg_rg_cma_labels.py" \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --gt-root /home/shang/datasets/DanceTrack/train \
    --output-dir "${LABEL_DIR}" \
    2>&1 | tee "${LOG_DIR}/prepare_labels.log"
else
  "${PY}" -c "import json; x=json.load(open('${LABEL_DIR}/summary.json')); assert x['complete'] and x['retained_labels'] > 0; print(json.dumps(x, indent=2, sort_keys=True))" \
    2>&1 | tee "${LOG_DIR}/prepare_labels.log"
fi

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  "${PY}" -m agentguard.cli build_iwg_attn_data \
    --dataset DanceTrack --mode train \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --label-dir "${LABEL_DIR}" --output-dir "${DATASET_DIR}" \
    --max-frame-gap 30 2>&1 | tee "${LOG_DIR}/build.log"
else
  "${PY}" -c "from agentguard.datasets.iwg_attn_dataset import StreamingIWGAttnDataset; d=StreamingIWGAttnDataset('${DATASET_DIR}', max_samples=1); print(d.metadata['dataset_sha256']); d.close()" \
    2>&1 | tee "${LOG_DIR}/build.log"
fi

TRAIN_COMMAND=(
  "${PY}" -m agentguard.cli train_iwg_attn
  --dataset-dir "${DATASET_DIR}" --checkpoint-dir "${CHECKPOINT_DIR}"
  --device cuda --epochs 100 --batch-size 1024 --num-workers 4
  --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1
  --grad-clip 1.0 --seed 42
  --memory-shards 5 --epochs-per-shard 10 --shard-cycles 2
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/train.command.txt"
printf '\n' >> "${PROVENANCE_DIR}/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

"${PY}" -m agentguard.cli validate_iwg_attn_checkpoint \
  --checkpoint "${CHECKPOINT}" --dataset-dir "${DATASET_DIR}" \
  --device cpu --max-batches 8 --output "${RUN_ROOT}/checkpoint_validation.json" \
  2>&1 | tee "${LOG_DIR}/checkpoint_validation.log"
sha256sum "${DATASET_DIR}/metadata.json" "${CHECKPOINT}" \
  > "${PROVENANCE_DIR}/formal_artifacts.sha256"

run_val_case() {
  local output="$1"
  local suffix="${RUN_NAME}_${output}_val"
  local command=(
    "${PY}" run.py --dataset DanceTrack --mode val
    --sequences "${VAL_SEQUENCES[@]}" --seed 10000 --legacy-output-naming
    --agentguard-mode iwg-attn --agentguard-checkpoint "${CHECKPOINT}"
    --iwg-attn-output "${output}" --agentguard-device cpu
    --detection-cache-root "${DETECTION_CACHE_ROOT}"
    --tracker-suffix "${suffix}" --use_post --skip-eval
    --resource-log "${RUN_ROOT}/resource_${output}_val.jsonl" --profile-every 500
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/eval_${output}.command.txt"
  printf '\n' >> "${PROVENANCE_DIR}/eval_${output}.command.txt"
  (cd "${ROOT}/3. Tracker" && "${command[@]}") \
    2>&1 | tee "${LOG_DIR}/track_${output}_val.log"

  local tracker_name="dance_val_0.80_${suffix}_agentguard_iwg_attn_${output}"
  (cd "${ROOT}/3. Tracker" && "${PY}" eval_only.py \
    --tracker_name "${tracker_name}" --dataset DanceTrack --mode val --no_post) \
    2>&1 | tee "${LOG_DIR}/eval_${output}_raw.log"
  (cd "${ROOT}/3. Tracker" && "${PY}" eval_only.py \
    --tracker_name "${tracker_name}" --dataset DanceTrack --mode val) \
    2>&1 | tee "${LOG_DIR}/eval_${output}_post.log"
}

run_val_case base
run_val_case final

pipeline_complete=1
