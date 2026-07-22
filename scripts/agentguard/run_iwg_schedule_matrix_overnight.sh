#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MASTER_NAME="${MASTER_NAME:-iwg_schedule_matrix_20260715}"
MASTER_ROOT="${ROOT}/outputs/agentguard/overnight/${MASTER_NAME}"
EXPERIMENT_ROOT="${ROOT}/outputs/agentguard/experiments"
DATASET_ROOT="${ROOT}/outputs/agentguard/datasets/iwg_rg_cma"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_ROOT="${ROOT}/outputs/3. track"

SPORTS_DATASET_DIR="${DATASET_ROOT}/SportsMOT/nsa_train_v3_compact"
SPORTS_DATA_LOG_ROOT="${MASTER_ROOT}/data_build/sportsmot"
SPORTS_LABEL_DIR="${ROOT}/outputs/agentguard/labels/iwg_rg_cma/SportsMOT/nsa_train_v3_compact"

SPORTS_OLD_NAME="iwg_rg_cma_v1_sportsmot_old_shard10x2_seed42_bs1024"
SPORTS_NEW_NAME="iwg_rg_cma_v1_sportsmot_interleaved_shard1x20_seed42_bs1024"
DANCE_NEW_NAME="iwg_rg_cma_v1_dancetrack_interleaved_shard1x20_seed42_bs1024"
MOT20_NEW_NAME="iwg_rg_cma_v1_mot20_interleaved_shard1x20_seed42_bs1024"

SPORTS_OLD_ROOT="${EXPERIMENT_ROOT}/${SPORTS_OLD_NAME}"
SPORTS_NEW_ROOT="${EXPERIMENT_ROOT}/${SPORTS_NEW_NAME}"
DANCE_NEW_ROOT="${EXPERIMENT_ROOT}/${DANCE_NEW_NAME}"
MOT20_NEW_ROOT="${EXPERIMENT_ROOT}/${MOT20_NEW_NAME}"

DANCE_DATASET_DIR="${DATASET_ROOT}/DanceTrack/nsa_train_v3_compact"
MOT20_DATASET_DIR="${DATASET_ROOT}/MOT20/nsa_all_v3_compact"

export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

mapfile -t SPORTS_VAL_SEQUENCES < <(
  sed '/^name$/d;/^$/d' "${ROOT}/3. Tracker/trackeval/seqmap/sportsmot/val.txt"
)
mapfile -t DANCE_VAL_SEQUENCES < <(
  sed '/^name$/d;/^$/d' "${ROOT}/3. Tracker/trackeval/seqmap/dancetrack/val.txt"
)
MOT20_ALL_SEQUENCES=(MOT20-01 MOT20-02 MOT20-03 MOT20-05)

mkdir -p "${MASTER_ROOT}"
if [[ -e "${MASTER_ROOT}/running" ]]; then
  echo "Refusing an already-running schedule matrix: ${MASTER_ROOT}" >&2
  exit 2
fi
if [[ -e "${MASTER_ROOT}/completed" ]]; then
  echo "Schedule matrix is already complete: ${MASTER_ROOT}" >&2
  exit 2
fi
echo "$$" > "${MASTER_ROOT}/pipeline.pid"
touch "${MASTER_ROOT}/running"
pipeline_complete=0
finish() {
  local code=$?
  if [[ ${pipeline_complete} -eq 0 && ${code} -eq 0 ]]; then
    code=130
  fi
  rm -f "${MASTER_ROOT}/running"
  echo "${code}" > "${MASTER_ROOT}/pipeline.exit_code"
  if [[ ${code} -eq 0 && ${pipeline_complete} -eq 1 ]]; then
    touch "${MASTER_ROOT}/completed"
  else
    touch "${MASTER_ROOT}/failed"
  fi
  exit "${code}"
}
trap finish EXIT

stage() {
  local message="$1"
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" | tee -a "${MASTER_ROOT}/stages.log"
}

git -C "${ROOT}" rev-parse HEAD > "${MASTER_ROOT}/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${MASTER_ROOT}/git_status_porcelain.txt"
"${PY}" -c "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" \
  > "${MASTER_ROOT}/environment.txt"

ensure_detection_cache() {
  local dataset="$1"
  local split="$2"
  local expected_sequences="$3"
  local source_pickle="$4"
  local target_pickle="$5"
  local log_path="$6"
  local manifest="${DETECTION_CACHE_ROOT}/${dataset}/${split}/manifest.json"
  local ready=false
  if [[ -f "${manifest}" ]]; then
    ready="$("${PY}" -c "import json; x=json.load(open('${manifest}')); print(str(bool(x.get('complete')) and len(x.get('sequences', {})) == ${expected_sequences} and int(x.get('reid_dim', 0)) == 2048).lower())")"
  fi
  if [[ "${ready}" == "true" ]]; then
    "${PY}" -c "import json; x=json.load(open('${manifest}')); print('${dataset}/${split}:', len(x['sequences']), 'sequences, reid_dim', x['reid_dim'])" \
      2>&1 | tee "${log_path}"
    return
  fi
  "${PY}" -u "${ROOT}/scripts/agentguard/build_detection_mmap_cache.py" \
    --dataset "${dataset}" --split "${split}" \
    --source-pickle "${source_pickle}" --target-pickle "${target_pickle}" \
    --output-root "${DETECTION_CACHE_ROOT}" \
    2>&1 | tee "${log_path}"
}

prepare_sportsmot_data() {
  mkdir -p "${SPORTS_DATA_LOG_ROOT}/logs"
  stage "SportsMOT data: build/verify train and val mmap detection caches"
  ensure_detection_cache \
    SportsMOT train 45 \
    "${ROOT}/outputs/2. det_feat/sportsmot_train_0.95.pickle" \
    "${ROOT}/outputs/2. det_feat/sportsmot_train_0.80.pickle" \
    "${SPORTS_DATA_LOG_ROOT}/logs/detection_train.log"
  ensure_detection_cache \
    SportsMOT val 45 \
    "${ROOT}/outputs/2. det_feat/sportsmot_val_0.95.pickle" \
    "${ROOT}/outputs/2. det_feat/sportsmot_val_0.80.pickle" \
    "${SPORTS_DATA_LOG_ROOT}/logs/detection_val.log"

  stage "SportsMOT data: cache complete train timelines"
  "${PY}" -u -m agentguard.cli cache_events \
    --dataset SportsMOT --mode train_custom \
    --event-cache-root "${EVENT_CACHE_ROOT}" \
    --detection-cache-root "${DETECTION_CACHE_ROOT}" \
    --data-dir /home/shang/datasets/ \
    2>&1 | tee "${SPORTS_DATA_LOG_ROOT}/logs/cache_events.log"

  local labels_ready=false
  if [[ -f "${SPORTS_LABEL_DIR}/summary.json" ]]; then
    labels_ready="$("${PY}" -c "import json; x=json.load(open('${SPORTS_LABEL_DIR}/summary.json')); print(str(bool(x.get('complete')) and len(x.get('sequences', [])) == 45 and int(x.get('retained_labels', 0)) > 0).lower())")"
  fi
  stage "SportsMOT data: build/verify safe-direct labels"
  if [[ "${labels_ready}" != "true" ]]; then
    "${PY}" -u "${ROOT}/scripts/agentguard/prepare_sportsmot_iwg_rg_cma_labels.py" \
      --event-cache-root "${EVENT_CACHE_ROOT}" \
      --detection-cache-root "${DETECTION_CACHE_ROOT}" \
      --gt-root /home/shang/datasets/SportsMOT/dataset/train \
      --output-dir "${SPORTS_LABEL_DIR}" \
      2>&1 | tee "${SPORTS_DATA_LOG_ROOT}/logs/prepare_labels.log"
  else
    "${PY}" -c "import json; x=json.load(open('${SPORTS_LABEL_DIR}/summary.json')); print('complete labels', x['retained_labels'], 'sequences', len(x['sequences']))" \
      2>&1 | tee "${SPORTS_DATA_LOG_ROOT}/logs/prepare_labels.log"
  fi

  stage "SportsMOT data: build/verify streaming six-event dataset"
  if [[ ! -f "${SPORTS_DATASET_DIR}/metadata.json" ]]; then
    "${PY}" -m agentguard.cli build_iwg_rg_cma_data \
      --dataset SportsMOT --mode train \
      --event-cache-root "${EVENT_CACHE_ROOT}" \
      --detection-cache-root "${DETECTION_CACHE_ROOT}" \
      --label-dir "${SPORTS_LABEL_DIR}" --output-dir "${SPORTS_DATASET_DIR}" \
      --max-frame-gap 30 \
      2>&1 | tee "${SPORTS_DATA_LOG_ROOT}/logs/build.log"
  else
    "${PY}" -c "from agentguard.datasets.iwg_rg_cma_dataset import StreamingIWGRGCMADataset; d=StreamingIWGRGCMADataset('${SPORTS_DATASET_DIR}', max_samples=1); print(d.metadata['dataset_sha256'], d.metadata['num_train_samples']); d.close()" \
      2>&1 | tee "${SPORTS_DATA_LOG_ROOT}/logs/build.log"
  fi
}

train_schedule() {
  local run_root="$1"
  local dataset_dir="$2"
  local epochs_per_shard="$3"
  local shard_cycles="$4"
  local schedule_name="$5"
  local checkpoint_dir="${run_root}/checkpoints"
  local checkpoint="${checkpoint_dir}/iwg_rg_cma_epoch100.pt"
  mkdir -p "${run_root}/logs" "${run_root}/provenance" "${checkpoint_dir}"
  printf '{"schedule":"%s","memory_shards":5,"epochs_per_shard":%d,"shard_cycles":%d,"epochs":100,"batch_size":1024,"seed":42}\n' \
    "${schedule_name}" "${epochs_per_shard}" "${shard_cycles}" \
    > "${run_root}/experiment_config.json"
  git -C "${ROOT}" rev-parse HEAD > "${run_root}/provenance/git_commit.txt"
  git -C "${ROOT}" status --porcelain > "${run_root}/provenance/git_status_porcelain.txt"
  if [[ ! -f "${checkpoint}" ]]; then
    stage "Train ${run_root##*/}: ${schedule_name}"
    local command=(
      "${PY}" -m agentguard.cli train_iwg_rg_cma
      --dataset-dir "${dataset_dir}" --checkpoint-dir "${checkpoint_dir}"
      --device cuda --epochs 100 --batch-size 1024 --num-workers 4
      --lr 0.0001 --weight-decay 0.0001 --warmup-epochs 1
      --grad-clip 1.0 --seed 42
      --memory-shards 5 --epochs-per-shard "${epochs_per_shard}"
      --shard-cycles "${shard_cycles}"
    )
    printf '%q ' "${command[@]}" > "${run_root}/provenance/train.command.txt"
    printf '\n' >> "${run_root}/provenance/train.command.txt"
    "${command[@]}" 2>&1 | tee "${run_root}/logs/train.log"
  else
    stage "Train ${run_root##*/}: checkpoint already complete, skip"
  fi
  "${PY}" -m agentguard.cli validate_iwg_rg_cma_checkpoint \
    --checkpoint "${checkpoint}" --dataset-dir "${dataset_dir}" \
    --device cpu --max-batches 8 --output "${run_root}/checkpoint_validation.json" \
    2>&1 | tee "${run_root}/logs/checkpoint_validation.log"
  sha256sum "${dataset_dir}/metadata.json" "${checkpoint}" \
    > "${run_root}/provenance/formal_artifacts.sha256"
  touch "${run_root}/training_completed"
}

validate_schedule() {
  local run_root="$1"
  local dataset="$2"
  local mode="$3"
  local result_prefix="$4"
  shift 4
  local epochs=("$@")
  local sequences=()
  case "${dataset}" in
    SportsMOT) sequences=("${SPORTS_VAL_SEQUENCES[@]}") ;;
    DanceTrack) sequences=("${DANCE_VAL_SEQUENCES[@]}") ;;
    MOT20) sequences=("${MOT20_ALL_SEQUENCES[@]}") ;;
    *) echo "unsupported validation dataset: ${dataset}" >&2; return 2 ;;
  esac
  local run_name="${run_root##*/}"
  for epoch in "${epochs[@]}"; do
    local tag
    tag="$(printf '%03d' "${epoch}")"
    local checkpoint="${run_root}/checkpoints/iwg_rg_cma_epoch${tag}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "missing validation checkpoint: ${checkpoint}" >&2
      return 2
    fi
    for gate in base final; do
      local suffix="${run_name}_e${tag}_${gate}_validation"
      local tracker_name="${result_prefix}_0.80_${suffix}_agentguard_iwg_rg_cma_${gate}"
      local raw_log="${run_root}/logs/eval_e${tag}_${gate}_raw.log"
      local post_log="${run_root}/logs/eval_e${tag}_${gate}_post.log"
      if [[ -s "${raw_log}" && -s "${post_log}" ]]; then
        stage "Validate ${run_name}: epoch${tag} ${gate} already complete, skip"
        continue
      fi
      stage "Validate ${run_name}: epoch${tag} ${gate} raw+post"
      local command=(
        "${PY}" run.py --dataset "${dataset}" --mode "${mode}"
        --sequences "${sequences[@]}" --seed 10000
        --agentguard-mode iwg-rg-cma --legacy-output-naming --agentguard-checkpoint "${checkpoint}"
        --iwg-rg-cma-output "${gate}" --agentguard-device cpu
        --detection-cache-root "${DETECTION_CACHE_ROOT}"
        --tracker-suffix "${suffix}" --use_post --skip-eval
        --resource-log "${run_root}/resource_e${tag}_${gate}.jsonl"
        --profile-every 500
      )
      printf '%q ' "${command[@]}" > "${run_root}/provenance/eval_e${tag}_${gate}.command.txt"
      printf '\n' >> "${run_root}/provenance/eval_e${tag}_${gate}.command.txt"
      (cd "${ROOT}/3. Tracker" && "${command[@]}") \
        2>&1 | tee "${run_root}/logs/track_e${tag}_${gate}.log"
      (cd "${ROOT}/3. Tracker" && "${PY}" eval_only.py \
        --tracker_name "${tracker_name}" --dataset "${dataset}" --mode "${mode}" \
        --sequences "${sequences[@]}" --no_post) \
        2>&1 | tee "${raw_log}"
      (cd "${ROOT}/3. Tracker" && "${PY}" eval_only.py \
        --tracker_name "${tracker_name}" --dataset "${dataset}" --mode "${mode}" \
        --sequences "${sequences[@]}") \
        2>&1 | tee "${post_log}"
    done
  done
  "${PY}" "${ROOT}/scripts/agentguard/summarize_iwg_schedule_validation.py" \
    --run-root "${run_root}" \
    2>&1 | tee "${run_root}/logs/evaluation_summary.log"
  touch "${run_root}/validation_completed"
}

prepare_sportsmot_data

train_schedule "${SPORTS_OLD_ROOT}" "${SPORTS_DATASET_DIR}" 10 2 old_contiguous_10_epochs
validate_schedule "${SPORTS_OLD_ROOT}" SportsMOT val sportsmot_val 50 100

train_schedule "${SPORTS_NEW_ROOT}" "${SPORTS_DATASET_DIR}" 1 20 interleaved_one_epoch
validate_schedule "${SPORTS_NEW_ROOT}" SportsMOT val sportsmot_val 25 50 75 100

if [[ ! -f "${DANCE_DATASET_DIR}/metadata.json" ]]; then
  echo "DanceTrack dataset not found: ${DANCE_DATASET_DIR}" >&2
  exit 2
fi
train_schedule "${DANCE_NEW_ROOT}" "${DANCE_DATASET_DIR}" 1 20 interleaved_one_epoch
validate_schedule "${DANCE_NEW_ROOT}" DanceTrack val dance_val 25 50 75 100

if [[ ! -f "${MOT20_DATASET_DIR}/metadata.json" ]]; then
  echo "MOT20 dataset not found: ${MOT20_DATASET_DIR}" >&2
  exit 2
fi
train_schedule "${MOT20_NEW_ROOT}" "${MOT20_DATASET_DIR}" 1 20 interleaved_one_epoch
validate_schedule "${MOT20_NEW_ROOT}" MOT20 all mot20_all 25 50 75 100

"${PY}" -c "import json; from pathlib import Path; roots=[Path(x) for x in ['${SPORTS_OLD_ROOT}','${SPORTS_NEW_ROOT}','${DANCE_NEW_ROOT}','${MOT20_NEW_ROOT}']]; data={p.name:json.loads((p/'evaluation_summary.json').read_text()) for p in roots}; Path('${MASTER_ROOT}/evaluation_matrix.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\\n'); print(json.dumps(data,indent=2,sort_keys=True))" \
  2>&1 | tee "${MASTER_ROOT}/evaluation_matrix.log"

pipeline_complete=1
stage "All schedule-matrix training and validation tasks completed"
