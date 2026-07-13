#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-iwg_v3_oldconfig_trainall_seed42}"
RUN_ROOT="${ROOT}/outputs/agentguard/experiments/${RUN_NAME}"
DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/dataset}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LABEL_DIR="${ROOT}/outputs/agentguard/experiments/iwg_tsrm_v1_phase_a/labels"
EVENT_CACHE_ROOT="${ROOT}/outputs/agentguard/event_cache_v3_iwg_v2"
DETECTION_CACHE_ROOT="${ROOT}/outputs/agentguard/detection_cache"
TRACKER_DIR="${ROOT}/3. Tracker"
TRACKER_SUFFIX="${TRACKER_SUFFIX:-${RUN_NAME}}"
TRACKER_FOLDER="mot17_all_0.80_${TRACKER_SUFFIX}_agentguard_iwg"
RESUME_FROM="${RESUME_FROM:-}"
SEQUENCES=(
  MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN
  MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN
)

export PYTHONPATH="${ROOT}:${TRACKER_DIR}:${ROOT}/agentguard/src:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
if [[ -e "${RUN_ROOT}/running" ]]; then
  existing_pid="$(cat "${RUN_ROOT}/pipeline.pid" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Refusing active old-config diagnostic: ${RUN_ROOT}" >&2
    exit 2
  fi
  rm -f "${RUN_ROOT}/running"
fi
if [[ -e "${RUN_ROOT}/completed" ]]; then
  if [[ -n "${RESUME_FROM}" && ! -e "${RUN_ROOT}/tracking_manifest.json" ]]; then
    rm -f "${RUN_ROOT}/completed"
  else
    echo "Refusing completed old-config diagnostic: ${RUN_ROOT}" >&2
    exit 2
  fi
fi
rm -f "${RUN_ROOT}/failed"
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

git -C "${ROOT}" rev-parse HEAD > "${RUN_ROOT}/provenance/git_commit.txt"
git -C "${ROOT}" status --porcelain > "${RUN_ROOT}/provenance/git_status_porcelain.txt"
"${PY}" -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" \
  > "${RUN_ROOT}/provenance/environment.txt"
sha256sum "${LABEL_DIR}"/*_labels.json > "${RUN_ROOT}/provenance/label_files.sha256"

BUILD_COMMAND=(
  "${PY}" -m agentguard.cli build_student_v0_data
  --dataset MOT17 --mode all --max-samples 0 --split-policy train_all
  --event-cache-root "${EVENT_CACHE_ROOT}"
  --detection-cache-root "${DETECTION_CACHE_ROOT}"
  --label-dir "${LABEL_DIR}" --output-dir "${DATASET_DIR}"
  --candidate-types A --candidate-weights A:1 --max-per-candidate-type 0
)
printf '%q ' "${BUILD_COMMAND[@]}" > "${RUN_ROOT}/provenance/build.command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/build.command.txt"
if [[ -f "${DATASET_DIR}/metadata.json" ]]; then
  "${PY}" - "${DATASET_DIR}/metadata.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
expected_sequences = [
    "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
    "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]
checks = {
    "split_policy": "train_all",
    "candidate_types": ["A"],
    "train_sequences": expected_sequences,
    "val_sequences": expected_sequences,
    "label_schema_version": 3,
    "cache_schema_version": 3,
}
for field, expected in checks.items():
    if metadata.get(field) != expected:
        raise SystemExit(
            f"Refusing incompatible diagnostic dataset: {field}="
            f"{metadata.get(field)!r}, expected {expected!r}"
        )
PY
  echo "Reusing strictly validated train-all dataset: ${DATASET_DIR}" \
    | tee "${RUN_ROOT}/logs/build.log"
else
  "${BUILD_COMMAND[@]}" 2>&1 | tee "${RUN_ROOT}/logs/build.log"
fi

TRAIN_COMMAND=(
  "${PY}" -m agentguard.cli train_student_v0
  --dataset MOT17 --mode all --device cuda
  --epochs 100 --batch-size 1024 --lr 0.0001
  --num-workers 2 --seed 42 --tgr-window-stride 1
  --skip-validation --full-val-every 0
  --dataset-dir "${DATASET_DIR}" --checkpoint-dir "${CHECKPOINT_DIR}"
  --skip-tgr-training
)
if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_COMMAND+=(--iwg-resume-checkpoint "${RESUME_FROM}")
fi
printf '%q ' "${TRAIN_COMMAND[@]}" > "${RUN_ROOT}/provenance/train.command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/train.command.txt"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train.log"

IWG_CKPT="${CHECKPOINT_DIR}/iwg/iwg_last.pt"
sha256sum "${DATASET_DIR}/metadata.json" "${IWG_CKPT}" \
  > "${RUN_ROOT}/provenance/artifacts.sha256"
EVAL_COMMAND=(
  "${PY}" run.py --dataset MOT17 --mode all
  --sequences "${SEQUENCES[@]}" --seed 10000
  --agentguard-mode iwg --iwg-checkpoint "${IWG_CKPT}"
  --agentguard-device cuda --detection-cache-root "${DETECTION_CACHE_ROOT}"
  --tracker-suffix "${TRACKER_SUFFIX}" --print-per-sequence-metrics
  --resource-log "${RUN_ROOT}/resource.jsonl" --profile-every 500
)
printf '%q ' "${EVAL_COMMAND[@]}" > "${RUN_ROOT}/provenance/eval.command.txt"
printf '\n' >> "${RUN_ROOT}/provenance/eval.command.txt"
(cd "${TRACKER_DIR}" && "${EVAL_COMMAND[@]}") 2>&1 | tee "${RUN_ROOT}/logs/eval.log"

"${PY}" - "${RUN_ROOT}" "${TRACKER_FOLDER}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

run_root = Path(sys.argv[1])
tracker_folder = sys.argv[2]
root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
root = Path.cwd()
tracker_root = root / "outputs" / "3. track"
sys.path.insert(0, str(root / "3. Tracker"))
from utils.etc import evaluate_sequences

sequences = [
    "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
    "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]
args = SimpleNamespace(
    mode="all",
    data_path="/home/shang/datasets/MOT17/train",
    output_dir=str(tracker_root),
    print_per_sequence_metrics=False,
)
metrics = evaluate_sequences(args, tracker_folder, "MOT17", sequences)
hashes = {}
for sequence in sequences:
    path = tracker_root / tracker_folder / f"{sequence}.txt"
    hashes[sequence] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "schema_version": 1,
    "purpose": "diagnostic_train_all_old_iwg_configuration",
    "not_holdout": True,
    "dataset": "MOT17",
    "mode": "all",
    "sequences": sequences,
    "training_sequences": sequences,
    "tracker_seed": 10000,
    "tracker_folder": tracker_folder,
    "source_sha256": hashes,
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    **metrics,
}
(run_root / "tracking_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
pipeline_complete=1
