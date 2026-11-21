#!/bin/bash
# Raw evaluation for the MOT20/MOT17 B/D checkpoints on the custom val videos.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="${EXPERIMENT_ROOT:-$WT/outputs/agentguard/experiments/shutrack_bd_mot_20261120}"
TRACKER="$WT/3. Tracker"
CACHE="$WT/outputs/agentguard/detection_cache"
mkdir -p "$EXP/logs"

eval_group() {
  local dataset="$1" group="$2" suffix="$3" epochs="$4"; shift 4
  local sequences=("$@")
  local ckpt="$EXP/$dataset/groups/$group/checkpoints/epoch_$(printf '%03d' "$epochs").pt"
  [ -f "$ckpt" ] || { echo "$dataset/$group: missing $ckpt"; return 1; }
  local out="$EXP/$dataset/evals/val_$suffix"
  mkdir -p "$out"
  local started
  started=$(date +%s)
  echo "$(date '+%F %T') | $dataset/$suffix: val eval (raw)" | tee -a "$EXP/logs/eval_driver.log"
  ( cd "$TRACKER" && "$PY" -u run.py \
      --dataset "$dataset" --mode all \
      --sequences "${sequences[@]}" \
      --seed 10000 --kf-type nsa --agentguard-mode shutrack-v2 \
      --shutrack-checkpoint "$ckpt" --agentguard-device cpu \
      --detection-cache-root "$CACHE" \
      --output_dir "$out" --tracker-suffix "$suffix" \
      --agentguard-stats-output "$out/gates.jsonl" \
      --print-per-sequence-metrics \
      --eval-json "$out/summary.json" \
      --torch-num-threads 4 --torch-num-interop-threads 1 ) \
    > "$EXP/logs/eval_${dataset}_${suffix}.log" 2>&1 || { echo "$dataset/$suffix: EVAL FAILED"; return 1; }
  echo "$(date '+%F %T') | $dataset/$suffix: eval done in $(( $(date +%s) - started ))s" \
    | tee -a "$EXP/logs/eval_driver.log"
}

eval_group MOT20 appearance_rank mot20b 100 MOT20-01 MOT20-05 || exit 1
eval_group MOT20 appearance_row_column mot20d 100 MOT20-01 MOT20-05 || exit 1
eval_group MOT17 appearance_rank mot17b 200 MOT17-09-FRCNN MOT17-10-FRCNN || exit 1
eval_group MOT17 appearance_row_column mot17d 200 MOT17-09-FRCNN MOT17-10-FRCNN || exit 1
echo "$(date '+%F %T') | EVAL_DONE"
