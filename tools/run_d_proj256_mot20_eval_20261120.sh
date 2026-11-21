#!/bin/bash
# Raw evaluation for the MOT20 D-256 (appearance_proj_dim=256) experiment.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_d_proj256_mot20_20261120"
TRACKER="$WT/3. Tracker"
CACHE="$WT/outputs/agentguard/detection_cache"
mkdir -p "$EXP/logs" "$EXP/evals"

CKPT="$EXP/groups/appearance_row_column/checkpoints/epoch_100.pt"
OUT="$EXP/evals/val_d256"
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
mkdir -p "$OUT"
started=$(date +%s)
echo "$(date '+%F %T') | D-256: val eval (raw)" | tee -a "$EXP/logs/eval_driver.log"
( cd "$TRACKER" && "$PY" -u run.py \
    --dataset MOT20 --mode all \
    --sequences MOT20-01 MOT20-05 \
    --seed 10000 --kf-type nsa --agentguard-mode shutrack-v2 \
    --shutrack-checkpoint "$CKPT" --agentguard-device cpu \
    --detection-cache-root "$CACHE" \
    --output_dir "$OUT" --tracker-suffix d256 \
    --agentguard-stats-output "$OUT/gates.jsonl" \
    --print-per-sequence-metrics \
    --eval-json "$OUT/summary.json" \
    --torch-num-threads 4 --torch-num-interop-threads 1 ) \
  > "$EXP/logs/eval_d256.log" 2>&1 || { echo "D-256: EVAL FAILED"; exit 1; }
echo "$(date '+%F %T') | D-256: eval done in $(( $(date +%s) - started ))s" \
  | tee -a "$EXP/logs/eval_driver.log"
