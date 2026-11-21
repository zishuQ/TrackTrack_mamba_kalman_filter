#!/bin/bash
# Raw evaluation for the SportsMOT D-256 (appearance_proj_dim=256) experiment.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_d_proj256_sportsmot_20261120"
TRACKER="$WT/3. Tracker"
CACHE="$WT/outputs/agentguard/detection_cache"
SEQMAP="$TRACKER/trackeval/seqmap/sportsmot/val.txt"
mapfile -t VAL_SEQUENCES < <(grep -v '^name$' "$SEQMAP")
mkdir -p "$EXP/logs" "$EXP/evals"

CKPT="$EXP/groups/appearance_row_column/checkpoints/epoch_200.pt"
OUT="$EXP/evals/val_d256"
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
mkdir -p "$OUT"
started=$(date +%s)
echo "$(date '+%F %T') | D-256 SportsMOT: val eval (raw)" | tee -a "$EXP/logs/eval_driver.log"
( cd "$TRACKER" && "$PY" -u run.py \
    --dataset SportsMOT --mode val \
    --sequences "${VAL_SEQUENCES[@]}" \
    --seed 10000 --kf-type nsa --agentguard-mode shutrack-v2 \
    --shutrack-checkpoint "$CKPT" --agentguard-device cpu \
    --detection-cache-root "$CACHE" \
    --output_dir "$OUT" --tracker-suffix d256sm \
    --agentguard-stats-output "$OUT/gates.jsonl" \
    --print-per-sequence-metrics \
    --eval-json "$OUT/summary.json" \
    --torch-num-threads 4 --torch-num-interop-threads 1 ) \
  > "$EXP/logs/eval_d256.log" 2>&1 || { echo "D-256 SportsMOT: EVAL FAILED"; exit 1; }
echo "$(date '+%F %T') | D-256 SportsMOT: eval done in $(( $(date +%s) - started ))s" \
  | tee -a "$EXP/logs/eval_driver.log"
