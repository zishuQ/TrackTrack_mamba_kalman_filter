#!/bin/bash
# Raw val evaluation for the SportsMOT B/D ablation checkpoints.
# Same detection cache, seed, NSA KF and raw (no post) protocol for both.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120"
TRACKER="$WT/3. Tracker"
CACHE="$WT/outputs/agentguard/detection_cache"
SEQMAP="$TRACKER/trackeval/seqmap/sportsmot/val.txt"
mapfile -t VAL_SEQUENCES < <(grep -v '^name$' "$SEQMAP")
mkdir -p "$EXP/logs"

eval_group() {
  local group="$1" suffix="$2"
  local ckpt="$EXP/groups/$group/checkpoints/epoch_200.pt"
  [ -f "$ckpt" ] || { echo "$group: missing $ckpt"; return 1; }
  local out="$EXP/evals/val_$suffix"
  mkdir -p "$out"
  echo "$(date '+%F %T') | $group: val eval (raw)" | tee -a "$EXP/logs/eval_driver.log"
  local started
  started=$(date +%s)
  ( cd "$TRACKER" && "$PY" -u run.py \
      --dataset SportsMOT --mode val \
      --sequences "${VAL_SEQUENCES[@]}" \
      --seed 10000 --kf-type nsa --agentguard-mode shutrack-v2 \
      --shutrack-checkpoint "$ckpt" --agentguard-device cpu \
      --detection-cache-root "$CACHE" \
      --output_dir "$out" --tracker-suffix "$suffix" \
      --agentguard-stats-output "$out/gates.jsonl" \
      --print-per-sequence-metrics \
      --eval-json "$out/summary.json" \
      --torch-num-threads 4 --torch-num-interop-threads 1 ) \
    > "$EXP/logs/eval_$suffix.log" 2>&1 || { echo "$suffix: EVAL FAILED"; return 1; }
  echo "$(date '+%F %T') | $suffix: eval done in $(( $(date +%s) - started ))s" \
    | tee -a "$EXP/logs/eval_driver.log"
}

eval_group appearance_rank b
eval_group appearance_row_column d
echo "$(date '+%F %T') | EVAL_DONE"
