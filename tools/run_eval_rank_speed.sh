#!/bin/bash
# Evaluate the two new supervision groups on the same 45-sequence SportsMOT
# val raw setting as the overnight groups (seed 10000, NSA, shared caches).
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN="$WT"
PY="${PYTHON:-$WT/.venv/bin/python}"
NEW="$MAIN/outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119"
CACHE="$MAIN/outputs/agentguard/detection_cache"
mkdir -p "$NEW/evals"
mapfile -t VAL < <(grep -v '^name$' "$WT/3. Tracker/trackeval/seqmap/sportsmot/val.txt")

for group in appearance_rank motion_diou_speed; do
  count=$(find "$NEW/evals/${group}_val" -name '*.txt' 2>/dev/null | wc -l)
  if [ ! -f "$NEW/evals/${group}_val.log" ] || [ "$count" -lt 45 ]; then
    echo "$(date '+%F %T') | $group: evaluating ${#VAL[@]} sequences"
    ( cd "$WT/3. Tracker" && "$PY" -u run.py --dataset SportsMOT --mode val \
        --sequences "${VAL[@]}" \
        --seed 10000 --kf-type nsa --agentguard-mode shutrack-v2 \
        --shutrack-checkpoint "$NEW/groups/$group/checkpoints/epoch_040.pt" \
        --agentguard-device cpu \
        --detection-cache-root "$CACHE" \
        --output_dir "$NEW/evals/${group}_val" --tracker-suffix "${group}_val" \
        --print-per-sequence-metrics \
        --torch-num-threads 4 --torch-num-interop-threads 1 ) \
        > "$NEW/evals/${group}_val.log" 2>&1 || { echo "$group: EVALUATION FAILED"; exit 1; }
  else
    echo "$(date '+%F %T') | $group: already evaluated"
  fi
done
echo "$(date '+%F %T') | EVAL_DONE"
