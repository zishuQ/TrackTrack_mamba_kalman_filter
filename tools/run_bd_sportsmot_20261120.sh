#!/bin/bash
# SportsMOT B/D paired ablation on the shared full-write column data.
# B (appearance_rank) then D (appearance_row_column, column_weight=50), serial.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120"
mkdir -p "$EXP/logs"
cd "$WT" || exit 1

common=(--dataset SportsMOT --split train --from-scratch --device cuda \
  --batch-size 1024 --lr 1e-4 --weight-decay 1e-4 --seed 42 \
  --max-epochs 200 --snapshot-every 20 \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 \
  --data-root "$EXP/data" --groups-root "$EXP/groups")

train_group() {
  local group="$1"; shift
  local dir="$EXP/groups/$group"
  if [ -f "$dir/complete.json" ]; then
    echo "$(date '+%F %T') | $group: already complete"
    return 0
  fi
  local resume=""
  [ -f "$dir/checkpoints/last.pt" ] && resume="--resume"
  local started
  started=$(date +%s)
  echo "$(date '+%F %T') | $group: training"
  "$PY" -u tools/train_supervision_groups.py --group "$group" "${common[@]}" "$@" \
    > "$EXP/logs/train_${group}.log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "$(date '+%F %T') | $group: TRAINING FAILED (status $status)"
    return 1
  fi
  echo "$(date '+%F %T') | $group: done in $(( $(date +%s) - started ))s"
}

train_group appearance_rank || exit 1
train_group appearance_row_column --column-weight 50 || exit 1
echo "$(date '+%F %T') | TRAIN_DONE"
