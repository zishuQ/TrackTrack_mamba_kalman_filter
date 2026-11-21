#!/bin/bash
# D-hinge: same D configuration, column penalty changes softplus -> relu.
# Reuses the accepted 45-sequence shared SportsMOT train data.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_d_hinge_sportsmot_20261120"
DATA="$WT/outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data"
mkdir -p "$EXP/logs" "$EXP/groups" "$EXP/evals"
cd "$WT" || exit 1

if [ -f "$EXP/groups/appearance_row_column/complete.json" ]; then
  echo "$(date '+%F %T') | appearance_row_column_hinge: already complete"
  exit 0
fi
resume=""
[ -f "$EXP/groups/appearance_row_column/checkpoints/last.pt" ] && resume="--resume"
started=$(date +%s)
echo "$(date '+%F %T') | appearance_row_column_hinge: training"
"$PY" -u tools/train_supervision_groups.py \
  --group appearance_row_column --column-hinge \
  --dataset SportsMOT --split train --from-scratch --device cuda \
  --batch-size 1024 --lr 1e-4 --weight-decay 1e-4 --seed 42 \
  --max-epochs 200 --snapshot-every 20 \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 \
  --column-weight 50 \
  --data-root "$DATA" --groups-root "$EXP/groups" $resume \
  > "$EXP/logs/train_appearance_row_column_hinge.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "$(date '+%F %T') | appearance_row_column_hinge: TRAINING FAILED (status $status)"
  exit 1
fi
echo "$(date '+%F %T') | appearance_row_column_hinge: done in $(( $(date +%s) - started ))s"
