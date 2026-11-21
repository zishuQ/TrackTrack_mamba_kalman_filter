#!/bin/bash
# MOT20 D-256 appearance-projection capacity experiment (single group).
# Reuses the accepted MOT20 train data; does not start the four-group driver.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_d_proj256_mot20_20261120"
DATA="$WT/outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT20/data"
mkdir -p "$EXP/logs"
cd "$WT" || exit 1

if [ -f "$EXP/groups/appearance_row_column/complete.json" ]; then
  echo "$(date '+%F %T') | D-256: already complete"
  exit 0
fi
resume=""
[ -f "$EXP/groups/appearance_row_column/checkpoints/last.pt" ] && resume="--resume"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 5 \
  > "$EXP/logs/gpu_memory.csv" 2>/dev/null &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null' EXIT

echo "$(date '+%F %T') | D-256: training"
/usr/bin/time -v "$PY" -u tools/train_supervision_groups.py \
  --dataset MOT20 --group appearance_row_column \
  --appearance-proj-dim 256 \
  --data-root "$DATA" --groups-root "$EXP/groups" --split train \
  --sequences MOT20-02 MOT20-03 \
  --from-scratch --device cuda --batch-size 1024 \
  --lr 1e-4 --weight-decay 1e-4 --seed 42 \
  --max-epochs 100 --snapshot-every 20 \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 \
  --column-weight 50 $resume \
  > "$EXP/logs/train.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "$(date '+%F %T') | D-256: TRAINING FAILED ($status)"
  exit 1
fi
echo "$(date '+%F %T') | D-256: training done"
