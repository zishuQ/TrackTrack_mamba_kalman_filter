#!/bin/bash
# Final B-recipe training: MOT20-B (from scratch) -> SportsMOT (from scratch)
# -> MOT17 (weights-only init from MOT20-B). Serial on GPU.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="$WT/outputs/agentguard/experiments/shutrack_b_final_20261120"
cd "$WT" || exit 1
mkdir -p "$EXP/logs"

common=(--group appearance_rank --device cuda --batch-size 1024 \
  --weight-decay 1e-4 --seed 42 --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 \
  --snapshot-every 20)

train_dataset() {
  local dataset="$1"; shift
  local dir="$EXP/$dataset"
  local ckpt="$dir/groups/appearance_rank/checkpoints/last.pt"
  if [ -f "$dir/groups/appearance_rank/complete.json" ]; then
    echo "$(date '+%F %T') | $dataset: already complete"
    return 0
  fi
  local resume=""
  [ -f "$ckpt" ] && resume="--resume"
  local started=$(date +%s)
  echo "$(date '+%F %T') | $dataset: training"
  "$PY" -u tools/train_supervision_groups.py "${common[@]}" --dataset "$dataset" \
    --data-root "$dir/data" --groups-root "$dir/groups" "$@" $resume || {
      echo "$(date '+%F %T') | $dataset: TRAINING FAILED"
      return 1
    }
  echo "$(date '+%F %T') | $dataset: done in $(( $(date +%s) - started ))s"
}

train_dataset MOT20 --split train --from-scratch --max-epochs 100 --lr 1e-4 || exit 1
train_dataset SportsMOT --split trainval --from-scratch --max-epochs 200 --lr 1e-4 || exit 1
M20_CKPT="$EXP/MOT20/groups/appearance_rank/checkpoints/last.pt"
[ -f "$M20_CKPT" ] || { echo "missing MOT20 checkpoint for MOT17 init"; exit 1; }
train_dataset MOT17 --split train --init "$M20_CKPT" --max-epochs 200 --lr 1e-5 || exit 1
echo "$(date '+%F %T') | TRAIN_DONE"
