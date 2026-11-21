#!/bin/bash
# MOT20/MOT17 B/D paired training on the custom split.
# MOT20: B and D from scratch, 100 shard-epochs (25 full passes).
# MOT17: B and D both initialized from the same MOT20-B final weights,
#        weights-only with refit stats, 200 shard-epochs (50 full passes).
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
EXP="${EXPERIMENT_ROOT:-$WT/outputs/agentguard/experiments/shutrack_bd_mot_20261120}"
DATA_ROOT="${TRAIN_DATA_ROOT:-$EXP}"
cd "$WT" || exit 1
mkdir -p "$EXP/logs"

MOT20_SEQ=(MOT20-02 MOT20-03)
MOT17_SEQ=(MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN)

train_group() {
  local dataset="$1" group="$2" epochs="$3" lr="$4"; shift 4
  local dir="$EXP/$dataset/groups/$group"
  if [ -f "$dir/complete.json" ]; then
    echo "$(date '+%F %T') | $dataset/$group: already complete"
    return 0
  fi
  local resume=""
  [ -f "$dir/checkpoints/last.pt" ] && resume="--resume"
  local started
  started=$(date +%s)
  echo "$(date '+%F %T') | $dataset/$group: training"
  "$PY" -u tools/train_supervision_groups.py \
    --dataset "$dataset" --group "$group" --split train \
    --data-root "$DATA_ROOT/$dataset/data" --groups-root "$EXP/$dataset/groups" \
    --device cuda --batch-size 1024 --lr "$lr" --weight-decay 1e-4 \
    --seed 42 --max-epochs "$epochs" --snapshot-every 20 \
    --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 \
    "$@" $resume > "$EXP/logs/train_${dataset}_${group}.log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "$(date '+%F %T') | $dataset/$group: TRAINING FAILED ($status)"
    return 1
  fi
  echo "$(date '+%F %T') | $dataset/$group: done in $(( $(date +%s) - started ))s"
}

train_group MOT20 appearance_rank 100 1e-4 --from-scratch --sequences "${MOT20_SEQ[@]}" || exit 1
train_group MOT20 appearance_row_column 100 1e-4 --from-scratch --column-weight 50 --sequences "${MOT20_SEQ[@]}" || exit 1

INIT="$EXP/MOT20/groups/appearance_rank/checkpoints/last.pt"
[ -f "$INIT" ] || { echo "missing MOT20-B final weights: $INIT"; exit 1; }
train_group MOT17 appearance_rank 200 1e-5 --init "$INIT" --sequences "${MOT17_SEQ[@]}" || exit 1
train_group MOT17 appearance_row_column 200 1e-5 --init "$INIT" --column-weight 50 --sequences "${MOT17_SEQ[@]}" || exit 1
echo "$(date '+%F %T') | TRAIN_DONE"
