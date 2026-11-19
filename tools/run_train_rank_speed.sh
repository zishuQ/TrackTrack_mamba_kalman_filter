#!/bin/bash
# Serial same-budget training for the two new SportsMOT supervision groups.
# 40 shard-epochs / 7,400 steps, same schedule as the overnight joint group.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN="$WT"
PY="${PYTHON:-$WT/.venv/bin/python}"
NEW="$MAIN/outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119"
cd "$WT" || exit 1
mkdir -p "$NEW/logs"

run_group() {
  local group="$1"; shift
  if [ -f "$NEW/groups/$group/complete.json" ] && [ -f "$NEW/groups/$group/checkpoints/epoch_040.pt" ]; then
    echo "$(date '+%F %T') | $group: already complete"
    return 0
  fi
  local resume=""
  if [ -f "$NEW/groups/$group/checkpoints/last.pt" ]; then
    resume="--resume"
  fi
  echo "$(date '+%F %T') | $group: training (weights: $*)"
  "$PY" -u tools/train_supervision_groups.py --group "$group" \
    --data-root "$NEW/data" --groups-root "$NEW/groups" "$@" $resume || {
      echo "$(date '+%F %T') | $group: TRAINING FAILED"
      exit 1
    }
}

run_group appearance_rank --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95
run_group motion_diou_speed --speed-weight 50
echo "$(date '+%F %T') | TRAIN_DONE"
