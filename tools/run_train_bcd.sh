#!/bin/bash
# Serial same-budget training for B (rank), C (rank + 4-D speed) and D
# (rank + next-frame appearance contention), all on the same schema-3 replay.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN="$WT"
PY="${PYTHON:-$WT/.venv/bin/python}"
NEW="$MAIN/outputs/agentguard/experiments/shutrack_a_supervision_bcd_20261119"
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
  local started=$(date +%s)
  echo "$(date '+%F %T') | $group: training"
  "$PY" -u tools/train_supervision_groups.py --group "$group" \
    --data-root "$NEW/data" --groups-root "$NEW/groups" "$@" $resume || {
      echo "$(date '+%F %T') | $group: TRAINING FAILED"
      exit 1
    }
  local ended=$(date +%s)
  echo "$(date '+%F %T') | $group: done in $((ended - started))s"
}

run_group appearance_rank \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95
run_group appearance_rank_speed4 \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 \
  --speed-weight 50 --size-weight 14
run_group appearance_row_column \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 --column-weight 50
echo "$(date '+%F %T') | TRAIN_DONE"
