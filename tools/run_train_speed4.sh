#!/bin/bash
# Same-budget training for the 4-D speed group (vw/vh L1 added on top of the
# existing centre-speed recipe). SIZE_WEIGHT must be chosen from the real-batch
# gradient check first (tools/verify_speed4.py).
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN="$WT"
PY="${PYTHON:-$WT/.venv/bin/python}"
NEW="$MAIN/outputs/agentguard/experiments/shutrack_a_supervision_speed4_20261119"
SIZE_WEIGHT="${SIZE_WEIGHT:?set SIZE_WEIGHT from the gradient measurement}"
cd "$WT" || exit 1
mkdir -p "$NEW/logs"
group=motion_diou_speed4
if [ -f "$NEW/groups/$group/complete.json" ] && [ -f "$NEW/groups/$group/checkpoints/epoch_040.pt" ]; then
  echo "$(date '+%F %T') | $group: already complete"
  exit 0
fi
resume=""
if [ -f "$NEW/groups/$group/checkpoints/last.pt" ]; then
  resume="--resume"
fi
echo "$(date '+%F %T') | $group: training size_weight=$SIZE_WEIGHT"
"$PY" -u tools/train_supervision_groups.py --group "$group" \
  --data-root "$NEW/data" --groups-root "$NEW/groups" \
  --speed-weight 50 --size-weight "$SIZE_WEIGHT" $resume || {
    echo "$(date '+%F %T') | $group: TRAINING FAILED"
    exit 1
  }
echo "$(date '+%F %T') | TRAIN_DONE"
