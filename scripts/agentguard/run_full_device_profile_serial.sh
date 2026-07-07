#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shang/workspace/TrackTrack
OUT="$REPO/outputs/agentguard/profile_device_cmp_full_serial"
PY="$REPO/.venv/bin/python"

mkdir -p "$OUT"
export PYTHONPATH="$REPO:$REPO/3. Tracker:$REPO/agentguard/src"
cd "$REPO/3. Tracker"

echo "=== CPU full MOT17 FRCNN start $(date -Is) ===" | tee "$OUT/cpu.log"
/usr/bin/time -f "WALL_SECONDS=%e" -o "$OUT/cpu.time" \
  "$PY" run.py \
    --dataset MOT17 \
    --mode all \
    --seed 42 \
    --sequences MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
    --skip-eval \
    --output_dir "$OUT/cpu_tracking" \
    --detection-cache-root "$REPO/outputs/agentguard/detection_cache" \
    --resource-log "$OUT/cpu_resource.jsonl" \
    --profile-every 500 \
    --agentguard-mode full \
    --iwg-checkpoint "$REPO/outputs/agentguard/checkpoints/MOT17/all/iwg/iwg_best.pt" \
    --tgr-checkpoint "$REPO/outputs/agentguard/checkpoints/MOT17/all/tgr/tgr_best.pt" \
    --agentguard-device cpu \
    --agentguard-tgr-frame-stride 4 \
    --agentguard-replay-diff-threshold 0.15 \
    --tracker-suffix profile_full_cpu \
    --print-per-sequence-metrics 2>&1 | tee -a "$OUT/cpu.log"
cat "$OUT/cpu.time" | tee -a "$OUT/cpu.log"
echo "=== CPU full MOT17 FRCNN end $(date -Is) ===" | tee -a "$OUT/cpu.log"

echo "=== CUDA full MOT17 FRCNN start $(date -Is) ===" | tee "$OUT/cuda.log"
/usr/bin/time -f "WALL_SECONDS=%e" -o "$OUT/cuda.time" \
  "$PY" run.py \
    --dataset MOT17 \
    --mode all \
    --seed 42 \
    --sequences MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
    --skip-eval \
    --output_dir "$OUT/cuda_tracking" \
    --detection-cache-root "$REPO/outputs/agentguard/detection_cache" \
    --resource-log "$OUT/cuda_resource.jsonl" \
    --profile-every 500 \
    --agentguard-mode full \
    --iwg-checkpoint "$REPO/outputs/agentguard/checkpoints/MOT17/all/iwg/iwg_best.pt" \
    --tgr-checkpoint "$REPO/outputs/agentguard/checkpoints/MOT17/all/tgr/tgr_best.pt" \
    --agentguard-device cuda \
    --agentguard-tgr-frame-stride 4 \
    --agentguard-replay-diff-threshold 0.15 \
    --tracker-suffix profile_full_cuda \
    --print-per-sequence-metrics 2>&1 | tee -a "$OUT/cuda.log"
cat "$OUT/cuda.time" | tee -a "$OUT/cuda.log"
echo "=== CUDA full MOT17 FRCNN end $(date -Is) ===" | tee -a "$OUT/cuda.log"
