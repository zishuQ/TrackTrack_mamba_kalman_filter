#!/bin/bash
# Test tracking + post-processing + packaging + local zip validation for the
# three final B checkpoints, CPU only. Uses the historical submission commands.
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-$WT/.venv/bin/python}"
MAIN="$WT"
EXP="$MAIN/outputs/agentguard/experiments/shutrack_b_final_20261120"
TRACKER="$MAIN/3. Tracker"
CACHE="$MAIN/outputs/agentguard/detection_cache"
VALIDATE="$WT/tools/codabench_submit.py"
MAKE_PACKAGE="$MAIN/outputs/agentguard/experiments/shutrack_a_submit_20261117/scripts/make_package.py"
mkdir -p "$EXP/logs"

track() {
  local dataset="$1"; shift
  local sequences=("$@")
  local dir="$EXP/$dataset"
  local ckpt="$dir/groups/appearance_rank/checkpoints/last.pt"
  local out="$dir/evals/tracking"
  mkdir -p "$out" "$dir/packages"
  [ -f "$ckpt" ] || { echo "$dataset: missing checkpoint"; return 1; }
  echo "$(date '+%F %T') | $dataset: tracking ${#sequences[@]} test sequences"
  ( cd "$TRACKER" && "$PY" -u run.py \
      --dataset "$dataset" --mode test \
      --sequences "${sequences[@]}" \
      --seed 10000 --kf-type nsa --agentguard-mode shutrack-v2 \
      --shutrack-checkpoint "$ckpt" --agentguard-device cpu \
      --detection-cache-root "$CACHE" \
      --output_dir "$out" --tracker-suffix final_b \
      --use_post --skip-eval \
      --torch-num-threads 4 --torch-num-interop-threads 1 ) \
    > "$dir/logs/track_test.log" 2>&1 || { echo "$dataset: TRACK FAILED"; return 1; }
  local post
  post=$(find "$out" -maxdepth 1 -type d -name '*_post' | head -1)
  [ -n "$post" ] || { echo "$dataset: no post dir"; return 1; }
  echo "$post" > "$dir/packages/post_dir.txt"
  return 0
}

mapfile -t MOT20_TEST <<< "MOT20-04 MOT20-06 MOT20-07 MOT20-08"
mapfile -t MOT17_TEST <<< "MOT17-01-FRCNN MOT17-03-FRCNN MOT17-06-FRCNN MOT17-07-FRCNN MOT17-08-FRCNN MOT17-12-FRCNN MOT17-14-FRCNN"
grep -v '^name$' "$TRACKER/trackeval/seqmap/sportsmot/test.txt" > "$EXP/SportsMOT/logs/test_sequences.txt"
mapfile -t SM_TEST < "$EXP/SportsMOT/logs/test_sequences.txt"

if [ ! -f "$EXP/MOT20/packages/MOT20.zip" ]; then
  track MOT20 "${MOT20_TEST[@]}" || exit 1
  POST=$(cat "$EXP/MOT20/packages/post_dir.txt")
  "$PY" "$MAKE_PACKAGE" --source "$POST" --zip "$EXP/MOT20/packages/MOT20.zip" --expected-count 4 \
    > "$EXP/MOT20/packages/package.json" 2>&1 || exit 1
  "$PY" "$VALIDATE" validate --zip "$EXP/MOT20/packages/MOT20.zip" --dataset mot20 \
    --seqinfo-root /home/shang/datasets/MOT20/test > "$EXP/MOT20/packages/validate.json" 2>&1 || exit 1
fi
if [ ! -f "$EXP/MOT17/packages/MOT17.zip" ]; then
  track MOT17 "${MOT17_TEST[@]}" || exit 1
  POST=$(cat "$EXP/MOT17/packages/post_dir.txt")
  count=$(find "$POST" -maxdepth 1 -name '*-FRCNN.txt' | wc -l)
  [ "$count" -eq 7 ] || { echo "MOT17 FRCNN count $count != 7"; exit 1; }
  for f in "$POST"/*-FRCNN.txt; do
    cp -f "$f" "${f%-FRCNN.txt}-DPM.txt"
    cp -f "$f" "${f%-FRCNN.txt}-SDP.txt"
  done
  "$PY" "$MAKE_PACKAGE" --source "$POST" --zip "$EXP/MOT17/packages/MOT17.zip" --expected-count 21 \
    > "$EXP/MOT17/packages/package.json" 2>&1 || exit 1
  "$PY" "$VALIDATE" validate --zip "$EXP/MOT17/packages/MOT17.zip" --dataset mot17 \
    --seqinfo-root /home/shang/datasets/MOT17/test > "$EXP/MOT17/packages/validate.json" 2>&1 || exit 1
fi
if [ ! -f "$EXP/SportsMOT/packages/SportsMOT.zip" ]; then
  track SportsMOT "${SM_TEST[@]}" || exit 1
  POST=$(cat "$EXP/SportsMOT/packages/post_dir.txt")
  "$PY" "$MAKE_PACKAGE" --source "$POST" --zip "$EXP/SportsMOT/packages/SportsMOT.zip" \
    --expected-count 150 --expected-names-file "$EXP/SportsMOT/logs/test_sequences.txt" \
    > "$EXP/SportsMOT/packages/package.json" 2>&1 || exit 1
  "$PY" "$VALIDATE" validate --zip "$EXP/SportsMOT/packages/SportsMOT.zip" --dataset sportsmot \
    --sequences-file "$EXP/SportsMOT/logs/test_sequences.txt" \
    --seqinfo-root /home/shang/datasets/SportsMOT/dataset/test \
    > "$EXP/SportsMOT/packages/validate.json" 2>&1 || exit 1
fi
echo "$(date '+%F %T') | PACKAGE_DONE"
