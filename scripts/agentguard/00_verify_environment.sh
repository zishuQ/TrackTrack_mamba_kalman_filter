#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MOT17}"
DATA_DIR="${DATA_DIR:-/home/shang/datasets}"
PICKLE_DIR="${PICKLE_DIR:-../outputs/2. det_feat}"

echo "=== AgentGuard Environment Verification ==="
echo "Dataset: $DATASET"
echo "Data dir: $DATA_DIR"
echo "Pickle dir: $PICKLE_DIR"

# Check dataset directory
if [ ! -d "$DATA_DIR/$DATASET" ]; then
    echo "ERROR: Dataset directory not found: $DATA_DIR/$DATASET"
    exit 1
fi

# Check GT files
GT_DIR="$DATA_DIR/$DATASET/train"
if [ -d "$GT_DIR" ]; then
    GT_COUNT=$(find "$GT_DIR" -name "gt.txt" | wc -l)
    echo "GT files found: $GT_COUNT"
else
    echo "WARNING: No training GT directory at $GT_DIR"
fi

# Check detection cache
if [ ! -d "$PICKLE_DIR" ]; then
    echo "ERROR: Detection cache directory not found: $PICKLE_DIR"
    exit 1
fi

# Check for existing .pickle files
PICKLE_COUNT=$(find "$PICKLE_DIR" -name "*.pickle" | wc -l)
echo "Pickle files found: $PICKLE_COUNT"

# Check ReID dimension
if [ $PICKLE_COUNT -gt 0 ]; then
    echo "Will detect ReID dimension at runtime"
fi

# Check CMC files
CMC_DIR="3. Tracker/trackers/cmc"
if [ -d "$CMC_DIR" ]; then
    CMC_COUNT=$(find "$CMC_DIR" -name "GMC-*.txt" | wc -l)
    echo "CMC files found: $CMC_COUNT"
fi

# Check image files
if [ -d "$DATA_DIR/$DATASET/train" ]; then
    IMG_COUNT=$(find "$DATA_DIR/$DATASET/train" -name "*.jpg" | head -1 | wc -l)
    echo "Image files accessible: $([ $IMG_COUNT -gt 0 ] && echo 'yes' || echo 'no')"
fi

# Check output dir write permissions
OUTPUT_DIR="outputs/agentguard"
mkdir -p "$OUTPUT_DIR"
if [ -w "$OUTPUT_DIR" ]; then
    echo "Output directory writable: $OUTPUT_DIR"
else
    echo "ERROR: Cannot write to $OUTPUT_DIR"
    exit 1
fi

echo "=== Environment verification complete ==="
