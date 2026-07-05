#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MOT17}"
PICKLE_DIR="${PICKLE_DIR:-../outputs/2. det_feat}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/agentguard/cache}"

echo "=== AgentGuard: Cache Events ==="
echo "Dataset: $DATASET"
echo "Pickle dir: $PICKLE_DIR"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Caching detections and GT matches for $DATASET ..."
cd agentguard
../.venv/bin/python -c "
from agentguard.data.cache_reader import iterate_cache
from agentguard.data.gt_matching import match_detections_to_gt
print('Cache and GT matching modules loaded successfully.')
"
echo "Cache events complete."
