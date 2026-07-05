#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MOT17}"
CACHE_DIR="${CACHE_DIR:-outputs/agentguard/cache}"

echo "=== AgentGuard: Validate Cache ==="
echo "Dataset: $DATASET"
echo "Cache dir: $CACHE_DIR"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Cache directory not found: $CACHE_DIR"
    exit 1
fi

# Count shard files
SHARD_COUNT=$(find "$CACHE_DIR" -name "*.json" -o -name "*.pt" -o -name "*.pkl" 2>/dev/null | wc -l)
echo "Cache shard files found: $SHARD_COUNT"

if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "WARNING: No cache shards found. Run 02_cache_events.sh first."
fi

echo "Validate cache complete."
