#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Building Evidence Packets: $DATASET ==="
cd "3. Tracker"
../.venv/bin/python -m agentguard.cli build_evidence_packets --dataset "$DATASET"
echo "=== Evidence packets complete ==="
