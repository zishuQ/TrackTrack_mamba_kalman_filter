#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
MODE="${MODE:-val_custom}"
ORACLE_TYPE="${ORACLE_TYPE:-full}"
echo "=== Running Oracle Evaluation: $DATASET $MODE ($ORACLE_TYPE) ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}/3. Tracker"
"${PYTHON_BIN}" -m agentguard.cli evaluate_oracle --dataset "$DATASET" --mode "$MODE" --oracle-type "$ORACLE_TYPE"
echo "=== Oracle evaluation complete ==="
