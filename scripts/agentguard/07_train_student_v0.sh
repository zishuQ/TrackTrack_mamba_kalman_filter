#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
DEVICE="${DEVICE:-cuda}"
echo "=== Training Student-V0: $DATASET on $DEVICE ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}/3. Tracker"
"${PYTHON_BIN}" -m agentguard.cli train_student_v0 --dataset "$DATASET" --device "$DEVICE"
echo "=== Student-V0 training complete ==="
