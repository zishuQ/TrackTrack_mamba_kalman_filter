#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-MOT17}"
echo "=== Building Student-V0 Dataset: $DATASET ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}/3. Tracker"
"${PYTHON_BIN}" -m agentguard.cli build_student_v0_data --dataset "$DATASET"
echo "=== Student-V0 data complete ==="
