#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_NAME="${RUN_NAME:-iwg_rg_cma_v1_sportsmot_trainval_seed42_bs1024_shard10x2_200e}"
export EPOCHS=200
export MEMORY_SHARDS=10
export EPOCHS_PER_SHARD=10
export SHARD_CYCLES=2

exec "${ROOT}/scripts/agentguard/run_iwg_rg_cma_sportsmot_trainval_100e.sh"
