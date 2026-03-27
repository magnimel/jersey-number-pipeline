#!/bin/bash
# Run this ONCE on a Narval LOGIN node (has internet access).
#
# Populates a uv cache at $SCRATCH/jersey-agg-uv-cache/ by syncing the
# locked environment. Compute nodes then run `uv sync --frozen` using that
# cache — no internet required, no full-venv rsync per job.
#
# Usage:
#   bash aggregation/setup_narval.sh
#
# After this, submit jobs with:
#   bash aggregation/sweep_narval.sh --cpu

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_CACHE="${SCRATCH}/jersey-agg-uv-cache"

echo "Loading modules..."
module load gcc/12.3 cuda/12.2 python/3.11

mkdir -p "${UV_CACHE}"

# ---------------------------------------------------------------------------
# Sync the locked environment into a throw-away venv.
# uv downloads each wheel and stores it in UV_CACHE_DIR.
# Subsequent `uv sync --frozen --cache-dir UV_CACHE` calls on compute nodes
# find everything locally — no network access needed.
# ---------------------------------------------------------------------------
cd "${REPO_DIR}/aggregation"

echo "Seeding uv cache at ${UV_CACHE} ..."
UV_CACHE_DIR="${UV_CACHE}" uv sync --frozen --python python3.11

echo ""
echo "Cache seeded. $(du -sh "${UV_CACHE}" | cut -f1) used."
echo ""
echo "Submit jobs with:"
echo "  bash aggregation/sweep_narval.sh --cpu"
