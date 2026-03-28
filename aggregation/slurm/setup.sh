#!/bin/bash
set -e
# Common setup sourced by every aggregation sbatch script.
# Expects REPO_NAME and REPO_PATH to be set by the calling script.

if [ -z "${SLURM_JOB_NAME}" ]; then
    SLURM_JOB_NAME=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)
    export SLURM_JOB_NAME
fi

RUN_DIR_NAME="${SLURM_JOB_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"
FULL_RUN_DIR="$REPO_PATH/aggregation/runs/${RUN_DIR_NAME}"
mkdir -p "${FULL_RUN_DIR}"

# Redirect stdout and stderr to run directory
exec > "${FULL_RUN_DIR}/log.out" 2> "${FULL_RUN_DIR}/log.err"

# Load W&B credentials and other env vars
ENV_FILE="$REPO_PATH/aggregation/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Rsync repo to local scratch (fast I/O during training)
NEW_REPO_PATH="$SLURM_TMPDIR/$REPO_NAME"
rsync -av \
    --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='out' --exclude='data' --exclude='models' \
    --exclude='aggregation/runs' --exclude='aggregation/checkpoints' \
    --exclude='.ruff_cache' --exclude='.mypy_cache' \
    "$REPO_PATH/" "$NEW_REPO_PATH"

cd "$NEW_REPO_PATH"
echo "Working directory: $(pwd)"

# Load modules
module load python/3.12.4
echo "Python: $(python --version)"

# Create venv and install aggregation dependencies
cd aggregation
python -m venv .venv
source .venv/bin/activate
# Install CPU-only torch first (~200MB vs ~2GB for CUDA) to stay within SLURM_TMPDIR quota
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv sync --active
echo "Dependencies installed"
cd "$NEW_REPO_PATH"
