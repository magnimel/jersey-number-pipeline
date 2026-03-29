#!/bin/bash
set -e
# Common setup sourced by every digit_classifier sbatch script.
# Expects REPO_NAME and REPO_PATH to be set by the calling script.

if [ -z "${SLURM_JOB_NAME}" ]; then
    SLURM_JOB_NAME=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)
    export SLURM_JOB_NAME
fi

RUN_DIR_NAME="${SLURM_JOB_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"
FULL_RUN_DIR="$REPO_PATH/digit_classifier/runs/${RUN_DIR_NAME}"
mkdir -p "${FULL_RUN_DIR}"

# Redirect stdout and stderr to run directory
exec > "${FULL_RUN_DIR}/log.out" 2> "${FULL_RUN_DIR}/log.err"

# Load W&B credentials and other env vars (including DIGIT_DATA_DIR)
ENV_FILE="$REPO_PATH/digit_classifier/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Rsync repo to local scratch (fast I/O during training)
NEW_REPO_PATH="$SLURM_TMPDIR/$REPO_NAME"
rsync -av \
    --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='out' --exclude='data' --exclude='models' \
    --exclude='digit_classifier/runs' --exclude='digit_classifier/checkpoints' \
    --exclude='.ruff_cache' --exclude='.mypy_cache' \
    "$REPO_PATH/" "$NEW_REPO_PATH"

cd "$NEW_REPO_PATH"
echo "Working directory: $(pwd)"

# Load modules
module load python/3.12.4
module load cuda/12.6
echo "Python: $(python --version)"

# Create venv and install digit_classifier dependencies
cd digit_classifier
python -m venv .venv
source .venv/bin/activate
unset GIT_ASKPASS SSH_ASKPASS
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0="" \
    TMPDIR="$SLURM_TMPDIR" uv sync --active
TMPDIR="$SLURM_TMPDIR" uv pip install -q https://github.com/Atze00/MoViNet-pytorch/archive/refs/heads/main.zip
echo "Dependencies installed"
cd "$NEW_REPO_PATH"
