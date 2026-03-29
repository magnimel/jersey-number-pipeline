#!/bin/bash
# Run a W&B sweep agent from an interactive GPU node.
# Mirrors batch job behaviour: rsyncs repo to $SLURM_TMPDIR, builds venv there.
#
# Usage (from repo root or anywhere):
#   bash digit_classifier/slurm/interactive_agent.sh [--count N] [--sweep-id ID]
#
# Defaults: --count 5, sweep ID from WANDB_SWEEP_ID env or hardcoded fallback

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
REPO_NAME="$(basename "$REPO_PATH")"
ENV_FILE="$REPO_PATH/digit_classifier/.env"

COUNT=5
SWEEP_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --count)    COUNT="$2";    shift 2 ;;
        --sweep-id) SWEEP_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Load env vars
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Error: $ENV_FILE not found."
    exit 1
fi

SWEEP_ID="${SWEEP_ID:-${WANDB_SWEEP_ID:-z4iyaqar}}"
SWEEP_PATH="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"

# Rsync repo to local scratch (fast I/O during training)
NEW_REPO_PATH="$SLURM_TMPDIR/$REPO_NAME"
echo "Syncing repo to $NEW_REPO_PATH ..."
rsync -a \
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

# Create venv and install deps
cd digit_classifier
python -m venv .venv
source .venv/bin/activate
if ! command -v uv &> /dev/null; then
    pip install -q uv
fi
unset GIT_ASKPASS SSH_ASKPASS
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0="" \
    TMPDIR="$SLURM_TMPDIR" uv sync --active
TMPDIR="$SLURM_TMPDIR" uv pip install -q https://github.com/Atze00/MoViNet-pytorch/archive/refs/heads/main.zip
echo "Dependencies ready."

cd "$NEW_REPO_PATH"

export PYTHONPATH="$NEW_REPO_PATH:${PYTHONPATH}"
echo "Starting agent for sweep: $SWEEP_PATH (count=$COUNT)"
wandb agent --count "$COUNT" "$SWEEP_PATH"
