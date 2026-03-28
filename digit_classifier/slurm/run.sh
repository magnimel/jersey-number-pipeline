#!/bin/bash
# Launcher: loads digit_classifier/.env for SLURM_ACCOUNT then calls sbatch.
#
# Usage:
#   bash digit_classifier/slurm/run.sh <path_to_sbatch_script>

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# digit_classifier/slurm/ -> digit_classifier/
DC_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$DC_ROOT/.env"

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_sbatch_script>"
    exit 1
fi

TARGET_SCRIPT="$1"

if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

echo "Using SLURM_ACCOUNT: ${SLURM_ACCOUNT}"
if [ -n "$SLURM_ACCOUNT" ]; then
    sbatch --account="$SLURM_ACCOUNT" "$TARGET_SCRIPT"
else
    sbatch "$TARGET_SCRIPT"
fi
