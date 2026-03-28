#!/bin/bash
# Launcher: loads aggregation/.env for SLURM_ACCOUNT then calls sbatch.
#
# Usage:
#   bash aggregation/slurm/run.sh <path_to_sbatch_script>

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# aggregation/slurm/ -> aggregation/
AGG_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$AGG_ROOT/.env"

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_sbatch_script>"
    exit 1
fi

TARGET_SCRIPT="$1"
shift
EXTRA_ARGS="$@"

if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

echo "Using SLURM_ACCOUNT: ${SLURM_ACCOUNT}"
if [ -n "$SLURM_ACCOUNT" ]; then
    sbatch --account="$SLURM_ACCOUNT" $EXTRA_ARGS "$TARGET_SCRIPT"
else
    sbatch $EXTRA_ARGS "$TARGET_SCRIPT"
fi
