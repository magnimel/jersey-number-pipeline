#!/bin/bash
# W&B sweep launcher: creates a sweep then submits N agent jobs.
#
# Usage:
#   bash digit_classifier/slurm/sweep.sh [--count N] [--sweep-id <id>]
#
# Examples:
#   # Create a new sweep and run 20 agents
#   bash digit_classifier/slurm/sweep.sh --count 20
#
#   # Resume an existing sweep (skip creation)
#   bash digit_classifier/slurm/sweep.sh --count 10 --sweep-id abc123

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
DC_ROOT="$REPO_PATH/digit_classifier"
ENV_FILE="$DC_ROOT/.env"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"
AGENT_SCRIPT="$SCRIPT_DIR/agent.sh"

COUNT=10
SWEEP_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --count) COUNT="$2"; shift 2 ;;
        --sweep-id) SWEEP_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Load env vars
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Error: $ENV_FILE not found. Copy from .env.template and fill in values."
    exit 1
fi

# Create sweep if no ID provided
if [ -z "$SWEEP_ID" ]; then
    echo "Creating W&B sweep..."
    SWEEP_OUTPUT=$(cd "$REPO_PATH" && wandb sweep digit_classifier/conf/sweep.yaml 2>&1)
    echo "$SWEEP_OUTPUT"
    # Parse sweep ID from output like "wandb: Created sweep with ID: abc123"
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'ID: \K\S+' | tail -1)
    if [ -z "$SWEEP_ID" ]; then
        echo "Error: could not parse sweep ID from wandb output."
        exit 1
    fi
fi

SWEEP_PATH="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
echo "Sweep: $SWEEP_PATH"
echo "Submitting $COUNT agent jobs..."

for i in $(seq 1 "$COUNT"); do
    SWEEP_PATH="$SWEEP_PATH" bash "$RUN_SCRIPT" "$AGENT_SCRIPT"
    sleep 1
done

echo "Done. Monitor at: https://wandb.ai/$SWEEP_PATH"
