#!/bin/bash
# W&B sweep launcher: creates a sweep then submits N agent jobs.
#
# Usage:
#   bash aggregation/slurm/sweep.sh [--count N] [--sweep-id <id>]
#
# Examples:
#   # Create a new sweep and run 20 agents
#   bash aggregation/slurm/sweep.sh --count 20
#
#   # Resume an existing sweep (skip creation)
#   bash aggregation/slurm/sweep.sh --count 10 --sweep-id abc123

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
AGG_ROOT="$REPO_PATH/aggregation"
ENV_FILE="$AGG_ROOT/.env"
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

if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Error: $ENV_FILE not found. Copy from .env.template and fill in values."
    exit 1
fi

if [ -z "$SWEEP_ID" ]; then
    echo "Creating W&B sweep..."
    SWEEP_OUTPUT=$(cd "$REPO_PATH" && wandb sweep \
        --project "$WANDB_PROJECT" \
        --entity "$WANDB_ENTITY" \
        aggregation/conf/sweep.yaml 2>&1)
    echo "$SWEEP_OUTPUT"
    # Parse the full path from the "wandb agent entity/project/id" line
    SWEEP_PATH=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K\S+' | tail -1)
    if [ -z "$SWEEP_PATH" ]; then
        echo "Error: could not parse sweep path from wandb output."
        exit 1
    fi
else
    SWEEP_PATH="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
fi
SWEEP_ID=$(basename "$SWEEP_PATH")
echo "Sweep: $SWEEP_PATH"
echo "Submitting $COUNT agent jobs..."

for i in $(seq 1 "$COUNT"); do
    SWEEP_PATH="$SWEEP_PATH" bash "$RUN_SCRIPT" "$AGENT_SCRIPT" --job-name="agg_${SWEEP_ID}_${i}"
    sleep 1
done

echo "Done. Monitor at: https://wandb.ai/$SWEEP_PATH"
