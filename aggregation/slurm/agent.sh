#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

REPO_NAME="jersey-number-pipeline"
REPO_PATH="/home/$USER/$REPO_NAME"
SETUP_SCRIPT="$REPO_PATH/aggregation/slurm/setup.sh"
source "$SETUP_SCRIPT"
# ONLY ADD COMMANDS BELOW THIS LINE

# SWEEP_PATH is injected by sweep.sh as an env var passed to sbatch
wandb agent --count 1 "$SWEEP_PATH"
