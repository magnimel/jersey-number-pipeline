#!/bin/bash
#SBATCH --job-name=digit_train
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

REPO_NAME="jersey-number-pipeline"
REPO_PATH="/home/$USER/$REPO_NAME"
SETUP_SCRIPT="$REPO_PATH/digit_classifier/slurm/setup.sh"
source "$SETUP_SCRIPT"
# ONLY ADD COMMANDS BELOW THIS LINE

python digit_classifier/train.py \
    --data "$DIGIT_DATA_DIR" \
    --output_dir "/scratch/$USER/jersey-digit-checkpoints/$SLURM_JOB_NAME" \
    --skip_test
