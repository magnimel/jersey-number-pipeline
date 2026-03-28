#!/bin/bash
#SBATCH --job-name=agg_train
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

REPO_NAME="jersey-number-pipeline"
REPO_PATH="/home/$USER/$REPO_NAME"
SETUP_SCRIPT="$REPO_PATH/aggregation/slurm/setup.sh"
source "$SETUP_SCRIPT"
# ONLY ADD COMMANDS BELOW THIS LINE

# Data paths — STR results from download_data.sh, GT from downloadables.py
DATA_DIR="$REPO_PATH/data/aggregation"
GT_DIR="$REPO_PATH/data/SoccerNet"

python aggregation/train.py \
    data.str_results="$DATA_DIR/jersey_id_results_train.json" \
    data.gt="$GT_DIR/train_gt.json" \
    data.test_str_results="$DATA_DIR/jersey_id_results_test.json" \
    data.test_gt="$GT_DIR/test_gt.json" \
    output.output_dir="/scratch/$USER/jersey-agg-checkpoints/$SLURM_JOB_NAME" \
    wandb.entity="$WANDB_ENTITY"
