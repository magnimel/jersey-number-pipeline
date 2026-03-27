#!/bin/bash
# Job name is set dynamically by sweep_narval.sh via: sbatch --job-name=<run_name>
# When submitting a single run manually, override with: sbatch --job-name=<name> submit_narval.sh
#SBATCH --job-name=agg-single
#SBATCH --account=def-fard
#SBATCH --gres=gpu:1
# NOTE: the model is small (~100K params, 33-dim input sequences). A single GPU
# is more than sufficient and could arguably run on CPU in reasonable time.
# 1 GPU is requested here as standard practice to get GPU memory bandwidth.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
# --output and --error are set dynamically by sweep_narval.sh via sbatch --output=...
# so that logs are organised as aggregation/logs/<run_name>/{out,err}.log
# Fallback for manual single-run submissions:
#SBATCH --output=aggregation/logs/manual-%j/out.log
#SBATCH --error=aggregation/logs/manual-%j/err.log

# ---------------------------------------------------------------------------
# Paths - adjust these to match your scratch layout
# ---------------------------------------------------------------------------
REPO_DIR="${HOME}/jersey-number-pipeline"
VENV_DIR="${SCRATCH}/envs/jersey-aggregation"
DATA_DIR="${SCRATCH}/jersey-agg-data"       # where you extracted the train data zip
OUTPUT_DIR="${HP_OUTPUT_DIR:-${SCRATCH}/jersey-agg-checkpoints}"
WANDB_OFFLINE_DIR="${SCRATCH}/jersey-agg-wandb"

STR_RESULTS="${DATA_DIR}/jersey_id_results_train.json"
GT_FILE="${DATA_DIR}/train_gt.json"
TEST_STR_RESULTS="${DATA_DIR}/jersey_id_results_test.json"
TEST_GT_FILE="${DATA_DIR}/test_gt.json"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Must match what was used in setup_narval.sh so the GPU torch wheel is found
module load gcc/12.3 cuda/12.2 python/3.11

source "${VENV_DIR}/bin/activate"

mkdir -p "${OUTPUT_DIR}" "${WANDB_OFFLINE_DIR}" "${REPO_DIR}/aggregation/logs"

# Load W&B secrets from .env (API key, entity, project)
# aggregation/.env is gitignored; place it manually in the repo before submitting
if [ -f "${REPO_DIR}/aggregation/.env" ]; then
    set -a
    source "${REPO_DIR}/aggregation/.env"
    set +a
fi

# Compute nodes have no internet; runs are saved locally and synced afterwards
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_OFFLINE_DIR}"

# ---------------------------------------------------------------------------
# Hyperparameters - set by sweep_narval.sh via --export; fall back to defaults
# for manual single-run submissions.
# ---------------------------------------------------------------------------
HP_LR="${HP_LR:-1e-3}"
HP_BS="${HP_BS:-64}"
HP_WD="${HP_WD:-1e-4}"
HP_EPOCHS="${HP_EPOCHS:-50}"
HP_PATIENCE="${HP_PATIENCE:-10}"
HP_CW="${HP_CW:-true}"           # use_class_weights
HP_P2D="${HP_P2D:-false}"        # use_digit_classifier
HP_RUN_NAME="${HP_RUN_NAME:-agg_lr${HP_LR}_bs${HP_BS}_wd${HP_WD}_ep${HP_EPOCHS}}"

echo "Hyperparameters:"
echo "  lr=${HP_LR}  batch_size=${HP_BS}  weight_decay=${HP_WD}"
echo "  epochs=${HP_EPOCHS}  use_class_weights=${HP_CW}  use_digit_classifier=${HP_P2D}"
echo "  run_name=${HP_RUN_NAME}"

# ---------------------------------------------------------------------------
# Run  (Hydra overrides)
# ---------------------------------------------------------------------------
cd "${REPO_DIR}"

python aggregation/train.py \
    data.str_results="${STR_RESULTS}" \
    data.gt="${GT_FILE}" \
    data.test_str_results="${TEST_STR_RESULTS}" \
    data.test_gt="${TEST_GT_FILE}" \
    output.output_dir="${OUTPUT_DIR}" \
    output.run_name="${HP_RUN_NAME}" \
    training.epochs="${HP_EPOCHS}" \
    training.batch_size="${HP_BS}" \
    training.lr="${HP_LR}" \
    training.weight_decay="${HP_WD}" \
    training.use_class_weights="${HP_CW}" \
    training.early_stopping_patience="${HP_PATIENCE}" \
    model.use_digit_classifier="${HP_P2D}"

echo ""
echo "Job done. Sync W&B offline run with:"
echo "  wandb sync ${WANDB_OFFLINE_DIR}/wandb/offline-run-*"
echo ""
echo "Update configuration.py with the best checkpoint path from:"
echo "  cat ${OUTPUT_DIR}/best_ckpt.txt"
