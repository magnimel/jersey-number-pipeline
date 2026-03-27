#!/bin/bash
# ---------------------------------------------------------------------------
# Hyperparameter sweep launcher for aggregation/submit_narval.sh
#
# Submits one SLURM job per hyperparameter combination. The job name and W&B
# run name are identical and encode the key hyperparameter values so that
# squeue output and the W&B dashboard both show what each run is testing.
#
# Run from the repo root:
#   bash aggregation/sweep_narval.sh [--dry-run] [--cpu]
#
# --cpu  : submit to CPU partition (no GPU); appends _cpu to job/run names and
#          uses a separate checkpoint subdir so GPU and CPU runs don't collide.
#
# After all jobs finish, sync offline W&B runs:
#   wandb sync $SCRATCH/jersey-agg-wandb/wandb/offline-run-*/
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_narval.sh"

DRY_RUN=0
CPU_MODE=0
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
    [[ "${arg}" == "--cpu" ]]     && CPU_MODE=1
done

[[ "${DRY_RUN}" -eq 1 ]] && echo "[dry-run] No jobs will be submitted."
[[ "${CPU_MODE}" -eq 1 ]] && echo "[cpu] Submitting to CPU partition (no GPU)."

# ---------------------------------------------------------------------------
# Sweep grid - edit these arrays to change what gets swept
# ---------------------------------------------------------------------------
LRS=(1e-3 5e-4 1e-4)
BATCH_SIZES=(32 64 128)
WEIGHT_DECAYS=(1e-4 1e-5)
EPOCHS=50
USE_CLASS_WEIGHTS=true   # keep fixed; set to "false" or add as sweep axis if needed
USE_DIGIT_CLASSIFIER=false

# ---------------------------------------------------------------------------
# Helper: format a float/sci-notation value into a compact slug for the name
# e.g. "1e-3" -> "1e-3", "5e-4" -> "5e-4"  (already compact, keep as-is)
# ---------------------------------------------------------------------------
slugify() { echo "$1" | tr '.' 'p'; }

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
TOTAL=0
for LR in "${LRS[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
        for WD in "${WEIGHT_DECAYS[@]}"; do

            # Build a name that is both human-readable and encodes all swept params.
            # Format: agg_lr<lr>_bs<bs>_wd<wd>_ep<epochs>[_cw][_p2d][_cpu]
            RUN_NAME="agg_lr${LR}_bs${BS}_wd${WD}_ep${EPOCHS}"
            [[ "${USE_CLASS_WEIGHTS}" == "true" ]]      && RUN_NAME="${RUN_NAME}_cw"
            [[ "${USE_DIGIT_CLASSIFIER}" == "true" ]]   && RUN_NAME="${RUN_NAME}_p2d"
            [[ "${CPU_MODE}" -eq 1 ]]                   && RUN_NAME="${RUN_NAME}_cpu"

            # Separate checkpoint subdir so GPU and CPU runs don't collide
            if [[ "${CPU_MODE}" -eq 1 ]]; then
                HP_OUTPUT_DIR="${SCRATCH}/jersey-agg-checkpoints/cpu"
            else
                HP_OUTPUT_DIR="${SCRATCH}/jersey-agg-checkpoints/gpu"
            fi

            echo "Submitting: ${RUN_NAME}"

            LOG_DIR="${SCRIPT_DIR}/logs/${RUN_NAME}"

            if [[ "${DRY_RUN}" -eq 0 ]]; then
                mkdir -p "${LOG_DIR}"
                if [[ "${CPU_MODE}" -eq 1 ]]; then
                    sbatch \
                        --job-name="${RUN_NAME}" \
                        --partition=cpubase_bycore_b1 \
                        --gres="" --mem=16G --time=0:30:00 \
                        --output="${LOG_DIR}/out.log" \
                        --error="${LOG_DIR}/err.log" \
                        --export=ALL,HP_LR="${LR}",HP_BS="${BS}",HP_WD="${WD}",HP_EPOCHS="${EPOCHS}",HP_CW="${USE_CLASS_WEIGHTS}",HP_P2D="${USE_DIGIT_CLASSIFIER}",HP_RUN_NAME="${RUN_NAME}",HP_OUTPUT_DIR="${HP_OUTPUT_DIR}" \
                        "${SUBMIT_SCRIPT}"
                else
                    sbatch \
                        --job-name="${RUN_NAME}" \
                        --output="${LOG_DIR}/out.log" \
                        --error="${LOG_DIR}/err.log" \
                        --export=ALL,HP_LR="${LR}",HP_BS="${BS}",HP_WD="${WD}",HP_EPOCHS="${EPOCHS}",HP_CW="${USE_CLASS_WEIGHTS}",HP_P2D="${USE_DIGIT_CLASSIFIER}",HP_RUN_NAME="${RUN_NAME}",HP_OUTPUT_DIR="${HP_OUTPUT_DIR}" \
                        "${SUBMIT_SCRIPT}"
                fi
            fi

            TOTAL=$(( TOTAL + 1 ))
        done
    done
done

echo ""
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] Would have submitted ${TOTAL} jobs."
else
    echo "Submitted ${TOTAL} jobs."
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "After all jobs finish, sync W&B offline runs:"
    echo "  wandb sync \$SCRATCH/jersey-agg-wandb/wandb/offline-run-*/"
fi
