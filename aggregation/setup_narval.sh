#!/bin/bash
# Run this ONCE on a Narval LOGIN node (has internet access).
# Creates a virtualenv in $SCRATCH and installs all dependencies using uv.
#
# Usage:
#   bash aggregation/setup_narval.sh
#
# After this, submit jobs with:
#   sbatch aggregation/submit_narval.sh

set -e

VENV_DIR="${SCRATCH}/envs/jersey-aggregation"

echo "Loading modules..."
module load gcc/12.3 cuda/12.2 python/3.11

CC_WHEELS_1="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3"
CC_WHEELS_2="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic"
CC_WHEELS_3="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
CC_FIND_LINKS="--find-links ${CC_WHEELS_1} --find-links ${CC_WHEELS_2} --find-links ${CC_WHEELS_3}"

echo "Creating virtualenv at ${VENV_DIR} ..."
rm -rf "${VENV_DIR}"
uv venv --python python3.11 "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Installing torch (GPU build) from ComputeCanada wheel cache..."
# --no-index + --find-links points uv at the CC wheel cache; CUDA module ensures GPU build is picked.
# shellcheck disable=SC2086
uv pip install --no-index ${CC_FIND_LINKS} torch torchvision tqdm numpy scikit-learn pytorch-lightning torchmetrics

echo "Installing remaining packages from PyPI..."
uv pip install wandb python-dotenv hydra-core

echo ""
echo "Setup complete. Virtualenv: ${VENV_DIR}"
echo "To test: source ${VENV_DIR}/bin/activate && python -c 'import torch; print(torch.__version__)'"
