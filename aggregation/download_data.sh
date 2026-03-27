#!/bin/bash
# Download the SoccerNet training pipeline outputs from Google Drive.
# Run this on a LOGIN node (has internet) or locally before rsyncing to Narval.
#
# The zip contains:
#   jersey_id_results_train.json  - PARSeq logits per crop
#   train_gt.json                 - tracklet -> jersey number ground truth
#
# Usage:
#   bash aggregation/download_data.sh [destination_dir]
#
# Default destination: $SCRATCH/jersey-agg-data  (matches submit_narval.sh)

set -e

DEST="${1:-${SCRATCH}/jersey-agg-data}"
FILE_ID="1bGPqNxN6G01tc8kunlXlk9ZNqhewugIK"
ZIP_PATH="${DEST}/train_pipeline_output.zip"

mkdir -p "${DEST}"

echo "Downloading to ${DEST} ..."

# Try gdown first (pip install gdown), fall back to curl
if command -v gdown &>/dev/null; then
    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${ZIP_PATH}"
else
    echo "gdown not found; trying curl (may fail for large Drive files without auth)..."
    curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o "${ZIP_PATH}"
fi

echo "Extracting..."
unzip -o "${ZIP_PATH}" -d "${DEST}"

echo ""
echo "Done. Files in ${DEST}:"
ls -lh "${DEST}"
