#!/bin/bash
# Download pre-computed SoccerNet pipeline outputs (PARSeq STR results) for
# aggregation model training/evaluation.
#
# The zips contain:
#   train_all_run.zip  -> jersey_id_results_train.json  (PARSeq logits per crop)
#   test_all_run.zip   -> jersey_id_results_test.json   (PARSeq logits per crop)
#
# Ground-truth files (train_gt.json / test_gt.json) come from the SoccerNet
# dataset splits downloaded by downloadables.py (train.zip / test.zip).
#
# Usage:
#   bash aggregation/download_data.sh [destination_dir]
#
# Default destination: data/aggregation/

set -e

DEST="${1:-data/aggregation}"
TRAIN_ID="1bGPqNxN6G01tc8kunlXlk9ZNqhewugIK"
TEST_ID="1yzuJaTx4m7rLW3c2inuudF7BGN6rmhNi"

mkdir -p "${DEST}"

download() {
    local file_id="$1"
    local zip_path="$2"
    local label="$3"

    echo "Downloading ${label} ..."
    if command -v gdown &>/dev/null; then
        gdown "https://drive.google.com/uc?id=${file_id}" -O "${zip_path}"
    else
        echo "gdown not found; install it with: pip install gdown"
        exit 1
    fi

    echo "Extracting ${label} ..."
    unzip -o "${zip_path}" -d "${DEST}"
}

download "${TRAIN_ID}" "${DEST}/train_all_run.zip" "train_all_run.zip"
download "${TEST_ID}"  "${DEST}/test_all_run.zip"  "test_all_run.zip"

echo ""
echo "Done. Files in ${DEST}:"
ls -lh "${DEST}"
