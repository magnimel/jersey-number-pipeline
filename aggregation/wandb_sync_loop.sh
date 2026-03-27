#!/bin/bash
# ---------------------------------------------------------------------------
# Watches $SCRATCH/jersey-agg-wandb for completed offline W&B runs and syncs
# them to wandb.ai automatically. Run this on a Narval LOGIN node (has internet)
# inside a tmux/screen session so it survives logout.
#
# Usage:
#   tmux new -s wandb-sync
#   bash aggregation/wandb_sync_loop.sh
#
# Stop with Ctrl+C or: tmux kill-session -t wandb-sync
# ---------------------------------------------------------------------------

WANDB_OFFLINE_DIR="${SCRATCH}/jersey-agg-wandb/wandb"
CHECK_INTERVAL=60  # seconds between checks

REPO_DIR="${HOME}/jersey-number-pipeline"
if [ -f "${REPO_DIR}/aggregation/.env" ]; then
    set -a; source "${REPO_DIR}/aggregation/.env"; set +a
fi

echo "Watching ${WANDB_OFFLINE_DIR} for completed runs (every ${CHECK_INTERVAL}s)..."
echo "Stop with Ctrl+C"
echo ""

SYNCED_RUNS=()

while true; do
    for run_dir in "${WANDB_OFFLINE_DIR}"/offline-run-*/; do
        [[ -d "${run_dir}" ]] || continue

        # Skip already synced runs
        already_synced=0
        for s in "${SYNCED_RUNS[@]+"${SYNCED_RUNS[@]}"}"; do
            [[ "${s}" == "${run_dir}" ]] && already_synced=1 && break
        done
        [[ "${already_synced}" -eq 1 ]] && continue

        # A run is complete when wandb-summary.json exists (written at run end)
        if [[ -f "${run_dir}/files/wandb-summary.json" ]]; then
            echo "[$(date '+%H:%M:%S')] Syncing: $(basename "${run_dir}")"
            if wandb sync "${run_dir}" 2>&1 | tail -1; then
                SYNCED_RUNS+=("${run_dir}")
                echo "[$(date '+%H:%M:%S')] Done: $(basename "${run_dir}")"
            else
                echo "[$(date '+%H:%M:%S')] FAILED: $(basename "${run_dir}") - will retry"
            fi
        fi
    done

    sleep "${CHECK_INTERVAL}"
done
