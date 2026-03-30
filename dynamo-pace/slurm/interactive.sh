#!/usr/bin/env bash
# interactive.sh — Request an interactive GPU node on PACE ICE.
# Edit NUM_GPUS and WALLTIME to match your experiment needs.
#
# Usage:
#   bash dynamo-pace/slurm/interactive.sh
#   bash dynamo-pace/slurm/interactive.sh 4   # request 4 GPUs

set -euo pipefail

NUM_GPUS="${1:-2}"   # default: 2 GPUs (enough for 1p1d)
WALLTIME="01:00:00"  # 4 hours — adjust as needed
PARTITION="gpu-h200" # PACE ICE H200 partition; change if your queue differs
ACCOUNT="ealbalas3"  # leave blank or set your PACE account: e.g. "GT-abc123"

SRUN_ARGS=(
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --gres="gpu:h200:${NUM_GPUS}"
    --time="$WALLTIME"
    --pty
)

if [[ -n "$ACCOUNT" ]]; then
    SRUN_ARGS+=(--account="$ACCOUNT")
fi

echo "[interactive] Requesting ${NUM_GPUS}x H200 GPU(s) for ${WALLTIME}..."
echo "[interactive] srun ${SRUN_ARGS[*]} bash"

srun "${SRUN_ARGS[@]}" bash --login -c "
    source /etc/profile
    source $SCRATCH/disaggregated-prefill-decode-research/dynamo-pace/scripts/env.sh
    echo '[interactive] Node ready. Environment loaded.'
    exec bash --login
"
