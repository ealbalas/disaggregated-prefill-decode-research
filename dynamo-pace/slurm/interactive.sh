#!/usr/bin/env bash
# interactive.sh — Request an interactive GPU node on PACE ICE.
# Edit NUM_GPUS and WALLTIME to match your experiment needs.
#
# Usage:
#   bash dynamo-pace/slurm/interactive.sh
#   bash dynamo-pace/slurm/interactive.sh 4   # request 4 GPUs

set -euo pipefail

NUM_GPUS="${1:-2}"   # default: 2 GPUs (enough for 1p1d)
WALLTIME="01:00:00"

echo "[interactive] Requesting ${NUM_GPUS}x H200 GPU(s) for ${WALLTIME}..."
echo "[interactive] salloc -N1 -G H200:${NUM_GPUS} --ntasks-per-node=4 --mem=256GB -t${WALLTIME}"

salloc -N1 -G "H200:${NUM_GPUS}" --ntasks-per-node=4 --mem=256GB -t"${WALLTIME}"
