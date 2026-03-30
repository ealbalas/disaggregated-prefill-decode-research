#!/usr/bin/env bash
# launch.sh — Launch the Dynamo serving stack inside Apptainer.
# Usage:
#   bash launch.sh [config.yaml]
#
# Defaults to disagg_1p1d.yaml if no config is given.
# All PACE bug workarounds are forwarded into the container via --env flags.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"   # Bug fix #4: source on every new node

CONFIG="${1:-$DYNAMO_PACE_DIR/configs/disagg_1p1d.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "[launch] ERROR: Config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "[launch] ERROR: Container SIF not found: $IMAGE" >&2
    echo "[launch] Run scripts/pull_image.sh first." >&2
    exit 1
fi

echo "[launch] Config:    $CONFIG"
echo "[launch] Image:     $IMAGE"
echo "[launch] Model:     $MODEL"
echo "[launch] Results:   $RESULTS_DIR"

# Resolve MODEL in the config by substituting the env var before passing to dynamo.
# Dynamo reads ${MODEL} literals from YAML, so we export MODEL into the container.

apptainer exec \
    --nv \
    --bind "$REPO_ROOT:$REPO_ROOT" \
    --bind "$MODEL:$MODEL" \
    --env MODEL="$MODEL" \
    --env RESULTS_DIR="$RESULTS_DIR" \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env VLLM_USE_DEEP_GEMM=0 \
    --env TORCHINDUCTOR_COMBO_KERNELS=0 \
    "$IMAGE" \
    dynamo serve "$CONFIG"
