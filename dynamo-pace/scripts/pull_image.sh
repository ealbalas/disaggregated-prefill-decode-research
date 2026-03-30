#!/usr/bin/env bash
# pull_image.sh — One-time setup: pull the Dynamo vLLM container to a local SIF.
# Run this from a login node or interactive node with apptainer available.
# Output: $IMAGE (set in env.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

REGISTRY_IMAGE="docker://nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0"

echo "[pull_image] Target SIF: $IMAGE"

if [[ -f "$IMAGE" ]]; then
    echo "[pull_image] SIF already exists. Delete it first to re-pull:"
    echo "  rm $IMAGE"
    exit 0
fi

mkdir -p "$CONTAINERS_DIR"

echo "[pull_image] Pulling $REGISTRY_IMAGE ..."
echo "[pull_image] This may take 10-20 minutes on first run."

# APPTAINER_CACHEDIR keeps layer cache under scratch to avoid filling $HOME
APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache \
apptainer pull "$IMAGE" "$REGISTRY_IMAGE"

echo "[pull_image] Done. Image saved to: $IMAGE"
