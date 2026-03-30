#!/usr/bin/env bash
# stop.sh — Gracefully stop all running Dynamo / vLLM processes on this node.
# Call this to clean up before re-launching with a different config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"   # Bug fix #4: source on every new node

echo "[stop] Sending SIGTERM to dynamo processes..."
pkill -TERM -f "dynamo serve" 2>/dev/null && echo "[stop] dynamo serve stopped." || echo "[stop] No dynamo serve process found."

echo "[stop] Sending SIGTERM to vllm processes..."
pkill -TERM -f "vllm" 2>/dev/null && echo "[stop] vllm stopped." || echo "[stop] No vllm process found."

# Give processes a moment to clean up before checking
sleep 3

# Force-kill any stragglers
pkill -KILL -f "dynamo serve" 2>/dev/null || true
pkill -KILL -f "vllm" 2>/dev/null || true

# Release any lingering GPU memory (best-effort)
if command -v fuser &>/dev/null; then
    for dev in /dev/nvidia[0-9]*; do
        fuser -k "$dev" 2>/dev/null || true
    done
fi

echo "[stop] Done."
