#!/usr/bin/env bash
#SBATCH --job-name=dynamo-1p1d-burst
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research/results/dynamo/slurm-%j-burst.out
#SBATCH --error=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research/results/dynamo/slurm-%j-burst.err

# job_burst_profile.sh — Launch Dynamo 1p1d, wait for it to be ready, then run
# BurstGPT's profile_vllm_trace.py against the live server.
#
# Submit with:
#   sbatch dynamo-pace/slurm/job_burst_profile.sh
#
# Override config at submission time:
#   sbatch --export=ALL,DYNAMO_CONFIG=dynamo-pace/configs/disagg_1p1d.yaml \
#          dynamo-pace/slurm/job_burst_profile.sh

set -euo pipefail

# Bug fix #4: source env on every new SLURM node
REPO_ROOT=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research
source "$REPO_ROOT/dynamo-pace/scripts/env.sh"

DYNAMO_CONFIG="${DYNAMO_CONFIG:-$DYNAMO_PACE_DIR/configs/disagg_1p1d.yaml}"

echo "========================================"
echo "SLURM job: $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_JOB_GPUS"
echo "Config:    $DYNAMO_CONFIG"
echo "========================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Launch Dynamo server in the background
# ---------------------------------------------------------------------------
bash "$DYNAMO_PACE_DIR/scripts/launch.sh" "$DYNAMO_CONFIG" &
SERVER_PID=$!

cleanup() {
    echo "[burst] Shutting down server..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "[burst] Done."
}
trap cleanup EXIT INT TERM

echo "[burst] Server PID: $SERVER_PID"

# ---------------------------------------------------------------------------
# Poll /health until the server is up
# ---------------------------------------------------------------------------
MAX_WAIT=600
WAITED=0
echo "[burst] Waiting for server to become healthy..."
while ! curl -sf 'http://localhost:8000/health' >/dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "[burst] ERROR: server did not start within ${MAX_WAIT}s (~10 min)." >&2
        exit 1
    fi
    echo "[burst]   still waiting... ${WAITED}s"
done

echo "[burst] /health OK"

# ---------------------------------------------------------------------------
# Confirm model is loaded via /v1/models
# ---------------------------------------------------------------------------
echo "[burst] Checking /v1/models..."
MODEL_NAME=$(curl -sf 'http://localhost:8000/v1/models' \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
echo "[burst] Model ready: $MODEL_NAME"

# ---------------------------------------------------------------------------
# Run BurstGPT trace profiler
# ---------------------------------------------------------------------------
echo "[burst] Running profile_vllm_trace.py from $BURST_GPT ..."
cd "$BURST_GPT"
python profile_vllm_trace.py

echo "[burst] profile_vllm_trace complete."
