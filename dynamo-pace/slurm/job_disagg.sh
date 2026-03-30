#!/usr/bin/env bash
#SBATCH --job-name=dynamo-disagg
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=08:00:00
#SBATCH --output=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research/results/dynamo/slurm-%j-disagg.out
#SBATCH --error=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research/results/dynamo/slurm-%j-disagg.err

# job_disagg.sh — SLURM batch job for disaggregated Dynamo serving + benchmark.
#
# Submit with:
#   sbatch dynamo-pace/slurm/job_disagg.sh [config.yaml]
#
# Override config at submission time:
#   sbatch --export=ALL,DYNAMO_CONFIG=dynamo-pace/configs/disagg_2p1d.yaml \
#          dynamo-pace/slurm/job_disagg.sh

set -euo pipefail

# Bug fix #4: source env on every new SLURM node
REPO_ROOT=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research
source "$REPO_ROOT/dynamo-pace/scripts/env.sh"

# Config can be overridden via --export at sbatch time
DYNAMO_CONFIG="${DYNAMO_CONFIG:-$DYNAMO_PACE_DIR/configs/disagg_1p1d.yaml}"

echo "========================================"
echo "SLURM job: $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_JOB_GPUS"
echo "Config:    $DYNAMO_CONFIG"
echo "========================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Launch Dynamo server in the background
bash "$DYNAMO_PACE_DIR/scripts/launch.sh" "$DYNAMO_CONFIG" &
SERVER_PID=$!

echo "[job_disagg] Server PID: $SERVER_PID"
echo "[job_disagg] Waiting for server to become ready..."

# Poll the health endpoint (bug #5: single-quoted JSON)
MAX_WAIT=180
WAITED=0
while ! curl -sf 'http://localhost:8000/health' >/dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "[job_disagg] ERROR: Server did not start within ${MAX_WAIT}s." >&2
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    echo "[job_disagg] Still waiting... (${WAITED}s)"
done

echo "[job_disagg] Server ready. Running benchmark..."

bash "$DYNAMO_PACE_DIR/benchmarks/run_benchmark.sh" \
    --tag "$(basename "$DYNAMO_CONFIG" .yaml)_job${SLURM_JOB_ID}"

echo "[job_disagg] Benchmark complete. Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "[job_disagg] Done."
