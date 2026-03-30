#!/usr/bin/env bash
#SBATCH --job-name=dynamo-agg
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research/results/dynamo/slurm-%j-agg.out
#SBATCH --error=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research/results/dynamo/slurm-%j-agg.err

# job_agg.sh — SLURM batch job for aggregated (baseline) Dynamo serving + benchmark.
# Uses agg_baseline.yaml: single worker, no disaggregation, 1 GPU.
#
# Submit with:
#   sbatch dynamo-pace/slurm/job_agg.sh

set -euo pipefail

# Bug fix #4: source env on every new SLURM node
REPO_ROOT=/storage/ice1/4/5/ealbalas3/disaggregated-prefill-decode-research
source "$REPO_ROOT/dynamo-pace/scripts/env.sh"

DYNAMO_CONFIG="$DYNAMO_PACE_DIR/configs/agg_baseline.yaml"

echo "========================================"
echo "SLURM job: $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_JOB_GPUS"
echo "Config:    $DYNAMO_CONFIG (aggregated baseline)"
echo "========================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Launch server in background
bash "$DYNAMO_PACE_DIR/scripts/launch.sh" "$DYNAMO_CONFIG" &
SERVER_PID=$!

echo "[job_agg] Server PID: $SERVER_PID"
echo "[job_agg] Waiting for server to become ready..."

MAX_WAIT=180
WAITED=0
while ! curl -sf 'http://localhost:8000/health' >/dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "[job_agg] ERROR: Server did not start within ${MAX_WAIT}s." >&2
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    echo "[job_agg] Still waiting... (${WAITED}s)"
done

echo "[job_agg] Server ready. Running benchmark..."

bash "$DYNAMO_PACE_DIR/benchmarks/run_benchmark.sh" \
    --tag "agg_baseline_job${SLURM_JOB_ID}"

echo "[job_agg] Benchmark complete. Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "[job_agg] Done."
