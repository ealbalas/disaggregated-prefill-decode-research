#!/bin/bash
#SBATCH --job-name=dynamo_burstgpt
#SBATCH -N1
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=dynamo_burstgpt_%j.out
#SBATCH --error=dynamo_burstgpt_%j.err

# Submit from the repo root: sbatch example/run_dynamo_burstgpt.slurm
# $SLURM_SUBMIT_DIR is automatically set to the directory where sbatch was called.
REPO_DIR="$SLURM_SUBMIT_DIR"

# 1. Environment — sets $MODEL, $SCRATCH, $SIF
RESULTS_DIR="$SCRATCH/disaggregated-prefill-decode-research/results/dynamo"
source "$REPO_DIR/dynamo-pace/scripts/env.sh"

# 2. Start Dynamo (disagg_1p1d: 1 prefill + 1 decode GPU).
# launch.sh spawns its own background processes and exits quickly,
# so we run it in the foreground and use the health poll below to detect readiness.
echo "Starting Dynamo (disagg_1p1d)..."
bash "$REPO_DIR/dynamo-pace/scripts/launch.sh" \
    "$REPO_DIR/dynamo-pace/configs/disagg_1p1d.yaml" \
    >"$SCRATCH/dynamo.launch.log" 2>&1

# 3. Poll health endpoint until the server accepts requests.
# If /health returns non-200, check the port in disagg_1p1d.yaml and update DYNAMO_URL.
DYNAMO_URL="http://localhost:8000/health"
MAX_WAIT=900
INTERVAL=5
ELAPSED=0

echo "Waiting for Dynamo to become ready..."
while true; do
    HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$DYNAMO_URL" || echo "000")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "Dynamo is ready."
        break
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "Timed out after ${MAX_WAIT}s. Check $SCRATCH/dynamo.launch.log"
        exit 1
    fi
done

# 4. Run BurstGPT benchmark
echo "Running BurstGPT benchmark..."
bash "$REPO_DIR/example/profile_dynamo_trace.sh"

# 5. Parse and print latency percentiles (p50/p90/p95/p99 TTFT and TPOT)
echo "Parsing results..."
python "$REPO_DIR/stats/get_latencies.py" "$SCRATCH/dynamo_1p1d_detail_log.json"

# 6. Stop Dynamo
bash "$REPO_DIR/dynamo-pace/scripts/stop.sh"
echo "Job complete."
