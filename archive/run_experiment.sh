#!/usr/bin/env bash
# experiments/run_experiment.sh
#
# Usage:
#   bash experiments/run_experiment.sh                           # 1p1d baseline
#   bash experiments/run_experiment.sh configs/2p3d_random.env  # 2p3d

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

CONFIG="${1:-$REPO_ROOT/configs/baseline_random.env}"
source "$CONFIG"

PIDS=()

# ── Helper: wait until a port is open ────────────────────────────────────────
wait_for_port() {
  local port=$1 label=$2
  echo "Waiting for $label on port $port..."
  until nc -z localhost "$port" 2>/dev/null; do sleep 2; done
  echo "  ✓ $label ready."
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

# ── Build proxy host/port args ────────────────────────────────────────────────
PREFILL_HOSTS="" PREFILL_PORT_ARGS="" DECODE_HOSTS="" DECODE_PORT_ARGS=""
for port in "${PREFILL_PORTS[@]}"; do PREFILL_HOSTS+="localhost "; PREFILL_PORT_ARGS+="$port "; done
for port in "${DECODE_PORTS[@]}";  do DECODE_HOSTS+="localhost ";  DECODE_PORT_ARGS+="$port ";  done

# ── Step 1: Proxy ─────────────────────────────────────────────────────────────
echo "Starting proxy..."
cd "$VLLM_PROXY_SRC"
python toy_proxy_server.py \
  --port "$PROXY_PORT" \
  --prefiller-hosts $PREFILL_HOSTS \
  --prefiller-ports $PREFILL_PORT_ARGS \
  --decoder-hosts $DECODE_HOSTS \
  --decoder-ports $DECODE_PORT_ARGS &
PIDS+=($!)
wait_for_port "$PROXY_PORT" "proxy"

# ── Step 2: Prefill instances ─────────────────────────────────────────────────
for i in "${!PREFILL_PORTS[@]}"; do
  echo "Starting prefill_$i (GPU ${PREFILL_GPUS[$i]}, port ${PREFILL_PORTS[$i]})..."
  CUDA_VISIBLE_DEVICES="${PREFILL_GPUS[$i]}" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNELS[$i]}" \
  vllm serve "$MODEL" \
    --port "${PREFILL_PORTS[$i]}" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
  PIDS+=($!)
  wait_for_port "${PREFILL_PORTS[$i]}" "prefill_$i"
done

# ── Step 3: Decode instances ──────────────────────────────────────────────────
for i in "${!DECODE_PORTS[@]}"; do
  echo "Starting decode_$i (GPU ${DECODE_GPUS[$i]}, port ${DECODE_PORTS[$i]})..."
  CUDA_VISIBLE_DEVICES="${DECODE_GPUS[$i]}" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNELS[$i]}" \
  vllm serve "$MODEL" \
    --port "${DECODE_PORTS[$i]}" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
  PIDS+=($!)
  wait_for_port "${DECODE_PORTS[$i]}" "decode_$i"
done

# ── Step 4: Benchmark ─────────────────────────────────────────────────────────
echo "All services up. Running benchmark..."
mkdir -p "$OUTDIR"
vllm bench serve \
  --backend vllm \
  --port "$PROXY_PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --ignore-eos \
  --request-rate "$REQUEST_RATE" \
  --num-prompts "$NUM_PROMPTS" \
  --burstiness "$BURSTINESS" \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "50,90,95,99" \
  --save-result \
  --goodput "ttft:$SLO_TTFT" "tpot:$SLO_TPOT" "e2el:$SLO_E2EL" \
  --result-dir "$OUTDIR" \
  --metadata "tag=$TAG"

echo "Done. Results saved to $OUTDIR"
