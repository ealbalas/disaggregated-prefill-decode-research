#!/usr/bin/env bash
# experiments/sweep.sh
#
# Runs a full parameter sweep across all ratios, input/output lengths,
# request rates, and concurrency levels defined in configs/sweep.env.
#
# Each ratio's servers are started once, all benchmarks for that ratio
# are run, then servers are torn down before moving to the next ratio.
#
# Usage:
#   bash experiments/sweep.sh
#   bash experiments/sweep.sh configs/sweep.env   # explicit config

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

SWEEP_CONFIG="${1:-$REPO_ROOT/configs/sweep.env}"
source "$SWEEP_CONFIG"

PIDS=()

# ── Helpers ───────────────────────────────────────────────────────────────────
wait_for_port() {
  local port=$1 label=$2
  echo "  Waiting for $label on port $port..."
  until nc -z localhost "$port" 2>/dev/null; do sleep 2; done
  echo "  ✓ $label ready."
}

stop_servers() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping servers (PIDs: ${PIDS[*]})..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    sleep 3
    PIDS=()
  fi
}

start_servers() {
  local ratio_config="$1"
  source "$ratio_config"

  # Build proxy args
  local prefill_hosts="" prefill_port_args="" decode_hosts="" decode_port_args=""
  for port in "${PREFILL_PORTS[@]}"; do prefill_hosts+="localhost "; prefill_port_args+="$port "; done
  for port in "${DECODE_PORTS[@]}";  do decode_hosts+="localhost ";  decode_port_args+="$port ";  done

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Starting servers for ratio: $RATIO"
  echo "  Prefill ports: ${PREFILL_PORTS[*]}"
  echo "  Decode  ports: ${DECODE_PORTS[*]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Proxy
  cd "$VLLM_PROXY_SRC"
  python toy_proxy_server.py \
    --port "$PROXY_PORT" \
    --prefiller-hosts $prefill_hosts \
    --prefiller-ports $prefill_port_args \
    --decoder-hosts $decode_hosts \
    --decoder-ports $decode_port_args &
  PIDS+=($!)
  wait_for_port "$PROXY_PORT" "proxy"

  # Prefill instances
  for i in "${!PREFILL_PORTS[@]}"; do
    CUDA_VISIBLE_DEVICES="${PREFILL_GPUS[$i]}" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNELS[$i]}" \
    vllm serve "$MODEL" \
      --port "${PREFILL_PORTS[$i]}" \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
    PIDS+=($!)
    wait_for_port "${PREFILL_PORTS[$i]}" "prefill_$i"
  done

  # Decode instances
  for i in "${!DECODE_PORTS[@]}"; do
    CUDA_VISIBLE_DEVICES="${DECODE_GPUS[$i]}" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNELS[$i]}" \
    vllm serve "$MODEL" \
      --port "${DECODE_PORTS[$i]}" \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
    PIDS+=($!)
    wait_for_port "${DECODE_PORTS[$i]}" "decode_$i"
  done
}

run_benchmark() {
  local input_len=$1 output_len=$2 rate=$3 concurrency=$4
  local tag="${RATIO}_${input_len}x${output_len}_rate${rate}_conc${concurrency}"

  echo "  → Running: $tag"

  vllm bench serve \
    --backend vllm \
    --port "$PROXY_PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --ignore-eos \
    --request-rate "$rate" \
    --max-concurrency "$concurrency" \
    --num-prompts "$NUM_PROMPTS" \
    --burstiness "$BURSTINESS" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,95,99" \
    --save-result \
    --save-detailed \
    --goodput "ttft:$SLO_TTFT" "tpot:$SLO_TPOT" "e2el:$SLO_E2EL" \
    --result-dir "$OUTDIR" \
    --metadata "tag=$tag"

  echo "  ✓ Done: $tag"
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() { stop_servers; }
trap cleanup EXIT

# ── Main sweep ────────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"

TOTAL=$(( ${#RATIOS[@]} * ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * ${#REQUEST_RATES[@]} * ${#CONCURRENCIES[@]} ))
echo "Starting sweep: $TOTAL total runs across ${#RATIOS[@]} ratios."

RUN=0
for ratio in "${RATIOS[@]}"; do
  RATIO_CONFIG="$REPO_ROOT/configs/ratios/${ratio}.env"

  if [[ ! -f "$RATIO_CONFIG" ]]; then
    echo "Warning: config not found for ratio '$ratio', skipping." >&2
    continue
  fi

  source "$RATIO_CONFIG"
  start_servers "$RATIO_CONFIG"

  for input_len in "${INPUT_LENS[@]}"; do
    for output_len in "${OUTPUT_LENS[@]}"; do
      for rate in "${REQUEST_RATES[@]}"; do
        for concurrency in "${CONCURRENCIES[@]}"; do
          RUN=$(( RUN + 1 ))
          echo "[$RUN/$TOTAL] ratio=$ratio input=$input_len output=$output_len rate=$rate concurrency=$concurrency"
          run_benchmark "$input_len" "$output_len" "$rate" "$concurrency"
        done
      done
    done
  done

  stop_servers
done

echo ""
echo "Sweep complete. Results in $OUTDIR"
echo "Run: python utils/plot_results.py to generate plots."
