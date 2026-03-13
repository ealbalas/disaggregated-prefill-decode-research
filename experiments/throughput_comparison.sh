#!/usr/bin/env bash
# experiments/throughput_comparison.sh
#
# Compares throughput across input/output length combinations for each config
# (colocated baseline + disaggregated ratios).
#
# Runs a full grid of INPUT_LENS × OUTPUT_LENS × REQUEST_RATES, making it easy
# to see how each setup handles different workload mixes side-by-side.
#
# Results saved to results/throughput_comparison/
#
# Usage:
#   bash experiments/throughput_comparison.sh
#   bash experiments/throughput_comparison.sh configs/throughput_comparison.env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

OUTDIR="$SCRATCH/disaggregated-prefill-decode-research/results/throughput_comparison"

SWEEP_CONFIG="${1:-$REPO_ROOT/configs/throughput_comparison.env}"
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
        wait "${PIDS[@]}" 2>/dev/null || true
        PIDS=()
    fi
}

# ── Benchmark runner ──────────────────────────────────────────────────────────
run_benchmark() {
    local port=$1 input_len=$2 output_len=$3 rate=$4 concurrency=$5 tag=$6

    echo "  → Running: $tag"

    if vllm bench serve \
        --backend vllm \
        --port "$port" \
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
        --goodput "ttft:$SLO_TTFT" "tpot:$SLO_TPOT" "e2el:$SLO_E2EL" \
        --result-dir "$OUTDIR" \
        --metadata "tag=$tag"; then
        latest=$(ls -t "$OUTDIR"/*.json 2>/dev/null | head -1)
        [[ -n "$latest" ]] && mv "$latest" "$OUTDIR/${tag}.json"
        echo "  ✓ Done: $tag"
    else
        echo "  ✗ Failed: $tag (skipping)" >&2
    fi
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() { stop_servers; }
trap cleanup EXIT

mkdir -p "$OUTDIR"
echo "Results dir: $OUTDIR"

# ── Defaults for phase-control vars ──────────────────────────────────────────
RUN_COLOCATED="${RUN_COLOCATED:-true}"
RUN_DISAGG="${RUN_DISAGG:-true}"

WORKLOADS=$((${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]}))
COMBOS=$((WORKLOADS * ${#REQUEST_RATES[@]} * ${#CONCURRENCIES[@]}))
ACTIVE_PHASES=0
[[ "$RUN_COLOCATED" == true ]] && ACTIVE_PHASES=$((ACTIVE_PHASES + 1))
[[ "$RUN_DISAGG"    == true ]] && ACTIVE_PHASES=$((ACTIVE_PHASES + ${#RATIOS[@]}))
TOTAL=$((ACTIVE_PHASES * COMBOS))

echo "Starting throughput comparison: $TOTAL total runs."
echo "  Input lens:  ${INPUT_LENS[*]}"
echo "  Output lens: ${OUTPUT_LENS[*]}"
echo "  Rates:       ${REQUEST_RATES[*]}"
echo "  Phases:      colocated=$RUN_COLOCATED  disagg=$RUN_DISAGG"

RUN=0

# ── Workload loop for a given port and config label ───────────────────────────
run_workloads() {
    local port=$1 config_label=$2
    for input_len in "${INPUT_LENS[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            for rate in "${REQUEST_RATES[@]}"; do
                for concurrency in "${CONCURRENCIES[@]}"; do
                    RUN=$((RUN + 1))
                    tag="${config_label}_${input_len}x${output_len}_rate${rate}_conc${concurrency}"
                    echo "[$RUN/$TOTAL] $tag"
                    run_benchmark "$port" "$input_len" "$output_len" "$rate" "$concurrency" "$tag"
                done
            done
        done
    done
}

# ══ Phase 1: Colocated ════════════════════════════════════════════════════════
if [[ "$RUN_COLOCATED" == true ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 1: Colocated (GPU $COLOCATED_GPU, port $COLOCATED_PORT)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES="$COLOCATED_GPU" \
        vllm serve "$MODEL" \
        --port "$COLOCATED_PORT" \
        --gpu-memory-utilization 0.8 \
        --max-model-len 2048 \
        --max-num-seqs 64 &
    PIDS+=($!)
    wait_for_port "$COLOCATED_PORT" "colocated"

    run_workloads "$COLOCATED_PORT" "colocated"

    stop_servers
else
    echo ""
    echo "Skipping Phase 1 (colocated): RUN_COLOCATED=false"
fi

# ══ Phase 2: Disaggregated ════════════════════════════════════════════════════
if [[ "$RUN_DISAGG" != true ]]; then
    echo ""
    echo "Skipping Phase 2 (disaggregated): RUN_DISAGG=false"
fi
for ratio in "${RATIOS[@]}"; do
    [[ "$RUN_DISAGG" == true ]] || continue
    RATIO_CONFIG="$REPO_ROOT/configs/ratios/${ratio}.env"

    if [[ ! -f "$RATIO_CONFIG" ]]; then
        echo "Warning: config not found for ratio '$ratio', skipping." >&2
        continue
    fi

    source "$RATIO_CONFIG"

    local_prefill_hosts="" local_prefill_ports="" local_decode_hosts="" local_decode_ports=""
    for port in "${PREFILL_PORTS[@]}"; do
        local_prefill_hosts+="localhost "
        local_prefill_ports+="$port "
    done
    for port in "${DECODE_PORTS[@]}"; do
        local_decode_hosts+="localhost "
        local_decode_ports+="$port "
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 2: Disaggregated — ratio $ratio"
    echo "  Prefill ports: ${PREFILL_PORTS[*]}"
    echo "  Decode  ports: ${DECODE_PORTS[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cd "$VLLM_PROXY_SRC"
    python toy_proxy_server.py \
        --port "$PROXY_PORT" \
        --prefiller-hosts $local_prefill_hosts \
        --prefiller-ports $local_prefill_ports \
        --decoder-hosts $local_decode_hosts \
        --decoder-ports $local_decode_ports &
    PIDS+=($!)
    wait_for_port "$PROXY_PORT" "proxy"

    for i in "${!PREFILL_PORTS[@]}"; do
        CUDA_VISIBLE_DEVICES="${PREFILL_GPUS[$i]}" \
            VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNELS[$i]}" \
            vllm serve "$MODEL" \
            --port "${PREFILL_PORTS[$i]}" \
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
            --gpu-memory-utilization 0.8 \
            --max-model-len 2048 \
            --max-num-seqs 64 &
        PIDS+=($!)
        wait_for_port "${PREFILL_PORTS[$i]}" "prefill_$i"
    done

    for i in "${!DECODE_PORTS[@]}"; do
        CUDA_VISIBLE_DEVICES="${DECODE_GPUS[$i]}" \
            VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNELS[$i]}" \
            vllm serve "$MODEL" \
            --port "${DECODE_PORTS[$i]}" \
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
            --gpu-memory-utilization 0.8 \
            --max-model-len 2048 \
            --max-num-seqs "${DECODE_MAX_NUM_SEQS:-64}" &
        PIDS+=($!)
        wait_for_port "${DECODE_PORTS[$i]}" "decode_$i"
    done

    run_workloads "$PROXY_PORT" "$ratio"

    stop_servers
done

echo ""
echo "Throughput comparison complete. Results in $OUTDIR"
echo "Run: python utils/plot_throughput_comparison.py to generate plots."
