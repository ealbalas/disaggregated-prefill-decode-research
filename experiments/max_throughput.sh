#!/usr/bin/env bash
# experiments/max_throughput.sh
#
# Max throughput characterization: separately measures prefill and decode
# capacity by isolating each stage.
#
# Prefill capacity: output_len=1 across input lengths × request rates.
#   Saturation = rate where request_throughput < rate (achieved < offered).
#
# Decode capacity: input_len=64 (fixed) across output lengths × request rates.
#   Same saturation detection.
#
# Results saved to results/max_throughput/
#
# Usage:
#   bash experiments/max_throughput.sh
#   bash experiments/max_throughput.sh configs/max_throughput.env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

OUTDIR="$SCRATCH/disaggregated-prefill-decode-research/results/max_throughput"

SWEEP_CONFIG="${1:-$REPO_ROOT/configs/max_throughput.env}"
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

PREFILL_WORKLOADS=${#PREFILL_INPUT_LENS[@]}
DECODE_WORKLOADS=${#DECODE_OUTPUT_LENS[@]}
TOTAL_WORKLOADS=$((PREFILL_WORKLOADS + DECODE_WORKLOADS))
COMBOS_PER_WORKLOAD=$((${#REQUEST_RATES[@]} * ${#CONCURRENCIES[@]}))
TOTAL=$(((1 + ${#RATIOS[@]}) * TOTAL_WORKLOADS * COMBOS_PER_WORKLOAD))
echo "Starting max throughput characterization: $TOTAL total runs."
echo "  Prefill sweep: ${PREFILL_INPUT_LENS[*]} input lengths (output_len=1)"
echo "  Decode sweep:  ${DECODE_OUTPUT_LENS[*]} output lengths (input_len=64)"

RUN=0

# ── Helper: run both sweep phases against a given port and config label ───────
run_sweeps() {
    local port=$1 config_label=$2

    # Prefill sweep
    for i in "${!PREFILL_INPUT_LENS[@]}"; do
        local input_len="${PREFILL_INPUT_LENS[$i]}"
        local output_len="${PREFILL_OUTPUT_LENS[$i]}"
        for rate in "${REQUEST_RATES[@]}"; do
            for concurrency in "${CONCURRENCIES[@]}"; do
                RUN=$((RUN + 1))
                tag="${config_label}_${input_len}x${output_len}_rate${rate}_conc${concurrency}"
                echo "[$RUN/$TOTAL] prefill sweep — $tag"
                run_benchmark "$port" "$input_len" "$output_len" "$rate" "$concurrency" "$tag"
            done
        done
    done

    # Decode sweep
    for i in "${!DECODE_INPUT_LENS[@]}"; do
        local input_len="${DECODE_INPUT_LENS[$i]}"
        local output_len="${DECODE_OUTPUT_LENS[$i]}"
        for rate in "${REQUEST_RATES[@]}"; do
            for concurrency in "${CONCURRENCIES[@]}"; do
                RUN=$((RUN + 1))
                tag="${config_label}_${input_len}x${output_len}_rate${rate}_conc${concurrency}"
                echo "[$RUN/$TOTAL] decode sweep — $tag"
                run_benchmark "$port" "$input_len" "$output_len" "$rate" "$concurrency" "$tag"
            done
        done
    done
}

# ══ Phase 1: Colocated ════════════════════════════════════════════════════════
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

run_sweeps "$COLOCATED_PORT" "colocated"

stop_servers

# ══ Phase 2: Disaggregated ════════════════════════════════════════════════════
for ratio in "${RATIOS[@]}"; do
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

    run_sweeps "$PROXY_PORT" "$ratio"

    stop_servers
done

echo ""
echo "Max throughput characterization complete. Results in $OUTDIR"
echo "Run: python utils/plot_max_throughput.py to generate plots."
