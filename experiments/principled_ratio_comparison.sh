#!/usr/bin/env bash
# experiments/principled_ratio_comparison.sh
#
# Principled ratio comparison: all 5 P:D ratios plus colocated baseline, at
# operating points derived from saturation_sweep results (~50% and ~90% of
# saturation), with per-workload E2EL SLOs derived from measured unloaded latency.
#
# Per-workload E2EL SLOs (2× colocated unloaded E2EL from existing baseline data):
#   1024x64  → 3000ms  (2 × 1509ms)
#   256x256  → 10000ms (2 × 4762ms)
#   64x256   → 10000ms (similar to balanced)
#
# Results saved to results/principled_ratio_comparison/
#
# Usage:
#   bash experiments/principled_ratio_comparison.sh
#   bash experiments/principled_ratio_comparison.sh configs/principled_ratio_comparison.env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

OUTDIR="$SCRATCH/disaggregated-prefill-decode-research/results/principled_ratio_comparison"

SWEEP_CONFIG="${1:-$REPO_ROOT/configs/principled_ratio_comparison.env}"
source "$SWEEP_CONFIG"

# Per-workload E2EL SLOs derived from 2× measured unloaded E2EL (colocated, low rate)
declare -A E2EL_SLO=(
    ["1024x64"]=3000
    ["256x256"]=10000
    ["64x256"]=10000
)

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
# Uses per-workload E2EL SLO from the associative array above.
run_benchmark() {
    local port=$1 input_len=$2 output_len=$3 rate=$4 concurrency=$5 tag=$6
    local workload_key="${input_len}x${output_len}"
    local e2el_slo="${E2EL_SLO[$workload_key]:-$SLO_E2EL}"

    echo "  → Running: $tag (E2EL SLO=${e2el_slo}ms)"

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
        --goodput "ttft:$SLO_TTFT" "tpot:$SLO_TPOT" "e2el:${e2el_slo}" \
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

TOTAL_WORKLOADS=${#INPUT_LENS[@]}
TOTAL_COMBOS=$((TOTAL_WORKLOADS * ${#REQUEST_RATES[@]} * ${#CONCURRENCIES[@]}))
TOTAL=$(((1 + ${#RATIOS[@]}) * TOTAL_COMBOS))
echo "Starting principled ratio comparison: $TOTAL total runs."
echo "  Colocated: $TOTAL_COMBOS runs"
echo "  Disaggregated (${#RATIOS[@]} ratios × $TOTAL_COMBOS): $((${#RATIOS[@]} * TOTAL_COMBOS)) runs"
echo "  Per-workload E2EL SLOs:"
for key in "${!E2EL_SLO[@]}"; do
    echo "    ${key}: ${E2EL_SLO[$key]}ms"
done

RUN=0

# ══ Phase 1: Colocated ═══════════════════════════════════════════════════════
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

for i in "${!INPUT_LENS[@]}"; do
    input_len="${INPUT_LENS[$i]}"
    output_len="${OUTPUT_LENS[$i]}"
    for rate in "${REQUEST_RATES[@]}"; do
        for concurrency in "${CONCURRENCIES[@]}"; do
            RUN=$((RUN + 1))
            tag="colocated_${input_len}x${output_len}_rate${rate}_conc${concurrency}"
            echo "[$RUN/$TOTAL] $tag"
            run_benchmark "$COLOCATED_PORT" "$input_len" "$output_len" "$rate" "$concurrency" "$tag"
        done
    done
done

stop_servers

# ══ Phase 2: Disaggregated ═══════════════════════════════════════════════════
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

    for i in "${!INPUT_LENS[@]}"; do
        input_len="${INPUT_LENS[$i]}"
        output_len="${OUTPUT_LENS[$i]}"
        for rate in "${REQUEST_RATES[@]}"; do
            for concurrency in "${CONCURRENCIES[@]}"; do
                RUN=$((RUN + 1))
                tag="${ratio}_${input_len}x${output_len}_rate${rate}_conc${concurrency}"
                echo "[$RUN/$TOTAL] $tag"
                run_benchmark "$PROXY_PORT" "$input_len" "$output_len" "$rate" "$concurrency" "$tag"
            done
        done
    done

    stop_servers
done

echo ""
echo "Principled ratio comparison complete. Results in $OUTDIR"
echo "Run: python utils/plot_principled_ratio_comparison.py to generate plots."
