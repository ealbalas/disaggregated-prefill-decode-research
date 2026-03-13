#!/usr/bin/env bash
# experiments/capacity_sweep.sh
#
# Adaptive capacity sweep: finds the maximum input/output length each config
# can sustain at a target throughput (TARGET_TOK_PER_S tok/s).
#
# Phase 1 — Prefill sweep: output_len=1, doubles input_len until throughput
#   drops below target. Records the last passing input length.
#
# Phase 2 — Decode sweep: fixes input_len from phase 1, doubles output_len
#   until throughput drops below target.
#
# Results saved to results/capacity_sweep/
#
# Usage:
#   bash experiments/capacity_sweep.sh
#   bash experiments/capacity_sweep.sh configs/capacity_sweep.env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

OUTDIR="$SCRATCH/disaggregated-prefill-decode-research/results/capacity_sweep"

SWEEP_CONFIG="${1:-$REPO_ROOT/configs/capacity_sweep.env}"
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
# Returns 0 on success, 1 on failure. Saves result to $OUTDIR/${tag}.json.
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
        return 0
    else
        echo "  ✗ Failed: $tag (skipping)" >&2
        return 1
    fi
}

# ── Throughput helpers ─────────────────────────────────────────────────────────
read_throughput() {
    local result_file=$1
    python3 -c "import json; d=json.load(open('$result_file')); print(d.get('output_throughput', 0))"
}

above_target() {
    local tok_s=$1
    python3 -c "import sys; sys.exit(0 if float('$tok_s') >= $TARGET_TOK_PER_S else 1)"
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() { stop_servers; }
trap cleanup EXIT

mkdir -p "$OUTDIR"
echo "Results dir: $OUTDIR"

# ── Defaults for phase-control vars ──────────────────────────────────────────
RUN_COLOCATED="${RUN_COLOCATED:-true}"
RUN_DISAGG="${RUN_DISAGG:-true}"

echo "Starting capacity sweep."
echo "  Target throughput: ${TARGET_TOK_PER_S} tok/s"
echo "  Request rate:      ${REQUEST_RATE} req/s  concurrency: ${CONCURRENCY}"
echo "  Prefill sweep:     input_len ${START_INPUT_LEN} → ${MAX_INPUT_LEN} (doubling), output_len=1"
echo "  Decode sweep:      output_len ${START_OUTPUT_LEN} → ${MAX_OUTPUT_LEN} (doubling), input_len=<phase1 result>"
echo "  Phases:            colocated=$RUN_COLOCATED  disagg=$RUN_DISAGG"

# ── Core capacity sweep logic ─────────────────────────────────────────────────
run_capacity_sweep() {
    local port=$1 config_label=$2
    local max_passing_input=0

    # ── Phase 1: Prefill sweep (output_len=1, double input_len) ───────────────
    echo ""
    echo "  [Prefill sweep] output_len=1, doubling input_len from $START_INPUT_LEN to $MAX_INPUT_LEN"
    local input_len=$START_INPUT_LEN

    while [[ $input_len -le $MAX_INPUT_LEN ]]; do
        local tag="${config_label}_prefill_${input_len}x1_rate${REQUEST_RATE}_conc${CONCURRENCY}"
        if run_benchmark "$port" "$input_len" 1 "$REQUEST_RATE" "$CONCURRENCY" "$tag"; then
            local result_file="$OUTDIR/${tag}.json"
            local tok_s
            tok_s=$(read_throughput "$result_file")
            echo "    input_len=$input_len → ${tok_s} tok/s (target: ${TARGET_TOK_PER_S})"

            if above_target "$tok_s"; then
                max_passing_input=$input_len
                input_len=$((input_len * 2))
            else
                echo "    ✗ Throughput dropped below target at input_len=$input_len — stopping prefill sweep."
                break
            fi
        else
            echo "    ✗ Benchmark failed at input_len=$input_len — stopping prefill sweep." >&2
            break
        fi
    done

    if [[ $max_passing_input -eq 0 ]]; then
        echo "  [Prefill sweep] No passing input length found — skipping decode sweep."
        return
    fi
    echo "  [Prefill sweep] Max passing input_len: $max_passing_input"

    # ── Phase 2: Decode sweep (fix input_len, double output_len) ──────────────
    echo ""
    echo "  [Decode sweep] input_len=$max_passing_input, doubling output_len from $START_OUTPUT_LEN to $MAX_OUTPUT_LEN"
    local output_len=$START_OUTPUT_LEN

    while [[ $output_len -le $MAX_OUTPUT_LEN ]]; do
        local tag="${config_label}_decode_${max_passing_input}x${output_len}_rate${REQUEST_RATE}_conc${CONCURRENCY}"
        if run_benchmark "$port" "$max_passing_input" "$output_len" "$REQUEST_RATE" "$CONCURRENCY" "$tag"; then
            local result_file="$OUTDIR/${tag}.json"
            local tok_s
            tok_s=$(read_throughput "$result_file")
            echo "    output_len=$output_len → ${tok_s} tok/s (target: ${TARGET_TOK_PER_S})"

            if above_target "$tok_s"; then
                output_len=$((output_len * 2))
            else
                echo "    ✗ Throughput dropped below target at output_len=$output_len — stopping decode sweep."
                break
            fi
        else
            echo "    ✗ Benchmark failed at output_len=$output_len — stopping decode sweep." >&2
            break
        fi
    done

    echo "  [Decode sweep] Complete."
}

# ══ Phase 1: Colocated ════════════════════════════════════════════════════════
if [[ "$RUN_COLOCATED" == true ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Colocated (GPU $COLOCATED_GPU, port $COLOCATED_PORT)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES="$COLOCATED_GPU" \
        vllm serve "$MODEL" \
        --port "$COLOCATED_PORT" \
        --gpu-memory-utilization 0.8 \
        --max-model-len 2048 \
        --max-num-seqs 64 &
    PIDS+=($!)
    wait_for_port "$COLOCATED_PORT" "colocated"

    run_capacity_sweep "$COLOCATED_PORT" "colocated"

    stop_servers
else
    echo ""
    echo "Skipping colocated: RUN_COLOCATED=false"
fi

# ══ Phase 2: Disaggregated ════════════════════════════════════════════════════
if [[ "$RUN_DISAGG" != true ]]; then
    echo ""
    echo "Skipping disaggregated: RUN_DISAGG=false"
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
    echo "Disaggregated — ratio $ratio"
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

    run_capacity_sweep "$PROXY_PORT" "$ratio"

    stop_servers
done

echo ""
echo "Capacity sweep complete. Results in $OUTDIR"
echo "Run: python utils/plot_capacity_sweep.py to generate plots."
