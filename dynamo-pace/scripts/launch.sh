#!/usr/bin/env bash
# launch.sh — Launch the Dynamo disaggregated serving stack inside Apptainer.
#
# Usage:
#   bash launch.sh [config.yaml]
#
# The config.yaml must define:
#   Frontend.port
#   Prefill.num_workers
#   Decode.num_workers
#   VllmWorker.*  (model, tensor_parallel_size, etc.)
#
# This script:
#   1. Starts nats-server and etcd inside the container
#   2. Starts the Dynamo frontend (python -m dynamo.frontend)
#   3. Starts N prefill workers (python -m dynamo.vllm --disaggregation-mode prefill)
#   4. Starts N decode workers  (python -m dynamo.vllm --disaggregation-mode decode)
#
# All processes are tracked; cleanup kills them all on exit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CONFIG="${1:-$DYNAMO_PACE_DIR/configs/disagg_1p1d.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "[launch] ERROR: Config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "[launch] ERROR: Container SIF not found: $IMAGE" >&2
    echo "[launch] Run scripts/pull_image.sh first." >&2
    exit 1
fi

echo "[launch] Config:  $CONFIG"
echo "[launch] Image:   $IMAGE"
echo "[launch] Model:   $MODEL"

# ---------------------------------------------------------------------------
# Parse config.yaml with Python (available on PACE via anaconda3 module)
# ---------------------------------------------------------------------------
read_yaml() {
    python3 - "$CONFIG" "$1" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
keys = sys.argv[2].split(".")
val = cfg
for k in keys:
    val = val[k]
print(val)
EOF
}

FRONTEND_PORT=$(read_yaml "Frontend.port")
NUM_PREFILL=$(read_yaml "Prefill.num_workers")
NUM_DECODE=$(read_yaml "Decode.num_workers")
TP_SIZE=$(read_yaml "VllmWorker.tensor_parallel_size")
MAX_NUM_SEQS=$(read_yaml "VllmWorker.max_num_seqs")
MAX_MODEL_LEN=$(read_yaml "VllmWorker.max_model_len")
DTYPE=$(read_yaml "VllmWorker.dtype")
ENFORCE_EAGER=$(read_yaml "VllmWorker.enforce_eager")

echo "[launch] Frontend port:   $FRONTEND_PORT"
echo "[launch] Prefill workers: $NUM_PREFILL"
echo "[launch] Decode workers:  $NUM_DECODE"
echo "[launch] Tensor parallel: $TP_SIZE"

# ---------------------------------------------------------------------------
# Build common vllm args
# ---------------------------------------------------------------------------
VLLM_ARGS=(
    --model "$MODEL"
    --tensor-parallel-size "$TP_SIZE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --dtype "$DTYPE"
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
    --disable-log-requests
)

if [[ "$ENFORCE_EAGER" == "True" || "$ENFORCE_EAGER" == "true" ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

# ---------------------------------------------------------------------------
# Common apptainer wrapper
# ---------------------------------------------------------------------------
APPTAINER_BASE=(
    apptainer exec
    --nv
    --bind "$REPO_ROOT:$REPO_ROOT"
    --bind "$MODEL:$MODEL"
    --env MODEL="$MODEL"
    --env CC=/usr/bin/gcc
    --env CXX=/usr/bin/g++
    --env VLLM_USE_DEEP_GEMM=0
    --env TORCHINDUCTOR_COMBO_KERNELS=0
    "$IMAGE"
)

# ---------------------------------------------------------------------------
# Cleanup: kill all background PIDs on exit
# ---------------------------------------------------------------------------
PIDS=()
cleanup() {
    echo "[launch] Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[launch] Done."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 1. Start nats-server
# ---------------------------------------------------------------------------
echo "[launch] Starting nats-server..."
"${APPTAINER_BASE[@]}" nats-server -js &
PIDS+=($!)
sleep 2

# ---------------------------------------------------------------------------
# 2. Start etcd
# ---------------------------------------------------------------------------
echo "[launch] Starting etcd..."
"${APPTAINER_BASE[@]}" \
    etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379 \
    --data-dir /tmp/etcd-dynamo &
PIDS+=($!)
sleep 2

# ---------------------------------------------------------------------------
# 3. Start frontend
# ---------------------------------------------------------------------------
echo "[launch] Starting frontend on port $FRONTEND_PORT..."
"${APPTAINER_BASE[@]}" \
    python3 -m dynamo.frontend \
    --http-port "$FRONTEND_PORT" &
PIDS+=($!)

# ---------------------------------------------------------------------------
# 4. Start prefill workers
# ---------------------------------------------------------------------------
for ((i=0; i<NUM_PREFILL; i++)); do
    echo "[launch] Starting prefill worker $((i+1))/$NUM_PREFILL..."
    "${APPTAINER_BASE[@]}" \
        python3 -m dynamo.vllm \
        "${VLLM_ARGS[@]}" \
        --disaggregation-mode prefill &
    PIDS+=($!)
done

# ---------------------------------------------------------------------------
# 5. Start decode workers
# ---------------------------------------------------------------------------
for ((i=0; i<NUM_DECODE; i++)); do
    echo "[launch] Starting decode worker $((i+1))/$NUM_DECODE..."
    "${APPTAINER_BASE[@]}" \
        python3 -m dynamo.vllm \
        "${VLLM_ARGS[@]}" \
        --disaggregation-mode decode &
    PIDS+=($!)
done

echo "[launch] All processes started. Waiting..."
echo "[launch] PIDs: ${PIDS[*]}"

# Wait until any process exits (crash or signal)
wait -n 2>/dev/null || wait
