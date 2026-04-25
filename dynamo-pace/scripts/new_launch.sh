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

# GPU assignment: each worker gets TP_SIZE consecutive GPUs.
# Worker 0 → GPUs 0..(TP_SIZE-1), worker 1 → GPUs TP_SIZE..(2*TP_SIZE-1), etc.
TOTAL_WORKERS=$(( NUM_PREFILL + NUM_DECODE ))
TOTAL_GPUS=$(( TOTAL_WORKERS * TP_SIZE ))

echo "[launch] Frontend port:   $FRONTEND_PORT"
echo "[launch] Prefill workers: $NUM_PREFILL"
echo "[launch] Decode workers:  $NUM_DECODE"
echo "[launch] Tensor parallel: $TP_SIZE"
echo "[launch] GPUs required:   $TOTAL_GPUS"

# ---------------------------------------------------------------------------
# Check that NIXL ports are free before launching
# ---------------------------------------------------------------------------
NIXL_BASE_PORT=5600
echo "[launch] Checking NIXL ports..."
for ((i=0; i<TOTAL_WORKERS; i++)); do
	    port=$(( NIXL_BASE_PORT + i ))
	        if fuser "$port/tcp" &>/dev/null; then
			        echo "[launch] ERROR: Port $port is already in use. Run: fuser -k $port/tcp" >&2
				        exit 1
					    fi
				    done
				    echo "[launch] All NIXL ports are free."

				    # ---------------------------------------------------------------------------
				    # Build common vllm args (no --kv-transfer-config here; set per-worker below)
				    # ---------------------------------------------------------------------------
				    VLLM_ARGS=(
					        --model "$MODEL"
						    --tensor-parallel-size "$TP_SIZE"
						        --max-num-seqs "$MAX_NUM_SEQS"
							    --max-model-len "$MAX_MODEL_LEN"
							        --dtype "$DTYPE"
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
												)
												# NOTE: $IMAGE is NOT included here so per-worker --env flags can be inserted before it

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
															    "${APPTAINER_BASE[@]}" "$IMAGE" nats-server -js &
															    PIDS+=($!)
															    sleep 2

															    # ---------------------------------------------------------------------------
															    # 2. Start etcd
															    # ---------------------------------------------------------------------------
															    echo "[launch] Starting etcd..."
															    "${APPTAINER_BASE[@]}" "$IMAGE" \
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
																																		"${APPTAINER_BASE[@]}" "$IMAGE" \
																																			    python3 -m dynamo.frontend \
																																			        --http-port "$FRONTEND_PORT" &
																																																																						PIDS+=($!)

																																																																						# ---------------------------------------------------------------------------
																																																																						# 4. Start prefill workers — each gets TP_SIZE GPUs and a unique NIXL port
																																																																						# ---------------------------------------------------------------------------
																																																																						GPU_IDX=0
																																																																						NIXL_PORT=$NIXL_BASE_PORT

																																																																						for ((i=0; i<NUM_PREFILL; i++)); do
																																																																							    GPU_LIST=$(seq -s, "$GPU_IDX" $(( GPU_IDX + TP_SIZE - 1 )))
																																																																							        echo "[launch] Starting prefill worker $((i+1))/$NUM_PREFILL on GPU(s) $GPU_LIST, NIXL port $NIXL_PORT..."
																																																																								    APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU_LIST" \
																																																																									        APPTAINERENV_VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
																																																																										    "${APPTAINER_BASE[@]}" "$IMAGE" \
																																																																										            python3 -m dynamo.vllm \
																																																																											            "${VLLM_ARGS[@]}" \
																																																																												            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
																																																																													            --disaggregation-mode prefill &
																																																																								        PIDS+=($!)
																																																																									    GPU_IDX=$(( GPU_IDX + TP_SIZE ))
																																																																									        NIXL_PORT=$(( NIXL_PORT + 1 ))
																																																																									done

																																																																									# ---------------------------------------------------------------------------
																																																																									# 5. Start decode workers — each gets TP_SIZE GPUs and a unique NIXL port
																																																																									# ---------------------------------------------------------------------------
																																																																									for ((i=0; i<NUM_DECODE; i++)); do
																																																																										    GPU_LIST=$(seq -s, "$GPU_IDX" $(( GPU_IDX + TP_SIZE - 1 )))
																																																																										        echo "[launch] Starting decode worker $((i+1))/$NUM_DECODE on GPU(s) $GPU_LIST, NIXL port $NIXL_PORT..."
																																																																											    APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU_LIST" \
																																																																												        APPTAINERENV_VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
																																																																													    "${APPTAINER_BASE[@]}" "$IMAGE" \
																																																																													            python3 -m dynamo.vllm \
																																																																														            "${VLLM_ARGS[@]}" \
																																																																															            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
																																																																																            --disaggregation-mode decode &
																																																																											        PIDS+=($!)
																																																																												    GPU_IDX=$(( GPU_IDX + TP_SIZE ))
																																																																												        NIXL_PORT=$(( NIXL_PORT + 1 ))
																																																																												done

																																																																												echo "[launch] All processes started. Waiting..."
																																																																												echo "[launch] PIDs: ${PIDS[*]}"

																																																																												# Wait until any process exits (crash or signal)
																																																																												wait -n 2>/dev/null || wait
