#!/usr/bin/env bash
# run_benchmark.sh — Drive the vLLM benchmark client against a running Dynamo server.
#
# Usage:
#   bash run_benchmark.sh [options]
#
# Options:
#   --host HOST          Server host (default: localhost)
#   --port PORT          Server port (default: 8000)
#   --input-len N        Prompt token length (default: 512)
#   --output-len N       Output token length (default: 128)
#   --rate R             Request rate req/s (default: 4)
#   --num-prompts N      Total number of prompts to send (default: 200)
#   --tag LABEL          Output filename tag (default: benchmark)
#   --results-dir DIR    Where to write JSON output (default: $RESULTS_DIR)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/env.sh"   # Bug fix #4

# Defaults
HOST="localhost"
PORT="8000"
INPUT_LEN="512"
OUTPUT_LEN="128"
RATE="4"
NUM_PROMPTS="200"
TAG="benchmark"
OUT_DIR="${RESULTS_DIR}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         HOST="$2";        shift 2 ;;
        --port)         PORT="$2";        shift 2 ;;
        --input-len)    INPUT_LEN="$2";   shift 2 ;;
        --output-len)   OUTPUT_LEN="$2";  shift 2 ;;
        --rate)         RATE="$2";        shift 2 ;;
        --num-prompts)  NUM_PROMPTS="$2"; shift 2 ;;
        --tag)          TAG="$2";         shift 2 ;;
        --results-dir)  OUT_DIR="$2";     shift 2 ;;
        *) echo "[benchmark] Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUT_DIR/${TAG}_in${INPUT_LEN}_out${OUTPUT_LEN}_rate${RATE}_${TIMESTAMP}.json"

echo "[benchmark] Host:        $HOST:$PORT"
echo "[benchmark] Input len:   $INPUT_LEN tokens"
echo "[benchmark] Output len:  $OUTPUT_LEN tokens"
echo "[benchmark] Rate:        $RATE req/s"
echo "[benchmark] Num prompts: $NUM_PROMPTS"
echo "[benchmark] Output:      $OUTFILE"

# Verify server is reachable before starting
# Bug fix #5: use single-quoted JSON in curl to avoid bash history expansion
if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
    echo "[benchmark] ERROR: Server not reachable at http://${HOST}:${PORT}/health" >&2
    exit 1
fi

# Fetch model name from server (bug #5: single-quoted JSON)
MODEL_NAME=$(curl -sf 'http://'"${HOST}:${PORT}"'/v1/models' \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
echo "[benchmark] Model: $MODEL_NAME"

# Run the vLLM benchmark inside the container
# The container has vLLM's benchmark_serving.py available
apptainer exec \
    --nv \
    --bind "$REPO_ROOT:$REPO_ROOT" \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env VLLM_USE_DEEP_GEMM=0 \
    --env TORCHINDUCTOR_COMBO_KERNELS=0 \
    "$IMAGE" \
    python3 -m vllm.entrypoints.cli.main bench serve \
        --backend vllm \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL_NAME" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --request-rate "$RATE" \
        --num-prompts "$NUM_PROMPTS" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-filename "$OUTFILE"

echo "[benchmark] Results saved to: $OUTFILE"
