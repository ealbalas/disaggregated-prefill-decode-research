# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project benchmarking **disaggregated LLM inference**, where the prefill and decode phases run on separate GPU instances connected via Nixl KV-cache transfer. The system measures latency/throughput tradeoffs across different prefill:decode server ratios.

## Environment Setup

The project runs on Georgia Tech's PACE ICE cluster. Before running any experiment script, `utils/env.sh` must be sourced — it sets key paths and activates the conda environment:

```bash
# Key env vars set by utils/env.sh
SCRATCH=/storage/ice1/4/5/ealbalas3
MODEL=$SCRATCH/disaggregated-prefill-decode-research/model
OUTDIR=$SCRATCH/disaggregated-prefill-decode-research/results/stats
VLLM_PROXY_SRC=$SCRATCH/disaggregated-prefill-decode-research/proxy

# HPC modules loaded
module load gcc cuda anaconda3
conda activate vllm_disagg
```

If running locally (not on PACE), you'll need to adapt `utils/env.sh` paths before sourcing.

## Running Experiments

**Download the model first (one-time setup):**
```bash
HF_TOKEN=hf_xxx bash utils/download_model.sh
```

**Run a single experiment (one P:D ratio):**
```bash
bash experiments/run_experiment.sh                              # 1p1d baseline
bash experiments/run_experiment.sh configs/2p3d_random.env     # custom topology
```

**Run the full parameter sweep:**
```bash
bash experiments/sweep.sh                              # uses configs/sweep.env
bash experiments/sweep.sh configs/custom_sweep.env    # explicit config
```

**Visualize results:**
```bash
python utils/plot_results.py
python utils/plot_results.py --results-dir /path/to/stats --plots-dir /path/to/plots
```

## Architecture

### Request Flow

```
Client
  ↓
[Proxy :8192]  (round-robin across prefill pool)
  ↓
[Prefill Instance(s)]  (vllm serve, max_tokens=1, returns KV params)
  ↓  (kv_transfer_params extracted from response)
[Decode Instance(s)]   (vllm serve, streams tokens back)
  ↓
Client
```

The proxy (`proxy/toy_proxy_server.py`) is a FastAPI server that:
1. Receives an OpenAI-compatible request
2. Forwards to a prefill instance with `max_tokens=1` and `do_remote_decode=True`
3. Extracts `kv_transfer_params` from prefill response
4. Forwards to a decode instance (with KV params injected) and streams the result back

KV cache is transferred between instances using **Nixl** (`NixlConnector`) over side-channel ports.

### Configuration System

Each ratio config (`configs/ratios/{ratio}.env`) defines:
- `PREFILL_PORTS`, `PREFILL_GPUS`, `PREFILL_SIDE_CHANNELS` — arrays, one per prefill instance
- `DECODE_PORTS`, `DECODE_GPUS`, `DECODE_SIDE_CHANNELS` — arrays, one per decode instance

The sweep config (`configs/sweep.env`) defines which ratios to iterate over and the benchmark parameter grid:
- `RATIOS` — must match filenames in `configs/ratios/`
- `INPUT_LENS`, `OUTPUT_LENS`, `REQUEST_RATES`, `CONCURRENCIES` — swept dimensions
- `SLO_TTFT`, `SLO_TPOT`, `SLO_E2EL` — goodput SLO thresholds (ms)

Result files are tagged as `{ratio}_{input}x{output}_rate{rate}_conc{concurrency}` and saved as JSON to `$OUTDIR`.

### Metrics

Benchmarks track: **TTFT** (time-to-first-token), **TPOT** (time-per-output-token), **ITL** (inter-token latency), **E2EL** (end-to-end latency), and throughput — at the 50th, 90th, 95th, and 99th percentiles.

## Adding a New Ratio Config

Create `configs/ratios/{ratio}.env` following the pattern of existing configs (e.g., `configs/ratios/2p1d.env`). Assign non-overlapping GPU indices and ports for each instance. Then add the ratio name to `RATIOS` in `configs/sweep.env`.
