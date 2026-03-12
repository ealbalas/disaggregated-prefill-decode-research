# Disaggregated Prefill-Decode Research

Benchmarking **disaggregated LLM inference**, where the prefill and decode phases run on separate GPU instances connected via Nixl KV-cache transfer. The system measures latency/throughput tradeoffs across different prefill:decode server ratios on Georgia Tech's PACE ICE cluster.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Experiments](#experiments)
- [Results Summary](#results-summary)
- [Configuration Reference](#configuration-reference)
- [Repository Structure](#repository-structure)

---

## Overview

In standard LLM inference, prefill (processing the input prompt) and decode (generating output tokens one by one) run on the same GPU. **Disaggregated inference** separates these phases onto dedicated GPU instances:

- **Prefill instances** process the prompt and produce KV-cache activations
- **Decode instances** receive those KV-cache parameters over the network (via Nixl) and generate tokens

This separation can improve hardware utilization and throughput at scale, but adds network overhead. This project benchmarks that tradeoff across five P:D ratios and multiple workload types, comparing against a colocated baseline.

---

## Architecture

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
3. Extracts `kv_transfer_params` from the prefill response
4. Forwards to a decode instance (with KV params injected) and streams tokens back

KV cache is transferred between instances using **Nixl** (`NixlConnector`) over side-channel ports.

### Ratios Tested

| Ratio | Prefill GPUs | Decode GPUs | Description |
|-------|-------------|-------------|-------------|
| colocated | 1 (shared) | 1 (shared) | Baseline — no disaggregation |
| 1p1d | 1 | 1 | Symmetric disaggregated |
| 1p2d | 1 | 2 | Extra decode capacity |
| 1p3d | 1 | 3 | Heavy decode scaling |
| 2p1d | 2 | 1 | Extra prefill capacity |
| 3p1d | 3 | 1 | Heavy prefill scaling |

---

## Setup

### Prerequisites

- Georgia Tech PACE ICE cluster access
- Conda environment `vllm_disagg` with vLLM + Nixl installed
- Model downloaded to `$SCRATCH`

### Environment

Before running any experiment, source the environment setup script:

```bash
source utils/env.sh
```

This loads GCC/CUDA/Anaconda modules, activates the conda environment, and sets key paths:

```bash
SCRATCH=/storage/ice1/4/5/ealbalas3
MODEL=$SCRATCH/disaggregated-prefill-decode-research/model
OUTDIR=$SCRATCH/disaggregated-prefill-decode-research/results/stats
VLLM_PROXY_SRC=$SCRATCH/disaggregated-prefill-decode-research/proxy
```

> If running outside PACE, edit the paths in `utils/env.sh` before sourcing.

### Download the Model

One-time setup to pull the model from HuggingFace (Llama-3.1-8B-Instruct):

```bash
HF_TOKEN=hf_xxx bash utils/download_model.sh
```

---

## Running Experiments

Each experiment follows this pattern:

```bash
# 1. Source environment
source utils/env.sh

# 2. Run experiment
bash experiments/<experiment>.sh [optional_config.env]

# 3. Plot results
python utils/plot_<experiment>.py
```

---

## Experiments

### 1. Ratio Characterization

**Script**: `experiments/ratio_characterization.sh`
**Config**: `configs/ratio_characterization.env`
**Purpose**: The main experiment. Compares all P:D ratios across three canonical workload types at two request rates.

**Workload archetypes:**
| Workload | Input tokens | Output tokens | Bottleneck |
|----------|-------------|--------------|------------|
| Prefill-heavy | 1024 | 64 | Prefill phase |
| Balanced | 256 | 256 | Both phases |
| Decode-heavy | 64 | 1024 | Decode phase |

**Parameters:**
- Request rates: 2, 10 req/s
- Concurrency: 32
- SLOs: TTFT < 500ms, TPOT < 50ms, E2EL < 5000ms

**Results**: `results/ratio_characterization/` — 36 JSON files, 4 plots

```bash
bash experiments/ratio_characterization.sh
python utils/plot_ratio_characterization.py
```

**Plots:**

| Plot | Description |
|------|-------------|
| `ratio_characterization_latency.png` | P50 TTFT and TPOT per ratio/workload/rate |
| `ratio_characterization_throughput.png` | Request throughput (req/s) |
| `ratio_characterization_output_throughput.png` | Output token throughput (tok/s) |
| `ratio_characterization_goodput.png` | Goodput — requests/s meeting all SLOs |

---

### 2. Prefill Length Sweep

**Script**: `experiments/prefill_test_1000.sh`
**Config**: `configs/prefill_test_1000.env`
**Purpose**: How does increasing input length affect TTFT, throughput, and goodput? Tests four input lengths at fixed output length.

**Parameters:**
- Input lengths: 1, 256, 512, 1024 tokens
- Output length: 256 tokens (fixed)
- Ratios: colocated, 1p1d, 1p2d, 2p1d
- Request rate: 10 req/s, concurrency: 32

**Results**: `results/prefill_test_1000/` — 16 JSON files, 4 plots

```bash
bash experiments/prefill_test_1000.sh
python utils/plot_prefill_test_1000.py
```

---

### 3. Decode Length Sweep

**Script**: `experiments/decode_test_1000.sh`
**Config**: `configs/decode_test_1000.env`
**Purpose**: How does increasing output length affect TPOT, E2EL, and goodput? Tests four output lengths at fixed input length.

**Parameters:**
- Output lengths: 1, 256, 512, 1024 tokens
- Input length: 256 tokens (fixed)
- Ratios: colocated, 1p1d, 1p2d, 2p1d
- Request rate: 10 req/s, concurrency: 32

**Results**: `results/decode_test_1000/` — 16 JSON files, 4 plots

```bash
bash experiments/decode_test_1000.sh
python utils/plot_decode_test_1000.py
```

---

### 4. Request Rate Sweep

**Script**: `experiments/request_rate_test.sh`
**Config**: `configs/request_rate_test.env`
**Purpose**: At what request rate does each ratio saturate? Sweeps from light to heavy load at a prefill-heavy workload.

**Parameters:**
- Request rates: 2, 4, 8, 10 req/s
- Input: 1024 tokens, output: 256 tokens (prefill-heavy)
- Ratios: colocated, 1p1d, 1p2d, 2p1d
- Concurrency: 32

**Results**: `results/request_rate_test/` — 16 JSON files, 4 plots

```bash
bash experiments/request_rate_test.sh
python utils/plot_request_rate_test.py
```

---

### 5. Prefill Stress Test

**Script**: `experiments/prefill_stress.sh`
**Config**: `configs/prefill_stress.env`
**Purpose**: Full cross-product stress test across all ratios. Used to find failure modes and stress the system under high load with varied input/output combinations.

**Parameters:**
- Input lengths and output lengths: large cross-product grid
- Request rates: multiple values
- Ratios: all 5 disaggregated + colocated
- Higher decode capacity: `max_num_seqs=256`

**Results**: `results/prefill_stress/` — JSON files, 4 plots

```bash
bash experiments/prefill_stress.sh
python utils/plot_prefill_stress.py
```

---

## Results Summary

### Key Findings

**1. Disaggregation adds ~90ms fixed TTFT overhead**

All disaggregated ratios see TTFT of 140–160ms minimum, vs. 57–59ms for colocated. This overhead is consistent across ratios and reflects the Nixl KV-cache transfer + serialization cost.

**2. Workload type dominates performance**

| Workload | Disaggregation impact |
|----------|-----------------------|
| Prefill-heavy (1024×64) | Works well — goodput stays near 100% at high load |
| Balanced (256×256) | Goodput collapses at rate=10; even colocated only hits 15% |
| Decode-heavy (64×1024) | Goodput is 0 across all disaggregated ratios |

**3. TPOT is unaffected by disaggregation**

Per-token generation latency stays at ~18–20ms regardless of ratio, since decode instances run independently after the KV transfer.

**4. Adding decode replicas helps little on prefill-heavy workloads**

The prefill instance is the bottleneck for long-input workloads. 1p2d and 1p3d show marginal improvement over 1p1d because the single prefill instance saturates first.

**5. Adding prefill replicas (2p1d, 3p1d) improves prefill-heavy throughput**

At rate=10 with 1024×64 inputs, 3p1d maintains 100% goodput vs. degradation in 1p3d. More prefill capacity directly helps prefill-bound workloads.

### Notable Anomalies

| Observation | Likely cause |
|-------------|-------------|
| 1p1d/1p2d TTFT spikes to 1700–2500ms at rate=2 (prefill-heavy) | Serialization stall in proxy or KV transfer at low request rate |
| 1p3d TTFT jumps to 14,000ms at rate=10 (prefill-heavy) | Single prefill instance cannot keep 3 decode replicas fed; unbounded queueing |
| Decode-heavy goodput uniformly 0 across all disaggregated ratios | Disaggregation overhead exceeds SLO budget for long-decode workloads |

---

## Configuration Reference

### Experiment Configs (`configs/*.env`)

Each experiment config defines the parameter sweep:

```bash
RATIOS=(1p1d 1p2d 2p1d)          # Which ratios to test
INPUT_LENS=(1024 256 64)          # Input lengths (paired with OUTPUT_LENS)
OUTPUT_LENS=(64 256 1024)         # Output lengths
REQUEST_RATES=(2 10)              # req/s
CONCURRENCIES=(32)
SLO_TTFT=500                      # ms
SLO_TPOT=50                       # ms
SLO_E2EL=5000                     # ms
```

### Ratio Configs (`configs/ratios/*.env`)

Each ratio config assigns non-overlapping GPU indices and ports:

```bash
# Example: configs/ratios/2p1d.env
PREFILL_PORTS=(8100 8101)
PREFILL_GPUS=(0 1)
PREFILL_SIDE_CHANNELS=(5600 5601)

DECODE_PORTS=(8200)
DECODE_GPUS=(2)
DECODE_SIDE_CHANNELS=(5700)
```

### Adding a New Ratio

1. Create `configs/ratios/{ratio}.env` following the pattern above
2. Assign non-overlapping GPU indices and ports
3. Add the ratio name to `RATIOS` in the relevant experiment config

### Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time-to-first-token — latency until the first output token |
| **TPOT** | Time-per-output-token — average inter-token latency during generation |
| **ITL** | Inter-token latency — latency between consecutive output tokens |
| **E2EL** | End-to-end latency — total request completion time |
| **Throughput** | Requests/second completed |
| **Output throughput** | Output tokens/second |
| **Goodput** | Requests/second meeting all SLO thresholds |

Results are reported at P50, P90, P95, and P99 percentiles.

---

## Repository Structure

```
.
├── proxy/
│   └── toy_proxy_server.py       # FastAPI round-robin proxy
├── experiments/
│   ├── ratio_characterization.sh # Main P:D ratio comparison
│   ├── prefill_test_1000.sh      # Input length sweep
│   ├── decode_test_1000.sh       # Output length sweep
│   ├── request_rate_test.sh      # Request rate sweep
│   └── prefill_stress.sh         # Full cross-product stress test
├── configs/
│   ├── ratio_characterization.env
│   ├── prefill_test_1000.env
│   ├── decode_test_1000.env
│   ├── request_rate_test.env
│   ├── prefill_stress.env
│   └── ratios/                   # Per-ratio GPU/port assignments
│       ├── 1p1d.env
│       ├── 1p2d.env
│       ├── 1p3d.env
│       ├── 2p1d.env
│       └── 3p1d.env
├── utils/
│   ├── env.sh                    # Cluster environment setup
│   ├── download_model.sh         # HuggingFace model download
│   ├── plot_results.py           # Generic results plotter
│   ├── plot_ratio_characterization.py
│   ├── plot_prefill_test_1000.py
│   ├── plot_decode_test_1000.py
│   ├── plot_request_rate_test.py
│   └── plot_prefill_stress.py
├── results/
│   ├── ratio_characterization/   # 36 JSON files + plots/
│   ├── prefill_test_1000/        # 16 JSON files + plots/
│   ├── decode_test_1000/         # 16 JSON files + plots/
│   ├── request_rate_test/        # 16 JSON files + plots/
│   └── prefill_stress/           # JSON files + plots/
├── model/                        # LLM model weights (gitignored)
└── CLAUDE.md                     # Claude Code project instructions
```
