# dynamo-pace

NVIDIA Dynamo + vLLM disaggregated prefill/decode serving on Georgia Tech PACE ICE cluster.

## Overview

This directory contains configs, launch scripts, SLURM job files, and benchmarking tools for
running Dynamo-based disaggregated inference on PACE. Prefill and decode phases run on separate
GPU workers; KV cache is transferred between them via Dynamo's built-in transport layer.

**Container:** `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0` (pulled to SIF via Apptainer)
**Model:** Qwen3 at `$SCRATCH/disaggregated-prefill-decode-research/model`
**Scheduler:** SLURM
**GPUs:** NVIDIA H200

## Directory Layout

```
dynamo-pace/
├── README.md
├── configs/                  # Dynamo serving graph YAML configs
│   ├── disagg_1p1d.yaml      # 1 prefill + 1 decode worker
│   ├── disagg_2p1d.yaml      # 2 prefill + 1 decode worker
│   ├── disagg_2p2d.yaml      # 2 prefill + 2 decode worker
│   └── agg_baseline.yaml     # Aggregated (no disaggregation) baseline
├── scripts/
│   ├── env.sh                # Source this first — sets paths and fixes PACE bugs
│   ├── pull_image.sh         # One-time: pull container image to SIF
│   ├── launch.sh             # Launch Dynamo serving stack inside Apptainer
│   └── stop.sh               # Kill all Dynamo/vLLM processes
├── slurm/
│   ├── interactive.sh        # Request an interactive GPU node
│   ├── job_disagg.sh         # Batch SLURM job for disaggregated serving
│   └── job_agg.sh            # Batch SLURM job for aggregated baseline
├── benchmarks/
│   ├── run_benchmark.sh      # Drive vLLM benchmark client against running server
│   └── parse_results.py      # Parse benchmark JSON → summary table + plots
└── notebooks/
    └── analyze_results.ipynb # Interactive analysis of parsed results
```

## Quick Start

### 1. Pull the container (one-time)

```bash
source dynamo-pace/scripts/env.sh
bash dynamo-pace/scripts/pull_image.sh
```

### 2. Interactive session

```bash
bash dynamo-pace/slurm/interactive.sh
```

### 3. Launch a serving configuration

```bash
# Inside the interactive node (env.sh is sourced automatically by launch.sh)
bash dynamo-pace/scripts/launch.sh dynamo-pace/configs/disagg_1p1d.yaml
```

### 4. Run a benchmark

```bash
bash dynamo-pace/benchmarks/run_benchmark.sh --input-len 512 --output-len 128 --rate 4
```

### 5. Parse and plot results

```bash
python dynamo-pace/benchmarks/parse_results.py --results-dir $RESULTS_DIR
```

## Known PACE Bugs (already handled in all scripts)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Triton compile failure | PACE spack GCC leaks into container | `--env CC=/usr/bin/gcc --env CXX=/usr/bin/g++` |
| DeepGEMM FP8 compile failure | Kernel incompatibility | `--env VLLM_USE_DEEP_GEMM=0` |
| Torch inductor crash | Combo kernels broken | `--env TORCHINDUCTOR_COMBO_KERNELS=0` + `enforce_eager: true` |
| Env vars lost on new node | SLURM resets environment | Every script sources `env.sh` at the top |
| curl `!` history expansion | bash history expansion | Always use single-quoted JSON in curl |

## Config Reference

Each `configs/*.yaml` file sets:

- `model` — absolute path to model weights
- `prefill.num_workers` / `decode.num_workers` — disaggregation topology
- `tensor_parallel_size` — GPUs per worker
- `max_num_seqs` — max concurrent sequences per worker
- `max_model_len` — max context length
- `enforce_eager` — disables CUDA graph capture (required for PACE)
- `environment` block — PACE env var workarounds baked in

## Adding a New Topology

1. Copy an existing config: `cp configs/disagg_1p1d.yaml configs/disagg_4p2d.yaml`
2. Update `prefill.num_workers`, `decode.num_workers`, and `tensor_parallel_size`
3. Ensure total GPUs ≤ GPUs allocated in your SLURM job
4. Run with: `bash scripts/launch.sh configs/disagg_4p2d.yaml`
