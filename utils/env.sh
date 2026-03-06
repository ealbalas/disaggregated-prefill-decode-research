#!/usr/bin/env bash
# utils/env.sh — source this at the top of every script
# Usage: source "$(dirname "$0")/../utils/env.sh"

export SCRATCH="/storage/ice1/4/5/ealbalas3"
export MODEL="$SCRATCH/disaggregated-prefill-decode-research/model"
export OUTDIR="$SCRATCH/disaggregated-prefill-decode-research/results/prefill_stress"
export VLLM_PROXY_SRC="$SCRATCH/disaggregated-prefill-decode-research/proxy"

export UCX_NET_DEVICES=all

module load gcc
module load cuda
module load anaconda3
conda activate vllm_disagg
