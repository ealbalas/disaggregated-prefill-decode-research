#!/usr/bin/env bash
# env.sh — Source this at the top of every script.
# Sets all paths and PACE-specific env var workarounds for Dynamo + vLLM.
# Bug fixes baked in (see README.md for root causes):
#   #1: CC/CXX override to prevent spack GCC leaking into container
#   #2: VLLM_USE_DEEP_GEMM=0 to skip failing DeepGEMM FP8 kernels
#   #3: TORCHINDUCTOR_COMBO_KERNELS=0 to prevent inductor crash
#   #4: This file must be sourced on every new SLURM node

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
export SCRATCH=/storage/ice1/4/5/ealbalas3
export REPO_ROOT=$SCRATCH/disaggregated-prefill-decode-research
export MODEL=$REPO_ROOT/model
export DYNAMO_PACE_DIR=$REPO_ROOT/dynamo-pace
export RESULTS_DIR=$REPO_ROOT/results/dynamo
export CONTAINERS_DIR=$SCRATCH/containers
export IMAGE=$REPO_ROOT/dynamo-vllm.sif

# ---------------------------------------------------------------------------
# PACE bug workarounds (#1, #2, #3)
# These are set in the host environment so launch.sh can forward them into
# the Apptainer container via --env flags.
# ---------------------------------------------------------------------------
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export VLLM_USE_DEEP_GEMM=0
export TORCHINDUCTOR_COMBO_KERNELS=0

# ---------------------------------------------------------------------------
# Load PACE spack modules
# ---------------------------------------------------------------------------
module load gcc cuda apptainer 2>/dev/null || true

# ---------------------------------------------------------------------------
# Create output directories if missing
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
mkdir -p "$CONTAINERS_DIR"
