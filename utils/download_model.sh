#!/usr/bin/env bash
# utils/download_model.sh
#
# Downloads a Llama model from HuggingFace into $MODEL.
# Will prompt for HF_TOKEN if not already set in the environment.
#
# Usage:
#   bash utils/download_model.sh
#   HF_TOKEN=hf_xxx bash utils/download_model.sh          # skip prompt
#   MODEL_REPO=meta-llama/Llama-3.1-8B-Instruct bash ...  # override model

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/utils/env.sh"

# ── Model to download (override with MODEL_REPO env var) ──────────────────────
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.1-8B-Instruct}"

# ── Prompt for token if not set ───────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "A HuggingFace token is required to download Llama models."
  echo "Get yours at: https://huggingface.co/settings/tokens"
  read -rsp "Enter HF_TOKEN: " HF_TOKEN
  echo
fi

if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: HF_TOKEN cannot be empty." >&2
  exit 1
fi

export HF_TOKEN

# ── Check huggingface_hub is available ────────────────────────────────────────
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "Installing huggingface_hub..."
  pip install -q huggingface-hub
fi

# ── Download ──────────────────────────────────────────────────────────────────
echo ""
echo "Downloading $MODEL_REPO → $MODEL"
echo "This may take a while for large models..."
echo ""

python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.pt", "original/*"],  # skip redundant formats
)
print("Download complete.")
PYEOF

echo ""
echo "Model saved to: $MODEL"
