#!/usr/bin/env bash
# Enter the stock NVIDIA NeMo container (NeMo 2.x preinstalled).
# Use this first — if the scripts run, you're done. If they fail on the
# adapter API (NeMo 2.x renamed a few things), switch to run_custom.sh
# which uses the Dockerfile that downgrades to NeMo 1.23.
#
# Usage: ./run_stock.sh [GPU_ID]
#   GPU_ID: 0/1/2/3 — defaults to 0
set -euo pipefail

GPU_ID="${1:-0}"
IMAGE="nvcr.io/nvidia/nemo:25.04"

echo "[run_stock] image=$IMAGE  gpu=$GPU_ID  workspace=$(pwd)"

docker run --gpus "\"device=${GPU_ID}\"" -it --rm \
    --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/.cache_hf:/root/.cache/huggingface" \
    -w /workspace \
    -e HF_HOME=/workspace/.cache_hf \
    -e NEMO_CACHE_DIR=/workspace/.cache_nemo \
    "$IMAGE" \
    bash
