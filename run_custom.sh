#!/usr/bin/env bash
# Build (once) and enter the custom image that pins NeMo 1.23.
# Use this if the scripts hit NeMo 2.x API drift in the stock container.
#
# Usage: ./run_custom.sh [GPU_ID]
set -euo pipefail

GPU_ID="${1:-0}"
IMAGE="meno-nemo123:latest"

# Build only if the image doesn't exist locally
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[run_custom] image $IMAGE not found — building (10-20 min first time)..."
    docker build -t "$IMAGE" -f docker/Dockerfile .
fi

echo "[run_custom] image=$IMAGE  gpu=$GPU_ID  workspace=$(pwd)"

docker run --gpus "\"device=${GPU_ID}\"" -it --rm \
    --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/.cache_hf:/root/.cache/huggingface" \
    -w /workspace \
    "$IMAGE" \
    bash
