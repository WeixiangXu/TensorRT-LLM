#!/bin/bash
# Build TensorRT-LLM for Blackwell with the block32 numerics changes.
#
# Run on a GPU node (the build needs the CUDA toolkit, not a GPU, but the
# verification right after does). Expect a couple of hours cold, minutes warm
# thanks to ccache.
#
# Usage: bash tools/nvfp4_block32/build_trtllm.sh [extra build_wheel args]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# SM100 = B200/GB200, SM103 = B300. Keep the list short; each arch costs time.
CUDA_ARCHS="${CUDA_ARCHS:-100-real}"
JOBS="${JOBS:-$(nproc)}"
CCACHE_DIR="${CCACHE_DIR:-/home/scratch.weixiangx_hw/workspace/.cache/ccache}"
export CCACHE_DIR

mkdir -p "$CCACHE_DIR"
cd "$REPO_ROOT"

echo "=== repo:    $REPO_ROOT"
echo "=== branch:  $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "=== archs:   $CUDA_ARCHS"
echo "=== jobs:    $JOBS"
echo "=== ccache:  $CCACHE_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv || echo "(no GPU visible)"

python3 ./scripts/build_wheel.py \
    --cuda_architectures "$CUDA_ARCHS" \
    --job_count "$JOBS" \
    --use_ccache \
    --install \
    "$@"

echo
echo "=== build done, running block32 verification ==="
python3 "$REPO_ROOT/tools/nvfp4_block32/verify_act_block32.py"
