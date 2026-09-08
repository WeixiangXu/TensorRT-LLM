#!/bin/bash
# Plan C: does the block32 branch actually run during real model inference?
# The trace line is printed from inside the branch, so it is immune to the
# routing noise that made output comparison useless.
set -uo pipefail
NAME=nvfp4smoke
OUT=/home/scratch.weixiangx_hw/workspace/nvfp4_block32_run/smoke
MODEL=/home/scratch.weixiangx_hw/workspace/models/DeepSeek-V4-Flash-0731-NVFP4-nvidia
WHEEL=/home/scratch.weixiangx_hw/workspace/repos/TensorRT-LLM/build/tensorrt_llm-1.3.0rc26-cp312-cp312-linux_x86_64.whl

echo "=== installing the freshly built wheel (with the trace) ==="
docker exec "$NAME" pip install --no-input --force-reinstall --no-deps "$WHEEL" 2>&1 | tail -2

for spec in "trace_on 1" "trace_off 0"; do
    set -- $spec
    ARM=$1; FLAG=$2
    echo
    echo "==================== $ARM (TRTLLM_NVFP4_ACT_BLOCK32=$FLAG) ===================="
    docker exec -e TRTLLM_NVFP4_ACT_BLOCK32=$FLAG "$NAME" bash -c "
cd /home/scratch.weixiangx_hw/workspace/repos/TensorRT-LLM/examples/llm-api
mpirun -n 8 --allow-run-as-root trtllm-llmapi-launch python3 quickstart_advanced.py \
  --model_dir $MODEL --moe_backend CUTLASS \
  --tp_size 8 --moe_ep_size 8 \
  --tokens_per_block 128 --max_num_tokens 8192 \
  --max_seq_len 4096 --kv_cache_fraction 0.5 --max_tokens 16 \
  --prompt 'Why is the sky blue?'
" > "$OUT/$ARM.log" 2>&1
    echo "exit: $?"
    echo "-- FC2 device trace:"; grep -a "nvfp4-block32. branch taken" "$OUT/$ARM.log" 2>/dev/null | head -2
    echo "-- FC1 host marker:"; grep -a "fp4_quantize widened" "$OUT/$ARM.log" 2>/dev/null | head -2
done

echo
echo "==================== VERDICT ===================="
ON=$(grep -ac "nvfp4-block32. branch taken" "$OUT/trace_on.log" 2>/dev/null)
OFF=$(grep -ac "nvfp4-block32. branch taken" "$OUT/trace_off.log" 2>/dev/null)
# hidden_size=4096 is FC1, inter_size=2048 is FC2. Both must appear: only FC2
# showed up before the Python-side quantize_input was widened.
FC1=$(grep -ac "fp4_quantize widened" "$OUT/trace_on.log" 2>/dev/null)
FC2=$(grep -ac "branch taken: num_cols=2048" "$OUT/trace_on.log" 2>/dev/null)
echo "flag=1 -> $ON trace line(s)  [FC1(num_cols=4096): $FC1, FC2(num_cols=2048): $FC2]"
echo "flag=0 -> $OFF trace line(s)"
if [ "$OFF" -ne 0 ]; then
    echo "BROKEN CONTROL: the trace appears with the flag off"
elif [ "$ON" -eq 0 ]; then
    echo "REFUTED: the branch never ran"
elif [ "$FC1" -eq 0 ]; then
    echo "PARTIAL: only FC2 is widened; the FC1 input is still block16"
elif [ "$FC2" -eq 0 ]; then
    echo "PARTIAL: only FC1 is widened; the FC2 input is still block16"
else
    echo "PROVEN: both MoE activation quantization points run at block32, and only when the flag is set"
fi
