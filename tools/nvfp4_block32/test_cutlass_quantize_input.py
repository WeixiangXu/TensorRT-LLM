"""Check CutlassFusedMoE.quantize_input, the layer the op-level test skips.

The FC1 input is quantized here, in Python, before any kernel runs, so an
op-level test can pass while FC1 is still block16 -- which is exactly what
happened until the widened size was threaded through. This drives the real
method (through a stub that carries only the attributes it reads) and asserts
on the arguments that reach fp4_quantize, for both communication branches.

Usage: python test_cutlass_quantize_input.py
"""
import os
import sys
import types

import torch

from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

SLOT = 16


def make_stub(quant_vec_size):
    """Minimal object exposing what quantize_input reads on the NVFP4 path."""
    # Every attribute quantize_input reads, so it walks its real branches
    # instead of dying on the first missing one.
    stub = types.SimpleNamespace(
        has_nvfp4=True,
        has_deepseek_fp8_block_scales=False,
        has_fp8_qdq=False,
        has_w4afp8=False,
        has_w4a16_mxfp4=False,
        has_w4a8_mxfp4_fp8=False,
        has_w4a8_mxfp4_mxfp8=False,
        has_int8_woq_per_channel=False,
        has_mxfp8=False,
        has_any_quant=True,
        force_dynamic_quantization=False,
        fc31_act_scale=None,
        fc31_alpha=None,
        fc31_input_dequant=None,
        fc31_weight_scale_2=None,
        quant_config=None,
        quant_method=None,
        fc31_input_scale=torch.tensor([1.0], device="cuda"),
        scaling_vector_size=SLOT,
        nvfp4_act_quant_vec_size=quant_vec_size,
        hidden_size=1024,
    )
    return stub


def run(quant_vec_size, post_quant_comm, calls):
    stub = make_stub(quant_vec_size)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    real = torch.ops.trtllm.fp4_quantize

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    torch.ops.trtllm.fp4_quantize = spy
    try:
        CutlassFusedMoE.quantize_input(stub, x, post_quant_comm=post_quant_comm)
    finally:
        torch.ops.trtllm.fp4_quantize = real


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    failures = []

    for post_quant_comm in (True, False):
        for quant_vec_size, expect in ((0, 0), (32, 32)):
            calls = []
            try:
                run(quant_vec_size, post_quant_comm, calls)
            except Exception as exc:  # noqa: BLE001
                failures.append("post_quant_comm=%s vec=%d raised %s"
                                % (post_quant_comm, quant_vec_size, exc))
                continue
            if not calls:
                failures.append("post_quant_comm=%s vec=%d never called "
                                "fp4_quantize" % (post_quant_comm, quant_vec_size))
                continue
            args = calls[0]
            got_slot = args[2]
            got_quant = args[5] if len(args) > 5 else None
            if got_slot != SLOT:
                failures.append("post_quant_comm=%s: sfVecSize is %s, must stay "
                                "%d so the GEMM still gets K/16 slots"
                                % (post_quant_comm, got_slot, SLOT))
            if got_quant != expect:
                failures.append("post_quant_comm=%s vec=%d: fp4_quantize got "
                                "sfQuantVecSize=%s, expected %d"
                                % (post_quant_comm, quant_vec_size, got_quant,
                                   expect))

    # A pre-quantized input carries no block size, so the widened path must
    # refuse it rather than leave FC1 at block16 while FC2 runs at block32.
    stub = make_stub(32)
    packed = torch.zeros(64, 512, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(64 * 1024 // SLOT, dtype=torch.uint8, device="cuda")
    try:
        CutlassFusedMoE.quantize_input(
            stub, Fp4QuantizedTensor(packed, sf, is_sf_swizzled=False),
            post_quant_comm=True)
        failures.append("a pre-quantized input was accepted on the widened path")
    except RuntimeError:
        pass

    if failures:
        for f in failures:
            print("FAIL: " + f)
        return 1
    print("PASS: quantize_input passes sfQuantVecSize on both communication "
          "branches, keeps sfVecSize at 16, and refuses pre-quantized input")
    return 0


if __name__ == "__main__":
    sys.exit(main())
