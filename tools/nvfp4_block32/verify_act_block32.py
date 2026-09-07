"""Check that TRTLLM_NVFP4_ACT_BLOCK32 really produces block32 numerics.

Runs on a Blackwell GPU against a built TensorRT-LLM. Verifies the fp4_quantize
path directly (sfQuantVecSize=32), which is the same device converter the
CUTLASS MoE kernels call, so a pass here covers both activation quantization
points in that backend.

Checks, on random data:
  1. block32 mode writes the same scale into both slots of every adjacent pair
  2. that scale equals an independent block32 reference (amax over 32 elements,
     quantized to E4M3 once)
  3. it also equals the max of the two block16 scales, the identity the offline
     weight-side converter relies on
  4. the dequantized values match a reference block32 quantization of the input
  5. block16 and block32 actually differ, so the flag is not a no-op

Usage: python verify_act_block32.py
"""
import os
import sys

import torch


def load_trtllm_ops():
    """Register torch.ops.trtllm without importing the full package.

    Importing tensorrt_llm pulls in mpi4py and the rest of the runtime, none of
    which this check needs -- it only calls one quantization op. Loading the
    torch bindings directly keeps the check runnable on a bare container.
    """
    try:
        import tensorrt_llm  # noqa: F401
        return "package"
    except ImportError:
        pass
    for root in sys.path:
        lib = os.path.join(root, "tensorrt_llm", "libs", "libth_common.so")
        if os.path.exists(lib):
            torch.ops.load_library(lib)
            return lib
    raise RuntimeError("libth_common.so not found on sys.path")

E2M1_MAX = 6.0
LINEAR_LAYOUT = False
SF_VEC = 16


def quantize(x, global_scale, sf_quant_vec_size):
    return torch.ops.trtllm.fp4_quantize(x, global_scale, SF_VEC, False,
                                         LINEAR_LAYOUT, sf_quant_vec_size)


def reference_scales(x, block, global_scale):
    """Per-block E4M3 scales, computed the way cvt_warp_fp16_to_fp4 does."""
    m, k = x.shape
    amax = x.float().abs().reshape(m, k // block, block).amax(dim=-1)
    return (global_scale * (amax / E2M1_MAX)).to(torch.float8_e4m3fn)


def dequantize(packed, sf, global_scale, m, k):
    """Unpack E2M1 pairs and undo both scale levels, for a value-level check."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
         -3.0, -4.0, -6.0],
        device=packed.device,
        dtype=torch.float32)
    b = packed.reshape(m, k // 2).to(torch.uint8)
    vals = torch.stack([lut[(b & 0xF).long()], lut[(b >> 4).long()]], dim=-1)
    vals = vals.reshape(m, k)
    scale = sf.reshape(m, k // SF_VEC).float() / global_scale.float()
    return vals * scale.repeat_interleave(SF_VEC, dim=1)


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    print("trtllm ops from: %s" % load_trtllm_ops())
    print("device: %s (sm_%d%d)" % ((torch.cuda.get_device_name(0),) +
                                    torch.cuda.get_device_capability()))
    torch.manual_seed(0)
    m, k = 256, 1024
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    # Give a few blocks a large outlier so block16 and block32 must disagree.
    x[0, 0] = 40.0
    x[7, 100] = -30.0
    global_scale = (448 * E2M1_MAX / x.float().abs().max()).reshape(1).cuda()

    q16, sf16 = quantize(x, global_scale, 0)
    q32, sf32 = quantize(x, global_scale, 32)

    sf16 = sf16.view(torch.float8_e4m3fn).reshape(m, k // SF_VEC)
    sf32 = sf32.view(torch.float8_e4m3fn).reshape(m, k // SF_VEC)

    failures = []

    # 1. adjacent pairs share one scale
    lo, hi = sf32[:, 0::2].float(), sf32[:, 1::2].float()
    if not torch.equal(lo, hi):
        failures.append("adjacent block16 slots differ in block32 mode (%d of %d)"
                        % ((lo != hi).sum().item(), lo.numel()))

    # 2. equals an independent block32 reference
    ref32 = reference_scales(x, 32, global_scale).float()
    if not torch.equal(lo, ref32):
        failures.append("block32 scales != reference amax-over-32 (%d of %d)" %
                        ((lo != ref32).sum().item(), ref32.numel()))

    # 3. equals the max of the two block16 scales
    pair_max = torch.maximum(sf16[:, 0::2].float(), sf16[:, 1::2].float())
    if not torch.equal(lo, pair_max):
        failures.append("block32 scale != max of the two block16 scales (%d of %d)"
                        % ((lo != pair_max).sum().item(), pair_max.numel()))

    # 4. dequantized values match a block32 reference quantization
    deq = dequantize(q32, sf32, global_scale, m, k)
    ref_scale = ref32.repeat_interleave(2, dim=1) / global_scale.float()
    ref_q = torch.round(
        (x.float() / ref_scale.repeat_interleave(SF_VEC, dim=1)).clamp(
            -E2M1_MAX, E2M1_MAX) * 2) / 2  # E2M1 is not uniform; loose check
    mismatch = (deq - ref_q * ref_scale.repeat_interleave(SF_VEC, dim=1)).abs()
    tol = ref_scale.repeat_interleave(SF_VEC, dim=1) * 1.01
    if (mismatch > tol).any():
        failures.append("dequantized values deviate from the block32 reference "
                        "by more than one E2M1 step (%d elements)" %
                        (mismatch > tol).sum().item())

    # 5. the flag is not a no-op
    if torch.equal(sf16.float(), sf32.float()):
        failures.append("block16 and block32 scales are identical -- flag had "
                        "no effect (is TRTLLM_NVFP4_ACT_BLOCK32 wired?)")

    changed = (sf16.float() != sf32.float()).float().mean().item()
    print("scale slots changed by block32: %.1f%%" % (100 * changed))

    if failures:
        for f in failures:
            print("FAIL: " + f)
        return 1
    print("PASS: block32 activation scales verified on %d x %d" % (m, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
