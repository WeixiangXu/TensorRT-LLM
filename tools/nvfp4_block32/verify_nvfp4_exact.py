"""Bit-exact NVFP4 block32 check, in both scale layouts.

Replaces the loose float comparison the first version did. It compares the
packed E2M1 bytes and the raw E4M3 scale bytes against an exact reference, so a
wrong code cannot pass on a tolerance. It also covers the swizzled layout, which
is what the MoE path actually stores -- the earlier check only ever ran linear.

Usage: python verify_nvfp4_exact.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_ref import quantize_reference, unswizzle_sf  # noqa: E402

E2M1_MAX = 6.0
SLOT = 16


def load_ops():
    try:
        import tensorrt_llm  # noqa: F401
        return "package"
    except ImportError:
        for root in sys.path:
            lib = os.path.join(root, "tensorrt_llm", "libs", "libth_common.so")
            if os.path.exists(lib):
                torch.ops.load_library(lib)
                return lib
    raise RuntimeError("libth_common.so not found")


def check(m, k, swizzled, quant_block, failures):
    tag = "%dx%d %s quant=%d" % (m, k, "swizzled" if swizzled else "linear",
                                 quant_block)
    torch.manual_seed(m * 1000 + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 40.0          # outliers, so the two granularities must disagree
    x[min(3, m - 1), 17] = -25.0
    gs = (448 * E2M1_MAX / x.float().abs().max()).reshape(1).cuda()

    packed, sf = torch.ops.trtllm.fp4_quantize(
        x, gs, SLOT, False, swizzled, 0 if quant_block == SLOT else quant_block)

    ref_packed, ref_sf = quantize_reference(x, gs, quant_block, SLOT)

    got_packed = packed.view(torch.uint8).reshape(m, k // 2)
    if not torch.equal(got_packed, ref_packed.to(got_packed.device)):
        n = (got_packed != ref_packed.to(got_packed.device)).sum().item()
        failures.append("%s: %d/%d packed E2M1 bytes differ from the exact "
                        "reference" % (tag, n, got_packed.numel()))

    raw = sf.view(torch.uint8).flatten()
    got_sf = (unswizzle_sf(raw, m, k // SLOT) if swizzled
              else raw[:m * (k // SLOT)].reshape(m, k // SLOT))
    if not torch.equal(got_sf, ref_sf.to(got_sf.device)):
        n = (got_sf != ref_sf.to(got_sf.device)).sum().item()
        failures.append("%s: %d/%d E4M3 scale bytes differ from the exact "
                        "reference" % (tag, n, got_sf.numel()))

    if quant_block == 32:
        pairs = got_sf.reshape(m, -1, 2)
        if not torch.equal(pairs[:, :, 0], pairs[:, :, 1]):
            failures.append("%s: adjacent slots do not share one scale" % tag)


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    print("ops from: %s" % load_ops())
    print("device: %s (sm_%d%d)" % ((torch.cuda.get_device_name(0),) +
                                    torch.cuda.get_device_capability()))

    failures = []
    # 4096 and 2048 are the DS V4 MoE FC1 and FC2 widths.
    for m, k in ((128, 4096), (256, 2048), (17, 1024)):
        for swizzled in (False, True):
            for quant_block in (16, 32):
                check(m, k, swizzled, quant_block, failures)

    if failures:
        for f in failures:
            print("FAIL: " + f)
        return 1
    print("PASS: packed E2M1 and E4M3 scales are bit-exact in both layouts, "
          "at block16 and block32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
