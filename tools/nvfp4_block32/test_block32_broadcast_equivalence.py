"""Verify: max(quantize_e4m3(a), quantize_e4m3(b)) == quantize_e4m3(max(a, b)).

This is the numeric identity the NVFP4 block32-on-block16-hardware trick
relies on: broadcasting the max of two already-quantized adjacent block16
E4M3 scales into both slots is bit-identical to recomputing amax over the
merged 32-element block and quantizing once, because E4M3 round-to-nearest
is a monotonic non-decreasing function of its input.

Standalone script (no pytest/numpy/torch dependency) so it runs anywhere.
"""
import random
import sys

from e4m3 import quantize_e4m3, MAX_E4M3, MIN_NORMAL_E4M3, SUBNORMAL_STEP_E4M3


def check(a: float, b: float) -> bool:
    naive = max(quantize_e4m3(a), quantize_e4m3(b))
    true_block32 = quantize_e4m3(max(a, b))
    return naive == true_block32


def main() -> int:
    random.seed(0)
    failures = []

    # Edge cases: exact E4M3 representable values, boundary/tie points,
    # subnormal range, overflow/clamp range, equal pairs, zero.
    edge_values = [
        0.0,
        SUBNORMAL_STEP_E4M3,
        SUBNORMAL_STEP_E4M3 * 1.5,  # exact tie -> round-to-even
        MIN_NORMAL_E4M3,
        MIN_NORMAL_E4M3 * 1.0625,  # halfway between two mantissa steps
        1.0,
        1.0 + 1.0 / 16.0,  # exact tie between two E4M3 mantissa steps
        447.9,
        448.0,
        500.0,  # above MAX_E4M3 -> clamps
        1e6,
    ]
    for a in edge_values:
        for b in edge_values:
            if not check(a, b):
                failures.append((a, b))

    # Randomized sweep across the representable dynamic range.
    for _ in range(200000):
        a = random.uniform(0, 600) if random.random() < 0.5 else random.uniform(0, 0.05)
        b = random.uniform(0, 600) if random.random() < 0.5 else random.uniform(0, 0.05)
        if not check(a, b):
            failures.append((a, b))

    if failures:
        print(f"FAIL: {len(failures)} mismatches, e.g. {failures[:10]}")
        return 1

    total = len(edge_values) ** 2 + 200000
    print(f"PASS: {total} cases, broadcast-max == recompute-and-requantize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
