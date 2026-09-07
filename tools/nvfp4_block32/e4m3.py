"""Pure-Python OCP FP8 E4M3 (e4m3fn, no Inf, 1-4-3, bias=7) quantizer.

Standalone (no numpy/torch dependency) so it can run on hosts without a
Python ML stack, e.g. for offline checkpoint tooling on a CPU-only node.
"""
import math

MAX_E4M3 = 448.0
MIN_NORMAL_E4M3 = 2.0**-6
SUBNORMAL_STEP_E4M3 = 2.0**-9


def quantize_e4m3(x: float) -> float:
    """Round x to the nearest representable E4M3 value (ties to even)."""
    if x == 0.0:
        return 0.0
    sign = -1.0 if x < 0.0 else 1.0
    ax = abs(x)

    if ax >= MAX_E4M3:
        return sign * MAX_E4M3

    if ax < MIN_NORMAL_E4M3:
        q = round(ax / SUBNORMAL_STEP_E4M3) * SUBNORMAL_STEP_E4M3
        return sign * q

    e = math.floor(math.log2(ax))
    mantissa = ax / (2.0**e)  # in [1, 2)
    steps = round(mantissa * 8.0)  # 3 mantissa bits -> 8 steps per octave
    if steps == 16:  # rounded up into the next octave
        steps = 8
        e += 1
    q = (steps / 8.0) * (2.0**e)
    if q > MAX_E4M3:
        q = MAX_E4M3
    return sign * q
