"""Exact NVFP4 reference: E2M1 lattice, E4M3 scales, and the swizzled layout.

The earlier check approximated E2M1 with round(x*2)/2, but the positive E2M1
values are 0, 0.5, 1, 1.5, 2, 3, 4, 6 -- not a uniform 0.5 grid. Combined with a
tolerance of about one scale, that let wrong codes pass. Everything here works
on exact codes so the comparison is bit-level.
"""
import torch

# Positive E2M1 values by code; the sign bit is the high bit of the nibble.
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAX = 6.0
# Midpoints for round-to-nearest; ties land on the even code, matching hardware.
_TIE_EVEN_UP = [False, True, False, True, False, True, False]  # code i -> i+1


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode to 4-bit E2M1 codes with round-to-nearest-even and satfinite."""
    sign = (x < 0).to(torch.uint8) << 3
    a = x.abs().clamp(max=E2M1_MAX)
    code = torch.zeros_like(a, dtype=torch.uint8)
    for i in range(7):
        lo, hi = E2M1_VALUES[i], E2M1_VALUES[i + 1]
        mid = (lo + hi) / 2
        # Above the midpoint always rounds up; exactly on it goes to the even code.
        up = (a > mid) | ((a == mid) & _TIE_EVEN_UP[i])
        in_bin = (a > lo) & (a <= hi)
        code = torch.where(in_bin, torch.full_like(code, i + 1) - (~up).to(torch.uint8),
                           code)
    code = torch.where(a >= E2M1_VALUES[7], torch.full_like(code, 7), code)
    return code | sign


def e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1_VALUES + [-v for v in E2M1_VALUES],
                       device=code.device, dtype=torch.float32)
    return lut[code.long()]


def quantize_reference(x: torch.Tensor, global_scale: torch.Tensor,
                       quant_block: int, slot_block: int):
    """Reference NVFP4 quantization, returning (packed codes, E4M3 scale bytes).

    quant_block elements share one scale; that scale is written into every
    slot_block-sized slot they span, which is how block32 numerics are produced
    on hardware that only reads a scale every 16 elements.
    """
    m, k = x.shape
    assert k % quant_block == 0 and quant_block % slot_block == 0
    amax = x.float().abs().reshape(m, k // quant_block, quant_block).amax(-1)
    # One E4M3 rounding, at the coarse granularity, exactly as the kernel does.
    sf = (global_scale.float() * (amax / E2M1_MAX)).to(torch.float8_e4m3fn)
    sf_slots = sf.repeat_interleave(quant_block // slot_block, dim=1)

    scale = sf.float() / global_scale.float()
    scaled = x.float() / scale.repeat_interleave(quant_block, dim=1).clamp(min=1e-30)
    codes = e2m1_encode(scaled)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, sf_slots.view(torch.uint8)


def unswizzle_sf(sf_flat: torch.Tensor, m: int, k_slots: int) -> torch.Tensor:
    """Undo the 128x4 scale-factor swizzle into a plain [m, k_slots] view.

    The MoE path stores scales swizzled, so a raw reshape would compare the
    wrong elements. Layout: [ceil(m/128)][ceil(k/4)][32][4][4].
    """
    m_tiles = (m + 127) // 128
    k_tiles = (k_slots + 3) // 4
    v = sf_flat[:m_tiles * k_tiles * 32 * 4 * 4].view(m_tiles, k_tiles, 32, 4, 4)
    # (mt, kt, inner_m32, inner_m4, inner_k4) -> row = mt*128 + m4*32 + m32
    v = v.permute(0, 3, 2, 1, 4).reshape(m_tiles * 4 * 32, k_tiles * 4)
    return v[:m, :k_slots]
