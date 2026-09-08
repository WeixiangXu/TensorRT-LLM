"""Decide whether a checkpoint's NVFP4 weight scales are genuinely block16.

A config that declares `group_size: 16` proves nothing: DS V4's NVFP4
checkpoint declares 16 while every adjacent scale pair is identical, because
the scales were derived from an MXFP4 (E8M0, block32) source and broadcast.
Such a checkpoint has no block16 weight arm to compare against, so this test
decides whether a weight-side experiment is possible at all.

Two independent fingerprints, both of which an MXFP4-derived checkpoint fails:

  adjacent-pair equality   scale[2i] == scale[2i+1] for every i. A real block16
                           quantization has no reason to satisfy this; a
                           block32 source broadcast into 16-slots satisfies it
                           everywhere.
  power-of-two             every E4M3 scale value is exactly 2^k. E8M0 encodes
                           only exponents, so a converted checkpoint keeps that
                           property; an E4M3 block16 quantization uses the
                           mantissa and will not.

Usage: fingerprint_weight_scales.py <model_dir> [--shards N] [--pattern SUBSTR]
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open


def is_power_of_two(t: torch.Tensor) -> torch.Tensor:
    """True where the value is exactly 2^k (k any integer), elementwise."""
    f = t.float()
    ok = f > 0
    m, e = torch.frexp(f)
    # frexp returns mantissa in [0.5, 1); an exact power of two has m == 0.5.
    return ok & (m == 0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--shards", type=int, default=4,
                    help="how many safetensors shards to sample")
    ap.add_argument("--pattern", default="",
                    help="only tensors whose name contains this")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    if not files:
        print(f"no safetensors under {args.model_dir}", file=sys.stderr)
        return 2
    step = max(1, len(files) // args.shards)
    sampled = files[::step][:args.shards]

    n_tensors = 0
    n_pairs = 0
    n_pairs_differ = 0
    n_vals = 0
    n_not_pow2 = 0
    examples = []
    seen_values = set()

    for path in sampled:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                if "weight_scale" not in name or name.endswith("_2"):
                    continue  # *_2 is the per-tensor FP32 scale, not per-block
                if args.pattern and args.pattern not in name:
                    continue
                t = f.get_tensor(name)
                if t.ndim < 1 or t.shape[-1] % 2:
                    continue
                n_tensors += 1
                flat = t.reshape(-1, t.shape[-1]).float()

                a, b = flat[:, 0::2], flat[:, 1::2]
                differ = (a != b)
                n_pairs += a.numel()
                n_pairs_differ += int(differ.sum())

                n_vals += flat.numel()
                n_not_pow2 += int((~is_power_of_two(flat)).sum())

                if len(seen_values) < 40:
                    seen_values.update(
                        flat.flatten()[:4096].unique().tolist()[:40])
                if len(examples) < 3:
                    examples.append((os.path.basename(path), name,
                                     tuple(t.shape), str(t.dtype),
                                     flat[0, :8].tolist()))

    if not n_tensors:
        print("no per-block weight_scale tensors matched", file=sys.stderr)
        return 2

    print(f"model:   {args.model_dir}")
    print(f"shards:  {len(sampled)} of {len(files)}")
    print(f"tensors: {n_tensors} per-block weight_scale tensors\n")
    for fn, name, shape, dtype, head in examples:
        print(f"  {fn}  {name}")
        print(f"    shape {shape} dtype {dtype}  first 8: "
              + ", ".join(f"{v:g}" for v in head))
    print()

    pct_differ = 100.0 * n_pairs_differ / n_pairs
    pct_pow2 = 100.0 * (n_vals - n_not_pow2) / n_vals
    print(f"adjacent pairs:  {n_pairs:,} checked, {n_pairs_differ:,} differ "
          f"({pct_differ:.4f}%)")
    print(f"power-of-two:    {n_vals:,} values, {pct_pow2:.4f}% are exactly 2^k")
    print(f"distinct values seen (sample): "
          + ", ".join(f"{v:g}" for v in sorted(seen_values)[:20]))
    print()

    block32_derived = pct_differ < 0.01 and pct_pow2 > 99.99
    if block32_derived:
        print("VERDICT: scales are block32-derived (MXFP4/E8M0 source).")
        print("  Adjacent slots are identical and every value is a power of")
        print("  two. There is no block16 weight arm in this checkpoint; a")
        print("  weight-side comparison would be a no-op.")
    else:
        print("VERDICT: scales are genuinely block16.")
        print("  Adjacent slots differ and the mantissa is in use, so the two")
        print("  weight groupings are distinguishable and a weight-side arm")
        print("  is possible on this checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
