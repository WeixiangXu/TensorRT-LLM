"""Rewrite an NVFP4 checkpoint's block16 weight scales into block32 semantics.

Blackwell NVFP4 tensor cores only read a scale every 16 elements. To measure
block32 numerics on that hardware, each pair of adjacent block16 E4M3 scales is
replaced by their maximum, so the two hardware blocks share one scale and the
resulting E2M1 encodings match a true block32 quantization bit-for-bit. See
test_block32_broadcast_equivalence.py for the identity that makes taking the max
of the two already-quantized scales exact.

Only ``*.weight_scale`` tensors of NVFP4 layers are touched: the packed FP4
weights, the FP32 per-tensor ``weight_scale_2``, and ``input_scale`` are copied
verbatim. ``hf_quant_config.json`` keeps ``group_size: 16`` because the
checkpoint stays structurally block16 -- it is only numerically block32 -- so
the runtime loads it through the unmodified NVFP4 path.

This measures accuracy only. Scale storage and metadata traffic are unchanged,
so it says nothing about the performance a real block32 kernel would deliver.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

MARKER_FILENAME = "nvfp4_block32_fake.json"


def read_header(path):
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return header, 8 + header_len


def nvfp4_layer_prefixes(src):
    """Prefixes of layers the quant config declares as NVFP4 with group_size 16."""
    with open(os.path.join(src, "hf_quant_config.json")) as fh:
        cfg = json.load(fh)["quantization"]
    prefixes = []
    for name, spec in cfg.get("quantized_layers", {}).items():
        if spec.get("quant_algo") != "NVFP4":
            continue
        group_size = spec.get("group_size", cfg.get("group_size"))
        if group_size != 16:
            raise ValueError(
                "%s has group_size %r; this script only converts 16 -> 32" %
                (name, group_size))
        prefixes.append(name + ".")
    if not prefixes:
        raise ValueError("no NVFP4 quantized_layers found in hf_quant_config.json")
    return tuple(prefixes)


def target_tensors(header, prefixes):
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not name.endswith(".weight_scale"):
            continue
        if not name.startswith(prefixes):
            continue
        if spec["dtype"] != "F8_E4M3":
            raise ValueError("%s: expected F8_E4M3, got %s" % (name, spec["dtype"]))
        if spec["shape"][-1] % 2:
            raise ValueError("%s: last dim %d is odd, cannot pair block16 scales" %
                             (name, spec["shape"][-1]))
        yield name, spec


def broadcast_pairwise_max(raw):
    """Pairwise max over adjacent bytes, each result written into both slots.

    E4M3 scales are non-negative, and for a clear sign bit the E4M3 bit pattern
    is monotonic in value, so an unsigned byte max is the numeric max.
    """
    if raw.max() > 0x7F:
        raise ValueError("negative E4M3 scale encountered (sign bit set)")
    pairs = raw.reshape(-1, 2)
    merged = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.repeat(merged, 2)


def convert_shard(src_path, dst_path, prefixes, verify_only=False):
    header, data_start = read_header(src_path)
    targets = list(target_tensors(header, prefixes))
    if not targets:
        if not verify_only and not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)
        return 0, 0

    if not verify_only:
        shutil.copyfile(src_path, dst_path)

    converted_bytes = 0
    mode = "rb" if verify_only else "r+b"
    with open(dst_path, mode) as fh:
        for name, spec in targets:
            begin, end = spec["data_offsets"]
            fh.seek(data_start + begin)
            raw = np.frombuffer(fh.read(end - begin), dtype=np.uint8)
            if verify_only:
                pairs = raw.reshape(-1, 2)
                if not np.array_equal(pairs[:, 0], pairs[:, 1]):
                    raise ValueError("%s in %s: adjacent scales differ" %
                                     (name, dst_path))
            else:
                fh.seek(data_start + begin)
                fh.write(broadcast_pairwise_max(raw).tobytes())
            converted_bytes += end - begin
    return len(targets), converted_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source NVFP4 checkpoint directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing --dst instead of writing it")
    args = ap.parse_args()

    prefixes = nvfp4_layer_prefixes(args.src)
    shards = sorted(f for f in os.listdir(args.src) if f.endswith(".safetensors"))
    if not args.verify:
        os.makedirs(args.dst, exist_ok=True)
        for entry in sorted(os.listdir(args.src)):
            path = os.path.join(args.src, entry)
            if entry.endswith(".safetensors") or entry.startswith(".") or os.path.isdir(path):
                continue
            shutil.copyfile(path, os.path.join(args.dst, entry))

    total_tensors = total_bytes = 0
    started = time.time()
    for i, shard in enumerate(shards, 1):
        n, nbytes = convert_shard(os.path.join(args.src, shard),
                                  os.path.join(args.dst, shard), prefixes,
                                  verify_only=args.verify)
        total_tensors += n
        total_bytes += nbytes
        print("[%2d/%d] %s: %d scale tensors, %.1f MB (%.0fs elapsed)" %
              (i, len(shards), shard, n, nbytes / 2**20, time.time() - started))
        sys.stdout.flush()

    if args.verify:
        print("VERIFIED: %d scale tensors, all adjacent block16 pairs equal" % total_tensors)
        return

    with open(os.path.join(args.dst, MARKER_FILENAME), "w") as fh:
        json.dump({
            "source_checkpoint": os.path.abspath(args.src),
            "transform": "adjacent block16 E4M3 weight scales replaced by their pairwise max",
            "effective_weight_block_size": 32,
            "stored_block_size": 16,
            "scale_tensors_converted": total_tensors,
            "note": "Numerically equivalent to block32 weight quantization. "
                    "Accuracy study only -- scale storage is unchanged, so this "
                    "carries no block32 performance benefit.",
        }, fh, indent=2)
    print("Converted %d scale tensors (%.1f GB of scales) in %.0fs" %
          (total_tensors, total_bytes / 2**30, time.time() - started))


if __name__ == "__main__":
    main()
