"""Dump prompt logprobs for a fixed prompt, for numeric arm comparison.

Generated text cannot show whether TRTLLM_NVFP4_ACT_BLOCK32 took effect: on 8
ranks the decode is non-deterministic even at temperature 0, so two runs of the
same arm already disagree. Prompt logprobs come from a single prefill pass over
a fixed input, so they are a direct numeric probe of the computation. Run this
twice per arm: the same-arm pair gives the noise floor, and the cross-arm pair
gives the effect. The flag is doing something iff the effect clears the noise.

Usage (rank 0 writes the file):
  mpirun -n 8 --allow-run-as-root trtllm-llmapi-launch \
      python3 dump_prompt_logits.py --model_dir DIR --out FILE [--tp 8]
"""
import argparse
import json

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

PROMPT = ("Rayleigh scattering explains why the daytime sky appears blue: "
          "shorter wavelengths scatter more strongly in the atmosphere.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.tp,
        moe_expert_parallel_size=args.tp,
        moe_config=MoeConfig(backend="CUTLASS"),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5,
                                      tokens_per_block=128),
        max_seq_len=4096,
        max_num_tokens=8192,
    )
    # One token of decode; everything we compare comes from the prefill.
    params = SamplingParams(max_tokens=1, temperature=0, top_k=1,
                            prompt_logprobs=1, return_context_logits=True)
    out = llm.generate([PROMPT], params)[0]

    # Context logits are the raw prefill output, the most direct probe. Prompt
    # logprobs are kept as a fallback in case the backend withholds the logits.
    values = []
    ctx = getattr(out, "context_logits", None)
    if ctx is not None:
        t = ctx.detach().float().cpu() if isinstance(ctx, torch.Tensor) else None
        if t is not None:
            flat = t.flatten()
            step = max(1, flat.numel() // 4096)  # subsample; full logits are huge
            values = [[i, float(flat[i])] for i in range(0, flat.numel(), step)]
            source = f"context_logits{tuple(t.shape)}"
    if not values:
        seq = out.outputs[0]
        for pos, slot in enumerate(getattr(seq, "prompt_logprobs", None) or []):
            if not slot:
                continue
            for tok, lp in sorted(slot.items()):
                values.append([pos * 1000000 + int(tok),
                               float(getattr(lp, "logprob", lp))])
        source = "prompt_logprobs"

    with open(args.out, "w") as fh:
        json.dump({"prompt": PROMPT, "source": source, "n": len(values),
                   "logprobs": values}, fh)
    print(f"wrote {len(values)} entries from {source} to {args.out}")


if __name__ == "__main__":
    main()
