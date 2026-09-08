"""Perplexity probe for the block32 activation study.

A single number per run says nothing here: multi-rank MoE inference is not
reproducible, and two runs of the same arm disagree (measured: 7.4% relative on
raw prefill logits, effect/noise 1.1x when comparing arms that way). So this
repeats the measurement inside one process and reports the spread, and the two
arms are compared as distributions rather than as single values.

Perplexity comes from one prefill pass -- cross-entropy of the context logits
against the actual next token -- so no sampling enters the number.

Usage (one arm per process, the flag is read once at startup):
  mpirun -n 8 --allow-run-as-root -x TRTLLM_NVFP4_ACT_BLOCK32=1 \
      trtllm-llmapi-launch python3 ppl_probe.py --model_dir DIR --repeats 3
"""
import argparse
import json
import math
import os

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

PASSAGE = (
    "The sky appears blue because of Rayleigh scattering. Sunlight reaching "
    "Earth is a mixture of all visible wavelengths, and when it meets the "
    "nitrogen and oxygen molecules of the atmosphere, the shorter wavelengths "
    "scatter far more strongly than the longer ones. The scattering rate rises "
    "as the inverse fourth power of the wavelength, so blue light, near four "
    "hundred and fifty nanometres, is redirected roughly ten times as often as "
    "red light near seven hundred nanometres. Looking away from the sun, an "
    "observer therefore sees light that has been scattered many times out of "
    "the direct beam, and that light is dominated by the blue end of the "
    "spectrum. Violet scatters even more strongly, but the sun emits less of "
    "it and the human eye is less sensitive to it, so the sky reads as blue "
    "rather than violet. Near sunrise and sunset the path through the "
    "atmosphere is much longer, the blue light is scattered out of the line of "
    "sight entirely, and what remains to reach the observer is red and orange.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    flag = os.environ.get("TRTLLM_NVFP4_ACT_BLOCK32", "0")
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.tp,
        moe_expert_parallel_size=args.tp,
        moe_config=MoeConfig(backend="CUTLASS"),
        # Block reuse must be off: with it on, a repeat of the same passage
        # hits the cache and prefills a single token, so context_logits comes
        # back with one row and the perplexity is computed over nothing.
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5,
                                      tokens_per_block=128,
                                      enable_block_reuse=False),
        max_seq_len=4096,
        max_num_tokens=8192,
    )
    tok = llm.tokenizer
    ids = tok.encode(PASSAGE)
    params = SamplingParams(max_tokens=1, temperature=0, top_k=1,
                            return_context_logits=True)

    ppls = []
    for _ in range(args.repeats):
        out = llm.generate([PASSAGE], params)[0]
        logits = out.context_logits
        if logits is None:
            print("FAIL: the backend returned no context logits")
            return 1
        lg = logits.detach().float().cpu()
        n = min(lg.shape[0], len(ids)) - 1
        if n <= 0:
            print("FAIL: prefill returned %d logit rows for %d tokens; the "
                  "passage was served from cache" % (lg.shape[0], len(ids)))
            return 1
        # Position i predicts token i+1.
        logprobs = torch.log_softmax(lg[:n], dim=-1)
        tgt = torch.tensor(ids[1:n + 1])
        nll = -logprobs.gather(1, tgt.unsqueeze(1)).squeeze(1)
        ppls.append(float(math.exp(nll.mean().item())))
        print("  ppl: %.6f" % ppls[-1])

    mean = sum(ppls) / len(ppls)
    spread = max(ppls) - min(ppls)
    print("ARM flag=%s  tokens=%d  repeats=%d" % (flag, n, len(ppls)))
    print("  mean ppl : %.6f" % mean)
    print("  spread   : %.6f  (%.3f%% of mean)" % (spread, 100 * spread / mean))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"flag": flag, "tokens": n, "ppls": ppls, "mean": mean,
                       "spread": spread}, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
