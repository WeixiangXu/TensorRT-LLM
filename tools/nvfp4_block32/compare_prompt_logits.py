"""Decide whether the block32 flag changed the computation, from logprob dumps.

Takes two same-arm dumps (the noise floor) and one dump from the other arm (the
effect). Equality is the wrong test here -- multi-rank inference is not bit
reproducible -- so the verdict is whether the cross-arm difference stands well
clear of the run-to-run difference within one arm.

Usage: compare_prompt_logits.py a1.json a2.json b1.json
"""
import json
import sys


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return {int(t): float(v) for t, v in d["logprobs"]}


def diff(x, y):
    keys = sorted(set(x) & set(y))
    if not keys:
        return None
    deltas = [abs(x[k] - y[k]) for k in keys]
    return {
        "n": len(keys),
        "max": max(deltas),
        "mean": sum(deltas) / len(deltas),
        "n_nonzero": sum(1 for d in deltas if d != 0.0),
    }


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    a1, a2, b1 = (load(p) for p in sys.argv[1:4])

    noise = diff(a1, a2)
    effect = diff(a1, b1)
    if noise is None or effect is None:
        print("FAIL: dumps share no token positions")
        return 1

    print("same-arm  (noise floor): n=%d max=%.3e mean=%.3e differing=%d" %
          (noise["n"], noise["max"], noise["mean"], noise["n_nonzero"]))
    print("cross-arm (effect):      n=%d max=%.3e mean=%.3e differing=%d" %
          (effect["n"], effect["max"], effect["mean"], effect["n_nonzero"]))

    if effect["max"] == 0.0:
        print("VERDICT: the arms are identical -- the flag changed nothing")
        return 1
    if noise["max"] == 0.0:
        print("VERDICT: same-arm runs agree exactly and the arms differ -- "
              "the flag demonstrably changes the computation")
        return 0
    ratio = effect["mean"] / noise["mean"] if noise["mean"] else float("inf")
    print("effect/noise mean ratio: %.1fx" % ratio)
    if ratio >= 10:
        print("VERDICT: the effect is far above the run-to-run noise -- "
              "the flag demonstrably changes the computation")
        return 0
    print("VERDICT: INCONCLUSIVE -- the effect is not clearly above the noise")
    return 1


if __name__ == "__main__":
    sys.exit(main())
