#!/usr/bin/env python3
"""Can this card host the teacher at all, and at what gpu_memory_utilization? (HPC, one GPU.)

    python scripts/probe_teacher_fits.py --model <local MiroThinker path>

THE CHEAPEST POSSIBLE REFUTATION OF A HARDWARE CHANNEL, which is why it runs first. The
teacher is a 30B-A3B MoE: ~61 GiB of bf16 weights before any KV cache. On an H20 (95.1 GiB)
the pinned TEACHER_GPU_MEMORY_UTILIZATION of 0.72 leaves 68.4 GiB and that is comfortable. On
an A800 (79.2 GiB) the same fraction is 57.0 GiB -- LESS THAN THE WEIGHTS. The teacher does
not load, and every gate below it is meaningless.

So this searches for a utilization that works instead of testing one guess, and reports the
KV budget that survives at it. A value that loads the weights but leaves no room to score a
full-length sequence is not a pass.

WHAT A PASS HERE DOES NOT MEAN. Only that the model loads and produces finite logprobs at the
target length. Batch invariance, chunk-boundary invariance and the online/offline equality
they underwrite are the determinism probe's job, and its verdict does not follow from this one.

THE VALUE THIS FINDS BECOMES PART OF THE CHANNEL'S IDENTITY. gpu_memory_utilization is a
BackendConfig ENGINE_RUNTIME_FIELD, so a channel that needs 0.88 produces a different
fingerprint from one running 0.72 -- which is exactly what stops a cache from one being fed
to the other. Record what this prints.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher.engine import (  # noqa: E402  (must precede `import vllm`)
    TEACHER_MAX_MODEL_LEN,
    TEACHER_MAX_NUM_BATCHED_TOKENS,
    pin_determinism_env,
)

GIB = 2 ** 30


def try_util(model: str, dtype: str, util: float, seq_len: int, max_model_len: int) -> tuple:
    """-> (loaded, scored_ok, note). Runs in THIS process; the caller retries in a fresh one
    because a failed vLLM init does not reliably free the card."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    total = torch.cuda.get_device_properties(0).total_memory / GIB
    print(f"[fits] trying gpu_memory_utilization={util:.2f} on a {total:.1f} GiB card "
          f"({util * total:.1f} GiB budget)")
    llm = LLM(model=model, dtype=dtype, gpu_memory_utilization=util,
              max_model_len=max_model_len,
              max_num_batched_tokens=TEACHER_MAX_NUM_BATCHED_TOKENS,
              enforce_eager=True, enable_prefix_caching=False)
    print(f"[fits]   loaded. now scoring a {seq_len}-token sequence "
          f"(the worst case this channel would ever see)")
    ids = list(range(1000, 1000 + seq_len))
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(max_tokens=1, prompt_logprobs=0))
    lp = out[0].prompt_logprobs
    ok = lp is not None and len(lp) == seq_len and lp[0] is None
    finite = ok and all(
        any(v.logprob == v.logprob for v in d.values()) for d in lp[1:] if d)  # not NaN
    return True, bool(ok and finite), f"prompt_logprobs len={0 if lp is None else len(lp)}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Does the teacher fit this card, and at what util?")
    ap.add_argument("--model", required=True, help="LOCAL MiroThinker path")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--utils", default="0.72,0.80,0.86,0.90,0.93",
                    help="utilizations to try, low to high. 0.72 is the H20 pin -- kept first "
                         "so the log records whether the existing value would have worked")
    ap.add_argument("--seq-len", type=int, default=TEACHER_MAX_MODEL_LEN - 1024,
                    help="length to score; default is just under max_model_len")
    ap.add_argument("--max-model-len", type=int, default=TEACHER_MAX_MODEL_LEN)
    ap.add_argument("--only", type=float, default=None,
                    help="try exactly this one value (used by the retry loop)")
    args = ap.parse_args()

    pin_determinism_env()
    import torch

    if not torch.cuda.is_available():
        print("[fits] FATAL: no usable CUDA device", file=sys.stderr)
        return 1

    utils = [args.only] if args.only is not None else [
        float(x) for x in args.utils.split(",")]
    for util in utils:
        try:
            loaded, scored, note = try_util(args.model, args.dtype, util, args.seq_len,
                                            args.max_model_len)
        except Exception as e:                            # noqa: BLE001 - OOM or anything else
            first = str(e).strip().splitlines()[0][:150]
            print(f"[fits]   FAILED at {util:.2f}: {type(e).__name__}: {first}")
            continue
        if loaded and scored:
            print()
            print(f"[fits] PASS: the teacher loads and scores {args.seq_len} tokens at "
                  f"gpu_memory_utilization={util:.2f} ({note})")
            print(f"[fits] RECORD THIS VALUE. It is a BackendConfig ENGINE_RUNTIME_FIELD, so a "
                  f"channel running {util:.2f} produces a different fingerprint from one "
                  f"running 0.72 -- which is what keeps their caches from being mixed.")
            return 0
        print(f"[fits]   loaded at {util:.2f} but scoring failed ({note}) -- KV budget too "
              f"small for {args.seq_len} tokens")

    print()
    print(f"[fits] FAIL: no utilization in {utils} both loads the teacher AND scores "
          f"{args.seq_len} tokens on this card.", file=sys.stderr)
    print("[fits] This card cannot host the teacher, so it cannot host either arm of an "
          "experiment. Every gate below this one is moot.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
