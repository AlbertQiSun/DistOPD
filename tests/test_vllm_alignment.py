"""Locks the vLLM prompt_logprobs off-by-one WITHOUT needing vLLM.

Fabricates a vLLM-shaped prompt_logprobs structure from known logits and asserts
the conversion reproduces the same align-to-t logprobs as the HF path. This is
the local proof of the "first entry is None, entry t = log p(token_t | <t)"
convention; the real numeric vLLM-vs-HF check is gate 3 (@slow).
"""
from __future__ import annotations

import torch

from opd.masking import token_aligned_logprobs, vllm_prompt_logprobs_to_aligned


class _FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def test_vllm_conversion_matches_hf_alignment():
    torch.manual_seed(0)
    n, v = 12, 64
    logits = torch.randn(n, v)
    ids = torch.randint(0, v, (n,))

    hf_aligned = token_aligned_logprobs(logits, ids)
    lsm = torch.log_softmax(logits.float(), dim=-1)

    # build vLLM-style prompt_logprobs: entry t = log p(token_t | <t) using row t-1
    prompt_logprobs = [None]
    for t in range(1, n):
        entry = {int(ids[t]): _FakeLogprob(float(lsm[t - 1, ids[t]]))}
        # a distractor top-logprob to mimic prompt_logprobs=k>0
        distractor = (int(ids[t]) + 1) % v
        entry[distractor] = _FakeLogprob(float(lsm[t - 1, distractor]))
        prompt_logprobs.append(entry)

    aligned = vllm_prompt_logprobs_to_aligned(prompt_logprobs, ids)
    assert torch.isnan(aligned[0])
    assert torch.allclose(aligned[1:], hf_aligned[1:], atol=1e-5)
