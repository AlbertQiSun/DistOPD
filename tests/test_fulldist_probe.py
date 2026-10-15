"""The full-distribution probe: every quantity checked against an independent computation.

WHY. These numbers pick K for a dump that cannot be regenerated cheaply, and decide whether
to spend a training run. Each closed form here (the cosine especially, which is algebra over
six reductions rather than a dot product of two 151936-vectors) is asserted against a
BRUTE-FORCE version built from the definitions, sharing no line with the implementation. A
sign slip in L or in the advantage would otherwise produce a plausible table.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import scripts.fulldist_probe as P


def _pair(V=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    lp = torch.log_softmax(torch.randn(3, V, generator=g), dim=-1)
    lq = torch.log_softmax(torch.randn(3, V, generator=g) * 1.7, dim=-1)
    y = torch.randint(0, V, (3,), generator=g)
    return lp, lq, y


def test_the_divergences_match_their_definitions():
    lp, lq, y = _pair()
    st = P.row_stats(lp, lq, y)
    p, q = lp.exp().double(), lq.exp().double()
    assert torch.allclose(st["rev_kl"], (p * (lp.double() - lq.double())).sum(-1), atol=1e-10)
    assert torch.allclose(st["fwd_kl"], (q * (lq.double() - lp.double())).sum(-1), atol=1e-10)
    assert torch.allclose(st["tv"], 0.5 * (p - q).abs().sum(-1), atol=1e-10)
    assert torch.allclose(st["bc"], (p * q).sqrt().sum(-1), atol=1e-9)


def test_the_two_KLs_are_not_accidentally_the_same_quantity():
    """The failure this project already had once: a field called KL that was not the KL the
    reader assumed. If these two ever coincide the fixture is degenerate and the direction
    convention is untested."""
    lp, lq, y = _pair()
    st = P.row_stats(lp, lq, y)
    assert not torch.allclose(st["rev_kl"], st["fwd_kl"], atol=1e-3)


def test_hellinger_and_renyi_are_recoverable_from_BC():
    """BC is stored instead of Hellinger and Renyi-0.5 because it yields both. If that were
    false the design would be storing too little."""
    lp, lq, y = _pair()
    bc = P.row_stats(lp, lq, y)["bc"]
    p, q = lp.exp().double(), lq.exp().double()
    hell = (0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum(-1)).sqrt()
    assert torch.allclose((1 - bc).clamp_min(0).sqrt(), hell, atol=1e-8)
    assert torch.allclose(-2 * bc.log(), -2 * torch.log(bc), atol=1e-12)


def test_the_advantage_matches_opd_loss_sign_convention():
    """A_y = log q_y - log p_y = teacher - student, exactly what opd/loss.py computes. A
    flipped sign here would invert the pre-check's correlation."""
    from opd.loss import compute_advantage

    lp, lq, y = _pair()
    st = P.row_stats(lp, lq, y)
    rows = torch.arange(3)
    ref = compute_advantage(lq[rows, y], lp[rows, y], torch.ones(3, dtype=torch.bool))
    assert torch.allclose(st["adv"].float(), ref, atol=1e-6)


def test_the_cosine_closed_form_equals_a_brute_force_dot_product():
    """THE check that matters. The cosine is computed from six scalar reductions; the
    reference builds both 151936-dim gradient vectors and takes their actual dot product,
    from the definitions grad_full = p*(L-R) and grad_samp = L_y*(e_y - p)."""
    lp, lq, y = _pair(V=97, seed=4)
    st = P.row_stats(lp, lq, y)
    p = lp.exp().double()
    L = (lp - lq).double()
    R = (p * L).sum(-1, keepdim=True)
    g_full = p * (L - R)
    e_y = torch.zeros_like(p)
    e_y[torch.arange(3), y] = 1.0
    L_y = L[torch.arange(3), y].unsqueeze(1)
    g_samp = L_y * (e_y - p)
    ref = ((g_full * g_samp).sum(-1)
           / (g_full.norm(dim=-1) * g_samp.norm(dim=-1)).clamp_min(1e-30))
    assert torch.allclose(st["cos"], ref, atol=1e-9), (st["cos"], ref)


def test_the_gradient_norms_match_their_brute_force_versions():
    lp, lq, y = _pair(V=53, seed=7)
    st = P.row_stats(lp, lq, y)
    p = lp.exp().double()
    L = (lp - lq).double()
    R = (p * L).sum(-1, keepdim=True)
    assert torch.allclose(st["grad_full_norm"], (p * (L - R)).norm(dim=-1), atol=1e-9)
    e_y = torch.zeros_like(p)
    e_y[torch.arange(3), y] = 1.0
    L_y = L[torch.arange(3), y].unsqueeze(1)
    assert torch.allclose(st["grad_samp_norm"], (L_y * (e_y - p)).norm(dim=-1), atol=1e-9)


def test_topk_mass_is_cumulative_and_hits_one_at_full_vocabulary():
    V = 200
    lp = torch.log_softmax(torch.randn(2, V, generator=torch.Generator().manual_seed(1)), -1)
    m = P.topk_mass(lp, max(P.K_GRID))
    assert m.shape == (2, len(P.K_GRID))
    assert torch.all(m[:, 1:] >= m[:, :-1] - 1e-12), "mass must be non-decreasing in K"
    assert torch.allclose(m[:, -1], torch.ones(2, dtype=torch.float64), atol=1e-9), (
        "K beyond the vocabulary must reach the whole mass")
    assert torch.allclose(m[:, 0], lp.max(-1).values.double().exp(), atol=1e-9)


def test_js_is_bounded_by_log2_and_symmetric():
    lp, lq, y = _pair(V=40, seed=11)
    a = P.row_stats(lp, lq, y)["js"]
    b = P.row_stats(lq, lp, y)["js"]
    assert torch.all(a <= math.log(2.0) + 1e-9)
    assert torch.allclose(a, b, atol=1e-10), "JS must not depend on argument order"


def test_models_are_placed_with_device_map_not_dot_to():
    """PRODUCER SIDE. `from_pretrained(...).to(dev)` builds the whole model in CPU RAM and
    then copies it, so the peak is ~2x the model; the 30B teacher was SIGKILLed at shard
    246/531 with 256 GB of RAM requested. device_map places each shard as it is read. This
    reads the source because there is no way to assert it from a unit test otherwise, and a
    silent revert costs an hour of queue time to rediscover."""
    with open("scripts/fulldist_probe.py", encoding="utf-8") as f:
        src = f.read()
    assert 'device_map={"": d_te}' in src and 'device_map={"": d_st}' in src
    # Match the MODEL-placement shape only. A bare ".to(d_te)" also matches legitimate TENSOR
    # moves inside the loop, which is what the first version of this assertion caught.
    assert "dtype=dt).to(" not in src, (
        "model placement reverted to from_pretrained(...).to(device), which doubles CPU peak")


# --------------------------------------------------------------------------- logits peak
class _Body:
    def __init__(self, W):
        self.W = W

    def __call__(self, input_ids=None):
        h = torch.nn.functional.one_hot(input_ids[0], self.W.shape[0]).float() @ self.W
        return type("O", (), {"last_hidden_state": h.unsqueeze(0)})()


class _Fake(torch.nn.Module):
    """Has the Qwen shape: .model returns post-norm hidden states, .lm_head projects them.
    __call__ returns logits the ordinary way, so the two paths can be compared."""

    def __init__(self, V=23, H=7):
        super().__init__()
        g = torch.Generator().manual_seed(2)
        self.W = torch.randn(V, H, generator=g)
        self.model = _Body(self.W)
        self.lm_head = torch.nn.Linear(H, V, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(V, H, generator=g))

    def forward(self, input_ids=None):
        h = self.model(input_ids=input_ids).last_hidden_state
        return type("O", (), {"logits": self.lm_head(h)})()


def test_the_hidden_state_path_reproduces_the_full_logits_path():
    """The peak fix must not change a number. head(body(x)) is logits for this layout, and
    that is asserted rather than assumed -- if a model applied its final norm inside the
    LM-head path instead, every log-prob in the probe would be quietly wrong."""
    m = _Fake()
    ids = torch.randint(0, 23, (40,), generator=torch.Generator().manual_seed(5))
    rows = torch.arange(1, 40, 3)
    chunks = list(P.chunked_logprobs(m, ids, rows, "cpu", chunk=5))
    got = torch.cat(chunks)
    ref = torch.log_softmax(m(input_ids=ids.unsqueeze(0)).logits[0][rows].float(), dim=-1)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()


def test_the_chunking_covers_every_requested_row_exactly_once():
    m = _Fake()
    ids = torch.randint(0, 23, (30,), generator=torch.Generator().manual_seed(6))
    rows = torch.arange(0, 29, 2)
    got = torch.cat(list(P.chunked_logprobs(m, ids, rows, "cpu", chunk=4)))
    assert got.shape[0] == rows.numel()


def test_the_probe_no_longer_materialises_full_logits_in_its_loop():
    """PRODUCER SIDE. The peak fix lives in one call site; a revert to `.logits[0]` there
    would restore a 4.98 GiB per-trajectory tensor and no unit test would notice."""
    with open("scripts/fulldist_probe.py", encoding="utf-8") as f:
        src = f.read()
    assert "chunked_logprobs(student" in src and "chunked_logprobs(teacher" in src
    assert ".logits[0]" not in src.split("def main(")[1], (
        "main() materialises the whole [seq, vocab] logits tensor again")
