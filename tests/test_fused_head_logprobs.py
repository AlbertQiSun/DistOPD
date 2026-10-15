"""Certification of the chunked-lm_head path (head_aligned_logprobs). (local, zero-GPU)

WHERE THIS COMES FROM. The full-logits path materialises [N, V] logits (bf16 4.35 GiB at
N=15360, V=151936) plus a same-sized gradient in backward. On the A100, with the steps>=1
resident base of 64.4 GiB (fp32 weights + accumulated grads + Adam moments), that pair is
~1 GiB more than the card has -- measured 2026-08-18/19: demand ~79.6 vs ~78.7 GiB, and the
real cache holds 14 max-length trajectories that hit it deterministically (census,
runs/cache_lengths.json). head_aligned_logprobs folds the lm_head matmul into the chunk
loop so the [N, V] never exists.

WHAT THE CERTIFICATION FOUND, AND WHAT IS THEREFORE ASSERTED. The first draft of this file
asserted bit-identity with the full-logits path. It FAILED: row-chunking the head matmul
changes the GEMM's M dimension and the kernel's blocking moves the bits -- measured ~5e-7,
1 ulp of the fp32 logits (pinned below as its own test, so the fact stays measured rather
than remembered). So the contract this file certifies is the honest one:

  1. VALUES against an ABSOLUTE longhand reference (logsumexp arithmetic, no code shared
     with opd.masking) -- per the test_logprob_chunking lesson;
  2. deviation from the exact pre-change reference path BOUNDED at the measured
     1-ulp-class scale, for values AND gradients (incl. tied embedding/head);
  3. DETERMINISM: same input, same weights -> bit-identical values and gradients. This,
     plus both arms sharing the path and the PINNED chunk size (HEAD_CHUNK_ROWS is a code
     constant, not a channel knob -- chunk size now moves bits, and a channel file may
     not change a number), is what the arm comparison actually needs;
  4. PRODUCER checks: the shipped Scorer/HFScorer path really is the fused one (the head
     never sees more than one chunk of rows -- delete _split_lm and these go red), and
     HFScorer REFUSES the silent full-logits fallback that would reintroduce the OOM.

GPU/bf16 on the real checkpoint is the @slow half: tests/test_fused_head_gpu_parity.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import opd.masking as masking
from opd.masking import head_aligned_logprobs, token_aligned_logprobs
from teacher.scorer import HFScorer, MockLM, Scorer

V, H, N = 613, 32, 97          # deliberately non-round; chunk sizes below never divide N-1

# Measured deviation scale of row-chunked GEMM vs full GEMM (1 ulp of these fp32 logits).
# The bound in the comparisons below is a few of these, not a loose tolerance.
ULP = 5e-7


def _mock(seed=3):
    return MockLM(V, dim=H, seed=seed)


def _ids(n=N, seed=11):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (n,), generator=g)


def _full_path_reference(lm, ids, chunk_rows=64):
    """The EXACT pre-change Scorer path: one full forward -> token_aligned_logprobs.
    Grad-carrying iff the caller's lm parameters require grad (the grad tests do)."""
    logits = lm(ids)
    out = token_aligned_logprobs(logits, ids, chunk_rows=chunk_rows)
    return out


# ---------------------------------------------------------------------------
# 0. the measured fact this file's contract is built on
# ---------------------------------------------------------------------------
def test_row_chunked_matmul_is_not_bitwise_and_we_know_it():
    """Row-chunked linear vs full linear on CPU fp32: NOT bit-identical, deviation ~1 ulp.
    This is the measurement that demoted the bit-identity claim to a bounded one and
    forced HEAD_CHUNK_ROWS to be a pinned code constant instead of a channel knob. If a
    torch upgrade ever makes it bitwise, this test says so (it asserts the BOUND, not the
    difference), and the contract can be revisited with evidence in hand."""
    lm = _mock()
    with torch.no_grad():
        hid = lm.emb(_ids())
        full = lm.head(hid)
        worst = max(float((lm.head(hid[a:min(a + 7, N)]) - full[a:min(a + 7, N)]).abs().max())
                    for a in range(0, N, 7))
    assert worst <= 4 * ULP, f"chunked-vs-full GEMM deviation {worst} exceeds the 1-ulp class"


# ---------------------------------------------------------------------------
# 1. ABSOLUTE reference: longhand logsumexp arithmetic, sharing no code with opd.masking
# ---------------------------------------------------------------------------
def test_fused_matches_longhand_logsumexp_reference():
    lm = _mock()
    ids = _ids()
    got = Scorer(lm, requires_grad=False).logprobs(ids)

    with torch.no_grad():
        hidden = lm.emb(ids)
        logits = hidden @ lm.head.weight.T                 # explicit matmul, full [N, V]
        ref = torch.full((ids.shape[0],), float("nan"))
        for t in range(1, ids.shape[0]):                   # longhand, row by row
            row = logits[t - 1].float()
            ref[t] = float(row[ids[t]] - torch.logsumexp(row, dim=0))
    assert torch.isnan(got[0])
    assert torch.allclose(got[1:], ref[1:], atol=1e-5), \
        f"max dev {(got[1:] - ref[1:]).abs().max()}"


# ---------------------------------------------------------------------------
# 2. bounded deviation from the pre-change path: values and gradients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [1, 7, 64, N + 5])
def test_fused_values_within_one_ulp_of_full_logits_path(chunk, monkeypatch):
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", chunk)
    lm = _mock()
    ids = _ids()
    fused = Scorer(lm, requires_grad=False).logprobs(ids)
    ref = _full_path_reference(lm, ids).detach()
    dev = (fused[1:] - ref[1:]).abs().max()
    # logp = logit - logsumexp(row): a 1-ulp logit shift moves logp by <= ~2 ulp
    assert float(dev) <= 8 * ULP, f"chunk={chunk}: deviation {dev} beyond the measured class"
    assert torch.isnan(fused[0]) and torch.isnan(ref[0])


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_fused_gradients_within_ulp_class_of_full_logits_path(chunk, monkeypatch):
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", chunk)
    ids = _ids()
    lm_a, lm_b = _mock(seed=5), _mock(seed=5)              # identical weights, two graphs
    for lm in (lm_a, lm_b):
        for p in lm.parameters():
            p.requires_grad_(True)

    Scorer(lm_a, requires_grad=True).logprobs(ids)[1:].sum().backward()
    _full_path_reference(lm_b, ids)[1:].sum().backward()

    for (na, pa), (_, pb) in zip(lm_a.named_parameters(), lm_b.named_parameters()):
        assert pa.grad is not None and pb.grad is not None, f"{na}: missing grad"
        scale = pb.grad.abs().max().clamp_min(1.0)
        rel = float((pa.grad - pb.grad).abs().max() / scale)
        assert rel <= 1e-5, f"chunk={chunk} grad of {na}: relative deviation {rel}"


def test_fused_gradients_with_tied_embedding_and_head(monkeypatch):
    """Qwen3-4B ties lm_head.weight to the input embedding, so the head's gradient
    accumulates into a weight the trunk also used. The chunk loop must not double-count
    or drop either contribution."""
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 7)

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(9)
            self.emb = nn.Embedding(V, H)
            self.head = nn.Linear(H, V, bias=False)
            with torch.no_grad():
                self.emb.weight.copy_(torch.randn(V, H, generator=g))
            self.head.weight = self.emb.weight            # the tie
        def forward(self, ids):
            return self.head(self.emb(ids))

    ids = _ids()
    lm_a, lm_b = Tied(), Tied()
    with torch.no_grad():
        lm_b.emb.weight.copy_(lm_a.emb.weight)
    Scorer(lm_a, requires_grad=True).logprobs(ids)[1:].sum().backward()
    _full_path_reference(lm_b, ids)[1:].sum().backward()
    ga, gb = lm_a.emb.weight.grad, lm_b.emb.weight.grad
    rel = float((ga - gb).abs().max() / gb.abs().max().clamp_min(1.0))
    assert rel <= 1e-5, f"tied-weight gradient relative deviation {rel}"


def test_fused_under_cpu_autocast_bf16_stays_in_the_bf16_ulp_class(monkeypatch):
    """The channel runs the forward under bf16 autocast; the fused path re-enters autocast
    per chunk for the head matmul. The deviation bound scales with bf16's ulp, not fp32's."""
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 13)
    lm = _mock()
    ids = _ids()
    fused = Scorer(lm, requires_grad=False, autocast_dtype="bfloat16").logprobs(ids)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = lm(ids)                                   # pre-change reference, autocast on
    ref = token_aligned_logprobs(logits, ids, chunk_rows=13)
    dev = float((fused[1:] - ref[1:]).abs().max())
    assert dev <= 0.05, f"autocast deviation {dev} beyond the bf16 ulp class"


# ---------------------------------------------------------------------------
# 3. determinism -- the property the arm comparison actually rests on
# ---------------------------------------------------------------------------
def test_fused_values_and_gradients_are_deterministic(monkeypatch):
    """Same weights, same input, twice -> bit-identical logp AND gradients. Combined with
    both arms sharing this path and the pinned chunk size, this is what replaces the
    (false) bit-identity-with-the-old-path claim."""
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 7)
    ids = _ids()
    outs, grads = [], []
    for _ in range(2):
        lm = _mock(seed=5)
        for p in lm.parameters():
            p.requires_grad_(True)
        out = Scorer(lm, requires_grad=True).logprobs(ids)
        out[1:].sum().backward()
        outs.append(out.detach().clone())
        grads.append({n: p.grad.clone() for n, p in lm.named_parameters()})
    assert torch.equal(outs[0][1:], outs[1][1:]), "fused values are not run-to-run stable"
    for n in grads[0]:
        assert torch.equal(grads[0][n], grads[1][n]), f"gradient of {n} not deterministic"


# ---------------------------------------------------------------------------
# 4. PRODUCER checks: the shipped path really is the fused one
# ---------------------------------------------------------------------------
class _RowRecorder:
    """Forward-hook that records how many rows the head sees per call."""
    def __init__(self):
        self.rows = []
    def __call__(self, module, args, output):
        self.rows.append(int(args[0].shape[0]))


def test_scorer_head_never_sees_more_than_one_chunk(monkeypatch):
    """Delete Scorer._split_lm (or make it return None) and this goes red: the fallback
    full-logits path would hand the head all N rows in one call."""
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 7)
    lm = _mock()
    rec = _RowRecorder()
    lm.head.register_forward_hook(rec)
    Scorer(lm, requires_grad=False).logprobs(_ids())
    assert rec.rows, "head was never called"
    assert max(rec.rows) <= 7, f"head saw {max(rec.rows)} rows -- [N, V] materialised"
    assert len(rec.rows) >= (N - 1) // 7, "fewer head calls than chunks"


def test_scorer_head_chunked_under_grad_too(monkeypatch):
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 7)
    lm = _mock()
    for p in lm.parameters():
        p.requires_grad_(True)
    rec = _RowRecorder()
    lm.head.register_forward_hook(rec)
    Scorer(lm, requires_grad=True).logprobs(_ids())[1:].sum().backward()
    assert max(rec.rows) <= 7, f"head saw {max(rec.rows)} rows under grad"


def test_channel_chunk_knob_does_not_reach_the_head(monkeypatch):
    """logprob_chunk_rows is a channel knob and channels may not move bits; the head
    chunk size must be the pinned constant regardless of what the channel says."""
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 7)
    lm = _mock()
    rec = _RowRecorder()
    lm.head.register_forward_hook(rec)
    Scorer(lm, requires_grad=False, logprob_chunk_rows=3).logprobs(_ids())
    assert max(rec.rows) == 7, \
        f"head chunk followed the channel knob (saw {max(rec.rows)} rows, pinned is 7)"


def _fake_hf(seed=21):
    class FakeTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.emb = nn.Embedding(V, H)
            with torch.no_grad():
                self.emb.weight.copy_(torch.randn(V, H, generator=g))
        def forward(self, ids):
            return SimpleNamespace(last_hidden_state=self.emb(ids))

    class FakeCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed + 1)
            self.model = FakeTrunk()
            self.lm_head = nn.Linear(H, V, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.randn(V, H, generator=g) * 0.1)
        def forward(self, ids):
            return SimpleNamespace(logits=self.lm_head(self.model(ids).last_hidden_state))
    return FakeCausalLM()


def test_hfscorer_fused_within_ulp_class_of_its_old_full_forward(monkeypatch):
    monkeypatch.setattr(masking, "HEAD_CHUNK_ROWS", 7)
    lm = _fake_hf()
    ids = _ids()
    fused = HFScorer(lm, requires_grad=False).logprobs(ids)
    with torch.no_grad():
        logits = lm(ids.unsqueeze(0)).logits[0]            # the pre-change HFScorer forward
    ref = token_aligned_logprobs(logits, ids, chunk_rows=7)
    dev = float((fused[1:] - ref[1:]).abs().max())
    assert dev <= 8 * ULP, f"HFScorer fused deviation {dev} beyond the measured class"


def test_hfscorer_refuses_to_fall_back_to_full_logits():
    """An lm without .model/.lm_head must be a LOUD error: the silent fallback is the
    exact [N, V] materialisation that OOMs on the 14 max-length trajectories -- a crash
    that would otherwise only fire deep into a real run."""
    with pytest.raises(AttributeError, match="lm_head"):
        HFScorer(MockLM(V, seed=0), requires_grad=True).logprobs(_ids())


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------
def test_single_token_sequence_is_all_nan():
    lm = _mock()
    out = Scorer(lm, requires_grad=False).logprobs(torch.tensor([5]))
    assert out.shape == (1,) and torch.isnan(out[0])


def test_head_aligned_logprobs_direct_contract():
    """Direct-call contract of the new masking primitive (device/dtype/nan-at-0),
    mirroring token_aligned_logprobs' documented surface."""
    lm = _mock()
    ids = _ids()
    hidden = lm.emb(ids)
    out = head_aligned_logprobs(lm.head, hidden, ids, chunk_rows=10)
    assert out.dtype == torch.float32 and out.device == hidden.device
    assert torch.isnan(out[0]) and not torch.isnan(out[1:]).any()
