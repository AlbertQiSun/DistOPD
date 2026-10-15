"""HARD GATE: chunked logprobs must be BIT-IDENTICAL to the whole-sequence computation.

Not "close". Bit-identical, values AND gradients. The claim licensing that is structural:
log_softmax normalises within a row and the gather takes one entry from that row, so row i's
output is a function of row i alone. How the rows are grouped into calls cannot change a
single bit. If the assertion below ever needs a tolerance, the premise is wrong and the
chunking is not the exact rewrite it is documented to be.

The gradient half is the part worth insisting on. Values could match while the backward
silently dropped or double-counted a chunk's contribution -- the loss would still look sane,
the model would train on a subtly wrong gradient, and nothing downstream would notice. That
is the failure mode this whole phase exists to make impossible, so `logits.grad` is compared
element-for-element, not just the forward output.

Chunk sizes are chosen to hit the awkward cases: bigger than the sequence (one chunk, the
old path), an exact divisor, a non-divisor with a short tail, and 1 (every row its own call).

A DIFFERENTIAL TEST ALONE IS NOT ENOUGH, and this file learned that the hard way. Injecting
an off-by-one into the chunk's target slice left every bit-identity assertion GREEN: the
"unchunked reference" is the same function called with a large chunk_rows, so a bug in the
SHARED body moves both sides equally and cancels. Differential tests can only see bugs that
depend on the axis being varied. So there is also an ABSOLUTE check against `_naive_reference`
below -- written out longhand, sharing no code with the function under test -- and every case
runs on BOTH the grad and no-grad paths, which are separate branches and were separately
breakable.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opd.loss import compute_opd_loss
from opd.masking import token_aligned_logprobs

CHUNKS = [10_000, 64, 37, 7, 1]


def _naive_reference(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[t] | row t-1), one row at a time, longhand.

    Shares NO code with token_aligned_logprobs -- that is the entire point. Compared with
    allclose rather than torch.equal: a per-row log_softmax and a batched one are the same
    mathematics but not necessarily the same kernel, so bit-identity is the right demand
    BETWEEN CHUNKINGS and closeness is the right demand against an independent reference.
    """
    n = int(ids.shape[0])
    out = torch.full((n,), float("nan"), dtype=torch.float32)
    for t in range(1, n):
        out[t] = torch.log_softmax(logits[t - 1].detach().float(), dim=-1)[int(ids[t])]
    return out


def _logits_ids(n=200, v=512, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, v, generator=g, dtype=torch.float32)
    if grad:
        logits.requires_grad_(True)
    ids = torch.randint(0, v, (n,), generator=g)
    return logits, ids


@pytest.mark.parametrize("grad", [False, True], ids=["nograd", "grad"])
@pytest.mark.parametrize("chunk", CHUNKS)
def test_values_are_bit_identical_to_the_unchunked_reference(chunk, grad):
    """Run on BOTH branches: checkpointed and not are separate code paths, and an injected
    break in one of them survived a suite that only exercised the other."""
    logits, ids = _logits_ids(grad=grad)
    ref = token_aligned_logprobs(logits, ids, chunk_rows=logits.shape[0] + 1)
    got = token_aligned_logprobs(logits, ids, chunk_rows=chunk)
    assert torch.isnan(ref[0]) and torch.isnan(got[0])
    assert torch.equal(ref[1:], got[1:]), f"chunk_rows={chunk} changed the values"


@pytest.mark.parametrize("grad", [False, True], ids=["nograd", "grad"])
@pytest.mark.parametrize("chunk", CHUNKS)
def test_values_match_an_INDEPENDENT_reference(chunk, grad):
    """The absolute check the differential ones cannot make. Catches any error in the shared
    body -- the alignment shift above all -- which moves both sides of a differential test
    together and therefore cancels out of it."""
    logits, ids = _logits_ids(n=60, v=128, grad=grad)
    got = token_aligned_logprobs(logits, ids, chunk_rows=chunk)
    want = _naive_reference(logits, ids)
    assert torch.allclose(got[1:].detach(), want[1:], atol=1e-6, rtol=0), (
        f"chunk_rows={chunk} grad={grad} disagrees with the longhand reference")


@pytest.mark.parametrize("chunk", CHUNKS)
def test_gradients_are_bit_identical_to_the_unchunked_reference(chunk):
    """The half that matters most: a chunked backward could drop or double a chunk's
    contribution while the forward still looked perfect."""
    la, ids = _logits_ids(grad=True)
    lb = la.detach().clone().requires_grad_(True)

    token_aligned_logprobs(la, ids, chunk_rows=la.shape[0] + 1)[1:].sum().backward()
    token_aligned_logprobs(lb, ids, chunk_rows=chunk)[1:].sum().backward()

    assert la.grad is not None and lb.grad is not None
    assert la.grad.abs().sum() > 0, "reference gradient is all zero -- test is vacuous"
    assert torch.equal(la.grad, lb.grad), f"chunk_rows={chunk} changed the gradient"


@pytest.mark.parametrize("chunk", CHUNKS)
def test_the_opd_loss_itself_is_bit_identical(chunk):
    """End of the chain: what the trainer actually optimises."""
    la, ids = _logits_ids(grad=True)
    lb = la.detach().clone().requires_grad_(True)
    teacher = torch.randn(200, generator=torch.Generator().manual_seed(7))
    mask = torch.zeros(200, dtype=torch.bool)
    mask[3:150] = True

    loss_a, adv_a = compute_opd_loss(
        teacher, token_aligned_logprobs(la, ids, chunk_rows=la.shape[0] + 1), mask, 10.0)
    loss_b, adv_b = compute_opd_loss(
        teacher, token_aligned_logprobs(lb, ids, chunk_rows=chunk), mask, 10.0)

    assert torch.equal(loss_a, loss_b), f"chunk_rows={chunk} changed the loss"
    assert torch.equal(adv_a[mask], adv_b[mask])
    loss_a.backward()
    loss_b.backward()
    assert torch.equal(la.grad, lb.grad), f"chunk_rows={chunk} changed the loss gradient"


def test_checkpointing_is_actually_engaged_when_grad_is_required():
    """Chunking the FORWARD alone would save nothing: autograd defines log_softmax's
    backward in terms of its OUTPUT, so every chunk's [rows, V] fp32 tensor would survive to
    backward and the peak would be unchanged. The saving comes from recomputing them. Assert
    the chunk function really is re-entered during backward."""
    import opd.masking as M

    calls = []
    real = M._logprob_chunk

    def counting(rows, targets):
        calls.append(rows.shape[0])
        return real(rows, targets)

    M._logprob_chunk = counting
    try:
        logits, ids = _logits_ids(n=100, grad=True)
        out = token_aligned_logprobs(logits, ids, chunk_rows=25)
        forward_calls = len(calls)
        out[1:].sum().backward()
        backward_calls = len(calls) - forward_calls
    finally:
        M._logprob_chunk = real

    assert forward_calls == 4, f"expected 4 chunks over 99 rows at 25/chunk, got {forward_calls}"
    assert backward_calls == forward_calls, (
        f"chunks were NOT recomputed in backward ({backward_calls} of {forward_calls}) -- "
        "checkpointing is not engaged, so the log_softmax outputs are being retained and "
        "the memory saving this change exists for is not happening")


def test_no_checkpoint_overhead_when_grad_is_not_required():
    """Teacher scoring and diagnostics run under no_grad; recomputation there would be pure
    waste, so that path must not checkpoint."""
    import opd.masking as M

    calls = []
    real = M._logprob_chunk
    M._logprob_chunk = lambda r, t: (calls.append(r.shape[0]), real(r, t))[1]
    try:
        logits, ids = _logits_ids(n=100, grad=False)
        with torch.no_grad():
            token_aligned_logprobs(logits, ids, chunk_rows=25)
    finally:
        M._logprob_chunk = real
    assert len(calls) == 4


@pytest.mark.parametrize("n", [1, 2, 3])
def test_short_sequences(n):
    logits, ids = _logits_ids(n=n, v=16)
    out = token_aligned_logprobs(logits, ids, chunk_rows=1)
    assert out.shape == (n,)
    assert torch.isnan(out[0])
    if n > 1:
        assert torch.isfinite(out[1:]).all()
        # compare from index 1: entry 0 is nan, and nan != nan makes torch.equal False
        assert torch.equal(out[1:], token_aligned_logprobs(logits, ids, chunk_rows=999)[1:])


@pytest.mark.parametrize("grad", [False, True], ids=["nograd", "grad"])
def test_chunking_does_not_change_the_alignment_convention(grad):
    """A chunked loop is an easy place to get the +1 shift wrong on the SECOND chunk only,
    which no aggregate would reveal. Score a one-hot sequence where the answer is known.

    Parametrised over grad because the checkpointed and plain branches slice the targets in
    two separate places -- an off-by-one injected into only the grad branch passed the whole
    original suite, which exercised this case on the no-grad branch alone."""
    n, v = 9, 5
    logits = torch.full((n, v), -1e9)
    ids = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3])
    for t in range(1, n):                       # row t-1 predicts ids[t], make that certain
        logits[t - 1, ids[t]] = 0.0
    if grad:
        logits = logits.requires_grad_(True)
    for chunk in (1, 2, 3, 4, 99):
        out = token_aligned_logprobs(logits, ids, chunk_rows=chunk)
        assert torch.allclose(out[1:].detach(), torch.zeros(n - 1), atol=1e-6), (
            f"chunk_rows={chunk} grad={grad} broke the align-to-t shift")
