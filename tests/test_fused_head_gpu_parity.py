"""GPU half of the chunked-lm_head certification: real checkpoint, bf16 autocast. (@slow)

The CPU half (tests/test_fused_head_logprobs.py) established the contract on the mock:
longhand-referenced values, ~1-ulp-class deviation from the retired full-logits path,
bit-level determinism. GEMM blocking is a KERNEL property, so none of that transfers to
cuBLAS/A100/bf16 by argument -- this file measures the same three claims where the run
actually happens, plus the claim the whole change was for: the peak-memory drop.

    OPD_STUDENT_CKPT=... pytest -m slow tests/test_fused_head_gpu_parity.py

Run it once per channel before the first fused-path training run on that channel.
"""
from __future__ import annotations

import os

import pytest
import torch

from opd.masking import token_aligned_logprobs
from teacher.scorer import HFScorer

pytestmark = pytest.mark.slow

N = 4096          # long enough for many 512-row chunks; short enough to run in minutes


@pytest.fixture(scope="module")
def student():
    ckpt = os.environ.get("OPD_STUDENT_CKPT")
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip("OPD_STUDENT_CKPT unset -- source slurm/env.sh")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # same as the arms
    return HFScorer.from_pretrained(ckpt, dtype="float32", autocast_dtype="bfloat16",
                                    requires_grad=True, device="cuda:0")


def _ids(student, n=N, seed=7):
    g = torch.Generator().manual_seed(seed)
    vocab = int(student.lm.config.vocab_size)
    return torch.randint(1000, vocab - 1000, (n,), generator=g)


def _full_path_reference(student, ids):
    """The retired pre-change HFScorer path, reconstructed verbatim: one full forward
    under autocast -> token_aligned_logprobs on the [N, V] logits."""
    dev = next(student.lm.parameters()).device
    student.lm.train()
    with torch.enable_grad(), student._autocast(dev):
        logits = student.lm(ids.unsqueeze(0).to(dev)).logits[0]
    return token_aligned_logprobs(logits, ids.to(dev), chunk_rows=512)


def test_fused_deviation_from_full_path_is_bf16_ulp_class(student):
    ids = _ids(student)
    fused = student.logprobs(ids).detach()
    ref = _full_path_reference(student, ids).detach()
    dev = (fused[1:] - ref[1:]).abs()
    # bf16 logits carry ~8 fractional bits; a blocking-order flip in the last place of a
    # logit of magnitude ~30 moves logp by ~0.25 at most. Anything beyond that is not
    # rounding, it is a bug.
    assert float(dev.max()) <= 0.25, f"max deviation {dev.max()} -- beyond bf16 rounding"
    assert float(dev.median()) <= 0.02, f"median deviation {dev.median()} -- systematic shift"


def test_fused_values_are_bitwise_deterministic_on_gpu(student):
    """Run-to-run bit-stability of the VALUES -- the forward has no atomic accumulation,
    so this must hold exactly. (Full-model gradients are a different story; see the two
    tests below.)"""
    ids = _ids(student)
    a = student.logprobs(ids).detach().clone()
    b = student.logprobs(ids).detach().clone()
    assert torch.equal(a[1:], b[1:]), "fused GPU values not deterministic"


def test_head_stage_gradients_are_bitwise_deterministic_in_isolation(student):
    """The gradients THIS change is responsible for: hidden -> chunked head -> logp.
    That stage is GEMMs plus checkpoint replay -- no atomic scatter -- so with
    CUBLAS_WORKSPACE_CONFIG set it must be bit-stable. Isolating it from the trunk
    (detached hidden) is what separates 'the chunk loop added nondeterminism' from the
    trunk's pre-existing atomics."""
    from opd.masking import head_aligned_logprobs
    ids = _ids(student)
    dev = next(student.lm.parameters()).device
    student.lm.train()
    with torch.no_grad(), student._autocast(dev):
        hidden0 = student.lm.model(ids.unsqueeze(0).to(dev)).last_hidden_state[0]
    head = student.lm.lm_head
    kw = student._head_kw()

    reps = []
    for _ in range(2):
        hidden = hidden0.detach().clone().requires_grad_(True)
        if head.weight.grad is not None:
            head.weight.grad = None
        out = head_aligned_logprobs(head, hidden, ids.to(dev), **kw)
        out[1:].sum().backward()
        reps.append((hidden.grad.detach().clone(), head.weight.grad.detach().clone()))
    head.weight.grad = None
    assert torch.equal(reps[0][0], reps[1][0]), "hidden-grad of the head stage not bit-stable"
    assert torch.equal(reps[0][1], reps[1][1]), "head-weight grad of the head stage not bit-stable"


def _grad_rep_diff(student, ids, fn):
    """Max |grad difference| on lm_head.weight between two identical runs of fn."""
    grads = []
    for _ in range(2):
        student.lm.zero_grad(set_to_none=True)
        fn()[1:].sum().backward()
        grads.append(student.lm.lm_head.weight.grad.detach().clone())
    student.lm.zero_grad(set_to_none=True)
    return float((grads[0] - grads[1]).abs().max())


def test_full_model_grad_nondeterminism_is_the_trunks_and_preexisting(student):
    """MEASURED FINDING (<node>, 2026-08-19): full-model gradients are NOT run-to-run
    bit-stable -- and that is the TRUNK's property, not the chunk loop's. Qwen3-4B ties
    lm_head.weight to the embedding, whose backward scatter-adds with atomics, and SDPA's
    backward accumulates atomically too; the retired full-logits path had both
    identically. This test pins that shape: the fused path's rep-to-rep grad wobble must
    be the SAME CLASS as the retired path's, not a new one. A fused wobble while the full
    path is clean would mean the chunk loop itself is nondeterministic -- that is the
    failure this test exists to catch."""
    ids = _ids(student)
    d_fused = _grad_rep_diff(student, ids, lambda: student.logprobs(ids))
    d_full = _grad_rep_diff(student, ids, lambda: _full_path_reference(student, ids))
    if d_full == 0.0:
        assert d_fused == 0.0, \
            f"full path is bit-stable but fused wobbles by {d_fused} -- chunk loop bug"
    else:
        assert d_fused <= 16 * d_full, \
            (f"fused rep-to-rep grad wobble {d_fused} vs full path {d_full} -- "
             "more than the pre-existing class, the chunk loop is adding nondeterminism")


def test_fused_peak_memory_beats_full_path_at_max_len(student):
    """The reason the change exists, measured AT THE PRODUCTION LENGTH. At N=4096 this
    test was a null instrument: both paths peaked at the same byte (34.46 GiB, <node>)
    because the [N, V] pair (~2.3 GiB) hides under the trunk's own backward transient --
    the head only dominates at length. At N=15360 (rollout.max_len, the length that
    killed the offline run at step 3) the pair is ~8.7 GiB and the drop must show.
    Passing here also directly certifies that a max_len trajectory's scoring stage fits
    this card with the fused path.

    MEASURED FROM ACCUMULATED-GRAD STATE, not from empty buffers. Round 2 on <node>
    reproduced gate 6's original blind spot inside this very test: with zero_grad'd
    buffers the full path read 34.61 GiB (vs fused 33.41) because at the logits-backward
    moment only lm_head's grads exist yet -- the [N, V] pair never stacked on the
    16.1 GiB of param grads, exactly the empty-buffer geometry that made the bench
    approve a length the real run then died on (grad accumulation has ALL grads resident
    when a long trajectory lands at micro>0). A short warmup backward pre-fills every
    param grad first, so the peak here has the production geometry."""
    ids = _ids(student, n=15360)
    dev = next(student.lm.parameters()).device

    def peak(fn):
        student.lm.zero_grad(set_to_none=True)
        student.logprobs(_ids(student, n=256, seed=13))[1:].sum().backward()  # fill grads
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        fn()[1:].sum().backward()          # accumulates into RESIDENT grads, as micro>0 does
        torch.cuda.synchronize(dev)
        student.lm.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2**30

    p_full = peak(lambda: _full_path_reference(student, ids))
    p_fused = peak(lambda: student.logprobs(ids))
    assert p_fused <= p_full - 4.0, \
        f"fused peak {p_fused:.2f} GiB vs full {p_full:.2f} GiB -- expected >=4 GiB drop at max_len"
