"""The aligned-logprob helper must return on the LOGITS' device.

This looks like tidiness and is not. `torch.full` defaults to CPU, so the original
implementation built the [N] result on CPU and filled it from a CUDA gather; PyTorch's
cross-device `copy_` accepts that without complaint, and autograd still works, so nothing
failed. What it changed is one caller up: `opd_train_step` derives the loss device from
`student_logp.device`, so on every real GPU run that device was `cpu` -- the whole OPD loss
was computed on CPU, the backward crossed the device boundary, and each trajectory paid a
synchronising transfer. HFScorer.logprobs asserted the opposite in a comment.

The CPU assertions below state the invariant but cannot FALSIFY it on a one-device machine
(cpu == cpu holds either way). The cuda test is the one that would have caught this, and it
only runs on the HPC node -- so treat a green local suite as "not disproven", not "proven".
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opd.masking import token_aligned_logprobs


def _case(device: str, n: int = 16, v: int = 64):
    g = torch.Generator(device="cpu").manual_seed(0)
    logits = torch.randn(n, v, generator=g).to(device)
    ids = torch.randint(0, v, (n,), generator=g).to(device)
    return logits, ids


def test_returns_on_the_logits_device_cpu():
    logits, ids = _case("cpu")
    out = token_aligned_logprobs(logits, ids)
    assert out.device == logits.device
    assert torch.isnan(out[0]) and not torch.isnan(out[1:]).any()


def test_ids_on_another_device_do_not_decide_the_output_device():
    """ids are allowed to arrive elsewhere; the LOGITS decide. On CPU this only pins the
    contract -- the cuda test below is what makes it non-vacuous."""
    logits, ids = _case("cpu")
    out = token_aligned_logprobs(logits, ids.cpu())
    assert out.device == logits.device


def test_values_are_unchanged_by_the_device_fix():
    """The fix must be a pure placement change: same numbers, to the bit."""
    logits, ids = _case("cpu", n=32, v=128)
    out = token_aligned_logprobs(logits, ids)
    lsm = torch.log_softmax(logits.float(), dim=-1)
    expect = torch.full((32,), float("nan"))
    expect[1:] = lsm[:-1].gather(1, ids[1:, None].long()).squeeze(1)
    assert torch.equal(out[1:], expect[1:])


def test_gradient_flows_back_to_the_logits():
    logits, ids = _case("cpu")
    logits.requires_grad_(True)
    token_aligned_logprobs(logits, ids)[1:].sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


@pytest.mark.slow
def test_cuda_output_stays_on_cuda_and_the_loss_follows():
    """The test that would have caught the bug: the returned tensor -- and therefore the
    device opd_train_step picks for the loss -- must be cuda, not cpu.

    require_gpu() rather than skipif: on a node whose driver is too old, cuda.is_available()
    is False and a skipif would report this gate as 'skipped' on the very hardware it was
    queued to run on."""
    from tests.gpu import require_gpu
    require_gpu()

    from opd.loss import compute_opd_loss

    logits, ids = _case("cuda", n=64, v=256)
    student_logp = token_aligned_logprobs(logits, ids)
    assert student_logp.is_cuda, "logprobs fell back to CPU -> the loss would run on CPU"

    # mirror opd_train_step: the loss device is DERIVED from this tensor
    dev = student_logp.device
    teacher_logp = torch.randn(64).to(dev)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[8:40] = True
    loss, adv = compute_opd_loss(teacher_logp, student_logp, mask.to(dev), 10.0)
    assert loss.is_cuda and adv.is_cuda
