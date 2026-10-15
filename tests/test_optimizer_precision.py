"""BEHAVIOURAL GATE: after N steps at the real lr, pi_theta must actually have left pi_ref.

This is the precision version of the STEP-3c behaviour gate, and it exists because the
failure it catches produces no error, no warning and no anomaly in any curve -- it produces
a FLAT LINE, which is indistinguishable from "offline is stable" and would have been read
as the experiment's headline result.

THE MECHANISM. AdamW's update is lr-sized: the normalised term m/(sqrt(v)+eps) is O(1) by
construction, so a weight moves by ~lr = 2e-6 per step. bfloat16 keeps 8 mantissa bits, so
near |w| = 0.02 the nearest representable neighbours are 7.8e-5 apart -- 39x further than
one step travels. Round-to-nearest returns the SAME number, every step, forever. It does not
accumulate: 150 steps of "add 2e-6" to a bf16 weight is 150 no-ops.

Lightning OPD's lr 2e-6 is not wrong. It assumes slime/Megatron, which keeps fp32 master
weights. Copying the learning rate without the optimizer precision that makes it work is the
same shape as the loss-normalisation question asked earlier -- a paper hyperparameter
carrying an implicit implementation context -- except that investigation ended in "we match
the reference" and this one ended in "we did not, and it silently disabled training".

The gate is a DIFFERENTIAL one on purpose: bf16 must FAIL the same assertion fp32 passes.
An absolute threshold alone could be met by a lucky scale; showing the discriminating power
directly is what makes it a gate rather than a smoke test.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REAL_LR = 2.0e-6           # configs/opd_qwen3.yaml optim.lr (Lightning OPD Table 6)
STEPS = 150                # configs/opd_qwen3.yaml optim.steps


def _train(param_dtype, lr=REAL_LR, steps=STEPS, n=4096, seed=0):
    """A slab of weights at realistic transformer scale, stepped by the REAL optimizer.

    -> (mean |dw|, fraction of weights that moved at all)
    """
    g = torch.Generator().manual_seed(seed)
    # std ~ 1/sqrt(hidden) with hidden=2560, i.e. the scale of a real Qwen3-4B linear weight
    w = (torch.randn(n, generator=g) * 0.02).to(param_dtype).requires_grad_(True)
    w0 = w.detach().float().clone()
    opt = torch.optim.AdamW([w], lr=lr, betas=(0.9, 0.98), weight_decay=0.1)
    base = torch.randn(n, generator=g) * 1e-3
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        w.grad = (base + torch.randn(n, generator=g) * 1e-4).to(param_dtype)
        opt.step()
    d = w.detach().float() - w0
    return d.abs().mean().item(), (d != 0).float().mean().item()


def test_fp32_parameters_actually_move_at_the_real_learning_rate():
    """THE GATE. Every weight must move, by an amount the lr can explain."""
    moved, frac = _train(torch.float32)
    assert frac == 1.0, f"only {frac:.1%} of fp32 weights moved in {STEPS} steps"
    assert moved > 1e-5, f"fp32 movement {moved:.2e} is too small to be a real update"


def test_bf16_parameters_FAIL_the_same_gate():
    """The discriminating half. If this ever starts passing, the gate above has stopped
    proving anything and the whole precision argument needs re-deriving."""
    moved, frac = _train(torch.bfloat16)
    assert frac < 0.2, (
        f"bf16 moved {frac:.1%} of weights -- this gate no longer discriminates, so "
        "test_fp32_parameters_actually_move is no longer evidence of anything")
    fp32_moved, _ = _train(torch.float32)
    assert moved < fp32_moved / 5, (
        f"bf16 update {moved:.2e} is not much smaller than fp32's {fp32_moved:.2e}")


def test_the_gap_closes_at_a_larger_learning_rate():
    """Attribution check: the failure must be lr-vs-spacing, NOT 'bf16 is broken'. At
    lr 1e-4 the step clears bf16's spacing and the two agree -- which is what pins the cause
    and rules out raising the lr as an alternative fix only on the grounds we chose, not on
    the grounds that it would not work."""
    big_bf16, frac_bf16 = _train(torch.bfloat16, lr=1e-4)
    big_fp32, _ = _train(torch.float32, lr=1e-4)
    assert frac_bf16 > 0.8, f"at lr 1e-4 bf16 should move freely, got {frac_bf16:.1%}"
    assert abs(big_bf16 - big_fp32) / big_fp32 < 0.3


def test_gradient_accumulation_needs_fp32_too():
    """The same rounding, one level down. global_batch=256 means each micro-step contributes
    1/256 = 3.9e-3 of the accumulator, and bf16's relative spacing is 2^-8 = 3.9e-3 -- the
    identical number. fp32 parameters give fp32 grads for free, which is why this is not a
    separate mitigation, but it must be asserted or a future 'save memory with bf16 grads'
    would reintroduce it."""
    n, micro = 4096, 256
    g = torch.Generator().manual_seed(1)
    parts = [torch.randn(n, generator=g, dtype=torch.float64) * 1e-3 for _ in range(micro)]
    exact = torch.stack(parts).sum(0) / micro

    def accumulate(dtype):
        acc = torch.zeros(n, dtype=dtype)
        for part in parts:
            acc += (part / micro).to(dtype)        # what .backward() does into p.grad
        return acc.double()

    err = {dt: ((accumulate(dt) - exact).norm() / exact.norm()).item()
           for dt in (torch.bfloat16, torch.float32)}
    assert err[torch.float32] < 1e-5, f"fp32 accumulation error {err[torch.float32]:.2e}"
    assert err[torch.bfloat16] > 1e-3, (
        "bf16 accumulation is no longer lossy at global_batch=256 -- re-derive before "
        "relying on this")


@pytest.mark.parametrize("field,expected", [("student_param_dtype", "float32"),
                                            ("student_autocast_dtype", "bfloat16")])
def test_the_real_config_actually_asks_for_this(field, expected):
    """The arithmetic above is worthless if the config still says bfloat16. Read the real
    file -- this is the link between the finding and the run."""
    import yaml

    cfg = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1]
                          / "configs" / "opd_qwen3.yaml").read_text(encoding="utf-8"))
    assert cfg["model"][field] == expected
    assert cfg["optim"]["lr"] == REAL_LR, "lr changed -- re-derive the spacing argument"


def test_student_param_dtype_is_independent_of_the_teacher_fingerprint():
    """model.dtype feeds BackendConfig and therefore gate 1. If the student's precision were
    taken from it, fixing the optimizer would have invalidated the precompute cache."""
    import yaml

    cfg = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1]
                          / "configs" / "opd_qwen3.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["dtype"] == "bfloat16", "teacher backend dtype must not have moved"
    assert cfg["model"]["student_param_dtype"] != cfg["model"]["dtype"]


def test_autocast_keeps_the_forward_in_bf16_while_params_stay_fp32():
    """The split has to be real: fp32 master weights would cost their memory twice over if
    the forward also ran in fp32."""
    from teacher.scorer import MockLM, Scorer

    lm = MockLM(vocab_size=64, dim=16, seed=0).to(torch.float32)
    s = Scorer(lm, requires_grad=True, autocast_dtype="bfloat16")
    assert all(p.dtype == torch.float32 for p in s.lm.parameters())
    ids = torch.randint(0, 64, (12,), generator=torch.Generator().manual_seed(2))
    with s._autocast(torch.device("cpu")):
        logits = s.lm(ids)
    assert logits.dtype == torch.bfloat16, "autocast did not narrow the forward"
    assert s.logprobs(ids).dtype == torch.float32, "logprobs must come back in fp32"
