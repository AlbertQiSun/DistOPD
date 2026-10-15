"""Gate on activation checkpointing: it must ENGAGE, and it must not change the numbers.

Both halves are needed, and the first is the one that would otherwise rot silently. HF
layers check `self.gradient_checkpointing and self.training`, and `from_pretrained` returns
a model in EVAL mode -- so calling `gradient_checkpointing_enable()` and stopping there is a
no-op with no warning, no error, and no visible difference except an OOM twenty minutes into
a job. A test that only compared losses would pass just as happily with the feature off.

So the engagement test counts LAYER RECOMPUTATION during backward, which is the observable
that distinguishes "on" from "quietly off".

Everything runs on a tiny randomly-initialised Qwen3 on CPU -- same architecture family as
the real 4B student (same layer class, same checkpointing machinery), small enough that the
whole gate is CPU-complete and does not wait for a GPU allocation.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opd.loss import compute_opd_loss
from opd.masking import token_aligned_logprobs
from teacher.scorer import assert_no_active_dropout, enable_grad_checkpointing

transformers = pytest.importorskip("transformers")


def _tiny(dropout=0.0, layers=3):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      attention_dropout=dropout)
    model = Qwen3ForCausalLM(cfg)
    # EVAL, because that is what from_pretrained returns and therefore what the real student
    # is in when enable_grad_checkpointing gets it. A freshly CONSTRUCTED nn.Module defaults
    # to train(), and leaving it there made this whole file pass with the mode switch deleted
    # -- the fixture was supplying the very condition under test.
    return model.eval()


def _run(model, ids, count_layers=True):
    """-> (forward layer calls, recomputed layer calls, logp, grad-norm)."""
    calls = {"n": 0}
    if count_layers:
        for layer in model.model.layers:
            orig = layer.forward

            def wrapped(*a, _o=orig, **k):
                calls["n"] += 1
                return _o(*a, **k)

            layer.forward = wrapped
    logits = model(ids.unsqueeze(0)).logits[0].float()
    logp = token_aligned_logprobs(logits, ids)
    fwd = calls["n"]
    logp[1:].sum().backward()
    gn = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    return fwd, calls["n"] - fwd, logp.detach(), gn


IDS = torch.randint(0, 64, (24,), generator=torch.Generator().manual_seed(3))


def test_checkpointing_actually_engages():
    """NON-VACUITY. Layers must run a SECOND time during backward; that recomputation IS the
    memory saving. Without it the feature is off and nothing else here would notice."""
    model = _tiny()
    enable_grad_checkpointing(model)
    fwd, recomputed, _, _ = _run(model, IDS)
    assert fwd == 3, f"expected one forward per layer, got {fwd}"
    assert recomputed == fwd, (
        f"layers were NOT recomputed in backward ({recomputed} of {fwd}) -- checkpointing "
        "is enabled but not engaged, so activations are still being retained")


def test_eval_mode_is_the_silent_noop_this_guards_against():
    """Pins the trap itself. If HF ever changes so that eval-mode checkpointing DOES engage,
    this goes red and enable_grad_checkpointing's train() can be revisited -- rather than the
    knowledge living only in a comment."""
    model = _tiny()
    enable_grad_checkpointing(model)
    model.eval()                                   # undo the mode enable_grad_checkpointing set
    _, recomputed, _, _ = _run(model, IDS)
    assert recomputed == 0, (
        "eval-mode checkpointing recomputed after all -- the documented reason "
        "enable_grad_checkpointing forces train() no longer holds")


def test_logprobs_and_gradients_match_the_uncheckpointed_run():
    """THE GATE. Recomputation replays the same ops on the same inputs, so the numbers must
    not move. Asserted on the per-token logprobs AND on the gradient -- a checkpointed
    backward that dropped a block would leave the forward perfectly intact."""
    plain = _tiny()
    plain.train()
    ckpt = _tiny()
    enable_grad_checkpointing(ckpt)
    assert all(torch.equal(a, b) for a, b in
               zip(plain.state_dict().values(), ckpt.state_dict().values())), "same init"

    _, rec_plain, lp_plain, gn_plain = _run(plain, IDS)
    _, rec_ckpt, lp_ckpt, gn_ckpt = _run(ckpt, IDS)

    assert rec_plain == 0 and rec_ckpt > 0, "the two arms must differ in checkpointing only"
    assert gn_plain > 0, "reference gradient is zero -- the comparison would be vacuous"
    assert torch.allclose(lp_plain[1:], lp_ckpt[1:], atol=1e-6, rtol=0), "logprobs moved"
    assert torch.allclose(gn_plain, gn_ckpt, rtol=1e-5), (
        f"gradient norm moved: {gn_plain.item()} vs {gn_ckpt.item()}")
    for (n, a), (_, b) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-4), f"gradient moved at {n}"


def test_the_opd_loss_is_unchanged():
    """End of the chain: what the trainer optimises."""
    teacher = torch.randn(24, generator=torch.Generator().manual_seed(11))
    mask = torch.zeros(24, dtype=torch.bool)
    mask[4:20] = True

    out = []
    for use_ckpt in (False, True):
        m = _tiny()
        if use_ckpt:
            enable_grad_checkpointing(m)
        else:
            m.train()
        logits = m(IDS.unsqueeze(0)).logits[0].float()
        loss, _ = compute_opd_loss(teacher, token_aligned_logprobs(logits, IDS), mask, 10.0)
        out.append(loss)
    assert torch.allclose(out[0], out[1], atol=1e-6, rtol=0), (
        f"checkpointing changed the loss: {out[0].item()} vs {out[1].item()}")


def test_train_mode_forward_is_deterministic():
    """train() is what checkpointing needs and also what would switch dropout on. Two
    identical passes must give identical numbers, or the online/offline comparison is
    measuring randomness."""
    m = _tiny()
    enable_grad_checkpointing(m)
    with torch.no_grad():
        a = m(IDS.unsqueeze(0)).logits
        b = m(IDS.unsqueeze(0)).logits
    assert torch.equal(a, b), "train-mode forward is not reproducible"


def test_dropout_guard_refuses_a_checkpoint_that_would_sample():
    """The guard has to bind, or it is decoration. A config with real dropout must be
    refused BEFORE the mode switch, not diagnosed afterwards from noisy losses."""
    model = _tiny(dropout=0.1)
    with pytest.raises(AssertionError, match="ACTIVE DROPOUT"):
        assert_no_active_dropout(model)

    assert not model.training, "_tiny must hand back eval mode, like from_pretrained"
    with pytest.raises(AssertionError, match="ACTIVE DROPOUT"):
        enable_grad_checkpointing(model)
    assert not model.training, (
        "the refusal came AFTER the mode switch: the model is sitting in train() with live "
        "dropout, which is the state the guard exists to prevent")


def test_dropout_guard_passes_the_real_configuration():
    assert_no_active_dropout(_tiny(dropout=0.0))         # Qwen3's shipped setting


def test_use_cache_is_turned_off():
    """A KV cache is meaningless under recomputation and warns loudly every step."""
    m = _tiny()
    enable_grad_checkpointing(m)
    assert m.config.use_cache is False
