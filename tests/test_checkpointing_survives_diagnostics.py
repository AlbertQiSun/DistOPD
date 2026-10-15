"""Gradient checkpointing must still be engaged AFTER a diagnostic pass has run.

THE BUG THIS EXISTS FOR. HF checkpointing engages only when `self.training` is True. The
diagnostic path (raw_logits / full_logprobs / log_dist_at) puts the model in eval() on every
call, and HFScorer.logprobs -- which overrides the base Scorer.logprobs -- did not set the
mode back. So the sequence the trainer actually runs,

    step 0: train -> diagnostics(eval) -> step 1: train -> ...

trained in EVAL mode from step 1 onward, retaining every activation instead of recomputing
them. Measured on the real 4B: 41.9 GiB of autograd graph for one 5000-token trajectory
against ~2.5 GiB with checkpointing live. It capped the trainable trajectory length at 5458
tokens while the sampler was configured to produce up to 16384, and it would have OOM'd the
real run one step in.

WHY THE EXISTING GATE MISSED IT. tests/test_grad_checkpointing.py calls
enable_grad_checkpointing() on a fresh model and measures recomputation immediately. It never
goes through HFScorer.logprobs, and it never runs a diagnostic first -- so it verifies the
function does its job and says nothing about whether anything UNDOES it. The trap was written
down in that file's docstring and the test still could not see this.

So this file tests the ORDER, on the surface the trainer uses.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opd.diagnostics import DiagnosticRunner
from opd.masking import build_score_mask
from opd.step import opd_train_step
from opd.trajectory import TokenType, Trajectory
from teacher.scorer import HFScorer, enable_grad_checkpointing

transformers = pytest.importorskip("transformers")


def _tiny_ckpt(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512))
    d = tmp_path / "ckpt"
    m.save_pretrained(d)
    return str(d)


def _traj(n=64, v=128, n_scored=20):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, v, (n,), generator=g)
    tt = torch.full((n,), int(TokenType.PROMPT), dtype=torch.long)
    tt[10:10 + n_scored] = int(TokenType.ACTION)
    return Trajectory(ids, tt)


def _count_recomputation(student, traj):
    """-> (forward layer calls, calls added during backward). Recomputation is the observable
    that distinguishes checkpointing 'on' from 'quietly off'."""
    calls = {"n": 0}
    originals = []
    for layer in student.lm.model.layers:
        originals.append((layer, layer.forward))
        orig = layer.forward

        def wrapped(*a, _o=orig, **k):
            calls["n"] += 1
            return _o(*a, **k)

        layer.forward = wrapped
    try:
        tlp = torch.randn(traj.input_ids.shape[0])
        out = opd_train_step(traj, tlp, student, tau=10.0)
        fwd = calls["n"]
        out.loss.backward()
        return fwd, calls["n"] - fwd, out
    finally:
        for layer, f in originals:
            layer.forward = f


def _student(tmp_path):
    return HFScorer.from_pretrained(_tiny_ckpt(tmp_path), dtype="float32",
                                    requires_grad=True, device="cpu")


def test_checkpointing_is_engaged_on_the_first_step(tmp_path):
    """Baseline: without this the ordering test below could not attribute a failure."""
    student = _student(tmp_path)
    fwd, recomputed, _ = _count_recomputation(student, _traj())
    assert fwd == 3
    assert recomputed == fwd, "checkpointing was never engaged even on step 0"


def test_checkpointing_SURVIVES_a_diagnostic_pass(tmp_path):
    """THE GATE. This is the trainer's real order, and the one that was broken."""
    student = _student(tmp_path)
    traj = _traj()

    # step 0, exactly as OPDTrainer.step runs it
    _, rec0, out = _count_recomputation(student, traj)
    assert rec0 == 3, "precondition: checkpointing must be on at step 0"

    # ...then the diagnostic pass, which puts the model in eval()
    ref = HFScorer.from_pretrained(_tiny_ckpt(tmp_path), dtype="float32",
                                   requires_grad=False, device="cpu")
    DiagnosticRunner(ref_scorer=ref, tau=10.0, embedder=None).run(
        0, "offline", traj, out, student, staleness=0, resync_every=1)

    # ...and step 1 must still recompute
    _, rec1, _ = _count_recomputation(student, traj)
    assert rec1 == 3, (
        f"after one diagnostic pass the student recomputed {rec1} of 3 layers -- it is "
        "training in eval mode, so every activation is being retained. On the real 4B that "
        "is 41.9 GiB of graph for a 5000-token trajectory instead of ~2.5 GiB.")


def test_the_model_is_left_in_train_mode_by_logprobs(tmp_path):
    """The mechanism, asserted directly: a scorer that carries grad must put itself in
    train() before the forward, no matter what left it in eval."""
    student = _student(tmp_path)
    student.lm.eval()
    student.logprobs(_traj().input_ids)
    assert student.lm.training, "logprobs did not restore train mode"


def test_a_grad_free_scorer_stays_in_eval(tmp_path):
    """The other half: the teacher and pi_ref must NEVER be flipped into train mode, which is
    where dropout would sample and where checkpointing would waste a recompute."""
    ref = HFScorer.from_pretrained(_tiny_ckpt(tmp_path), dtype="float32",
                                   requires_grad=False, device="cpu")
    ref.lm.train()
    ref.logprobs(_traj().input_ids)
    assert not ref.lm.training, "a grad-free scorer was left in train mode"


def test_from_pretrained_does_not_checkpoint_a_grad_free_model(tmp_path):
    """Recomputation under no_grad is pure waste."""
    ref = HFScorer.from_pretrained(_tiny_ckpt(tmp_path), dtype="float32",
                                   requires_grad=False, device="cpu")
    assert not getattr(ref.lm, "is_gradient_checkpointing", False)
