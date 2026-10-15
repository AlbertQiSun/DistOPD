"""Milestone-4 end-to-end: the unified trainer runs N steps in BOTH modes.

Not a convergence test -- the point is that one loop drives online and offline
through the identical opd_train_step + optimizer update, a config flag being the
only switch. We check the loop actually steps (len==steps, losses finite) and
that the optimizer touches the student params (proves grad flows through the
shared path). Offline additionally exercises the precompute->disk->readback source.
"""
from __future__ import annotations

import copy
import tempfile

import torch

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.trainer import OPDConfig, OPDTrainer, build_source
from rollout.policy import MockPolicy
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]
QUESTIONS = ["What is the capital of France?", "How tall is the Eiffel Tower?"]


def _components():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = Scorer(MockLM(tok.vocab_size, seed=5678), requires_grad=True)
    return tok, retr, policy, teacher, student


def _run(mode: str, precompute_dir=None):
    tok, retr, policy, teacher, student = _components()
    cfg = OPDConfig(mode=mode, steps=3, lr=1e-2, precompute_dir=precompute_dir)
    before = copy.deepcopy(student.lm.state_dict())
    source = build_source(cfg, QUESTIONS, policy, retr, tok, teacher)
    trainer = OPDTrainer(student, source, cfg)
    outs = list(trainer.train())
    after = student.lm.state_dict()
    return cfg, outs, before, after


def _assert_stepped(cfg, outs, before, after):
    assert len(outs) == cfg.steps
    for o in outs:
        assert torch.isfinite(o.loss), "loss went non-finite"
        assert int(o.score_mask.sum()) > 0, "no scored tokens -- nothing to learn from"
    # optimizer actually moved at least one student parameter => grad flowed
    moved = any(not torch.equal(before[k], after[k]) for k in before)
    assert moved, "optimizer did not update student params -- grad did not flow"


def test_online_trainer_steps():
    cfg, outs, before, after = _run("online")
    _assert_stepped(cfg, outs, before, after)


def test_offline_trainer_steps():
    with tempfile.TemporaryDirectory() as d:
        cfg, outs, before, after = _run("offline", precompute_dir=d)
    _assert_stepped(cfg, outs, before, after)
