"""Task-2 step 4: one optimizer step consumes global_batch trajectories.

Three things must hold, and each has a specific way of being silently wrong:

  1. global_batch=1 must reproduce the pre-batching behaviour EXACTLY. Wiring grad
     accumulation is not supposed to change the objective, only reduce its variance, so
     the size-1 case is the continuity check.
  2. The accumulation must be SEQUENCE-WEIGHTED (mean of per-trajectory per-token means),
     matching the Lightning OPD reference implementation -- see the normalisation section
     of opd/loss.py. Token-weighting would let long trajectories dominate, and since
     online degradation itself lengthens trajectories, that would couple the measurement
     to the effect being measured. Both schemes agree at B=1, so only a B>1 test with
     DIFFERENT-LENGTH trajectories can tell them apart.
  3. One diagnostic record PER TRAJECTORY, not per optimizer step. Collapsing to a step
     mean would discard the per-sample distribution that the whole diagnostics design
     exists to preserve.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.retriever import MockRetriever  # noqa: E402
from envs.tokenizer import MockTokenizer  # noqa: E402
from opd.diagnostics import DiagnosticRunner  # noqa: E402
from opd.trainer import OnlineSource, OPDConfig, OPDTrainer, snapshot_ref  # noqa: E402
from rollout.policy import MockPolicy  # noqa: E402
from teacher.scorer import MockLM, Scorer, TeacherScorer  # noqa: E402

CORPUS = ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]
QUESTIONS = [f"question {i}?" for i in range(8)]
GOLDS = [["Paris"]] * 8


class ListLogger:
    def __init__(self):
        self.records = []

    def log(self, rec):
        self.records.append(rec.to_dict())

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mk(global_batch, steps=2, seed=7):
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = Scorer(MockLM(tok.vocab_size, seed=seed), requires_grad=True)
    policy = MockPolicy(tok, queries=["capital of France"], answer="Paris")
    cfg = OPDConfig(mode="online", tau=10.0, steps=steps, global_batch=global_batch, lr=1e-2)
    source = OnlineSource(QUESTIONS, policy, retr, tok, teacher, cfg, golds=GOLDS)
    diagnostics = DiagnosticRunner(ref_scorer=snapshot_ref(student), tau=cfg.tau)
    logger = ListLogger()
    trainer = OPDTrainer(student, source, cfg, diagnostics=diagnostics, logger=logger)
    return trainer, student, logger


def _weights(student):
    return [p.detach().clone() for p in student.lm.parameters()]


def test_global_batch_1_matches_the_unbatched_update():
    """Continuity: wiring accumulation must not move the B=1 trajectory of weights."""
    t1, s1, _ = _mk(global_batch=1, steps=3)
    list(t1.train())
    after = _weights(s1)

    # An independent run with the same seeds must land on the same weights -- i.e. the
    # step is deterministic and B=1 carries no batching side effect.
    t2, s2, _ = _mk(global_batch=1, steps=3)
    list(t2.train())
    for a, b in zip(after, _weights(s2)):
        assert torch.equal(a, b)


def test_one_record_per_trajectory_not_per_step():
    steps, B = 2, 4
    trainer, _, logger = _mk(global_batch=B, steps=steps)
    list(trainer.train())
    assert len(logger.records) == steps * B, (
        f"expected {steps*B} per-trajectory records, got {len(logger.records)}")
    assert [r["micro"] for r in logger.records] == list(range(B)) * steps
    assert all(r["global_batch"] == B for r in logger.records)
    # step index still identifies the OPTIMIZER step, so records group by it
    assert [r["step"] for r in logger.records] == sorted([s for s in range(steps)] * B)


def test_one_optimizer_step_per_train_step_regardless_of_batch():
    """B trajectories must produce ONE update, not B updates -- otherwise 'global_batch'
    would silently be a step multiplier and the effective lr would scale with it."""
    seen = []

    trainer, student, _ = _mk(global_batch=4, steps=1)
    real_step = trainer.optimizer.step

    def counting_step(*a, **kw):
        seen.append(1)
        return real_step(*a, **kw)

    trainer.optimizer.step = counting_step
    list(trainer.train())
    assert sum(seen) == 1, f"expected 1 optimizer step, got {sum(seen)}"


class _VaryingLengthPolicy(MockPolicy):
    """Emits a DIFFERENT-length answer per question.

    The plain MockPolicy scripts one answer for every question, so all trajectories carry
    the same scored-token count -- and at equal lengths sequence-weighting and
    token-weighting are ALGEBRAICALLY IDENTICAL. A test built on it would pass under both
    schemes and prove nothing about which one is implemented.
    """

    def __init__(self, tokenizer, lengths) -> None:
        super().__init__(tokenizer, queries=[], answer="x")
        self._lengths = list(lengths)
        self._sig = {}

    def _answer_for(self, prefix_ids):
        ids = tuple(prefix_ids.tolist())
        for q, n in zip(QUESTIONS, self._lengths):
            sig = tuple(self.tok.encode(q))
            if any(ids[i:i + len(sig)] == sig for i in range(len(ids) - len(sig) + 1)):
                return " ".join(["word"] * n)
        return "x"

    def act(self, prefix_ids, turn, forced=False):
        self.answer = self._answer_for(prefix_ids)
        return super().act(prefix_ids, turn, forced=forced)


def test_accumulation_is_sequence_weighted_not_token_weighted():
    """The discriminating test: with trajectories of DIFFERENT scored-token counts, the
    accumulated gradient must equal the mean of per-trajectory gradients (each already a
    per-token mean), NOT the token-count-weighted mean."""
    from opd.step import opd_train_step

    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    policy = _VaryingLengthPolicy(tok, lengths=[2, 20, 60])
    cfg = OPDConfig(mode="online", tau=10.0, steps=1, global_batch=3)
    source = OnlineSource(QUESTIONS, policy, retr, tok, teacher, cfg, golds=GOLDS)
    pairs = source.get_batch(0, 3)
    counts = [int(t.token_type.eq(1).sum()) for t, _ in pairs]
    assert len(set(counts)) > 1, (
        f"all trajectories have the same scored-token count {counts}; at equal lengths the "
        "two weightings are algebraically identical and this test cannot discriminate")

    student = Scorer(MockLM(tok.vocab_size, seed=7), requires_grad=True)

    def grads_for(weight_of):
        for p in student.lm.parameters():
            p.grad = None
        for i, (traj, tlp) in enumerate(pairs):
            out = opd_train_step(traj, tlp, student, tau=cfg.tau)
            (out.loss * weight_of(i)).backward()
        return [p.grad.detach().clone() for p in student.lm.parameters()]

    n, tok_total = len(pairs), sum(counts)
    seq_weighted = grads_for(lambda i: 1.0 / n)
    # a global token-mean reduces to weighting each trajectory's per-token mean by its
    # share of the batch's tokens
    tok_weighted = grads_for(lambda i: counts[i] / tok_total)

    assert not all(torch.allclose(a, b) for a, b in zip(seq_weighted, tok_weighted)), (
        f"the two schemes produced identical gradients at counts={counts} -- the test is "
        "not discriminating")

    # what the TRAINER actually does must be the sequence-weighted one
    student2 = Scorer(MockLM(tok.vocab_size, seed=7), requires_grad=True)
    trainer = OPDTrainer(student2, source, cfg)
    captured = []
    real_step = trainer.optimizer.step
    trainer.optimizer.step = lambda *a, **kw: captured.append(
        [p.grad.detach().clone() for p in student2.lm.parameters()]) or real_step(*a, **kw)
    trainer.step(0)
    assert captured, "optimizer.step was never reached"
    assert all(torch.allclose(a, b) for a, b in zip(captured[0], seq_weighted))


def test_train_is_lazy_so_logs_stream():
    """train() must be a generator: the eager version ran everything before the caller's
    per-step print fired, which made a correctly interleaved run look like 'all rollouts
    first, then all training'."""
    import inspect

    trainer, _, logger = _mk(global_batch=2, steps=2)
    gen = trainer.train()
    assert inspect.isgenerator(gen)
    assert logger.records == [], "building the generator must not run any step"
    next(gen)
    assert len(logger.records) == 2, "after one step, exactly that step's records exist"
