"""Confound #1, last variant: is "online" actually on-policy? (local, zero-GPU)

vLLM holds its OWN copy of the weights; the trainer updates the training-side copy.
If they are not resynced, the engine keeps sampling from stale weights and the online
arm silently degrades into a frozen (offline-like) run -- and a resync BUG produces an
"online worse than offline" curve indistinguishable from real poisoning. The mock has
no separate inference engine, so M2-M6 could not exercise this. Here a MockInferenceEngine
stands in for vLLM (a real weight COPY) so the resync semantics are proven without a GPU:

  1. after resync, the engine weights are bit-for-bit the training weights (and it is a
     SNAPSHOT, not a live alias -- mutating the student diverges them);
  2. staleness is exactly step - last_sync_step, so a rollout at step k is provably drawn
     from step-k weights under resync_every=1, and provably STALE on skipped steps;
  3. resync_every is an explicit, logged config -- "how online is online" is a reported,
     ablatable number, and offline is honestly recorded as maximally stale (staleness==step).

Real vLLM weight hot-update (does it land, latency) is the HPC half -- see
rollout/vllm_policy.py::sync_from.
"""
from __future__ import annotations

import tempfile

import torch

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.diagnostics import DiagnosticRunner
from opd.step import opd_train_step
from opd.trainer import OfflineSource, OnlineSource, OPDConfig, OPDTrainer, snapshot_ref
from rollout.policy import MockPolicy
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
]
QUESTIONS = ["What is the capital of France?"]
GOLDS = [["Paris"]]


class MockInferenceEngine:
    """Stand-in for the vLLM engine: holds its OWN copy of the student's weights."""

    def __init__(self) -> None:
        self.state = None

    def load_weights(self, student) -> None:
        self.state = {k: v.clone() for k, v in student.lm.state_dict().items()}


class MockResyncPolicy(MockPolicy):
    """Scripted policy whose sync_from pushes weights into a mock inference engine."""

    def __init__(self, engine, tokenizer, queries, answer) -> None:
        super().__init__(tokenizer, queries, answer)
        self.engine = engine

    def sync_from(self, student) -> None:
        self.engine.load_weights(student)


class ListLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, rec) -> None:
        self.records.append(rec.to_dict())

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _state_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def _student(seed=7):
    return Scorer(MockLM(MockTokenizer(vocab_size=8192).vocab_size, seed=seed),
                  requires_grad=True)


# --- assertion 1: resync => bit-equal, and it is a SNAPSHOT not a live alias --------
def test_resync_makes_engine_bit_equal_then_snapshot_diverges():
    student = _student()
    engine = MockInferenceEngine()
    engine.load_weights(student)
    assert _state_equal(engine.state, student.lm.state_dict())

    with torch.no_grad():                       # training weights move on...
        list(student.lm.parameters())[0].add_(1.0)
    assert not _state_equal(engine.state, student.lm.state_dict())   # ...engine did NOT (snapshot)

    engine.load_weights(student)                # resync closes the gap
    assert _state_equal(engine.state, student.lm.state_dict())


# --- assertion 2 (strong): the sampler holds step-k weights under resync_every=1,
#     and is provably STALE on skipped steps (a resync bug would be caught here) -----
def test_minimal_online_loop_sampler_uses_current_weights():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))

    for resync_every, expected in [(1, [True, True, True]), (2, [True, False, True])]:
        student = _student(seed=7)
        engine = MockInferenceEngine()
        policy = MockResyncPolicy(engine, tok, queries=["capital of France"], answer="Paris")
        cfg = OPDConfig(mode="online", tau=10.0, resync_every=resync_every)
        source = OnlineSource(QUESTIONS, policy, retr, tok, teacher, cfg, golds=GOLDS)
        opt = torch.optim.SGD(student.lm.parameters(), lr=1.0)   # big lr => weights surely move

        sampler_is_current = []
        for step in range(3):
            if step % cfg.resync_every == 0:
                source.resync(student, step)
            # does the engine that SAMPLES hold the current training weights?
            sampler_is_current.append(_state_equal(engine.state, student.lm.state_dict()))
            traj, tlp = source.get(step)
            out = opd_train_step(traj, tlp, student, tau=cfg.tau)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
        assert sampler_is_current == expected


# --- assertion 3: staleness is logged per record and driven by resync_every ---------
def _run_online(resync_every, steps):
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = _student(seed=7)
    policy = MockPolicy(tok, queries=["capital of France"], answer="Paris")
    cfg = OPDConfig(mode="online", tau=10.0, steps=steps, resync_every=resync_every)
    source = OnlineSource(QUESTIONS, policy, retr, tok, teacher, cfg, golds=GOLDS)
    diagnostics = DiagnosticRunner(ref_scorer=snapshot_ref(student), tau=cfg.tau)
    logger = ListLogger()
    # train() is a GENERATOR now (so job logs stream per step); a bare call would
    # build it and run nothing. Consume it.
    list(OPDTrainer(student, source, cfg, diagnostics=diagnostics, logger=logger).train())
    return logger.records


def test_trainer_logs_staleness_by_resync_cadence():
    r1 = _run_online(resync_every=1, steps=4)
    assert [x["policy"]["staleness"] for x in r1] == [0, 0, 0, 0]         # fully on-policy
    assert all(x["policy"]["on_policy"] for x in r1)

    r2 = _run_online(resync_every=2, steps=4)
    assert [x["policy"]["staleness"] for x in r2] == [0, 1, 0, 1]         # stale on odd steps
    assert [x["policy"]["on_policy"] for x in r2] == [True, False, True, False]
    assert all(x["policy"]["resync_every"] == 2 for x in r2)


def test_offline_is_recorded_as_maximally_stale():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = _student(seed=7)
    policy = MockPolicy(tok, queries=["capital of France"], answer="Paris")
    with tempfile.TemporaryDirectory() as d:
        cfg = OPDConfig(mode="offline", tau=10.0, steps=3, precompute_dir=d)
        source = OfflineSource(QUESTIONS, policy, retr, tok, teacher, cfg,
                               expected_backend=teacher.config, golds=GOLDS)
        diagnostics = DiagnosticRunner(ref_scorer=snapshot_ref(student), tau=cfg.tau)
        logger = ListLogger()
        list(OPDTrainer(student, source, cfg, diagnostics=diagnostics, logger=logger).train())
    # frozen rollouts: never resynced => staleness == step, on_policy only at step 0
    assert [x["policy"]["staleness"] for x in logger.records] == [0, 1, 2]
    assert [x["policy"]["on_policy"] for x in logger.records] == [True, False, False]
