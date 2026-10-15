"""The memory decisions of commit 4, gated as BEHAVIOUR rather than as VRAM numbers.

None of these can be measured as free bytes on a CPU box, and none of them needs to be. Each
is a placement or lifetime rule with an observable consequence:

  * the student is loaded in the dtype the CONFIG asks for -- not the teacher's
  * pi_ref sits on the host and is paged in only for the diagnostic pass
  * the gradient buffer is released before that pass, not carried through it

They are here because reverse-verification found all three ungated: undoing each one left
the whole suite green. A memory decision with no test is a memory decision that comes back
as an OOM in hour nine of a twenty-four hour job.

The first is the same shape as the `global_batch` field that sat in the config unread: the
config saying the right thing proves nothing about what the code loads.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.diagnostics import DiagnosticRunner
from opd.trainer import OnlineSource, OPDConfig, OPDTrainer, snapshot_ref
from rollout.policy import MockPolicy
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = ["Paris is the capital of France."]


# ---------------------------------------------------------------------------
# 1. does make_student actually REQUEST what the config specifies?
# ---------------------------------------------------------------------------
def test_make_student_requests_the_config_student_dtype_not_the_teacher_dtype(monkeypatch):
    """The config carries student_param_dtype: float32 and dtype: bfloat16 (the teacher's,
    which feeds gate 1's fingerprint). Taking the student's precision from the latter is a
    one-word edit that silently restores the no-op trainer, and no config test would see it.
    Capture the actual call."""
    import scripts.train as T

    seen = {}

    class _Fake:
        @classmethod
        def from_pretrained(cls, name, **kw):
            seen.update(kw)
            seen["name"] = name
            return object()

    import teacher.scorer as S
    monkeypatch.setattr(S, "HFScorer", _Fake)

    cfg = {"model": {"student": "/some/ckpt", "dtype": "bfloat16",
                     "student_param_dtype": "float32",
                     "student_autocast_dtype": "bfloat16",
                     "student_device": "cuda:1"}}
    T.make_student(cfg, dry_run=False, tokenizer=None)

    assert seen["dtype"] == "float32", (
        f"the student was loaded as {seen['dtype']!r}. bf16 parameters cannot represent an "
        "AdamW step at lr 2e-6 -- training becomes a no-op that reads as a stable flat line")
    assert seen["autocast_dtype"] == "bfloat16", "the forward must still compute in bf16"
    assert seen["requires_grad"] is True


# ---------------------------------------------------------------------------
# 2. pi_ref placement
# ---------------------------------------------------------------------------
def _spy_on_to(module):
    """Record every device `.to()` is asked for, so placement is testable on a one-device
    box -- where comparing p.device against 'cpu' would be trivially true either way."""
    asked = []
    real = module.to

    def spy(*a, **k):
        if a:
            asked.append(str(a[0]))
        return real(*a, **k)

    module.to = spy
    return asked


def test_snapshot_ref_parks_the_reference_on_the_host():
    """pi_ref is a full copy of the student -- ~15 GiB in fp32 for the 4B -- and it is idle
    for the whole forward/backward, which is exactly when the card is tightest."""
    tok = MockTokenizer(vocab_size=256)
    student = Scorer(MockLM(tok.vocab_size, seed=1), requires_grad=True)
    asked = _spy_on_to(student.lm)

    ref = snapshot_ref(student)
    assert all(p.device.type == "cpu" for p in ref.lm.parameters())
    # the deepcopy inherits the spy, so the recorded request is the ref's own
    assert any("cpu" in a for a in asked), (
        "snapshot_ref never asked for the host -- on a GPU run pi_ref would stay resident "
        "on the card through every forward/backward for nothing")


def test_reference_precision_is_not_narrowed():
    """The drift this measures is ~1e-4 KL between two nearly identical distributions. bf16
    weights carry ~4e-3 relative error, so a narrowed reference would report its own
    quantisation as policy drift."""
    tok = MockTokenizer(vocab_size=256)
    lm = MockLM(tok.vocab_size, seed=1).to(torch.float32)
    ref = snapshot_ref(Scorer(lm, requires_grad=True))
    assert all(p.dtype == torch.float32 for p in ref.lm.parameters())


# ---------------------------------------------------------------------------
# 3. lifetimes across one real optimizer step
# ---------------------------------------------------------------------------
def _trainer(steps=1, global_batch=2):
    tok = MockTokenizer(vocab_size=8192)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = Scorer(MockLM(tok.vocab_size, seed=7), requires_grad=True)
    cfg = OPDConfig(mode="online", steps=steps, global_batch=global_batch, seed=3)
    src = OnlineSource([f"q{i}?" for i in range(4)], MockPolicy(tok, ["Paris"], "Paris"),
                       MockRetriever(CORPUS, seed=0), tok, teacher, cfg,
                       golds=[["Paris"]] * 4)
    ref = snapshot_ref(student)
    diag = DiagnosticRunner(ref_scorer=ref, tau=cfg.tau, embedder=None)

    class _Log:
        def __init__(self): self.records = []
        def log(self, rec): self.records.append(rec.to_dict())
        def close(self): pass

    log = _Log()
    return OPDTrainer(student, src, cfg, diagnostics=diag, logger=log), student, ref, log


def test_gradients_are_released_before_the_diagnostic_pass():
    """set_to_none frees a parameter-sized buffer (~15 GiB in fp32). Zeroing in place would
    keep it resident through diagnostics -- the other moment the card is tight, since pi_ref
    pages back in and kl_importance materialises [N, V] twice."""
    trainer, student, _, log = _trainer()
    trainer.step(0)
    live = [n for n, p in student.lm.named_parameters() if p.grad is not None]
    assert not live, (
        f"gradient buffers still resident after the step: {live}. They are dead weight "
        "through the diagnostic pass, which is when memory is tightest.")
    assert log.records, "the step must still have produced diagnostics"


def test_reference_is_paged_in_for_diagnostics_and_parked_again():
    """One transfer per OPTIMIZER step, not per trajectory -- and it must come back off the
    card afterwards, or the saving lasts exactly one step."""
    trainer, student, ref, log = _trainer(global_batch=3)
    asked = _spy_on_to(ref.lm)
    trainer.step(0)

    assert asked, "pi_ref was never moved -- the diagnostic pass ran against a host copy"
    assert "cpu" in asked[-1], f"pi_ref was left on {asked[-1]} after the step"
    # one round trip for the step, not one per trajectory
    assert len(asked) == 2, f"expected one page-in and one page-out, got {asked}"
    assert len(log.records) == 3, "one diagnostic record per trajectory, as before"


def test_the_reference_is_back_on_the_host_even_if_diagnostics_raise():
    """A leaked 15 GiB reference is not a thing to leave to control flow."""
    trainer, student, ref, log = _trainer()
    asked = _spy_on_to(ref.lm)

    def boom(*a, **k):
        raise RuntimeError("diagnostic exploded")

    trainer.diagnostics.run = boom
    try:
        trainer.step(0)
    except RuntimeError:
        pass
    assert asked and "cpu" in asked[-1], (
        "pi_ref stayed on the card after a diagnostic failure -- the next step would then "
        "run with 15 GiB less headroom than the budget assumes")
