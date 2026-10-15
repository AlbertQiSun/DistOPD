"""A ~31-hour run must not end with nothing to evaluate.

WHAT SURVIVES A CRASH TODAY, measured rather than assumed: opd/diag_logger.py flushes after
every record, so the per-step diagnostics -- the online-vs-offline curve, which is the paper's
main artifact -- are durable to the last completed step. The WEIGHTS were not: train.py wrote
the student once, after the loop. A job killed at hour 30 of 31 kept the curve and lost the
checkpoint, and the checkpoint is what the held-out dev evaluation needs (the in-run
diagnostics cannot answer "did OPD change the model?" -- online task.em is self-poisoning
onset, offline task.em is a frozen-input property).

THE PROPERTY THESE TESTS ARE FOR is not "a file appears". It is that the pointer never names
an incomplete checkpoint. `save_pretrained` writes several files and there is no instant at
which that is atomic, so a single overwritten directory would leave a job killed mid-write
with something that is neither the old checkpoint nor a new one. Two alternating slots plus a
pointer written LAST is what makes "read the pointer, get a complete checkpoint" true, and
test_a_failed_save_leaves_the_previous_pointer_intact is the test that this holds under the
failure it was built for.

NO GPU AND NO MODEL: the student and tokenizer are fakes that record calls, which is what lets
the crash case be provoked at all -- it is not reachable by running the real thing and hoping.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    CKPT_POINTER,
    ckpt_slot,
    save_periodic_checkpoint,
    save_student_checkpoint,
)


class _FakeLM:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.saved_to = fail, []

    def save_pretrained(self, d) -> None:
        self.saved_to.append(str(d))
        if self.fail:
            raise RuntimeError("simulated kill during save_pretrained")
        (pathlib.Path(d) / "model.safetensors").write_text("weights", encoding="utf-8")


class _FakeStudent:
    def __init__(self, fail: bool = False) -> None:
        self.lm = _FakeLM(fail)


class _FakeTok:
    def __init__(self) -> None:
        self.saved_to = []

    def save_pretrained(self, d) -> None:
        self.saved_to.append(str(d))
        (pathlib.Path(d) / "tokenizer.json").write_text("tok", encoding="utf-8")


class _Adapter:
    """HFTokenizerAdapter's shape: the real tokenizer hangs off `.tok`."""

    def __init__(self) -> None:
        self.tok = _FakeTok()


# ---------------------------------------------------------------- the slot arithmetic
def test_no_save_when_disabled():
    assert all(ckpt_slot(s, 0) is None for s in range(1, 200))


def test_only_multiples_of_save_every_save():
    saving = [s for s in range(1, 101) if ckpt_slot(s, 25) is not None]
    assert saving == [25, 50, 75, 100]


def test_consecutive_saves_alternate_slots():
    """THE POINT OF TWO SLOTS. If two consecutive saves shared a directory, the second would
    overwrite the only complete checkpoint on disk while writing."""
    slots = [ckpt_slot(s, 25) for s in (25, 50, 75, 100)]
    assert slots == [1, 0, 1, 0]
    assert all(a != b for a, b in zip(slots, slots[1:]))


def test_step_zero_never_saves():
    """Step 0 is pi_ref, which the run already has -- and `0 % n == 0` would save it."""
    assert ckpt_slot(0, 25) is None


# ---------------------------------------------------------------- the save itself
def test_a_checkpoint_carries_the_tokenizer_too(tmp_path):
    """scripts/stage3_eval.py --model <ckpt> loads both. Weights alone are not evaluable."""
    student, tok = _FakeStudent(), _Adapter()
    d = save_student_checkpoint(student, tok, str(tmp_path / "ckpt"))
    assert (pathlib.Path(d) / "model.safetensors").exists()
    assert (pathlib.Path(d) / "tokenizer.json").exists()
    assert tok.tok.saved_to == [d], "the adapter must be unwrapped, not saved as itself"


def test_a_tokenizer_without_save_pretrained_is_skipped_not_fatal(tmp_path):
    """MockTokenizer has none. The guard must not turn a dry-run into a crash."""
    class _Bare:
        pass

    d = save_student_checkpoint(_FakeStudent(), _Bare(), str(tmp_path / "ckpt"))
    assert (pathlib.Path(d) / "model.safetensors").exists()


# ---------------------------------------------------------------- crash safety
def test_the_pointer_names_the_slot_that_was_just_written(tmp_path):
    save_periodic_checkpoint(_FakeStudent(), _Adapter(), str(tmp_path), "online", 50, 0)
    p = json.loads((tmp_path / CKPT_POINTER.format(mode="online")).read_text(encoding="utf-8"))
    assert p == {"mode": "online", "step": 50, "dir": "ckpt_online_p0"}
    assert (tmp_path / p["dir"] / "model.safetensors").exists()


def test_a_failed_save_leaves_the_previous_pointer_intact(tmp_path):
    """THE FAILURE THIS DESIGN EXISTS FOR. Slot 0 completed at step 50; the step-75 save into
    slot 1 dies partway. The pointer must still name slot 0 -- if it had been written first,
    or if both saves shared a directory, the run would end holding a checkpoint that cannot be
    loaded and no way to know that from the pointer."""
    ok_student, tok = _FakeStudent(), _Adapter()
    save_periodic_checkpoint(ok_student, tok, str(tmp_path), "online", 50, 0)

    with pytest.raises(RuntimeError):
        save_periodic_checkpoint(_FakeStudent(fail=True), tok, str(tmp_path), "online", 75, 1)

    p = json.loads((tmp_path / CKPT_POINTER.format(mode="online")).read_text(encoding="utf-8"))
    assert p["step"] == 50 and p["dir"] == "ckpt_online_p0", (
        f"the pointer followed a save that never finished: {p}")
    assert (tmp_path / "ckpt_online_p0" / "model.safetensors").exists(), (
        "the failed save destroyed the previous complete checkpoint")


def test_the_two_arms_do_not_share_a_pointer_or_a_slot(tmp_path):
    """Both arms write into the same runs/ directory. A shared name would have the online
    arm's pointer describing the offline arm's weights."""
    save_periodic_checkpoint(_FakeStudent(), _Adapter(), str(tmp_path), "online", 50, 0)
    save_periodic_checkpoint(_FakeStudent(), _Adapter(), str(tmp_path), "offline", 75, 1)
    on = json.loads((tmp_path / CKPT_POINTER.format(mode="online")).read_text(encoding="utf-8"))
    off = json.loads((tmp_path / CKPT_POINTER.format(mode="offline")).read_text(encoding="utf-8"))
    assert on["dir"] != off["dir"] and on["step"] != off["step"]


# ---------------------------------------------------------------- producer side
def test_the_training_loop_actually_calls_the_periodic_save():
    """PRODUCER TEST, per the project conventions. Everything above tests functions; this tests that the run
    LOOP invokes them. Delete the call in train.py and every test above stays green while no
    checkpoint is ever written -- the exact shape of gate this repo keeps finding to be
    decoration."""
    src = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    loop = src.split("for i, out in enumerate(trainer.train()):", 1)
    assert len(loop) == 2, "the training loop moved -- this test can no longer see it"
    body = loop[1].split("\n    print(f\"done ->", 1)[0]
    assert "ckpt_slot(" in body, "the loop does not decide whether to checkpoint"
    assert "save_periodic_checkpoint(" in body, "the loop never saves"


def test_periodic_and_final_saves_share_one_code_path():
    """A periodic checkpoint written differently from the final one is one you cannot evaluate
    the same way -- and the difference would only surface after losing the run that wrote it.
    Exactly one place may call student.lm.save_pretrained."""
    src = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    calls = [ln for ln in src.splitlines()
             if "save_pretrained(" in ln and "student.lm" in ln and not ln.strip().startswith("#")]
    assert len(calls) == 1, f"student weights are saved from {len(calls)} places: {calls}"


def test_save_every_is_reachable_from_the_command_line():
    src = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert '"--save-every"' in src
    assert "save_every=args.save_every" in src, "the flag is parsed but never passed to run()"
