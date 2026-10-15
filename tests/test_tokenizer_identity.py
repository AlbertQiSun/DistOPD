"""The teacher/student tokenizer-identity gate: does it actually discriminate?

A comparison that only ever sees one real pair is untestable in CI and untrusted anywhere. So
compare_tokenizers takes OBJECTS, and this file feeds it pairs whose answer is known by
construction: identical, differing only in a special-token id, and differing only in how one
string segments.

THE POINT OF THE THIRD CASE. Equal vocab_size and equal eos would pass a pair that MERGES
differently, and merges are precisely what shifts token boundaries -- which is the failure the
gate exists to catch, and the one that produces no exception. A gate that checks only the
scalar attributes would look thorough and catch nothing.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_tokenizer_identity import compare_tokenizers, probe_strings


class FakeTok:
    """Character-code tokenizer. `merges` maps a substring to a single id, which is how a real
    BPE difference presents: same alphabet, different segmentation."""

    def __init__(self, vocab_size=151936, eos_token_id=151645, bos_token_id=None,
                 pad_token_id=None, merges=()):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.merges = tuple(merges)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for m in self.merges:
                if text.startswith(m, i):
                    out.append(900000 + self.merges.index(m))
                    i += len(m)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_the_probe_set_is_not_trivial():
    probes = probe_strings()
    assert len(probes) > 25, "too few probes to be meaningful"
    assert any("<" in p for p in probes), "no template markers in the probe set"
    # the adversarial pairs: some probe must be a strict prefix of another, or the boundary
    # hazard the whole pure-id rule is about is never exercised
    assert any(a != b and b.startswith(a) for a in probes for b in probes), (
        "no prefix pairs -- a merge that only appears with a continuation cannot be seen")


def test_an_identical_pair_passes():
    a, b = FakeTok(), FakeTok()
    res = compare_tokenizers(a, b, probe_strings())
    assert res["ok"] and res["first_divergence"] is None
    assert res["n"] == len(probe_strings())


@pytest.mark.parametrize("field", ["eos_token_id", "bos_token_id", "pad_token_id"])
def test_a_differing_special_token_is_REPORTED_but_not_fatal(field):
    """A DELIBERATE RELAXATION, and the reason has to be in the test or it will read as a bug.

    The first version of this gate failed on any special-token difference and printed the
    SEGMENTATION warning when it did. On the real pair that produced a FAIL saying "the teacher
    and the student segment text differently" while every one of its own probes had matched --
    the sole difference being eos_token_id: 151645 <|im_end|> on a Thinking-lineage teacher vs
    151643 <|endoftext|> inherited by a student SFT'd from Qwen3-4B-Base. Both are correct for
    their lineage, and 151643 is positive evidence that the intended Base model was used rather
    than the Instruct one.

    It is not fatal because NOTHING READS IT: the rollout appends the literal T.IM_END
    (rollout/rollout.py) and vLLM stops on stop=[T.SEARCH_CLOSE, T.ANSWER_CLOSE]
    (rollout/vllm_policy.py). Those are frozen template constants whose SEGMENTATION is checked
    by the probes. The relaxation is to the metadata, never to the property that matters -- see
    the test below, which is what stops it from becoming a hole.
    """
    res = compare_tokenizers(FakeTok(**{field: 151645}), FakeTok(**{field: 2}), probe_strings())
    assert res["ok"], f"{field} alone must not fail the gate"
    assert res["attrs"][field]["equal"] is False, "and it must still be REPORTED"
    assert res["attrs"][field]["fatal"] is False
    assert field not in res["fatal_attrs"]


def test_the_relaxation_does_not_mask_a_real_divergence():
    """THE GUARD ON THE RELAXATION. A pair that differs in eos AND merges differently must still
    fail -- otherwise "eos is advisory" would have quietly become "the gate is advisory"."""
    res = compare_tokenizers(FakeTok(eos_token_id=151645),
                             FakeTok(eos_token_id=2, merges=("Paris",)), probe_strings())
    assert not res["ok"] and res["segmentation_ok"] is False
    assert res["first_divergence"] is not None


def test_a_differing_vocab_size_IS_fatal():
    """vocab_size is the one attribute the pipeline consumes: it sets the width of every logit
    row the teacher returns, so a mismatch misaligns the whole distribution rather than shifting
    it. Kept fatal for that reason, not by inertia."""
    res = compare_tokenizers(FakeTok(vocab_size=151936), FakeTok(vocab_size=32000),
                             probe_strings())
    assert not res["ok"]
    assert "vocab_size" in res["fatal_attrs"]
    assert res["segmentation_ok"] is True, "the fixture must isolate vocab_size from merges"


def test_a_MERGE_difference_is_caught_though_every_scalar_agrees():
    """THE GATE. Both tokenizers report the same vocab size and the same special tokens; they
    differ only in that one fuses "Paris" into a single id. That is the shape of a real
    tokenizer mismatch, it passes every attribute check, and it silently misaligns the
    per-token advantage from that position onward."""
    plain, merging = FakeTok(), FakeTok(merges=("Paris",))
    res = compare_tokenizers(plain, merging, probe_strings())
    assert not res["ok"], "a merge difference slipped through every scalar check"
    d = res["first_divergence"]
    assert d is not None and "Paris" in d["text"]
    assert d["teacher_ids"] != d["student_ids"]


def test_a_merge_that_only_bites_in_context_is_still_caught():
    """The nastier case: the merge covers a string that appears only as part of a longer probe,
    so isolated-token checks would miss it."""
    res = compare_tokenizers(FakeTok(), FakeTok(merges=("capital of France",)), probe_strings())
    assert not res["ok"]


def test_the_first_divergence_is_reported_not_just_a_boolean():
    """The operator has to be able to act on it. A bare False sends them re-downloading
    checkpoints; the offending string and both id lists say which side is wrong."""
    res = compare_tokenizers(FakeTok(), FakeTok(merges=("Paris",)), probe_strings())
    d = res["first_divergence"]
    assert set(d) == {"text", "teacher_ids", "student_ids"}
    assert all(isinstance(i, int) for i in d["teacher_ids"])


@pytest.mark.slow
def test_the_real_staged_checkpoints_agree():
    """The gate against the ACTUAL pair, on whatever transformers the channel resolved. Skips
    where the checkpoints are not staged -- so it is the certify run, not CI, that settles it."""
    import os

    teacher = os.environ.get("OPD_TEACHER_MODEL")
    student = os.environ.get("OPD_STUDENT_CKPT")
    if not teacher or not student:
        pytest.skip("OPD_TEACHER_MODEL / OPD_STUDENT_CKPT unset -- source slurm/env.sh")
    if not (pathlib.Path(teacher).exists() and pathlib.Path(student).exists()):
        pytest.skip("checkpoints not staged on this machine")

    from transformers import AutoTokenizer

    res = compare_tokenizers(AutoTokenizer.from_pretrained(teacher),
                             AutoTokenizer.from_pretrained(student), probe_strings())
    assert res["ok"], f"teacher/student tokenizers diverge: {res['first_divergence']}"
