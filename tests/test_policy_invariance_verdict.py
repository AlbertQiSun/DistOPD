"""The verdict must not contradict the measurement printed next to it.

WHAT HAPPENED. gates 3a/3b ran on the A100 on 2026-08-17 and the probe printed:

    batch_invariant=OFF  greedy_flips=0  sampled_flips=6  max|dlogp|=1.528e-01
    batch_invariant=ON   greedy_flips=0  sampled_flips=0  max|dlogp|=0.000e+00
    VERDICT: cause (ii) RNG STREAM ... The flag is NOT the remedy ... this needs
             per-request derived seeds

Two lines of evidence, and the third line contradicted both. The flag took a 100% sampled
flip rate to zero and a 0.15 logprob residual to exactly zero -- which is precisely what
cause (ii) says CANNOT happen ("Remedy: NONE from that flag"). Acting on that verdict would
have meant rewriting the sampler's seeding, on a channel that was in fact already certified.

THE BUG was one branch reading `greedy_flips == 0` as "greedy is clean". Greedy was not
clean: its LOGITS differed by 0.15 and only the argmax failed to cross. The probe's own
opening paragraph says matching ids with a non-zero residual "is NOT a pass -- it is a near
miss that will bite later", and its own `else` branch said to treat that as cause (i) -- but
`elif s_off > 0` came first and swallowed it.

These tests are written against the REAL numbers from that run, so the case that produced the
wrong answer is the case that is pinned. The pure classify_cause() exists so the
interpretation of an expensive GPU run can be checked on a CPU -- which is what it could not
be before.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.policy_invariance_probe import classify_cause  # noqa: E402


def _run(setting, *, g_flips, g_resid, s_flips, s_resid):
    return {
        "batch_invariant": setting,
        "correctness": {
            "greedy/turn0": {"flips": g_flips, "max_logprob_residual": g_resid},
            "greedy/late": {"flips": 0, "max_logprob_residual": 0.0},
            "sampled/turn0": {"flips": s_flips, "max_logprob_residual": s_resid},
            "sampled/late": {"flips": 0, "max_logprob_residual": 0.0},
        },
    }


# The A100 run, verbatim.
A100_OFF = _run("off", g_flips=0, g_resid=1.528e-01, s_flips=6, s_resid=1.076e-01)
A100_ON = _run("on", g_flips=0, g_resid=0.0, s_flips=0, s_resid=0.0)


def test_the_A100_run_is_numerics_not_rng():
    """THE REGRESSION. greedy ids matched, greedy LOGITS did not -- that is cause (i)."""
    cause, sentence = classify_cause(A100_OFF, A100_ON)
    assert cause == "numerics", f"got {cause}: {sentence}"
    assert "per-request derived seeds" not in sentence, (
        "the verdict still sends the reader to rewrite the sampler's seeding")


def test_the_A100_verdict_says_the_flag_is_the_remedy():
    """Because it measurably was: 6/6 sampled flips and a 0.15 residual both went to zero."""
    _, sentence = classify_cause(A100_OFF, A100_ON)
    assert "flag IS the remedy" in sentence
    assert "ON run confirms" in sentence, "the ON measurement must be cited, not just implied"


def test_a_greedy_residual_alone_is_enough_to_call_it_numerics():
    """No id flips anywhere, but the logits moved. The old code fell through to 'near miss';
    the point is that a near miss and cause (i) have the SAME remedy, so the flag applies."""
    off = _run("off", g_flips=0, g_resid=5.7e-02, s_flips=0, s_resid=1.2e-01)
    cause, _ = classify_cause(off, A100_ON)
    assert cause == "numerics"


def test_a_greedy_flip_is_still_numerics():
    """The unambiguous case the old code got right must stay right."""
    off = _run("off", g_flips=3, g_resid=2e-01, s_flips=6, s_resid=2e-01)
    assert classify_cause(off, A100_ON)[0] == "numerics"


def test_rng_stream_requires_greedy_to_be_BIT_IDENTICAL():
    """The only shape that genuinely implicates the RNG: greedy identical in ids AND in
    logits, sampling still moving. Then and only then is the flag not the remedy."""
    off = _run("off", g_flips=0, g_resid=0.0, s_flips=6, s_resid=0.0)
    on_still_flipping = _run("on", g_flips=0, g_resid=0.0, s_flips=6, s_resid=0.0)
    cause, sentence = classify_cause(off, on_still_flipping)
    assert cause == "rng-stream"
    assert "per-request derived seeds" in sentence
    assert "CONTRADICTION" not in sentence, "nothing to contradict -- the ON run also flipped"


def test_an_rng_verdict_against_a_clean_ON_run_is_flagged_as_contradictory():
    """THE SELF-AUDIT. If the classification ever again says 'the flag cannot help' while the
    ON run came back perfectly clean, the printout must say so rather than let a reader act on
    the inference over the direct measurement."""
    off = _run("off", g_flips=0, g_resid=0.0, s_flips=6, s_resid=0.0)
    cause, sentence = classify_cause(off, A100_ON)
    assert cause == "rng-stream"
    assert "CONTRADICTION" in sentence
    assert "direct evidence" in sentence


def test_already_invariant_needs_zero_flips_AND_zero_residual():
    """`already-invariant` recommends NOT enabling the flag, so a 0.15 residual must never
    reach it -- that is the near miss wearing a pass."""
    off = _run("off", g_flips=0, g_resid=0.0, s_flips=0, s_resid=0.0)
    assert classify_cause(off, A100_ON)[0] == "already-invariant"

    nearly = _run("off", g_flips=0, g_resid=0.0, s_flips=0, s_resid=1e-03)
    assert classify_cause(nearly, A100_ON)[0] != "already-invariant"


def test_the_verdict_is_written_to_the_gate_log_with_its_numbers():
    """PRODUCER SIDE. The sentence is an inference over four numbers; persisting only the
    sentence leaves a later reader unable to re-derive it -- or to catch it being wrong, which
    is exactly how this one was caught. the project's rule, applied to a verdict."""
    src = (ROOT / "scripts" / "policy_invariance_probe.py").read_text(encoding="utf-8")
    call = src.split('append_gate_run("policy_batch_invariance_verdict"', 1)
    assert len(call) == 2, "the verdict is not recorded in the gate log at all"
    block = call[1][:600]
    for field in ('"off"', '"on"', "_agg(off", "_agg(on", '"sentence"'):
        assert field in block, f"the gate-log record omits {field}"
