"""The policy probe's VERDICT logic, tested on CPU.

The probe itself needs a GPU, but the part that decides what an expensive run MEANS is
plain Python -- and it is the part that would quietly mislead. A verdict that says
"numerics" when the cause is the RNG stream sends the next hours into enabling a flag that
is not the remedy; a verdict that says PASS on matching ids while the logprobs already
differ blesses a near miss that a longer generation would flip.

Each branch is exercised with a synthetic payload of the shape the probe writes.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "policy_probe", ROOT / "scripts" / "policy_invariance_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # must import WITHOUT argv / GPU
    return mod


PP = _load()


@pytest.fixture(autouse=True)
def no_real_gate_log(monkeypatch):
    """Never let these tests reach runs/gate_log.jsonl.

    _report() appends a gate record when cost data is present, and the first version of
    this file therefore wrote 8 fabricated "policy_batch_invariance_cost" entries into the
    tracked provenance log -- indistinguishable, to a later reader, from real measurements.
    Collect them instead, which also lets the tests assert what WOULD be recorded."""
    recorded = []
    monkeypatch.setattr(PP, "append_gate_run",
                        lambda gate, verdict, **kw: recorded.append((gate, verdict, kw)))
    return recorded


def _case(greedy_flips=0, sampled_flips=0, resid=0.0):
    return {
        "greedy/turn0": {"flips": greedy_flips, "max_logprob_residual": resid},
        "sampled/turn0": {"flips": sampled_flips, "max_logprob_residual": resid},
    }


def _payload(setting, greedy=0, sampled=0, resid=0.0, cost=None):
    return {"batch_invariant": setting, "correctness": _case(greedy, sampled, resid),
            "cost": cost}


def _verdict(capsys, off_kw, on_kw=None):
    PP._report(_payload("off", **off_kw), _payload("on", **(on_kw or {})), "cfg")
    return capsys.readouterr().out


def test_clean_off_means_do_not_enable_the_flag(capsys):
    out = _verdict(capsys, dict(greedy=0, sampled=0, resid=0.0))
    assert "ALREADY batch-invariant" in out
    assert "do not enable" in out


def test_greedy_flip_is_attributed_to_numerics(capsys):
    """Greedy has no RNG, so a flip there can only be tiling -- and the flag IS the fix."""
    out = _verdict(capsys, dict(greedy=3, sampled=5))
    assert "cause (i) NUMERICS" in out
    assert "flag IS the remedy" in out


def test_sampled_only_flip_is_attributed_to_rng_and_disables_the_thresholds(capsys):
    """The branch that matters most: greedy clean + sampled flipping means the flag is NOT
    the remedy, so the <2x / 2-4x / >4x cost thresholds must not be applied at all."""
    out = _verdict(capsys, dict(greedy=0, sampled=4))
    assert "cause (ii) RNG STREAM" in out
    assert "NOT the remedy" in out
    assert "per-request derived seeds" in out


def test_matching_ids_with_nonzero_residual_is_not_a_pass(capsys):
    """Discrete equality is weak evidence: the logits already differ, the argmax boundary
    just has not been crossed yet."""
    out = _verdict(capsys, dict(greedy=0, sampled=0, resid=1e-6))
    assert "NEAR MISS" in out
    assert "ALREADY batch-invariant" not in out


@pytest.mark.parametrize("ratio,expect", [(1.4, "<2x"), (2.7, "2-4x"), (6.0, ">4x")])
def test_cost_bands_match_the_pre_agreed_thresholds(capsys, ratio, expect, no_real_gate_log):
    off_cost = {"policy_seconds": 100.0, "end_to_end_seconds": 200.0, "batch_sizes": [8, 5, 3]}
    on_cost = {"policy_seconds": 100.0 * ratio, "end_to_end_seconds": 220.0,
               "batch_sizes": [8, 5, 3]}
    PP._report(_payload("off", greedy=1, cost=off_cost),
               _payload("on", cost=on_cost), "cfg")
    out = capsys.readouterr().out
    assert "POLICY-ONLY cost ratio" in out
    assert expect in out
    # the end-to-end number must be reported but labelled as diluted, never used for the band
    assert "retrieval-diluted" in out
    # and the gate record carries the ratio + band, not the diluted number
    gate, verdict, kw = no_real_gate_log[-1]
    assert gate == "policy_batch_invariance_cost"
    assert verdict.startswith(f"{ratio:.2f}")
    assert kw["details"]["band"].startswith(expect)


def test_round_survival_is_reported(capsys):
    """How much the batch shrinks per round is how many DISTINCT batch compositions a
    trajectory meets in a real rollout -- i.e. how many chances the correctness question
    gets that the one-shot probe does not cover."""
    cost = {"policy_seconds": 10.0, "end_to_end_seconds": 20.0, "batch_sizes": [32, 21, 9, 4]}
    PP._report(_payload("off", cost=cost), _payload("on", cost=cost), "cfg")
    out = capsys.readouterr().out
    assert "round survival" in out
    assert "[32, 21, 9, 4]" in out


def test_gate_log_refuses_synthetic_writes_from_a_test_run():
    """The structural backstop, in case a future test forgets to inject a collector.

    A fabricated entry in the provenance log is worse than a missing one: nobody reading it
    later can tell it from a real gate run. This exact thing happened -- these very tests
    wrote 8 fake cost records into the tracked log before the guard existed."""
    from opd.gate_log import DEFAULT_PATH, append_gate_run

    with pytest.raises(RuntimeError, match="refusing to append to the real gate log"):
        append_gate_run("fake_gate", "PASS")           # defaults to DEFAULT_PATH

    # an explicit path is the sanctioned escape hatch and must still work
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gl.jsonl")
        append_gate_run("fake_gate", "PASS", path=p)
        assert os.path.exists(p)
    assert DEFAULT_PATH.endswith("gate_log.jsonl")
