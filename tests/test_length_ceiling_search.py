"""The length search must not report a ceiling it never hit.

WHERE THIS COMES FROM. certify gate 6 on the A100 printed

    --> longest trajectory that fits on this card: ~16169 tokens

and the H20 had printed the same 16169. Two cards with different memory budgets cannot share
a memory-determined limit to the token, and that is the whole tell: the number was `hi` minus
the bisection's stopping tolerance, produced by a loop in which EVERY probe fit. 16384 - 256
brackets it exactly; the printed probes were 9529, 12956, 14670, 15527, 15955, 16169, all
"fits", none OOM.

The corroborating measurement is in the same log: peak memory was 74.9 GiB at 708 tokens and
74.9 GiB at 13750. Length barely moves the peak, because the peak is the pi_ref page-in during
the diagnostic (15 weights + 15 grads + 30 Adam + 15 pi_ref). A run whose peak is
length-independent is not going to OOM by getting longer, so the search was always going to
walk to its own upper bound.

`rollout.max_len` is set from this number, and it is the kind of value that ends up in a paper
as "the trainable length ceiling on an A100". the project's rule is that a claim must point at
evidence; this file is the evidence for the shape of the claim.

NO GPU NEEDED: `fits` is injected, so the arithmetic that produced the wrong verdict is
testable directly, which is exactly what it was not before.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_train_step import search_length_ceiling  # noqa: E402


def test_everything_fits_reports_NO_ceiling_not_hi_minus_tolerance():
    """THE BUG, at the exact numbers that produced it. Nothing OOMs anywhere."""
    r = search_length_ceiling(lambda n: True, 2674, 16384)
    assert r["located"] is False, (
        "the search claimed to have located a ceiling although no probe ever OOMed")
    assert r["largest_that_fits"] == 16384, (
        f"with everything fitting, the established fact is 'up to the search bound', "
        f"not {r['largest_that_fits']}")
    assert r["largest_that_fits"] != 16169, "16169 is hi minus the tolerance, not a measurement"


def test_it_costs_ONE_probe_when_everything_fits():
    """Probing hi first is also what makes the no-ceiling case cheap: the old loop spent six
    full training passes at 9529..16169 tokens to conclude nothing."""
    seen = []
    search_length_ceiling(lambda n: seen.append(n) or True, 2674, 16384)
    assert seen == [16384], f"expected a single probe at hi, got {seen}"


def test_a_REAL_ceiling_is_still_found_and_bracketed():
    """The refusal above must not have cost the function its job."""
    r = search_length_ceiling(lambda n: n <= 8000, 2674, 16384)
    assert r["located"] is True
    assert r["largest_that_fits"] <= 8000 < r["smallest_that_ooms"], r
    assert r["smallest_that_ooms"] - r["largest_that_fits"] <= 256, "bracket wider than the tol"


def test_the_bracket_is_reported_not_just_the_lower_end():
    """`largest_that_fits` alone cannot say whether 8001 was tried. The pair can."""
    r = search_length_ceiling(lambda n: n <= 12000, 2674, 16384)
    assert r["smallest_that_ooms"] is not None
    assert not any(p["fits"] for p in r["probes"] if p["tokens"] >= r["smallest_that_ooms"])
    assert all(p["fits"] for p in r["probes"] if p["tokens"] <= r["largest_that_fits"])


def test_nothing_fits_at_all_does_not_return_lo_as_if_it_fit():
    """If even the floor OOMs, `lo` was never probed and must not be reported as fitting."""
    r = search_length_ceiling(lambda n: False, 2674, 16384)
    assert r["located"] is True
    assert not any(p["fits"] for p in r["probes"])
    assert r["smallest_that_ooms"] <= 2674 + 256


def test_every_probe_is_persisted_not_just_the_verdict():
    """the project conventions: persist the raw materials. The probe list is what lets a later reader ask a
    question this run did not -- e.g. where the peak started growing -- without a GPU re-run."""
    r = search_length_ceiling(lambda n: n <= 10000, 2674, 16384)
    assert len(r["probes"]) >= 2
    assert all(set(p) == {"tokens", "fits"} for p in r["probes"])
    assert r["search_upper_bound"] == 16384


def test_the_verdict_and_the_number_cannot_drift_apart():
    """`located` is the field that distinguishes "= X" from "> X". A caller reading only
    largest_that_fits gets the same integer in both cases, which is how the wrong sentence got
    printed for months on two different clusters."""
    a = search_length_ceiling(lambda n: True, 2674, 16384)
    b = search_length_ceiling(lambda n: n <= 16383, 2674, 16384)
    assert a["located"] != b["located"], "the two cases must be distinguishable"
    assert a["smallest_that_ooms"] is None and b["smallest_that_ooms"] is not None
