"""Binning the damage curve: on the STEP axis, so a truncated arm still lines up.

WHY THIS IS TESTED AND THE PLOTTING IS NOT. Everything else in plot_diagnostics draws; this
one turns 121 MB of per-sample rows into the six numbers a decision gets made on, and the
comparison it feeds is between arms that DO NOT have the same number of rows. miro's online
arm ran all 150 steps; its first attempt at the MoE line died at step 38, and a run killed by
the 72 h wall stops wherever it stops. Binning by record position would put a 150-step arm
and a 38-step arm on different x-axes and print them as if they were the same axis -- the
resulting "online degraded and offline did not" would be an artifact of where the jobs
stopped.

The diagnostics file also carries global_batch rows PER step, so record index is not step
index even for a complete run.
"""
from __future__ import annotations

import pytest

from scripts.plot_diagnostics import binned


def _rows(steps, value, per_step=4):
    """`per_step` rows for each step, as the real diagnostics writer produces."""
    return [{"step": s, "task": {"em": value(s)}} for s in steps for _ in range(per_step)]


def test_bins_are_step_ranges_not_record_ranges():
    rows = _rows(range(150), lambda s: float(s))
    out = binned(rows, ("task", "em"), bins=6, total_steps=150)
    assert [(lo, hi) for lo, hi, _, _ in out] == [
        (0, 24), (25, 49), (50, 74), (75, 99), (100, 124), (125, 149)]
    assert out[0][2] == pytest.approx(12.0)      # mean of 0..24
    assert out[-1][2] == pytest.approx(137.0)    # mean of 125..149
    assert all(n == 25 * 4 for _, _, _, n in out), "each bin holds 25 steps x 4 rows"


def test_an_arm_that_stopped_early_keeps_the_same_axis():
    """THE regression. A 38-step arm must occupy the FIRST bins and leave the rest empty --
    not be stretched across all six as record-order binning would do."""
    short = binned(_rows(range(38), lambda s: 1.0), ("task", "em"), bins=6, total_steps=150)
    assert short[0][2] == 1.0 and short[1][2] == 1.0
    assert [m for _, _, m, _ in short[2:]] == [None, None, None, None]
    # ... and the bin edges are identical to a complete arm's, so the two print row for row.
    full = binned(_rows(range(150), lambda s: 1.0), ("task", "em"), bins=6, total_steps=150)
    assert [(lo, hi) for lo, hi, _, _ in short] == [(lo, hi) for lo, hi, _, _ in full]


def test_an_empty_bin_is_none_rather_than_dropped():
    """Dropping it would shift every later column left, silently comparing step 100 against
    step 75 in the printed table."""
    rows = _rows([0, 1, 2, 140, 141], lambda s: 1.0)
    out = binned(rows, ("task", "em"), bins=6, total_steps=150)
    assert len(out) == 6
    assert out[0][2] == 1.0 and out[5][2] == 1.0
    assert [m for _, _, m, _ in out[1:5]] == [None, None, None, None]


def test_the_last_bin_closes_at_the_declared_total():
    for bins in (3, 6, 7, 11):
        out = binned(_rows(range(150), lambda s: 1.0), ("task", "em"), bins, 150)
        assert len(out) == bins
        assert out[-1][1] == 149, f"{bins} bins left the axis open at {out[-1][1]}"
        assert out[0][0] == 0


def test_a_step_past_the_declared_total_lands_in_the_last_bin_not_a_new_one():
    """--steps is what the operator declares; a run that somehow logged beyond it must not
    silently add a column that the other arm does not have."""
    out = binned(_rows([0, 200], lambda s: float(s)), ("task", "em"), bins=4, total_steps=150)
    assert len(out) == 4
    assert out[-1][2] == pytest.approx(200.0)


def test_missing_values_are_skipped_without_shifting_the_step_axis():
    """kl_importance returns None for a trajectory with no scored tokens. Those rows must not
    count toward a bin's mean, and must not push other rows into the wrong bin."""
    rows = [{"step": 0, "task": {"em": None}},
            {"step": 0, "task": {"em": 1.0}},
            {"step": 149, "task": {"em": 3.0}}]
    out = binned(rows, ("task", "em"), bins=6, total_steps=150)
    assert out[0][2] == pytest.approx(1.0) and out[0][3] == 1
    assert out[-1][2] == pytest.approx(3.0)


def test_no_rows_at_all_yields_a_full_row_of_none():
    out = binned([], ("task", "em"), bins=6, total_steps=150)
    assert len(out) == 6 and all(m is None for _, _, m, _ in out)


# --------------------------------------------------------------------------- matched window
def test_max_step_drops_the_longer_arm_s_tail_instead_of_folding_it_in():
    """THE regression. Comparing a 57-step online arm with a 150-step offline one, --steps 57
    alone put offline's steps 57..149 into the last bin, so a column labelled 38-56 was really
    the mean over 38-149. Flat series barely move; kl_pertok, which rises by 100x, was read
    completely wrong."""
    rows = _rows(range(150), lambda s: float(s))
    folded = binned(rows, ("task", "em"), bins=3, total_steps=57)
    windowed = binned(rows, ("task", "em"), bins=3, total_steps=57, max_step=56)
    assert folded[-1][2] == pytest.approx(sum(range(38, 150)) / 112)
    assert windowed[-1][2] == pytest.approx(sum(range(38, 57)) / 19)
    assert windowed[-1][2] < folded[-1][2]


def test_max_step_leaves_a_short_arm_untouched():
    """The shorter arm has nothing past the window, so the two calls must agree -- otherwise
    the filter would be changing the very series it exists to make comparable."""
    rows = _rows(range(57), lambda s: float(s))
    assert (binned(rows, ("task", "em"), 3, 57)
            == binned(rows, ("task", "em"), 3, 57, max_step=56))


def test_without_max_step_the_clamping_behaviour_is_unchanged():
    """The default must not silently start dropping data: a run that logged a step past the
    declared total still lands in the last bin rather than vanishing."""
    out = binned(_rows([0, 200], lambda s: float(s)), ("task", "em"), bins=4, total_steps=150)
    assert out[-1][2] == pytest.approx(200.0)


# ---------------------------------------------------------------- the importance weight
# `iw_mean 637746287.4441` is what the dense arm's first bin printed. It is not a policy that
# drifted nine orders of magnitude: iw is exp(student_logp - ref_logp), a single sample at a
# log-ratio of 20 contributes 4.8e8, and the mean of 6400 numbers belongs to whichever one is
# that size. A mean of exponentials cannot summarise this series -- so the summary reports the
# MEDIAN (where the bulk sits) and p90 (whether the tail is growing).
def test_the_median_ignores_a_single_enormous_sample():
    from scripts.plot_diagnostics import _reduce
    vals = [0.9] * 99 + [1e9]
    assert _reduce(vals, "median") == 0.9
    assert _reduce(vals, "mean") > 1e7, "the mean is the thing being avoided; keep it honest"


def test_p90_reports_the_tail_that_the_median_hides():
    from scripts.plot_diagnostics import _reduce
    # 85/15, not 90/10: at exactly 90/10 the 90th percentile sits ON the boundary and
    # nearest-rank vs interpolating definitions disagree, so the test would be asserting a
    # convention rather than the behaviour.
    vals = [0.9] * 85 + [50.0] * 15
    assert _reduce(vals, "median") == 0.9
    assert _reduce(vals, "p90") == 50.0


def test_p90_of_a_single_sample_does_not_index_off_the_end():
    from scripts.plot_diagnostics import _reduce
    assert _reduce([7.0], "p90") == 7.0


def test_binned_passes_the_statistic_through_rather_than_always_averaging():
    """The reducer existing is not enough -- binned() has to use it. Hardcode `mean` back
    inside binned and this goes red while _reduce's own tests stay green."""
    from scripts.plot_diagnostics import binned
    rows = [{"step": 0, "kl_importance": {"iw_mean": v}} for v in ([0.9] * 9 + [1e9])]
    med = binned(rows, ("kl_importance", "iw_mean"), bins=1, total_steps=1, agg="median")
    assert med[0][2] == 0.9


def test_no_summary_row_reports_the_importance_weight_as_a_mean():
    """The producer-side gate. A future edit that adds an iw row back with agg="mean" puts
    an uninterpretable 6e8 into the table that gets read as a result."""
    from scripts.plot_diagnostics import SUMMARY_ROWS
    for name, keys, _why, agg in SUMMARY_ROWS:
        if keys == ("kl_importance", "iw_mean"):
            assert agg in ("median", "p90"), f"{name} summarises iw with {agg}"
    aggs = {agg for _n, keys, _w, agg in SUMMARY_ROWS if keys == ("kl_importance", "iw_mean")}
    assert aggs == {"median", "p90"}, "the summary must show both the bulk and the tail"
