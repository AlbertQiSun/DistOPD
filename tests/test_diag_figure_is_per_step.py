"""The figure must draw one point per STEP, not one per sample.

THE FAILURE THIS ENCODES. runs/fig_miro.png and runs/fig_dense.png came back unreadable --
solid blocks of colour with no visible curve. The data was fine; the plotting was not.
scripts/plot_diagnostics.py drew `_series`, which yields a point for every ROW of
diag_*.jsonl, and the trainer writes global_batch = 256 rows per step. A 150-step arm is
38,400 markers, 256 of them stacked on each x value -- and `answered`, `cover_em` and
`is_truncated` are per-sample 0/1, so each column spanned the entire y range. Every panel
was a filled rectangle. The --summary text mode was legible for exactly one reason: binned()
aggregated before printing, and the figure path did not.

TWO MORE THINGS THIS FILE PINS DOWN:

  - the IW panel used the per-step MEAN of exp(student_logp - ref_logp). One sample with a
    log-ratio of 20 contributes 4.8e8 and owns the mean; miro's panel was a single spike at
    4e12 with every other step on the floor, and the same pathology printed
    "iw_mean 637,746,287" in dense's summary. It is now the MEDIAN, on a log axis.
  - the suptitle read "dry-run mock => flat is expected", left from when this script only
    ever ran on the CPU smoke. On a figure of a real 150-step arm that sentence tells the
    reader to dismiss the result the figure exists to show.

WHY THE ASSERTIONS GO THROUGH A RENDERED FIGURE. Testing per_step alone would leave the
figure free to call _series again -- which is precisely the bug. These tests run main(),
reopen the saved Figure objects, and count the vertices matplotlib actually received. Delete
the aggregation and the vertex count jumps from n_steps to n_rows, in red.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import pytest                                                     # noqa: E402

import scripts.plot_diagnostics as P                              # noqa: E402

ROWS_PER_STEP = 8          # production is 256; 8 is enough to tell a row count from a step
N_STEPS = 12


def _row(step, i):
    """One diagnostics row, shaped like the real writer's.

    `answered` and `is_truncated` alternate per sample inside a step: that is what makes a
    per-sample plot a solid block and a per-step plot a line at 0.5.
    """
    return {
        "step": step,
        "task": {"answered": float(i % 2), "em": 0.0, "cover_em": float(i % 2), "f1": 0.5},
        "retrieval": {"per_turn_hit_rate": 0.4, "union_hit": float(i % 2),
                      "num_search_calls": 2},
        # one sample per step carries a huge weight -- the mean-of-exponentials trap
        "kl_importance": {"iw_mean": 1e9 if i == 0 else 0.9, "kl_pertok_mean": 0.2},
        "termination": {"is_truncated": float(i % 2)},
        "policy": {"staleness": 0},
    }


@pytest.fixture
def figure(tmp_path, monkeypatch):
    """Render the real figure from a real jsonl pair and hand back the live Figure."""
    on = tmp_path / "opd_x_online" / "diag_online.jsonl"
    off = tmp_path / "opd_x_offline" / "diag_offline.jsonl"
    for path in (on, off):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in range(N_STEPS):
                for i in range(ROWS_PER_STEP):
                    f.write(json.dumps(_row(s, i)) + "\n")
    out = tmp_path / "fig.png"
    plt.close("all")
    monkeypatch.setattr(sys, "argv",
                        ["plot_diagnostics.py", "--online", str(on), "--offline", str(off),
                         "--out", str(out)])
    P.main()
    assert out.exists(), "main() did not write a figure"
    figs = [plt.figure(n) for n in plt.get_fignums()]
    assert len(figs) == 1
    yield figs[0]
    plt.close("all")


def _curves(fig):
    """Every plotted Line2D that carries a label, excluding the axhline/axhspan guides."""
    out = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            lab = ln.get_label()
            if lab and not lab.startswith("_") and len(ln.get_xdata()) > 1:
                out.append((ax, lab, ln))
    return out


# --------------------------------------------------------------------------- THE regression
def test_no_curve_carries_more_points_than_there_are_steps(figure):
    """38,400 markers on a 150-step axis is what made the production figures a solid block."""
    n_rows = N_STEPS * ROWS_PER_STEP
    for ax, lab, ln in _curves(figure):
        n = len(ln.get_xdata())
        assert n != n_rows, (
            f"panel {ax.get_title()!r} curve {lab!r} has one point per ROW ({n}); "
            f"the writer emits {ROWS_PER_STEP} rows per step")
        assert n <= N_STEPS, f"{lab!r} has {n} points for {N_STEPS} steps"


def test_a_per_sample_binary_field_becomes_its_per_step_rate(figure):
    """`answered` alternates 0/1 within each step, so the aggregated curve must sit at 0.5.
    A per-sample plot would show only the values 0 and 1 and never 0.5 -- and this is the
    assertion that goes red if the mean is replaced by "take the first row of the step"."""
    got = [ln for _, lab, ln in _curves(figure) if lab == "online answered"]
    assert got, "the task panel no longer draws the answered rate"
    assert list(got[0].get_ydata()) == pytest.approx([0.5] * N_STEPS)


def test_every_step_present_in_the_file_appears_on_the_axis(figure):
    """Aggregation must not silently drop steps -- a curve of 3 points would also pass a
    'fewer points than rows' check."""
    got = [ln for _, lab, ln in _curves(figure) if lab == "online answered"]
    assert list(got[0].get_xdata()) == list(range(N_STEPS))


# --------------------------------------------------------------------------- the IW panel
def _iw_panel(fig):
    for ax in fig.axes:
        if "importance weight" in ax.get_title():
            return ax
    pytest.fail("the importance-weight panel is gone")


def test_the_importance_weight_is_a_median_so_one_outlier_cannot_own_it(figure):
    """Every step here contains one sample at 1e9 and seven at 0.9. The mean is 1.25e8; the
    median is 0.9. The production figure showed the former as a 4e12 spike."""
    ys = [ln.get_ydata() for ln in _iw_panel(figure).get_lines()
          if ln.get_label() == "online IW median"]
    assert ys, "the IW curve is no longer labelled as a median"
    assert list(ys[0]) == pytest.approx([0.9] * N_STEPS)


def test_the_importance_weight_panel_uses_a_log_axis(figure):
    """iw spans orders of magnitude across arms; on a linear axis the ~0.94 Lightning
    reference band and anything above it cannot be read on the same plot."""
    assert _iw_panel(figure).get_yscale() == "log"


# --------------------------------------------------------------------------- the title
def test_the_title_no_longer_calls_real_data_a_dry_run_mock(figure):
    t = figure.get_suptitle()
    assert "dry-run mock" not in t and "flat is expected" not in t


def test_the_title_names_both_arms_and_how_far_each_got(figure):
    """A figure that does not say which arm it is, or that one arm stopped early, is a figure
    that gets compared to the wrong thing -- dense's online arm reached 145/150 and its
    offline arm 150/150."""
    t = figure.get_suptitle()
    assert "opd_x_online" in t and "opd_x_offline" in t
    assert f"({N_STEPS} steps)" in t


# --------------------------------------------------------------------------- per_step itself
def test_per_step_median_is_the_middle_value_not_the_mean():
    recs = [{"step": 0, "k": {"v": v}} for v in (1.0, 2.0, 300.0)]
    assert P.per_step(recs, ("k", "v"), agg="median") == ([0], [2.0])


def test_per_step_median_of_an_even_count_averages_the_middle_pair():
    recs = [{"step": 0, "k": {"v": v}} for v in (1.0, 2.0, 4.0, 300.0)]
    assert P.per_step(recs, ("k", "v"), agg="median")[1] == [3.0]


def test_per_step_honours_max_step_so_arms_of_different_length_compare():
    """Same distinction as binned(): --steps sets the axis, --max-step DROPS rows. moe's
    online arm died at 57 while its offline arm ran to 150."""
    recs = [{"step": s, "k": {"v": 1.0}} for s in range(10)]
    assert P.per_step(recs, ("k", "v"), max_step=4)[0] == [0, 1, 2, 3, 4]


def test_per_step_skips_rows_missing_the_field_without_shifting_the_axis():
    """A row whose diagnostic was not computed must not be read as a zero."""
    recs = [{"step": 0, "k": {"v": 1.0}}, {"step": 0}, {"step": 1, "k": {"v": 3.0}}]
    assert P.per_step(recs, ("k", "v")) == ([0, 1], [1.0, 3.0])
