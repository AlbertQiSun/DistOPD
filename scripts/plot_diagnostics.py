"""Turn diag_{online,offline}.jsonl into the confound-diagnostic figures.

Each panel is tied to the hypothesis it discriminates (see opd/diagnostics.py):

  1. answered rate + cover_em vs step      -- TASK PERFORMANCE headline. `answered` is
     whether the policy emitted <answer> at all, and it is the metric that actually moves:
     miro's online arm falls 0.974 -> 0.923 over 150 steps while cover_em RISES. Strict EM
     (what this panel used to draw) is floored by verbosity and shows nothing.
  2. per-hop retrieval hit rate vs step    -- CONTEXT QUALITY (confound #2), now a
     SUPPORTING signal not the task headline. (union hit overlaid, dashed, as the
     task-success proxy that would MISS progressive per-hop rot.)
  3. importance weight pi_theta/pi_ref     -- Lightning OPD Fig 3a analog (#2). Their
     static setting sits ~0.94 mean, std<0.1 (drawn as a reference band); if agentic
     online blows past it, that is "interactive breaks offline-stays-on-policy". Plotted
     as the per-step MEDIAN on a log axis: iw is exp(logp difference), so the mean is a
     mean of exponentials and one sample owns it (dense bin 0 read 637,746,287).
  4. termination / truncation rate vs step -- diverge-vs-truncation split (#2).
  5. boundary-bucketed clip rate           -- teacher miscalibration right after
     retrieval injection (#3); x = action-token offset AFTER the observation edge.
  6. online on-policy sanity: staleness    -- MUST be 0 every online step (resync bug
     guard); a nonzero line means "online is secretly stale", discard the run.

EVERY PANEL AGGREGATES PER STEP before drawing (see per_step). The diagnostics file holds
global_batch = 256 rows per step, so a 150-step arm is 38,400 rows; drawing one marker each
turned every 0/1 panel into a solid block and made the figures unreadable.

TWO PANEL SETS, chosen with --panels:

  confound  (default) the six above -- the PRE-REGISTERED hypothesis discriminators.
  mechanism            EM / F1 / truncation rate / search calls per trajectory /
                       trajectory length / importance-weight median. This is the set that
                       shows HOW the online arm degrades rather than WHETHER: truncation
                       0.02 -> 0.23, trajectory length 2000 -> 2360 tokens, iw median
                       1 -> 20, all on the same step axis, with search calls and EM/F1
                       drifting DOWN alongside. Confound figure answers "is it real";
                       mechanism figure answers "what is it doing".

--smooth W draws the raw per-step series faint and a centred rolling mean of width W heavy
on top (default 9, 0 disables). Per-step means over 256 samples still carry visible sampling
noise -- EM swings +-0.05 step to step -- and a trend of 0.02/150 steps is invisible under it.
The raw line is kept underneath so the smoothing can never hide the spread it is averaging.

matplotlib is optional; if missing the script prints a clear install hint and exits 0
(so it never breaks a headless HPC job that only needed the jsonl).

    python scripts/plot_diagnostics.py --run-dir runs
    python scripts/plot_diagnostics.py --panels mechanism --online A --offline B --out f.png
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _series(records, *keys):
    """Extract (steps, values) for a nested key path, skipping null values."""
    xs, ys = [], []
    for r in records:
        v = r
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            xs.append(r["step"])
            ys.append(v)
    return xs, ys


def _reduce(vals, agg: str) -> float:
    """mean / median / p90 of a bin's samples.

    WHY MORE THAN A MEAN. iw is exp(student_logp - ref_logp). The dense arm's first bin
    printed `iw_mean 637746287.4441` -- not a policy that drifted by nine orders of magnitude
    but a handful of samples with large log-ratios, since e^20 = 4.8e8 and the mean of 6400
    numbers is owned by any one of them that size. A mean of exponentials is not a summary of
    the distribution, and the number it produced was unreadable as a curve or as a scalar.
    The median says where the bulk sits; p90 says whether the tail is growing, which is the
    part that actually matters for clipping.
    """
    v = sorted(vals)
    n = len(v)
    if agg == "median":
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    if agg == "p90":
        return v[min(n - 1, int(0.9 * (n - 1) + 0.5))]
    return sum(v) / n


def binned(records, keys, bins: int, total_steps: int, max_step: int | None = None,
           agg: str = "mean"):
    """One statistic of a nested series per equal-width STEP bin -> [(lo, hi, val, n), ...].

    Binned on the STEP axis rather than on record order, because there are global_batch rows
    per step and a run can stop early: ordering by position would put a 150-step arm and a
    38-step arm on incomparable x-axes, which is exactly the comparison being made.

    A bin with no rows yields mean None rather than being dropped, so two arms always line up
    row for row in the printed table.
    """
    xs, ys = _series(records, *keys)
    width = max(1, -(-total_steps // bins))          # ceil, so the last bin closes at total
    acc = defaultdict(list)
    for x, y in zip(xs, ys):
        # max_step DROPS rows; total_steps only sets the axis. The distinction is the whole
        # point of a matched-window comparison: reading a 57-step online arm against a
        # 150-step offline one with --steps 57 alone folded offline's steps 57..149 into the
        # last bin, so "offline kl_pertok 1.2503 at steps 38-56" was really the mean over
        # 38-149. Flat series barely move; anything that drifts is read completely wrong.
        if max_step is not None and int(x) > max_step:
            continue
        acc[min(int(x) // width, bins - 1)].append(float(y))
    out = []
    for b in range(bins):
        lo, hi = b * width, min((b + 1) * width - 1, total_steps - 1)
        vals = acc.get(b, [])
        out.append((lo, hi, _reduce(vals, agg) if vals else None, len(vals)))
    return out


def per_step(records, keys, agg: str = "mean", max_step: int | None = None):
    """One point per TRAINING STEP, not one per sample.

    WHY THE FIGURES WERE UNREADABLE WITHOUT THIS. _series returns a point for every ROW, and
    there are global_batch = 256 rows per step. A 150-step arm therefore drew 38,400 markers,
    256 of them stacked on each x -- and since `answered`, `em` and `is_truncated` are per-
    sample 0/1, every panel came out a solid block of colour spanning the whole y range. The
    text summary was legible for exactly this reason: it aggregated first.

    `agg="median"` exists for the importance weight. iw is exp(student_logp - ref_logp), so a
    single sample with a log-ratio of 20 contributes e^20 = 4.8e8 and the MEAN becomes that
    sample. One such outlier flattened the whole IW panel to a spike at 4e12, and it is the
    same pathology that made "iw_mean 637,746,287" appear in the dense summary.
    """
    xs, ys = _series(records, *keys)
    acc = defaultdict(list)
    for x, y in zip(xs, ys):
        if max_step is not None and int(x) > max_step:
            continue
        acc[int(x)].append(float(y))
    steps = sorted(acc)
    return steps, [_reduce(acc[s], agg) for s in steps]


def smooth(ys, w: int):
    """Centred rolling mean of width `w`, with the window SHRINKING at the edges.

    Shrinking rather than padding or emitting NaN: a padded edge invents data (it pulls the
    first points toward whatever the pad value is) and a NaN edge amputates the first and
    last w//2 steps -- which on a 150-step arm is the first 4 and the last 4, i.e. exactly
    the region a "did it drift by the end" reading depends on.

    w <= 1 returns the input unchanged, so --smooth 0 is a real off switch.
    """
    if w <= 1 or not ys:
        return list(ys)
    h = w // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - h), min(len(ys), i + h + 1)
        seg = ys[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


ARM_COLOR = {"online": "tab:blue", "offline": "tab:orange"}


def draw(ax, xs, ys, label, arm, w: int, ls: str = "-", lw: float = 2.0):
    """Raw series faint, smoothed series heavy, both in the arm's colour.

    Drawing ONLY the smoothed line would be a quiet lie: a rolling mean of width 9 over a
    series whose step-to-step spread is +-0.05 renders a 0.02 trend as a confident curve and
    shows nothing of the spread it came from. Keeping the raw line underneath makes the
    signal-to-noise visible in the figure itself rather than in a caption.
    """
    if not xs:
        return
    c = ARM_COLOR.get(arm, None)
    if w > 1:
        ax.plot(xs, ys, ls, lw=0.7, alpha=0.35, color=c)
        ax.plot(xs, smooth(ys, w), ls, lw=lw, color=c, label=label)
    else:
        ax.plot(xs, ys, ls, lw=max(1.2, lw * 0.7), color=c, label=label)


# ---------------------------------------------------------------------------
# MECHANISM panel set: not "is the damage real" but "what does it consist of".
# ---------------------------------------------------------------------------
# The fields are the ones opd/diagnostics.termination_record actually writes --
# is_truncated, num_search_calls and length all live under `termination`, NOT under
# `retrieval`. SUMMARY_ROWS asked for ("retrieval", "num_search_calls") for weeks and the
# row silently never printed, because print_summary drops a row whose bins are all None.
MECHANISM_PANELS = [
    ("answer EM (per-step mean)", ("task", "em"), "mean", False),
    ("answer F1 (per-step mean)", ("task", "f1"), "mean", False),
    ("truncation rate", ("termination", "is_truncated"), "mean", False),
    ("search calls per trajectory", ("termination", "num_search_calls"), "mean", False),
    ("trajectory length (tokens)", ("termination", "length"), "mean", False),
    ("importance-weight median (log)", ("kl_importance", "iw_mean"), "median", True),
]


SUMMARY_ROWS = [
    ("answered", ("task", "answered"), "did it emit <answer> at all -- THE headline", "mean"),
    ("em", ("task", "em"), "strict exact match", "mean"),
    ("cover_em", ("task", "cover_em"), "gold appears in the answer span", "mean"),
    ("f1", ("task", "f1"), "token F1", "mean"),
    ("union_hit", ("retrieval", "union_hit"), "gold retrieved at least once", "mean"),
    ("per_turn_hit", ("retrieval", "per_turn_hit_rate"), "per-hop retrieval quality", "mean"),
    # iw is REPORTED AS MEDIAN + p90, never as a mean -- see _reduce. The mean read
    # 637,746,287 on dense's first bin, which is one sample, not a drifting policy.
    ("iw_median", ("kl_importance", "iw_mean"), "pi_theta/pi_ref bulk; Lightning ~0.94",
     "median"),
    ("iw_p90", ("kl_importance", "iw_mean"), "same series, tail -- is clipping growing?",
     "p90"),
    ("kl_pertok", ("kl_importance", "kl_pertok_mean"), "drift from pi_ref", "mean"),
    # ("retrieval", "num_search_calls") for weeks -- termination_record writes it, so the
    # row was all-None and print_summary dropped it without a word.
    ("n_searches", ("termination", "num_search_calls"), "searches per trajectory", "mean"),
    ("traj_len", ("termination", "length"), "trajectory length in tokens", "mean"),
    ("truncated", ("termination", "is_truncated"), "hit the length ceiling", "mean"),
]


def print_summary(online, offline, bins: int, total_steps: int,
                  max_step: int | None = None) -> None:
    """The curve as numbers. Figures are for the paper; this is for reading over ssh."""
    window = "" if max_step is None else f"; rows beyond step {max_step} DROPPED"
    print(f"steps binned into {bins} ranges of {max(1, -(-total_steps // bins))}; "
          f"online rows={len(online)} offline rows={len(offline)}{window}")
    print("(offline replays a FROZEN cache, so a flat offline line is the DESIGN, not a "
          "result. What carries information is online's shape, and the online-offline gap.)")
    for name, keys, why, agg in SUMMARY_ROWS:
        on = binned(online, keys, bins, total_steps, max_step, agg)
        off = binned(offline, keys, bins, total_steps, max_step, agg)
        if all(m is None for _, _, m, _ in on) and all(m is None for _, _, m, _ in off):
            continue
        print()
        print(f"{name}   -- {why}")
        # Width 9, not 7: traj_len runs to 2303.3036 and iw_p90 to 198.9554, and a value
        # wider than its field pushes every later column right -- so the row that most needs
        # to be read against the header is the one that stops lining up with it.
        head = "  ".join(f"{f'{lo}-{hi}':^9}" for lo, hi, _, _ in on)
        print(f"    step      {head}")
        for label, series in (("online ", on), ("offline", off)):
            cells = "  ".join("    --   " if m is None else f"{m:9.4f}" for _, _, m, _ in series)
            print(f"    {label}   {cells}")
        first = next((m for _, _, m, _ in on if m is not None), None)
        last = next((m for _, _, m, _ in reversed(on) if m is not None), None)
        if first is not None and last is not None:
            print(f"    online first->last bin: {first:.4f} -> {last:.4f} "
                  f"(delta {last - first:+.4f})")


def arm_label(path: str) -> str:
    """`runs/opd_miro_online/diag_online.jsonl` -> `opd_miro_online`. A figure that does not
    say which arm it is gets compared to the wrong one."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) or os.path.basename(path)


def line_name(on_path: str, off_path: str) -> str:
    """The teacher line shared by the two arms -- `miro`, `dense`, `moe` -- or "" if the two
    paths do not agree on one. Never guessed: three lines run concurrently and a mislabelled
    figure is worse than an unlabelled one."""
    def mid(p):
        parts = arm_label(p).split("_")
        return parts[1] if len(parts) >= 3 and parts[0] == "opd" else None

    a, b = mid(on_path), mid(off_path)
    return a if a and a == b else ""


def figure_title(on_path, off_path, n_on, n_off, args) -> str:
    if args.title:
        return args.title
    line = line_name(on_path, off_path)
    head = f"Row 10 {line} REAL" if line else "Row 10 REAL"
    window = "" if args.max_step is None else f", steps > {args.max_step} dropped"
    sm = f"w={args.smooth} smoothing" if args.smooth > 1 else "no smoothing"
    # When both arms sit in ONE directory (--run-dir runs) the per-arm label is the same
    # string for both, and printing it twice reads as "online runs vs offline runs". Naming
    # the directory adds nothing there; naming it is only useful when the arms differ.
    a, b = arm_label(on_path), arm_label(off_path)
    on_tag = f"online {a} " if a != b else "online "
    off_tag = f"offline {b} " if a != b else "offline "
    return (f"{head}: {on_tag}({n_on} steps) vs {off_tag}({n_off} steps), "
            f"per-step means, {sm}{window}")


def mechanism_figure(plt, online, offline, args):
    """EM / F1 / truncation / search calls / trajectory length / importance weight.

    WHAT THIS SET IS FOR. The confound figure answers "is the online degradation real and not
    a retrieval or measurement artefact". It cannot answer "what is the degradation MADE OF",
    because the quantities that move are not among its panels. Put on one step axis, these six
    say it in one look: the online arm's trajectories get LONGER, hit the length ceiling more
    often, issue FEWER searches, and drift further from pi_ref -- while EM and F1 sag. That is
    a policy learning to spend its budget on text instead of on terminating, which is a
    different failure from "the retrieved context rotted".
    """
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    flat = [ax for row in axes for ax in row]
    for ax, (title, keys, agg, logy) in zip(flat, MECHANISM_PANELS):
        drew = False
        for recs, arm in ((online, "online"), (offline, "offline")):
            xs, ys = per_step(recs, keys, agg=agg, max_step=args.max_step)
            if xs:
                lab = f"{arm} {keys[-1]}(med)" if agg == "median" else arm
                draw(ax, xs, ys, lab, arm, args.smooth)
                drew = True
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("training step")
        if drew:
            ax.legend(fontsize=9)
        else:
            # SAY SO rather than leave a blank frame. An empty panel and a flat panel look
            # alike at a glance, and one of them means the field was never written.
            ax.text(0.5, 0.5, f"no {'.'.join(keys)} in either arm", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="crimson")
    n_on = _n_steps(online, args.max_step)
    n_off = _n_steps(offline, args.max_step)
    fig.suptitle(figure_title(args.online_path, args.offline_path, n_on, n_off, args),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _n_steps(recs, max_step):
    return len({r["step"] for r in recs
                if max_step is None or int(r["step"]) <= max_step})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--online", default=None,
                    help="explicit diag_online.jsonl; the three lines keep their arms in "
                         "SEPARATE directories (runs/opd_<line>_{online,offline}), so a "
                         "single --run-dir cannot name both")
    ap.add_argument("--offline", default=None, help="explicit diag_offline.jsonl")
    ap.add_argument("--summary", action="store_true",
                    help="print the curves as numbers instead of rendering a figure")
    ap.add_argument("--bins", type=int, default=6, help="step bins for --summary (default 6)")
    ap.add_argument("--steps", type=int, default=150, help="total steps the arms declare")
    ap.add_argument("--max-step", type=int, default=None,
                    help="DROP rows past this step. Use it to compare arms that got different "
                         "distances: --steps N alone only sets the axis, and folds a longer "
                         "arm's tail into the last bin, which silently mislabels it.")
    ap.add_argument("--panels", choices=("confound", "mechanism"), default="confound",
                    help="confound = the pre-registered hypothesis discriminators; "
                         "mechanism = EM/F1/truncation/search calls/trajectory length/iw")
    ap.add_argument("--smooth", type=int, default=9,
                    help="width of the centred rolling mean drawn over the raw per-step "
                         "series (default 9; 0 or 1 disables). The raw series stays visible "
                         "underneath so the smoothing cannot hide the spread.")
    ap.add_argument("--title", default=None, help="override the figure suptitle")
    ap.add_argument("--out", default=None, help="output png (default: <run-dir>/diagnostics.png)")
    args = ap.parse_args()

    on_path = args.online or os.path.join(args.run_dir, "diag_online.jsonl")
    off_path = args.offline or os.path.join(args.run_dir, "diag_offline.jsonl")
    args.online_path, args.offline_path = on_path, off_path

    if args.summary:
        online, offline = _load(on_path), _load(off_path)
        if not online and not offline:
            print(f"no diagnostics found at {on_path} or {off_path}")
            return
        print_summary(online, offline, args.bins, args.steps, args.max_step)
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; the jsonl is still valid. "
              "pip install matplotlib to render figures.")
        return

    online = _load(on_path)
    offline = _load(off_path)
    if not online and not offline:
        print(f"no diag_*.jsonl found in {args.run_dir}; run scripts/dry_run_train.py first.")
        return

    if args.panels == "mechanism":
        fig = mechanism_figure(plt, online, offline, args)
        out = args.out or os.path.join(args.run_dir, "diagnostics.png")
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")
        return

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # (1) TASK PERFORMANCE headline: answer EM (strict) + F1 (lenient), together
    ax = axes[0][0]
    for recs, arm in ((online, "online"), (offline, "offline")):
        xs, ys = per_step(recs, ("task", "answered"), max_step=args.max_step)
        draw(ax, xs, ys, f"{arm} answered", arm, args.smooth)
        xs, ys = per_step(recs, ("task", "cover_em"), max_step=args.max_step)
        draw(ax, xs, ys, f"{arm} cover_em", arm, args.smooth, ls="--", lw=1.2)
    # `answered` is promoted to the headline and drawn heavy: across all three lines the
    # damage appears ONLY in whether the policy still emits <answer>, while cover_em rises.
    # Plotting EM alone -- as this panel used to -- shows the metric that does not move.
    ax.set_title("Task: answered rate (heavy) + cover_em (light)")
    ax.set_xlabel("training step"); ax.set_ylabel("score"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # (2) per-hop retrieval hit rate -- context-quality (supporting)
    ax = axes[0][1]
    for recs, arm in ((online, "online"), (offline, "offline")):
        xs, ys = per_step(recs, ("retrieval", "per_turn_hit_rate"), max_step=args.max_step)
        draw(ax, xs, ys, f"{arm} per-hop", arm, args.smooth)
    # overlay union (task-success proxy) faint, to make the smoothing hazard visible
    xs, ys = per_step(online, ("retrieval", "union_hit"), max_step=args.max_step)
    draw(ax, xs, ys, "online union (proxy)", "online", args.smooth, ls=":", lw=1.2)
    ax.set_title("Context quality: per-hop retrieval hit rate")
    ax.set_xlabel("training step"); ax.set_ylabel("hit rate"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # (3) importance weight -- Lightning OPD Fig 3a analog
    ax = axes[1][0]
    ax.axhspan(0.94 - 0.1, 0.94 + 0.1, color="green", alpha=0.12,
               label="Lightning OPD static band (~0.94, std<0.1)")
    ax.axhline(0.94, color="green", alpha=0.4, lw=1)
    for recs, arm in ((online, "online"), (offline, "offline")):
        xs, ys = per_step(recs, ("kl_importance", "iw_mean"), agg="median",
                          max_step=args.max_step)
        draw(ax, xs, ys, f"{arm} IW median", arm, args.smooth)
    # MEDIAN, and a log axis. iw is exp(student_logp - ref_logp): one sample at a log-ratio
    # of 20 contributes 4.8e8, and the per-step MEAN becomes that one sample -- which is how
    # this panel came out as a single spike at 4e12 with everything else on the floor.
    ax.set_yscale("log")
    ax.set_title("Policy drift: importance weight pi_theta/pi_ref (MEDIAN, log axis)")
    ax.set_xlabel("training step"); ax.set_ylabel("median IW"); ax.legend(fontsize=8)

    # (4) truncation rate -- diverge vs truncation
    ax = axes[1][1]
    for recs, arm in ((online, "online"), (offline, "offline")):
        xs, ys = per_step(recs, ("termination", "is_truncated"), max_step=args.max_step)
        draw(ax, xs, ys, f"{arm} truncated", arm, args.smooth)
    ax.set_title("Termination: truncation RATE vs step")
    ax.set_xlabel("training step"); ax.set_ylabel("fraction truncated")
    ax.legend(fontsize=8)

    # (5) boundary-bucketed clip rate -- teacher miscalibration after retrieval
    ax = axes[2][0]
    order = ["1", "2", "3", "4", "5", "6+"]
    for recs, label, style in ((online, "online", "-"), (offline, "offline", "--")):
        agg = defaultdict(list)
        for r in recs:
            for k, v in (r.get("boundary") or {}).items():
                agg[k].append(v["clip_rate"])
        xs = [k for k in order if k in agg]
        if xs:
            ax.plot(xs, [sum(agg[k]) / len(agg[k]) for k in xs], style, marker="o",
                    label=f"{label}")
    ax.set_title("Confound #3: clip rate by action offset after obs boundary")
    ax.set_xlabel("action-token offset after observation"); ax.set_ylabel("clip rate")
    ax.legend(fontsize=8)

    # (6) online on-policy sanity: staleness MUST be 0 every online step
    ax = axes[2][1]
    for recs, arm in ((online, "online"), (offline, "offline")):
        xs, ys = per_step(recs, ("policy", "staleness"), max_step=args.max_step)
        draw(ax, xs, ys, f"{arm} staleness", arm, args.smooth)
    ax.axhline(0, color="red", alpha=0.3, lw=1)
    ax.set_title("On-policy sanity: staleness (online MUST be 0; nonzero = resync bug)")
    ax.set_xlabel("training step"); ax.set_ylabel("step - last_sync_step")
    ax.legend(fontsize=8)

    # NAME THE ARMS. The title used to read "dry-run mock => flat is expected", left over
    # from when this only ever ran on the CPU smoke. On a real figure that sentence tells the
    # reader to dismiss the one thing the figure is for.
    n_on = _n_steps(online, args.max_step)
    n_off = _n_steps(offline, args.max_step)
    fig.suptitle(figure_title(on_path, off_path, n_on, n_off, args), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or os.path.join(args.run_dir, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
