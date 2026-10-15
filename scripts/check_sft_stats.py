"""M-SFT Stage 1 HEALTH REPORT (not a gate): read a generated SFT shard's manifest and
report MiroThinker's actual behavior distribution in the OPD env. Zero quality filtering
means this script does NOT decide whether the shard is "good enough" for Stage 2 -- that
judgment is the human's. The ONE thing it still FAILS on is a DEVICE red flag: too many
zero-search trajectories, which means MiroThinker often couldn't search at all
(template / tool-protocol broken), not that the data is low quality.

WHY A REPORT, NOT A GATE
------------------------
SFT takes the teacher distribution as-is (any quality filter re-introduces sigma_Delta).
So a low productive-multihop fraction, a high spinning ratio, or "fewer kept than target"
are NOT failures -- they are the teacher's real behavior, and the response is a HUMAN
decision about the INPUTS, not an automated drop:
  - productive_multihop low / spinning high => questions too hard OR retriever too weak
        -> change the question mix / swap the index / retune top_k (NOT tighten a threshold)
  - retrieval_empty high                     => retriever returning nothing
        -> fix the index / retriever endpoint
  - kept < target                            => generate more questions
The productive-vs-spinning ratio among high-hop trajectories is still the diagnostic the
user asked for; it is printed, not enforced.

THE ONE RED FLAG THAT FAILS: no_search_frac
-------------------------------------------
A single no-search trajectory is KEPT (MiroThinker judged it didn't need to search -- real
behavior). But if a LARGE FRACTION of trajectories never searched, that is a DEVICE failure
(agent can't invoke the tool) -- a structural problem to fix in env/prompt BEFORE Stage 2.
That is why no_search_frac (and only it) can FAIL. Note a high no_search_frac has TWO
possible causes -- (a) env/tool broken, or (b) the question mix is trivially easy so
MiroThinker legitimately doesn't search -- so on a red flag, INSPECT (the printed hop
distribution tells them apart: env-broken collapses ~everything to hop 0), don't just
retune the threshold.

    python scripts/check_sft_stats.py <out_dir> [--floor 0.40] [--target-kept 3000]
                                      [--spin-min-hops 3] [--max-no-search-frac 0.10]

--floor / --target-kept / --spin-min-hops shape the REPORT only; --max-no-search-frac is
the sole FAIL threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _load_manifest(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "manifest.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_gen_max_len(out_dir: str) -> int | None:
    """The max_len the shard was ACTUALLY generated with (datagen persists it in
    stats.json under gen_config.max_len). Returns None if absent (older shard)."""
    path = os.path.join(out_dir, "stats.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            gc = (json.load(f) or {}).get("gen_config") or {}
        v = gc.get("max_len")
        return int(v) if v is not None else None
    except (FileNotFoundError, ValueError, KeyError):
        return None


def _read_gen_action_cap(out_dir: str) -> int | None:
    """The max_action_tokens this shard was produced under (stats.json engine block).
    None for a shard written before the field existed -- reported as unknown, never guessed
    at 512, because "unknown" and "512" are exactly the two things a reader must not
    confuse here."""
    try:
        with open(os.path.join(out_dir, "stats.json"), "r", encoding="utf-8") as f:
            v = ((json.load(f) or {}).get("engine") or {}).get("max_action_tokens")
        return int(v) if v is not None else None
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def _action_cap_pressure(rows: list[dict], cap: int | None) -> dict:
    """How often an action ran out of sampling budget instead of finishing.

    WHY THIS IS PERMANENT REPORTING AND NOT A ONE-OFF SCRIPT. vLLM samples each action with
    stop=["</search>","</answer>"], so a finished action is SHORTER than the cap and a
    starved one has no tag at all -- which rollout files as empty_query, indistinguishable
    from a model that cannot follow the protocol. That is exactly how max_action_tokens=512
    went unnoticed: it clipped 100% of Qwen3-30B-A3B-Thinking's empty_query turns and 78% of
    Qwen3-32B's while never touching MiroThinker, and it read as a teacher-quality difference
    for weeks.

    2560 was chosen (2026-08-21) to clear the most verbose teacher's observed MAX of 2372 --
    but that max came from 222 actions over 64 questions, and a 3000-question shard will run
    a longer tail. So the cap is MONITORED here rather than assumed: every shard states what
    fraction of its actions ended at the ceiling. Counted over ALL generated trajectories,
    kept and dropped alike, because a clipped action is the usual REASON for a drop and
    counting only survivors would hide precisely the cases this exists to find."""
    if cap is None:
        return {"cap": None, "n_actions": 0, "at_cap": 0, "at_cap_frac": None}
    # SPLIT BY WHETHER THE TURN WAS PRODUCTIVE, because hitting the ceiling means opposite
    # things on the two sides and a pooled rate is unreadable. A turn that emitted a query (or
    # the final answer) and still hit the cap is the cap CLIPPING legitimate behaviour -- the
    # length distribution is censored and every percentile beside it is a lie. A turn that
    # emitted nothing and hit the cap is the cap BOUNDING a runaway action, which is the job
    # of a backstop; raising it would only make those longer.
    #
    # Measured 2026-08-22 on runs/prod_moe_sft, which is why the split exists: pooled at-cap
    # was 1.48% and tripped the alarm, but it was 0.04% on query turns (3 of 7933) against
    # 73.3% on no-query turns (176 of 240). The pooled number said "regenerate an 11-hour
    # shard". The split said the cap is working.
    n = at = nq = atq = 0
    for r in rows:
        for t in (r.get("turns_summary") or []):
            v = t.get("n_tok")
            if v is None:
                continue
            hit = int(v) >= cap - 8
            n += 1
            at += hit
            if (t.get("q") or "").strip():
                nq += 1
                atq += hit
        v = r.get("final_action_n_tok")
        if v is not None:
            hit = int(v) >= cap - 8
            n += 1
            at += hit
            nq += 1
            atq += hit
    return {"cap": cap, "n_actions": n, "at_cap": at,
            "at_cap_frac": (at / n) if n else None,
            "n_productive": nq, "at_cap_productive": atq,
            "at_cap_productive_frac": (atq / nq) if nq else None}


def _pctl(xs: list[int], q: float) -> int | None:
    """Nearest-rank percentile of a small integer sample (no numpy dep)."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return int(s[k])


def _token_length_dist(kept: list[dict], *, max_len: int) -> dict:
    """Token-length distribution over KEPT trajectories -- the teacher-complexity vs
    4B-student-capacity readout. n_tokens is the FULL trajectory length (prompt+action+
    observation), n_action_tokens is the supervised span. This is a REPORT: long
    MiroThinker trajectories are a signal the 4B student may struggle to imitate and a
    heads-up for the Stage-2 cutoff_len, NOT a drop reason (nothing is filtered on length).
    """
    tot = [int(r.get("n_tokens") or 0) for r in kept if r.get("n_tokens") is not None]
    act = [int(r.get("n_action_tokens") or 0) for r in kept if r.get("n_action_tokens") is not None]

    def _summary(xs: list[int]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1),
                "p50": _pctl(xs, 0.50), "p90": _pctl(xs, 0.90),
                "p99": _pctl(xs, 0.99), "max": max(xs)}

    # how close the longest trajectories run to the rollout budget (max_len): a high
    # near-cap fraction means MiroThinker is being TRUNCATED, so the shard under-represents
    # its longest reasoning -- a report signal to raise max_len or accept the clip.
    near_cap = sum(1 for x in tot if x >= int(0.98 * max_len))
    return {"max_len": max_len,
            "n_tokens": _summary(tot),
            "n_action_tokens": _summary(act),
            "near_cap": near_cap,
            "near_cap_frac": (near_cap / len(tot)) if tot else 0.0}


def _source_order(tag) -> tuple:
    """Sort key so the per-source block reads easy->hard: 1hop, 2hop, multihop, other
    named sources (alpha), then the untagged/single-source fallback last."""
    order = {"1hop": 0, "2hop": 1, "multihop": 2}
    if tag is None:
        return (9, "")
    return (order.get(str(tag), 3), str(tag))


def _by_source(rows: list[dict]) -> list[dict]:
    """Split the SAME per-source signals the human gate reads across the difficulty layers.
    The aggregate hides the real signal, so compare EACH layer against its MEASURED baseline
    (not an invented threshold). MiroThinker is an exhaustive-confirmation teacher: a low
    natural_frac_of_answered is its real signature, NOT a defect. The alarm is a RELATIVE
    REGRESSION (a layer materially below its measured baseline), never an absolute miss of a
    from-thin-air number. Measured baselines (full regen, n in parens):
      1hop natural_frac_of_answered: ~25% (old prompt) -> ~39% (anti-confirmation, n=1075)
      2hop: ~25% -> ~31% (n=1775)
      multihop: ~15% -> ~19% (n=693)
    The anti-confirmation SYSTEM_PROMPT (fingerprint f9da237f476eb71e) lifted all three; if a
    future regen drops materially BELOW these, THAT is the true alarm. PER-LAYER, report-only."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(r.get("source_tag"), []).append(r)
    out = []
    for tag in sorted(groups, key=_source_order):
        grp = groups[tag]
        n = len(grp)
        kept = [r for r in grp if r.get("kept")]
        n_kept = len(kept)
        natural = sum(1 for r in kept if not r.get("forced_answer_fired"))
        prefill_rescued = sum(1 for r in grp
                              if r.get("forced_answer_fired") and r.get("forced_answer_answered"))
        n_think = sum(1 for r in grp if r.get("has_think"))
        prod_mh = sum(1 for r in kept if r.get("productive_multihop") is True)
        out.append({
            "source_tag": tag if tag is not None else "(untagged)",
            "n_generated": n,
            "answered": n_kept,
            "answered_frac": (n_kept / n) if n else 0.0,
            "natural": natural,
            "natural_frac_of_answered": (natural / n_kept) if n_kept else None,
            "prefill_rescued": prefill_rescued,
            "think_frac": (n_think / n) if n else 0.0,
            "productive_multihop": prod_mh,
            "productive_multihop_frac": (prod_mh / n_kept) if n_kept else 0.0,
        })
    return out


def analyze(rows: list[dict], *, floor: float, target_kept: int,
            spin_min_hops: int, max_no_search_frac: float, max_len: int = 4096,
            action_cap_tokens: int | None = None) -> dict:
    total = len(rows)
    kept = [r for r in rows if r.get("kept")]
    n_kept = len(kept)

    # drop breakdown (over ALL generated). With zero quality filtering this only ever
    # contains STRUCTURAL reasons: no_action / empty_query / empty_answer.
    drops: dict[str, int] = {}
    for r in rows:
        if not r.get("kept"):
            drops[r.get("reason", "?")] = drops.get(r.get("reason", "?"), 0) + 1

    # productive multi-hop fraction is computed over KEPT trajectories (REPORT ONLY).
    prod_mh = sum(1 for r in kept if r.get("productive_multihop") is True)
    prod_mh_frac = (prod_mh / n_kept) if n_kept else 0.0

    # productive-vs-spinning ratio among HIGH-HOP trajectories, over ALL generated, so the
    # "too hard / retriever weak" signal isn't hidden (REPORT ONLY).
    def _bucket(min_hops: int) -> dict:
        hi = [r for r in rows if int(r.get("hops") or 0) >= min_hops]
        prod = sum(1 for r in hi if r.get("productive") is True)
        spin = sum(1 for r in hi if r.get("productive") is False)
        unk = sum(1 for r in hi if r.get("productive") is None)
        n = len(hi)
        return {"n": n, "productive": prod, "spinning": spin, "unknown_gold": unk,
                "productive_frac": (prod / n) if n else None,
                "spinning_frac": (spin / n) if n else None}

    multihop2 = _bucket(2)
    highhop = _bucket(spin_min_hops)

    # token-length distribution over KEPT (teacher-complexity vs 4B-student-capacity).
    token_lengths = _token_length_dist(kept, max_len=max_len)
    action_cap = _action_cap_pressure(rows, action_cap_tokens)

    # hop distribution over ALL generated (kept + dropped) -- lets a reader tell a device
    # collapse (~everything at hop 0) from a legitimately-easy mix.
    hop_hist: dict[str, int] = {}
    for r in rows:
        h = int(r.get("hops") or 0)
        hop_hist[str(h)] = hop_hist.get(str(h), 0) + 1

    # ---- termination-mode split (post prefilled-<answer> forced seam). ----
    # answered == kept (kept <=> terminated ANSWERED with a non-empty <answer>). Every kept
    # trajectory is either a NATURAL answer (the model emitted <answer> on its own before the
    # budget ran out) or a PREFILL-RESCUED answer (Guard 1 fired: warning + prefilled <answer>
    # open tag, then the model closed it). forced_failed trajectories are DROPPED not_terminated
    # (Guard 1 fired but the model never closed </answer>). Watch the NATURAL fraction, not just
    # answered% total: a healthy shard terminates mostly on its own, with the forced seam as a
    # safety net -- if prefill-rescued dominates, pi_ref is being taught to lean on the crutch.
    natural = sum(1 for r in kept if not r.get("forced_answer_fired"))
    forced_fired = sum(1 for r in rows if r.get("forced_answer_fired"))
    prefill_rescued = sum(1 for r in rows
                          if r.get("forced_answer_fired") and r.get("forced_answer_answered"))
    forced_failed = forced_fired - prefill_rescued
    term_split = {
        "answered": n_kept,
        "answered_frac": (n_kept / total) if total else 0.0,
        "natural": natural,
        "natural_frac": (natural / total) if total else 0.0,
        "natural_frac_of_answered": (natural / n_kept) if n_kept else None,
        "forced_fired": forced_fired,
        "prefill_rescued": prefill_rescued,
        "prefill_rescued_frac": (prefill_rescued / total) if total else 0.0,
        "forced_failed": forced_failed,
    }

    # think-appearance rate: fraction of generated trajectories whose native <think> channel
    # fired in ANY action span. With enable_thinking=True this should be ~1.0; a low value is a
    # regression toward the empty-<think></think> suppression bug (report-only).
    n_think = sum(1 for r in rows if r.get("has_think"))
    think_frac = (n_think / total) if total else 0.0

    # no_search / retrieval_empty are now KEPT (not drops); count them from the row flags.
    no_search = sum(1 for r in rows if r.get("no_search"))
    no_search_frac = (no_search / total) if total else 0.0
    retrieval_empty = sum(1 for r in rows if r.get("retrieval_empty"))
    retrieval_empty_frac = (retrieval_empty / total) if total else 0.0

    # ---- verdict: ONLY the device red flag can FAIL. ----
    reasons_fail = []
    if no_search_frac > max_no_search_frac:
        reasons_fail.append(
            f"no_search {no_search_frac:.2%} > {max_no_search_frac:.0%} "
            "=> MiroThinker often isn't searching (likely template/tool-protocol problem); "
            "INSPECT the hop distribution (device-broken collapses to hop 0) before Stage 2")

    # ---- report-only signals: printed, never fail (thresholds do not coerce the data). ----
    reports = []
    if n_kept < target_kept:
        reports.append(f"kept {n_kept} < target {target_kept} => generate more questions")
    if prod_mh_frac < floor:
        reports.append(
            f"productive_multihop_frac {prod_mh_frac:.2%} < floor {floor:.0%} "
            "=> questions may be too hard / retriever too weak (human call: adjust INPUTS)")
    if highhop["n"] and (highhop["spinning_frac"] or 0.0) > 0.5:
        reports.append(
            f"among hops>={spin_min_hops}, spinning {highhop['spinning_frac']:.2%} > 50% "
            "=> questions too hard OR retriever too weak (human call: adjust INPUTS)")
    if retrieval_empty_frac > 0.10:
        reports.append(
            f"retrieval_empty {retrieval_empty_frac:.2%} > 10% "
            "=> retriever returning no docs (fix index/top_k; not a data-quality drop)")
    if token_lengths["near_cap_frac"] > 0.10:
        reports.append(
            f"near_cap {token_lengths['near_cap_frac']:.2%} of kept run >= 98% of max_len "
            f"({max_len}) => MiroThinker is being truncated; its longest reasoning is "
            "under-represented (raise rollout.max_len or accept the clip -- NOT a drop)")

    return {
        "total_generated": total,
        "kept": n_kept,
        "target_kept": target_kept,
        "dropped": drops,
        "termination_split": term_split,
        "by_source": _by_source(rows),
        "think_appearance": n_think,
        "think_appearance_frac": think_frac,
        "no_search": no_search,
        "no_search_frac": no_search_frac,
        "retrieval_empty": retrieval_empty,
        "retrieval_empty_frac": retrieval_empty_frac,
        "hop_hist": hop_hist,
        "productive_multihop": prod_mh,
        "productive_multihop_frac": prod_mh_frac,
        "floor": floor,
        "token_lengths": token_lengths,
        "action_cap": action_cap,
        "multihop_ge2": multihop2,
        f"highhop_ge{spin_min_hops}": highhop,
        "pass": not reasons_fail,
        "fail_reasons": reasons_fail,
        "reports": reports,
    }


def _print_report(a: dict) -> None:
    print("=" * 70)
    print(f"SFT Stage 1 HEALTH REPORT   ->   {'PASS' if a['pass'] else 'FAIL (device red flag)'}")
    print("=" * 70)
    print(f"generated={a['total_generated']}  kept={a['kept']} (target {a['target_kept']}, "
          "report-only)")
    print(f"dropped (structural only)={a['dropped']}")
    ts = a["termination_split"]
    naf = "None" if ts["natural_frac_of_answered"] is None else f"{ts['natural_frac_of_answered']:.2%}"
    print(f"answered={ts['answered']} ({ts['answered_frac']:.2%})  "
          f"natural={ts['natural']} ({ts['natural_frac']:.2%}; {naf} of answered)  "
          f"prefill_rescued={ts['prefill_rescued']} ({ts['prefill_rescued_frac']:.2%})  "
          f"forced_failed={ts['forced_failed']}")
    print(f"think_appearance={a['think_appearance']} ({a['think_appearance_frac']:.2%})")
    bs = a.get("by_source") or []
    # per-source split is only meaningful for the difficulty-mixed shard; a single-source /
    # dry-run shard collapses to one (untagged) group -> skip the block (the aggregate is it).
    if len(bs) > 1 or (len(bs) == 1 and bs[0]["source_tag"] != "(untagged)"):
        print("-" * 70)
        print("PER-SOURCE (report-only; the human gate is PER-LAYER vs its MEASURED baseline):")
        print("  natural_frac_of_answered baselines (anti-confirmation prompt f9da237f476eb71e):")
        print("  1hop ~39% (was ~25%), 2hop ~31% (was ~25%), multihop ~19% (was ~15%). MiroThinker")
        print("  is exhaustive-confirmation: low natural_frac is its SIGNATURE, not a defect. The")
        print("  alarm is a layer materially BELOW its baseline (regression), NOT any absolute miss.")
        for s in bs:
            naf = ("None" if s["natural_frac_of_answered"] is None
                   else f"{s['natural_frac_of_answered']:.2%}")
            print(f"  [{s['source_tag']}] n={s['n_generated']}  "
                  f"answered={s['answered']} ({s['answered_frac']:.2%})  "
                  f"natural={s['natural']} ({naf} of answered)  "
                  f"prefill_rescued={s['prefill_rescued']}  "
                  f"think={s['think_frac']:.2%}  "
                  f"prod_multihop={s['productive_multihop']} ({s['productive_multihop_frac']:.2%})")
    print(f"hop_hist (all generated)={a['hop_hist']}")
    print(f"no_search={a['no_search']} ({a['no_search_frac']:.2%})  "
          f"retrieval_empty={a['retrieval_empty']} ({a['retrieval_empty_frac']:.2%})")
    print(f"productive_multihop={a['productive_multihop']} "
          f"({a['productive_multihop_frac']:.2%})  floor={a['floor']:.0%} (report-only)")
    tl = a["token_lengths"]
    nt, na = tl["n_tokens"], tl["n_action_tokens"]
    print(f"token_len (kept, teacher-complexity vs 4B capacity; max_len={tl['max_len']}):")
    print(f"  n_tokens        p50={nt['p50']} p90={nt['p90']} p99={nt['p99']} "
          f"max={nt['max']} mean={nt['mean']}")
    print(f"  n_action_tokens p50={na['p50']} p90={na['p90']} p99={na['p99']} "
          f"max={na['max']} mean={na['mean']}")
    print(f"  near_cap (>=98% max_len)={tl['near_cap']} ({tl['near_cap_frac']:.2%})")
    ac = a.get("action_cap") or {}
    if ac.get("at_cap_frac") is None:
        print("  action cap: NOT RECORDED by this shard (predates engine.max_action_tokens). "
              "Whether\n              its empty_query drops are truncation or format cannot "
              "be told from disk.")
    else:
        print(f"  action cap: {ac['at_cap']}/{ac['n_actions']} actions ended AT "
              f"max_action_tokens={ac['cap']} ({ac['at_cap_frac']:.2%})")
        pf = ac.get("at_cap_productive_frac")
        if pf is not None:
            print(f"              of which PRODUCTIVE (emitted a query or the answer): "
                  f"{ac['at_cap_productive']}/{ac['n_productive']} ({pf:.2%})"
                  f"   <- THIS is the one that means clipping")
        if pf is not None and pf > 0.01:
            print("              *** the cap is CLIPPING PRODUCTIVE actions. Their length")
            print("              distribution is censored at the ceiling, and each clipped")
            print("              action loses its closing tag and is dropped empty_query,")
            print("              which reads as a format failure. Re-measure with a larger")
            print("              cap and raise it. ***")
        elif ac["at_cap_frac"] > 0.01:
            print("              (pooled rate is above 1% but the PRODUCTIVE rate is not: the")
            print("              cap is bounding runaway actions, which is its job. Raising it")
            print("              would only make those longer.)")
    mh = a["multihop_ge2"]
    mh_pf = "None" if mh["productive_frac"] is None else f"{mh['productive_frac']:.2%}"
    print(f"hops>=2: n={mh['n']} productive={mh['productive']} spinning={mh['spinning']} "
          f"unknown_gold={mh['unknown_gold']} prod_frac={mh_pf}")
    hi_key = [k for k in a if k.startswith("highhop_ge")][0]
    hi = a[hi_key]
    hi_sf = "None" if hi["spinning_frac"] is None else f"{hi['spinning_frac']:.2%}"
    print(f"{hi_key}: n={hi['n']} productive={hi['productive']} spinning={hi['spinning']} "
          f"spin_frac={hi_sf}")
    if a["reports"]:
        print("-" * 70)
        print("REPORTS (human decides on INPUTS; NOT failures):")
        for r in a["reports"]:
            print(f"  - {r}")
    if not a["pass"]:
        print("-" * 70)
        print("FAIL reasons (device red flag -- fix env/prompt before Stage 2):")
        for r in a["fail_reasons"]:
            print(f"  - {r}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="M-SFT Stage 1 shard HEALTH REPORT "
                                             "(only no_search can FAIL).")
    ap.add_argument("out_dir")
    ap.add_argument("--floor", type=float, default=0.40,
                    help="report threshold: productive-multihop fraction (NOT a gate)")
    ap.add_argument("--target-kept", type=int, default=3000,
                    help="report threshold: min kept trajectories to unblock Stage 2 (NOT a gate)")
    ap.add_argument("--spin-min-hops", type=int, default=3,
                    help="hop count at/above which never-hit is called spinning (report bucket)")
    ap.add_argument("--max-no-search-frac", type=float, default=0.10,
                    help="the ONLY FAIL threshold: red flag if more than this fraction blurted "
                         "with zero search (device/tool-protocol failure)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="rollout.max_len for the near-cap (truncation) report. Default: read "
                         "the ACTUAL value from the shard's stats.json (gen_config.max_len); "
                         "pass this only to override.")
    args = ap.parse_args()

    # single source of truth: prefer the max_len the shard was ACTUALLY generated with
    # (persisted in stats.json by datagen), so the near-cap report can't drift from rollout.
    max_len = args.max_len
    if max_len is None:
        max_len = _read_gen_max_len(args.out_dir)
    if max_len is None:
        max_len = 4096
        print(f"[warn] no gen_config.max_len in {args.out_dir}/stats.json and no --max-len; "
              "falling back to 4096 for the near-cap report (may be stale).")

    rows = _load_manifest(args.out_dir)
    a = analyze(rows, floor=args.floor, target_kept=args.target_kept,
                spin_min_hops=args.spin_min_hops, max_no_search_frac=args.max_no_search_frac,
                max_len=max_len,
                # NOT defaulted to 512 when absent: an old shard reports "not recorded", so a
                # reader can tell "the cap never bit" from "nobody knows what the cap was".
                action_cap_tokens=_read_gen_action_cap(args.out_dir))
    _print_report(a)
    with open(os.path.join(args.out_dir, "check.json"), "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2)
    sys.exit(0 if a["pass"] else 1)


if __name__ == "__main__":
    main()
