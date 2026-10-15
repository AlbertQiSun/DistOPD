"""The teacher HEAD at one position: what /score_head returns and what the delta target reads.

Shared by slurm/teacher_server.py (producer, vLLM prompt_logprobs=k_wide entries) and
scripts/teacher_head_dump.py (consumer), so the selection rule lives in exactly one place and
is unit-tested on synthetic rows.

Per position the server holds vLLM's entry {token_id: Logprob} -- the top-k_wide tokens plus
the realised token -- and the client's `extra_ids` (S \\ K_q candidates: pi_ref's top-K' and y).
    K_q     the smallest prefix of the teacher's ranking whose mass reaches `cover`, capped at
            `k_cap`; both id and fp32 log q are kept.
    extras  log q at every extra id NOT already in K_q; None when the id is outside the
            top-k_wide the engine returned -- then all that is known is log q < floor_lp.
    floor   the smallest log-prob in the entry (the k_wide-th): the upper bound on any unknown
    lp_y    log q at the realised token (always in the entry: the alignment gate's number)
"""
from __future__ import annotations

import math

KQ_COVER = 0.99
KQ_CAP = 32


def _lp(v):
    return float(getattr(v, "logprob", v))


def select_kq(sorted_lps, cover=KQ_COVER, k_cap=KQ_CAP):
    """|K_q| for log-probs sorted descending: smallest k with cumulative mass >= cover, capped."""
    acc, k = 0.0, 0
    for lp in sorted_lps:
        k += 1; acc += math.exp(lp)
        if acc >= cover or k >= k_cap:
            break
    return max(k, 1) if sorted_lps else 0


def head_from_entry(entry, y, extra_ids, k_cap=KQ_CAP, cover=KQ_COVER):
    """entry: {token_id: Logprob-or-float} for ONE position. Returns a JSON-ready dict."""
    if y not in entry:
        raise ValueError(f"realised token {y} missing from the engine's entry")
    items = sorted(((int(t), _lp(v)) for t, v in entry.items()), key=lambda kv: -kv[1])
    lps = [lp for _, lp in items]
    kq = select_kq(lps, cover, k_cap)
    top = items[:kq]
    top_set = {t for t, _ in top}
    lookup = dict(items)
    ex_ids, ex_lp, seen = [], [], set()
    for t in extra_ids:
        t = int(t)
        if t in top_set or t in seen:                            # K' ∪ {y} arrives with y possibly twice:
            continue                                             # a duplicate would be counted twice in the residual
        seen.add(t); ex_ids.append(t); ex_lp.append(lookup.get(t))   # None = outside the wide list
    return dict(kq=kq, top_ids=[t for t, _ in top], top_lp=[lp for _, lp in top],
                extra_ids=ex_ids, extra_lp=ex_lp, lp_y=lookup[int(y)],
                floor_lp=lps[-1], n_entry=len(items), cover_reached=bool(sum(math.exp(lp) for lp in lps[:kq]) >= cover))


def residual_log_mass(top_lp, extra_lp):
    """log(1 - sum_S q) with unknown extras counted as 0 (an UPPER bound on the residual);
    clamped at log(1e-12) so a fully-covered position stays finite."""
    s = sum(math.exp(v) for v in top_lp) + sum(math.exp(v) for v in extra_lp if v is not None)
    return math.log(max(1.0 - s, 1e-12))
