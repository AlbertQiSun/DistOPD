"""delta-OPD: the forward KL from a delta-reweighted pi_ref target, on a per-position support
S_t = K' (pi_ref top-K') ∪ K_q (teacher top-K_q) ∪ {y_t} plus one residual bucket B.

THE TARGET, in the delta form the design fixes:
    q_hat(j) ∝ pi_ref(j) · exp(beta · delta_j),      delta_j = log q(j) − log pi_ref(j)  on J, 0 off J
J is where the teacher is KNOWN.  Soft positions (x2 = log pi_ref(rank-2) > −8): the teacher
head (scripts/teacher_head_dump.py) holds log q on all of S and log(1 − Σ_S q), so J = S and
the bucket carries its own delta_B = log q_B − log pi_B.  Hard positions: only log q(y) exists
(the existing precompute), so J = {y}, delta_y is CLIPPED to ±tau exactly as compute_advantage
clips the Lightning advantage, every other token of S and the bucket keep pi_ref (delta = 0).
Off S the target is pi_ref rescaled by the bucket's factor -- which is what makes the loss
computable on S ∪ {B} without the vocabulary: the bucket cell is Σ_{j∉S} of that target.

    loss_t = Σ_{j∈S} q_hat_j (log q_hat_j − log p_θ(j)) + q_hat_B (log q_hat_B − log p_θ(B)),
    log p_θ(j) = logit_j − logsumexp(row),   p_θ(B) = 1 − Σ_S p_θ(j)
averaged over the trajectory's scored positions -- the same per-sequence token mean as
compute_opd_loss, over the SAME mask (asserted, delta_train_step).  beta = 1 reproduces the
teacher on S ∪ {B} at soft positions; beta = 0.5 is the geometric interpolation
pi_ref^0.5 q^0.5 renormalised (tests/test_delta_loss.py::test_4).

WHAT THE STUDENT SUPPLIES.  Not a [N, V] row: opd.masking.head_gathered_logits gives the fp32
logits at the S ids and each row's log-sum-exp from chunk-sized head slabs, exactly like the
existing head_aligned_logprobs path (same pinned chunk, same checkpoint, same autocast).

CACHE PROVENANCE. pi_ref on K_q \\ K' is not stored anywhere (ref_topk keeps top-K'); it is
floored at pi_ref's K'-th value and counted (`pi_floored`).  log q on a K' token outside the
engine's top-k_wide is floored at the k_wide-th value and counted (`q_floored`).  Both only
matter for beta < 1 and for the pi_ref residual, and the report prints how often they fire.

Two module-level hooks exist ONLY so tests/test_delta_loss.py::test_7 can mutate the loss
(drop the bucket; fill with sg(p_θ) instead of pi_ref) and prove the tests that must go red do.

ABLATIONS (MUST_RUN M3 / R21-R23, 2026-09-21) are the same three mutations as first-class,
trainable options, named in ABLATIONS and threaded through DeltaCacheReader (support) and
delta_train_step (include_bucket, anchor):
    teacher_head   S = K_q ∪ {y}: pi_ref's top-K' is dropped from the support (and the client's
                   extras, which ARE that top-K'); hard positions keep S = {y}. The union is the
                   one unprecedented component of the loss, so this is the control the claim
                   "the union, not the teacher's own head, is load-bearing" rests on.
    no_bucket      the residual cell is dropped from the target and the loss (test_7a's mutation).
    no_anchor      where delta = 0 (hard rows off y, and their bucket) the target is filled with
                   sg(p_θ) instead of pi_ref, i.e. no pull back to pi_ref there (test_7b's mutation).
    supp1 / supp2 / supp4   (batch 3, P0-A / P1-1, 2026-09-22) the NESTED support of size n:
                   S = {y} plus pi_ref's top-ranked tokens (K' order, y skipped) until |S| = n --
                   K' is truncated, K_q is NOT unioned in. Everything a truncation drops goes to
                   the bucket: log_pi_rest is summed over the truncated S as always, and log_q_rest
                   is RE-DERIVED from the teacher values known on S (the server's log_rest was the
                   residual of the union set and would leave the dropped tokens' teacher mass out of
                   B). n = 1 is {y} ∪ B, the one-coordinate member of the family (prop:projection):
                   the forward KL on {y, B} pins log p(y) alone. Hard rows: J = {y} as everywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .loss import DEFAULT_TAU
from .masking import build_score_mask, head_gathered_logits, scored_indices

SOFT_X2 = -8.0            # tail_slope's soft-bin rule: log pi_ref(rank-2) > -8
ABLATIONS = {"none":         dict(support="union",   include_bucket=True,  anchor="pi_ref"),
             "teacher_head": dict(support="teacher", include_bucket=True,  anchor="pi_ref"),
             "no_bucket":    dict(support="union",   include_bucket=False, anchor="pi_ref"),
             "no_anchor":    dict(support="union",   include_bucket=True,  anchor="student"),
             "supp1":        dict(support="nested:1", include_bucket=True, anchor="pi_ref"),
             "supp2":        dict(support="nested:2", include_bucket=True, anchor="pi_ref"),
             "supp4":        dict(support="nested:4", include_bucket=True, anchor="pi_ref"),
             # the rho sweep (2026-10-05): teacher-only support at coverage rho -- S = K_q(rho) ∪ {y}, K_q(rho) = the
             # shortest teacher-rank prefix whose mass reaches rho, read off the cached rho = 0.99 head (so rho <= 0.99
             # only); the bucket's teacher mass is re-derived on the truncated S, as for supp<n>
             "rho50":        dict(support="rho:0.50", include_bucket=True, anchor="pi_ref"),
             "rho80":        dict(support="rho:0.80", include_bucket=True, anchor="pi_ref"),
             "rho90":        dict(support="rho:0.90", include_bucket=True, anchor="pi_ref"),
             "rho95":        dict(support="rho:0.95", include_bucket=True, anchor="pi_ref"),
             "rho99":        dict(support="rho:0.99", include_bucket=True, anchor="pi_ref")}


def rho_cover(support):
    """rho for support "rho:<rho>" (0 < rho <= 0.99: the cached head reaches 0.99 and no more), else None."""
    m = re.fullmatch(r"rho:([0-9.]+)", str(support))
    if not m:
        return None
    rho = float(m.group(1))
    if not (0.0 < rho <= 0.99):
        raise ValueError(f"rho must lie in (0, 0.99] -- the cached teacher head covers 0.99; got {rho}")
    return rho


def nested_size(support):
    """n for support "nested:<n>" (n >= 1), None for "union" / "teacher" / "rho:<rho>"; anything else is refused."""
    if support in ("union", "teacher") or rho_cover(support) is not None:
        return None
    m = re.fullmatch(r"nested:(\d+)", str(support))
    if not m or int(m.group(1)) < 1:
        raise ValueError(f"support must be 'union', 'teacher', 'nested:<n>' (n >= 1) or 'rho:<rho>', got {support!r}")
    return int(m.group(1))


def prefix_for_cover(top_lp, rho):
    """Length of the shortest prefix of a descending log-prob list whose mass reaches rho (at least 1)."""
    acc = 0.0
    for k, lp in enumerate(top_lp):
        acc += float(np.exp(lp))
        if acc >= rho:
            return k + 1
    return max(len(top_lp), 1)
LOG_EPS = -27.631021115928547        # log(1e-12): the floor for an empty bucket / an empty residual
NEG = -1e30


@dataclass
class DeltaCache:
    """Everything the loss needs at one trajectory's scored positions (row order = position order)."""
    pos: torch.Tensor            # [N] long  -- the token position t whose prediction is scored
    is_soft: torch.Tensor        # [N] bool
    s_ids: torch.Tensor          # [N, M] long, -1 = pad
    s_valid: torch.Tensor        # [N, M] bool
    log_pi_ref: torch.Tensor     # [N, M] fp32 (0 at pads)
    log_q: torch.Tensor          # [N, M] fp32, nan where the teacher is unknown (hard rows off y)
    in_J: torch.Tensor           # [N, M] bool
    y_col: torch.Tensor          # [N] long  -- column of y in s_ids
    log_pi_rest: torch.Tensor    # [N] fp32  log(1 - Σ_S pi_ref)
    log_q_rest: torch.Tensor     # [N] fp32  log(1 - Σ_S q) (soft), nan (hard)
    teacher_argmax: torch.Tensor  # [N] long, -1 where unknown (hard rows)
    tau: float = DEFAULT_TAU
    pi_floored: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))   # [N, M]
    q_floored: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))    # [N, M]

    def to(self, device):
        kw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()}
        return DeltaCache(**kw)

    @property
    def n(self):
        return int(self.pos.shape[0])


# ---------------------------------------------------------------- builder (pure, tested)
def build_delta_cache(ref, head, teacher_lp_y, tau=DEFAULT_TAU, soft_x2=SOFT_X2, support="union"):
    """One trajectory.  support: "union" (S = K' ∪ K_q ∪ {y}), "teacher" (S = K_q ∪ {y}) or
    "nested:<n>" (S = {y} ∪ pi_ref's top ranks to |S| = n, K_q not unioned; ABLATIONS supp1/2/4).
    ref: dict of numpy arrays over its scored positions in order -- pos [N], y [N],
         top_ids [N, K'], top_lp [N, K'] (pi_ref rank order), lp_y [N]   (scripts/ref_topk_dump.py)
    head: {pos: dict(top_ids, top_lp, extra_ids, extra_lp (None = unknown), floor_lp, log_rest)}
         for the SOFT positions only                                   (scripts/teacher_head_dump.py)
    teacher_lp_y: [N] the existing precompute's log q(y) at each pos (record['teacher_logp'][pos])."""
    pos = np.asarray(ref["pos"]).astype(np.int64); y = np.asarray(ref["y"]).astype(np.int64)
    top_ids = np.asarray(ref["top_ids"]).astype(np.int64); top_lp = np.asarray(ref["top_lp"]).astype(np.float64)
    lp_y = np.asarray(ref["lp_y"]).astype(np.float64); tq_y = np.asarray(teacher_lp_y).astype(np.float64)
    N, K = top_ids.shape
    soft = top_lp[:, 1] > soft_x2 if K > 1 else np.zeros(N, bool)
    n_nested = nested_size(support); rho = rho_cover(support)
    rows = []
    for i in range(N):
        # pi_ref's known values on its top-K' are always available for LOOKUP; whether those
        # tokens are in S is the support choice
        lpr = {int(t): float(v) for t, v in zip(top_ids[i], top_lp[i])}
        ids = [int(t) for t in top_ids[i]] if support == "union" else []
        floor_pi = float(top_lp[i, -1])
        pi_fl, q_fl = {}, {}
        lq, inJ = {}, {}
        if rho is not None:
            # RHO: the teacher's own head truncated to coverage rho, plus y; pi_ref's top-K' is only a LOOKUP table
            yy = int(y[i])
            lpr[yy] = float(lp_y[i]); lq[yy] = float(tq_y[i]); inJ[yy] = True
            if soft[i]:
                h = head[int(pos[i])]
                k = prefix_for_cover(h["top_lp"], rho)
                ids = [int(t) for t in h["top_ids"][:k]]
                for t, v in zip(ids, h["top_lp"][:k]):
                    if t != yy:
                        lq[t] = float(v); inJ[t] = True
                    if t not in lpr:
                        lpr[t] = floor_pi; pi_fl[t] = True
                if yy not in ids:
                    ids.append(yy)
                log_q_rest = float(np.log(max(1.0 - sum(np.exp(lq[t]) for t in ids), 1e-12)))     # every member's q is known here
                t_am = int(h["top_ids"][0])
            else:
                ids = [yy]; log_q_rest = float("nan"); t_am = -1
            rows.append((ids, lpr, lq, inJ, pi_fl, q_fl, ids.index(yy), log_q_rest, t_am))
            continue
        if n_nested is not None:
            # NESTED: y first, then pi_ref's ranks (y skipped) until |S| = n; nothing else joins S
            yy = int(y[i]); ids = [yy]
            for t in top_ids[i]:
                if len(ids) >= n_nested:
                    break
                if int(t) != yy:
                    ids.append(int(t))
            lpr[yy] = float(lp_y[i])
            lq[yy] = float(tq_y[i]); inJ[yy] = True
            if soft[i]:
                h = head[int(pos[i])]
                known = {int(t): float(v) for t, v in zip(h["top_ids"], h["top_lp"])}
                known.update({int(t): float(v) for t, v in zip(h["extra_ids"], h["extra_lp"]) if v is not None})
                for t in ids[1:]:
                    if t in known:
                        lq[t] = known[t]
                    else:
                        lq[t] = float(h["floor_lp"]); q_fl[t] = True
                    inJ[t] = True
                # the bucket's teacher mass is 1 - the mass KNOWN on S (a floored member counts 0,
                # as the server's residual_log_mass counts an unknown extra): whatever the
                # truncation dropped is in here by construction
                log_q_rest = float(np.log(max(1.0 - sum(np.exp(lq[t]) for t in ids if not q_fl.get(t, False)), 1e-12)))
                t_am = int(h["top_ids"][0])
            else:
                log_q_rest = float("nan"); t_am = -1
            uniq = ids
            rows.append((uniq, lpr, lq, inJ, pi_fl, q_fl, 0, log_q_rest, t_am))
            continue
        if soft[i]:
            h = head[int(pos[i])]
            for t, v in zip(h["top_ids"], h["top_lp"]):
                t = int(t); lq[t] = float(v); inJ[t] = True
                if t not in ids:
                    ids.append(t)
                if t not in lpr:
                    lpr[t] = floor_pi; pi_fl[t] = True
            # the client's extras are pi_ref's top-K' ∪ {y} (teacher_head_dump asked for them):
            # under the teacher-only support they are not part of S and their teacher values
            # are not used (y's own value comes from the precompute below)
            for t, v in (zip(h["extra_ids"], h["extra_lp"]) if support == "union" else ()):
                t = int(t)
                if t not in ids:                     # an extra the client asked for beyond K' ∪ {y}: keep it
                    ids.append(t)
                if t not in lpr:
                    lpr[t] = floor_pi; pi_fl[t] = True
                if v is None:
                    lq[t] = float(h["floor_lp"]); q_fl[t] = True
                else:
                    lq[t] = float(v)
                inJ[t] = True
            yy = int(y[i])
            if yy not in ids:
                ids.append(yy)
            lpr[yy] = float(lp_y[i])                # y's pi_ref is exact from the full row
            if yy not in lq:
                lq[yy] = float(tq_y[i]); inJ[yy] = True
            log_q_rest = float(h["log_rest"]); t_am = int(h["top_ids"][0])
        else:
            yy = int(y[i])
            if yy not in ids:
                ids.append(yy)
            lpr[yy] = float(lp_y[i])
            lq[yy] = float(tq_y[i]); inJ[yy] = True
            log_q_rest = float("nan"); t_am = -1
        seen, uniq = set(), []
        for t in ids:
            if t not in seen:
                seen.add(t); uniq.append(t)
        rows.append((uniq, lpr, lq, inJ, pi_fl, q_fl, uniq.index(yy), log_q_rest, t_am))
    M = max(len(r[0]) for r in rows) if rows else 1
    s_ids = torch.full((N, M), -1, dtype=torch.long); s_valid = torch.zeros((N, M), dtype=torch.bool)
    log_pi = torch.zeros((N, M)); log_q = torch.full((N, M), float("nan")); in_J = torch.zeros((N, M), dtype=torch.bool)
    pi_fl_t = torch.zeros((N, M), dtype=torch.bool); q_fl_t = torch.zeros((N, M), dtype=torch.bool)
    y_col = torch.zeros(N, dtype=torch.long); lpr_rest = torch.zeros(N); lq_rest = torch.zeros(N); t_argmax = torch.zeros(N, dtype=torch.long)
    for i, (uniq, lpr, lq, inJ, pi_fl, q_fl, yc, lqr, t_am) in enumerate(rows):
        m = len(uniq)
        s_ids[i, :m] = torch.tensor(uniq); s_valid[i, :m] = True
        log_pi[i, :m] = torch.tensor([lpr[t] for t in uniq])
        log_q[i, :m] = torch.tensor([lq.get(t, float("nan")) for t in uniq])
        in_J[i, :m] = torch.tensor([bool(inJ.get(t, False)) for t in uniq])
        pi_fl_t[i, :m] = torch.tensor([bool(pi_fl.get(t, False)) for t in uniq])
        q_fl_t[i, :m] = torch.tensor([bool(q_fl.get(t, False)) for t in uniq])
        y_col[i] = yc; lq_rest[i] = lqr; t_argmax[i] = t_am
        lpr_rest[i] = float(np.log(max(1.0 - sum(np.exp(v) for v in (lpr[t] for t in uniq)), 1e-12)))
    return DeltaCache(pos=torch.from_numpy(pos), is_soft=torch.from_numpy(soft.copy()), s_ids=s_ids, s_valid=s_valid,
                      log_pi_ref=log_pi.float(), log_q=log_q.float(), in_J=in_J, y_col=y_col, log_pi_rest=lpr_rest.float(),
                      log_q_rest=lq_rest.float(), teacher_argmax=t_argmax, tau=tau, pi_floored=pi_fl_t, q_floored=q_fl_t)


# ---------------------------------------------------------------- the target (pure, tested)
def _fill_log_pi_ref(cache: DeltaCache, log_p_S: torch.Tensor, log_p_B: torch.Tensor):
    """What fills the target where delta = 0 (hard rows off y, and their bucket): pi_ref, on S and
    on the bucket. (test_7 replaces this with sg(p_θ).)"""
    return cache.log_pi_ref, cache.log_pi_rest


_INCLUDE_BUCKET = True      # test_7 sets False


def delta_targets(cache: DeltaCache, beta: float, fill: Optional[torch.Tensor] = None, fill_B: Optional[torch.Tensor] = None,
                  include_bucket: bool = True):
    """log q_hat on S ([N, M], -inf at pads) and on the bucket ([N]), normalised together.
    include_bucket=False (ABLATIONS["no_bucket"]) normalises over S alone and returns -inf for the bucket."""
    fill = cache.log_pi_ref if fill is None else fill
    fill_B = cache.log_pi_rest if fill_B is None else fill_B
    delta = torch.where(cache.in_J, cache.log_q - cache.log_pi_ref, torch.zeros_like(cache.log_pi_ref))
    delta = torch.nan_to_num(delta, nan=0.0)
    # HARD rows: J = {y} and delta_y is clipped like the Lightning advantage; SOFT rows are not clipped
    hard = ~cache.is_soft
    clipped = delta.clamp(-cache.tau, cache.tau)
    delta = torch.where(hard[:, None], clipped, delta)
    lw_S = torch.where(cache.in_J, cache.log_pi_ref + beta * delta, fill)
    lw_S = torch.where(cache.s_valid, lw_S, torch.full_like(lw_S, NEG))
    delta_B = torch.nan_to_num(cache.log_q_rest - cache.log_pi_rest, nan=0.0)
    lw_B = torch.where(cache.is_soft, cache.log_pi_rest + beta * delta_B, fill_B)
    if not (_INCLUDE_BUCKET and include_bucket):
        lw_B = torch.full_like(lw_B, NEG)
    logZ = torch.logsumexp(torch.cat([lw_S, lw_B[:, None]], 1), dim=1)
    return lw_S - logZ[:, None], lw_B - logZ


# ---------------------------------------------------------------- the loss
def delta_kl_loss(student_logits_gathered: torch.Tensor, student_logsumexp: torch.Tensor, cache: DeltaCache,
                  beta: float = 1.0, mode: str = "kl", include_bucket: bool = True, anchor: str = "pi_ref"):
    """student_logits_gathered [N, M] fp32 at cache.s_ids (any value at pads), student_logsumexp [N].
    Returns (loss = per-row mean, info).  mode:
      "kl"         the delta-KL loss above
      "lightning"  the existing Lightning surrogate with delta_y in place of the advantage:
                   −sg(clip(delta_y)) · log p_θ(y), same normalisation (equals compute_opd_loss when p_θ = pi_ref)
      "y_square"   ½ (log p_θ(y) − sg(log pi_ref(y) + beta·clip(delta_y)))², the y-coordinate projection
    include_bucket / anchor: ABLATIONS (no_bucket drops the residual cell; anchor="student" fills the
    delta = 0 cells with sg(p_θ) instead of pi_ref).
    """
    if anchor not in ("pi_ref", "student"):
        raise ValueError(f"anchor must be 'pi_ref' or 'student', got {anchor!r}")
    log_p_S = student_logits_gathered - student_logsumexp[:, None]
    log_p_S = torch.where(cache.s_valid, log_p_S, torch.full_like(log_p_S, NEG))
    n = max(cache.n, 1)
    r = torch.arange(cache.n, device=log_p_S.device)
    log_p_y = log_p_S[r, cache.y_col]
    if mode in ("lightning", "y_square"):
        d_y = (cache.log_q - cache.log_pi_ref)[r, cache.y_col].clamp(-cache.tau, cache.tau).detach()
        if mode == "lightning":
            per_row = -d_y * log_p_y
        else:
            per_row = 0.5 * (log_p_y - (cache.log_pi_ref[r, cache.y_col] + beta * d_y).detach()) ** 2
        loss = per_row.sum() / n
        return loss, dict(per_row=per_row.detach(), mode=mode)
    p_S_sum = torch.where(cache.s_valid, log_p_S.exp(), torch.zeros_like(log_p_S)).sum(1)
    log_p_B = torch.log((1.0 - p_S_sum).clamp_min(1e-12))
    if anchor == "student":
        fill, fill_B = log_p_S.detach(), log_p_B.detach()
    else:
        fill, fill_B = _fill_log_pi_ref(cache, log_p_S, log_p_B)
    lq_S, lq_B = delta_targets(cache, beta, fill, fill_B, include_bucket=include_bucket)
    q_S, q_B = lq_S.exp(), lq_B.exp()
    kl_S = torch.where(cache.s_valid, q_S * (lq_S - log_p_S), torch.zeros_like(q_S)).sum(1)
    kl_B = q_B * (lq_B - log_p_B) if (_INCLUDE_BUCKET and include_bucket) else torch.zeros_like(kl_S)
    per_row = kl_S + kl_B
    loss = per_row.sum() / n
    with torch.no_grad():
        soft = cache.is_soft
        pr = per_row.detach()
        stud_am = torch.where(cache.s_valid, log_p_S, torch.full_like(log_p_S, NEG)).argmax(1)
        stud_am_id = cache.s_ids[r, stud_am]
        known = cache.teacher_argmax >= 0
        ref_am = cache.s_ids[r, torch.where(cache.s_valid, cache.log_pi_ref, torch.full_like(cache.log_pi_ref, NEG)).argmax(1)]
        info = dict(per_row=pr, mode=mode, n_soft=int(soft.sum()), n_hard=int((~soft).sum()),
                    loss_soft=float(pr[soft].mean()) if soft.any() else float("nan"),
                    loss_hard=float(pr[~soft].mean()) if (~soft).any() else float("nan"),
                    argmax_student_eq_teacher=float((stud_am_id[known] == cache.teacher_argmax[known]).float().mean()) if known.any() else float("nan"),
                    argmax_ref_eq_teacher=float((ref_am[known] == cache.teacher_argmax[known]).float().mean()) if known.any() else float("nan"),
                    log_p_y=log_p_y.detach())
    return loss, info


# ---------------------------------------------------------------- the step (the trainer's plug point)
@dataclass
class DeltaStepOutput:
    loss: torch.Tensor
    advantage: torch.Tensor        # log q(y) − log p_θ(y) at scored positions (diagnostics compatibility), 0 elsewhere
    score_mask: torch.Tensor
    student_logp: torch.Tensor     # [N] log p_θ(y_t) at scored positions, nan elsewhere
    teacher_logp: torch.Tensor
    info: dict


def delta_train_step(trajectory, teacher_logp: torch.Tensor, cache: DeltaCache, student_scorer,
                     beta: float = 1.0, mode: str = "kl", include_bucket: bool = True, anchor: str = "pi_ref") -> DeltaStepOutput:
    """The delta arm's opd_train_step: same mask, same normalisation, the loss swapped."""
    score_mask = build_score_mask(trajectory.token_type)
    idx = scored_indices(score_mask)
    if not torch.equal(idx.cpu(), cache.pos.cpu()):
        raise AssertionError(f"delta cache positions differ from the score mask: cache {cache.n} rows, mask {int(idx.numel())} "
                             f"(first diff at {int((idx.cpu() != cache.pos.cpu()[:idx.numel()]).nonzero()[0]) if idx.numel() == cache.n else 'length'})")
    gathered, lse = student_scorer.gathered_logits(trajectory.input_ids, cache.pos - 1, cache.s_ids)
    dev = gathered.device
    c = cache.to(dev)
    loss, info = delta_kl_loss(gathered, lse, c, beta, mode, include_bucket=include_bucket, anchor=anchor)
    n_tok = int(trajectory.input_ids.shape[0])
    student_logp = torch.full((n_tok,), float("nan"), dtype=torch.float32, device=dev)
    student_logp[c.pos] = info["log_p_y"].float()
    teacher_logp = teacher_logp.to(dev)
    adv = torch.zeros(n_tok, dtype=torch.float32, device=dev)
    adv[c.pos] = (teacher_logp[c.pos] - info["log_p_y"]).clamp(-c.tau, c.tau)
    return DeltaStepOutput(loss=loss, advantage=adv, score_mask=score_mask.to(dev), student_logp=student_logp,
                           teacher_logp=teacher_logp, info=info)
