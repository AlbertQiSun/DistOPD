"""Confound diagnostics -- the measurement instruments whose output IS the paper's
evidence. Every function declares the proposition it discriminates and the curve
shape each competing hypothesis predicts; a "busy but non-discriminating" log is
worse than none. These run on REAL training (drifting pi_theta + real retriever);
the deterministic dry-run only proves the schema/collection code does not crash --
it CANNOT show poisoning, and a flat dry-run curve is expected, not a bug.

Metric conventions are aligned to prior work (not invented): answer correctness and
retrieval hit both use the SQuAD/DPR `normalize_answer`; retrieval quality is
answer-recall@k (DPR "Top-k accuracy": gold answer string appears in retrieved
passages), the established convention in the Search-R1 / FlashRAG lineage. The only
declared extensions are (a) per-turn/per-hop hit rate as the mechanism headline
(union hit is retained but only as a task-success proxy -- see retrieval_quality)
and (b) tracking these across RL steps, which is unmeasured in that literature.
"""
from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from opd.step import StepOutput
from opd.trajectory import TokenType, Trajectory

# ---------------------------------------------------------------------------
# answer-string matching -- EXACT SQuAD/DPR normalizer, shared by task-EM and hit@k
# so retrieval "hit" and answer "correct" can never use divergent conventions.
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _has_alias(passage: str, alias: str) -> bool:
    """DPR has_answer for a SINGLE alias: normalized alias token sequence is a
    contiguous sublist of the normalized passage tokens."""
    g = normalize_answer(alias).split()
    p = normalize_answer(passage).split()
    if not g:
        return False
    for i in range(len(p) - len(g) + 1):
        if p[i:i + len(g)] == g:
            return True
    return False


def has_answer(passage: str, gold) -> bool:
    """DPR has_answer with alias support. `gold` is a single string OR a list of
    aliases; a passage HITS if ANY alias matches (OR, not all) -- and every alias
    goes through the SAME `normalize_answer` as the passage, so the whole answer-
    recall@k curve cannot drift from an and/or or normalizer split."""
    aliases = [gold] if isinstance(gold, str) else list(gold or [])
    return any(_has_alias(passage, a) for a in aliases)


# ---------------------------------------------------------------------------
# TASK PERFORMANCE headline -- EM + F1 on the model's FINAL <answer> span vs gold.
# This (not retrieval hit) is the task metric: poisoning rots synthesis-from-context
# FIRST, and union_hit ("answer string present in retrieved docs") is blind to that.
# ---------------------------------------------------------------------------
def _f1_tokens(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    """SQuAD token-level F1 over already-normalized token lists."""
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)   # both empty -> 1.0; one empty -> 0.0
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_em_f1(final_answer, gold) -> dict:
    """HEADLINE task performance: exact-match (strict) + token-F1 (lenient) + cover-EM
    (length-INSENSITIVE substring hit) of the final <answer> span vs gold, all through the
    shared SQuAD/DPR `normalize_answer` and OR-maxed over gold aliases.

    EM is the strict lower bound, F1 the lenient middle, cover_em the length-insensitive
    ceiling; reported TOGETHER because their relationship is itself diagnostic:
      em + f1 both fall   -> genuine answer degradation (poisoning)
      F1 flat, EM falls   -> answers drift to partially-correct
      EM flat, F1 falls   -> answers get verbose / padded (recall noise)
      cover_em~1, EM~0    -> answer is CORRECT but VERBOSE (a padded <answer> span), NOT a
                             wrong answer -- separates "found it but chatty" from "wrong".
    cover_em exists to strip the answer-STYLE confound out of strict EM/F1: strict EM/F1 are
    length-sensitive (F1's precision penalizes long spans), so a verbose imitator (pi_ref,
    modelled on a chatty teacher) is scored unfairly low vs a terse Instruct baseline even
    when it FOUND the answer. cover_em is length-blind, so its delta isolates capability from
    style. NOTE: it is a plain normalized SUBSTRING test (`gold in pred`), NOT the stricter
    DPR token-subsequence `has_answer` -- so it can sub-word-match ("york" in "yorker"); swap
    to has_answer if a given gold set makes that a real risk.
    `answered` separates a WRONG answer from NO answer (truncation confound guard)."""
    aliases = [gold] if isinstance(gold, str) else list(gold or [])
    pred = normalize_answer(final_answer or "")
    if not aliases:
        return {"answered": bool(pred), "em": None, "f1": None, "cover_em": None,
                "pred": pred or None}
    pred_tokens = pred.split()
    gold_norms = [normalize_answer(a) for a in aliases]
    em = max(float(pred == g) for g in gold_norms)
    f1 = max(_f1_tokens(pred_tokens, g.split()) for g in gold_norms)
    cover_em = max(float(bool(g) and g in pred) for g in gold_norms)
    return {"answered": bool(pred), "em": em, "f1": f1, "cover_em": cover_em,
            "pred": pred or None}


# ---------------------------------------------------------------------------
# CONFOUND #2 -- context quality (mode B: retrieval outcome)
# ---------------------------------------------------------------------------
def retrieval_quality(meta: dict) -> dict:
    """Proposition: does the retrieved context still CONTAIN the evidence to answer?

    HEADLINE = per-turn/per-hop hit rate. Poisoning degrades progressively (early
    hop hits before drift, later hops collapse); the UNION ("any turn hit") would
    still mark such a trajectory a hit and stay flat while context truly rots --
    which would let the judge misread real poisoning as truncation. So per-hop is
    the mechanism signal; union is kept ONLY as a task-success proxy.

    Predicted (online, over steps): poisoning -> per_turn_hit_rate DOWN, first_hit
    pushed later or lost; truncation confound -> per_turn_hit_rate ~flat. Offline:
    flat by construction (frozen context).

    MEASURED (Row 10 real, 147/150 steps, 2026-08-20): the poisoning shape did NOT
    appear -- online per-hop hits were flat-to-RISING (hop-4 0.187->0.217; still up
    +5pts on the paired both-arms-reached-hop-4 question set, offline flat on the same
    questions). Context does not rot; the online degradation lives in TERMINATION
    (answered 0.92->0.77). See the objective-mis-specification note on
    boundary_bucketed_advantage below."""
    gold = meta.get("gold_answer")
    turns = meta.get("turns", []) or []
    if not gold or not turns:
        return {"per_turn_hit": [], "per_turn_hit_rate": None,
                "union_hit": None, "first_hit_turn": -1, "n_turns": len(turns)}
    per_turn = [bool(any(has_answer(d, gold) for d in t.get("docs", []))) for t in turns]
    hit_rate = sum(per_turn) / len(per_turn)
    first = per_turn.index(True) if any(per_turn) else -1
    return {"per_turn_hit": per_turn,           # headline: per-hop, not smoothed
            "per_turn_hit_rate": hit_rate,      # mechanism curve
            "union_hit": bool(any(per_turn)),   # task-success proxy ONLY
            "first_hit_turn": first,
            "n_turns": len(turns)}


# ---------------------------------------------------------------------------
# CONFOUND #2 -- query health (mode A: generation side). Our novel diagnostic,
# no prior-paper convention; kept simple and string-based.
# ---------------------------------------------------------------------------
def _bigram_repetition(text: str) -> float:
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def query_health(meta: dict) -> dict:
    """Proposition: is pi_theta still emitting SENSIBLE searches, or collapsing?

    Predicted (online, over steps): poisoning -> degeneracy_rate UP (empty / verbatim
    repeats / self-repetition), query_len collapses or balloons. Offline: flat.

    MEASURED (Row 10 real, 2026-08-20): the OPPOSITE -- online degeneracy_rate HALVED
    (0.133->0.063, query_len 6.6->7.0) while offline sat frozen at 0.15. The poisoning
    chain's first link never fired; queries got cleaner as the policy drifted."""
    turns = meta.get("turns", []) or []
    q_norm = normalize_answer(meta.get("question", ""))
    queries = [t.get("query", "") for t in turns]
    seen: set[str] = set()
    flags = []
    for q in queries:
        qn = normalize_answer(q)
        degenerate = (len(qn) == 0) or (qn == q_norm) or (qn in seen) or (_bigram_repetition(q) > 0.5)
        flags.append(bool(degenerate))
        seen.add(qn)
    lengths = [len(q.split()) for q in queries]
    return {"degeneracy_rate": (sum(flags) / len(flags)) if flags else None,
            "query_len_mean": (sum(lengths) / len(lengths)) if lengths else None,
            "n_queries": len(queries)}


# ---------------------------------------------------------------------------
# CONFOUND #2 -- termination (diverge vs truncation, the raw split)
# ---------------------------------------------------------------------------
def termination_record(traj: Trajectory) -> dict:
    """Proposition: are failures driven by TRUNCATION or by genuine divergence?
    Read WITH context-quality: truncation confound -> is_truncated/truncation-rate
    rises while context stays clean; poisoning -> context rots first."""
    r = traj.terminated
    reason = r.value if r is not None else None
    return {"reason": reason,
            "is_truncated": bool(reason is not None and "truncated" in reason),
            "num_search_calls": int(traj.num_search_calls),
            "length": int(traj.length)}


# ---------------------------------------------------------------------------
# CONFOUND #2 -- policy drift: KL(pi_theta||pi_ref) + importance weight.
# ALGORITHM PINNED: per-token MEAN (not sum -> length can't leak in); ONLY at
# ACTION positions (consistent with score_mask; observations must not dilute).
# The importance-weight mean/std is the direct analog of Lightning OPD Fig 3a
# (their static setting sits at ~0.94 mean, std<0.1); if agentic training blows
# it up, same quantity/same plot = "interactive breaks the offline-stays-on-policy
# premise", flat vs exploding.
# ---------------------------------------------------------------------------
def _aligned_from_full(full_lp: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Realized-token aligned logprob from a full [N,V] log-dist: out[t]=full[t-1, id_t]."""
    out = torch.full((ids.shape[0],), float("nan"), dtype=full_lp.dtype)
    if ids.shape[0] > 1:
        rows = full_lp[:-1]                       # rows 0..N-2 predict tokens 1..N-1
        out[1:] = rows.gather(1, ids[1:].unsqueeze(1)).squeeze(1)
    return out


KL_ROW_CHUNK = 128


def kl_importance(student_scorer, ref_scorer, traj: Trajectory,
                  score_mask: torch.Tensor) -> dict:
    """Exact per-token KL(pi_theta || pi_ref) and the importance weight, at SCORED tokens.

    Same quantities as before, computed without the three costs that made this diagnostic
    more expensive than the training step it was measuring:

      1. IT ASKED FOR EVERY ROW. KL is wanted at the rows that generated scored tokens --
         p50 is 270 of 2241, p90 487 of 2674. The rest were computed and thrown away.
      2. IT MOVED THEM TO THE HOST. full_logprobs forced .cpu(), so one trajectory pushed
         2 x [N, V] fp32 across the bus: 2.7 GB at p50, 16.7 GB at the longest. At
         global_batch=256 that is ~700 GB per optimizer step, for data used once.
      3. IT REDUCED IN PYTHON. A loop over scored tokens, each doing exp/sub/mul/sum over a
         151,936-wide vector: ~69,000 iterations per step at global_batch=256.

    Row-restricted, on-device, and vectorised in row chunks.

    WHAT IS AND IS NOT GUARANTEED, stated precisely because an earlier version of this
    docstring said "the numbers are unchanged", which is stronger than what holds:

      * ROW RESTRICTION IS EXACT. log_softmax normalises within a row, so computing only the
        rows that are read cannot alter any value. That part is bit-identical.
      * THE REDUCTION ORDER AND THE DEVICE BOTH CHANGED. The original summed Python floats
        one token at a time on the HOST; this sums fp32 tensors per chunk on the DEVICE.
        Floating-point addition is not associative and the kernels are not the same, so the
        results agree to a TOLERANCE (the gate asserts rel=1e-5), not to the last bit.

    Bit-identity was never available here -- the host-side Python loop is exactly what made
    this diagnostic cost more than the training step it measures (~700 GB/step of transfer at
    global_batch=256, and all of it outside the timer). The tolerance is ~1e-5 relative on a
    quantity of order 1e-4, i.e. ~1e-9 absolute, which is far below the drift being measured.
    That is the trade, and it is recorded rather than implied.
    """
    ids = traj.input_ids
    m = score_mask.to(ids.device).clone()
    m[0] = False
    idx = torch.nonzero(m, as_tuple=False).flatten()
    if idx.numel() == 0:
        return {"kl_pertok_mean": None, "iw_mean": None, "iw_std": None, "n": 0}

    rows = idx - 1                                # the row that PREDICTED each scored token
    targets = ids.to(idx.device).index_select(0, idx)

    # ONE forward each, and the fp32 widening happens PER CHUNK.
    #
    # THIS USED TO CALL log_dist_at TWICE, which returns [n_scored, V] fp32. Both results are
    # needed at the same time (KL compares them row by row), so two of them were live at once
    # -- 4.99 GiB each at n_scored = 8800, on the card that also holds the student and its
    # optimizer states. It killed opd-moe-on at step 38 after 16 hours:
    #
    #     torch.OutOfMemoryError: Tried to allocate 4.99 GiB. GPU 1 ... 4.12 GiB is free
    #
    # The MoE line hits it first because its actions are the longest (mean 1070 action tokens
    # per question against MiroThinker's 311), so a trajectory offers far more scored rows.
    # Same shape as the teacher OOM in the project conventions: a [rows, vocab] fp32 tensor materialised in
    # full when only a slice of it is ever read at a time.
    #
    # The loop below ALREADY chunked at KL_ROW_CHUNK; it just chunked two tensors that had
    # already been built. Moving the log_softmax inside it keeps the bf16 [N, V] logits (which
    # the forward produces anyway) and never widens more than KL_ROW_CHUNK rows to fp32.
    # Peak drops from 2 x n_scored x V x 4 bytes to 2 x N x V x 2 + 2 x chunk x V x 4.
    st_logits = student_scorer.raw_logits(ids)    # [N, V] in the model's dtype, on device
    rf_logits = ref_scorer.raw_logits(ids)
    rows = rows.to(st_logits.device)
    targets = targets.to(st_logits.device)

    kl_sum, n = 0.0, int(idx.numel())
    st_lp_parts, rf_lp_parts = [], []
    for a in range(0, n, KL_ROW_CHUNK):
        r = rows[a:a + KL_ROW_CHUNK]
        p = torch.log_softmax(st_logits[r].float(), dim=-1)
        q = torch.log_softmax(rf_logits[r].float(), dim=-1)
        t = targets[a:a + KL_ROW_CHUNK].unsqueeze(1)
        st_lp_parts.append(p.gather(1, t).squeeze(1))   # realized-token logprobs, align-to-t
        rf_lp_parts.append(q.gather(1, t).squeeze(1))
        kl_sum += float((p.exp() * (p - q)).sum(dim=-1).sum())
        del p, q
    st_lp = torch.cat(st_lp_parts)
    rf_lp = torch.cat(rf_lp_parts)
    w = torch.exp(st_lp - rf_lp)                  # importance weight pi_theta/pi_ref
    return {"kl_pertok_mean": kl_sum / n,         # MEAN over action tokens
            "iw_mean": float(w.mean()),
            "iw_std": float(w.std(unbiased=False)),
            "n": n}


# ---------------------------------------------------------------------------
# CONFOUND #3 -- advantage scale + clip consistency
# ---------------------------------------------------------------------------
def advantage_stats(step_out: StepOutput, tau: float) -> dict:
    """Proposition: is the advantage well-scaled and is clipping rare/consistent?
    clip_rate = fraction of scored tokens whose UNCLIPPED |teacher-student| exceeds
    tau (would-be clip). A blown-up clip_rate means the clip is doing heavy lifting
    -> advantages are not comparable across online/offline unless clip behaves same."""
    m = step_out.score_mask
    adv = step_out.advantage[m]
    if adv.numel() == 0:
        return {"adv_mean": None, "adv_std": None, "adv_absmax": None,
                "clip_rate": None, "n_scored": 0}
    raw = (step_out.teacher_logp - step_out.student_logp)[m]
    clip = (raw.abs() > tau)
    return {"adv_mean": float(adv.mean()),
            "adv_std": float(adv.std(unbiased=False)),
            "adv_absmax": float(adv.abs().max()),
            "clip_rate": float(clip.float().mean()),
            "n_scored": int(m.sum())}


def boundary_bucketed_advantage(traj: Trajectory, step_out: StepOutput, tau: float,
                                max_fine: int = 5) -> dict:
    """Proposition: does the teacher MISCALIBRATE right after retrieval injection?

    Mechanism: the first few action tokens generated immediately AFTER an observation
    (retrieved docs just spliced in) are the most likely to have an anomalous
    teacher-vs-student gap -> spiky advantage / clipping. Bucketing is ONE-DIRECTIONAL:
    offset = 1,2,3,... counting FORWARD from each observation's right edge into the
    following action span (fine buckets 1..max_fine, coarse `>max_fine`). We do NOT
    use symmetric absolute distance -- that would fold the action TAIL before an
    observation into the same bucket and smooth away the post-boundary spike.

    MEASURED (Row 10 real, 2026-08-20): NO spike -- offset-1 |A| declined in both arms
    (online mean_abs 0.728->0.520, p90 1.375->1.017), clip ~1e-4 throughout. The teacher
    is well-calibrated on post-injection states, which KILLED the "off-manifold teacher
    is unreliable" reading of off-dynamics. What Row 10 established instead: the
    teacher's signal in the keep-searching region is clean, confident, and points AWAY
    from terminating -- RELIABLY HARMFUL, not unreliable (online answered collapsed
    0.92->0.77 while retrieval and answered-conditioned quality held or improved).
    Direct preference measurement: scripts/probe_teacher_termination.py."""
    tt = traj.token_type
    adv = step_out.advantage
    raw = step_out.teacher_logp - step_out.student_logp
    ACTION, OBS = int(TokenType.ACTION), int(TokenType.OBSERVATION)
    buckets: dict[str, dict[str, list]] = {}
    offset = 0
    for t in range(tt.shape[0]):
        is_action = int(tt[t]) == ACTION
        if is_action and t > 0 and int(tt[t - 1]) == OBS:
            offset = 1                                   # first token after obs boundary
        elif is_action and offset > 0 and int(tt[t - 1]) == ACTION:
            offset += 1                                  # continue the post-boundary run
        else:
            offset = 0                                   # pre-obs action / non-action: not counted
        if offset > 0:
            key = str(offset) if offset <= max_fine else f"{max_fine + 1}+"
            b = buckets.setdefault(key, {"adv": [], "clip": []})
            b["adv"].append(float(adv[t]))
            b["clip"].append(bool(float(raw[t].abs()) > tau))

    def _summarize(key: str, v: dict) -> dict:
        adv = v["adv"]
        mags = sorted(abs(x) for x in adv)
        n = len(mags)
        out = {
            "mean_adv": sum(adv) / n,          # SIGNED (kept: existing readers use it)
            "clip_rate": sum(v["clip"]) / n,
            "count": n,
            # |A_t|. A SIGNED mean CANCELS -- a bucket holding +5 and -5 reports mean_adv 0
            # while the tokens are individually extreme -- and a post-observation spike is
            # exactly the case where the disagreement can go both ways. Without these, the
            # magnitude question could not be answered from the artifact at all.
            "mean_abs": sum(mags) / n,
            "p50_abs": mags[n // 2],
            "p90_abs": mags[min(n - 1, int(0.9 * n))],
            "max_abs": mags[-1],
        }
        # FINE buckets keep the RAW per-token advantages. They are tiny (one token per
        # observation boundary per bucket, so ~n_turns values each), and keeping them means
        # any future statistic -- a different quantile, a histogram, a sign breakdown -- is
        # computable OFFLINE instead of costing a GPU re-run. That is the the project conventions test:
        # can a new metric be built from the artifact without re-running generation? For the
        # coarse `6+` bucket (hundreds of tokens per step) the summary above is the
        # deliberate trade -- persisting all of it would bloat every diagnostic record.
        if key != f"{max_fine + 1}+":
            out["adv"] = list(adv)
        return out

    return {k: _summarize(k, v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# CONFOUND #1 -- masking runtime guard (structurally locked by M2-M4 tests; this
# is only a real-training tripwire).
# ---------------------------------------------------------------------------
def masking_guard(score_mask: torch.Tensor, traj: Trajectory) -> dict:
    assert not bool(score_mask[0]), "position 0 must never be scored"
    n_action = int((traj.token_type == int(TokenType.ACTION)).sum())
    n_scored = int(score_mask.sum())
    return {"n_scored": n_scored, "n_action_tokens": n_action,
            "scored_equals_action": (n_scored == n_action)}


# ---------------------------------------------------------------------------
# SECONDARY (embedding-based, labeled): A1 query drift + B3 doc relevance.
# Optional -- needs an encoder (E5 on HPC). None in dry-run => emitted as null.
# E5 cosine is partly circular (it is what the retriever optimizes) => supporting
# evidence only, never headline.
# ---------------------------------------------------------------------------
def embedding_metrics(meta: dict, embedder) -> Optional[dict]:
    if embedder is None:
        return None
    turns = meta.get("turns", []) or []
    question = meta.get("question", "")
    if not turns:
        return {"query_question_cos_mean": None, "doc_relevance_cos_mean": None}
    import torch as _t

    def _enc(texts):
        v = embedder.encode(texts)
        v = v if isinstance(v, _t.Tensor) else _t.tensor(v)
        return _t.nn.functional.normalize(v.float(), dim=-1)

    q_vec = _enc([question])[0]
    queries = [t.get("query", "") for t in turns]
    docs = [d for t in turns for d in t.get("docs", [])]
    qy = _enc(queries)
    qq_cos = float((qy @ q_vec).mean())
    dr_cos = None
    if docs:
        dv = _enc(docs)
        dr_cos = float((dv @ q_vec).mean())
    return {"query_question_cos_mean": qq_cos, "doc_relevance_cos_mean": dr_cos}


# ---------------------------------------------------------------------------
# CONFOUND #1 -- policy freshness (is "online" actually on-policy?)
# ---------------------------------------------------------------------------
def policy_freshness(staleness: int, resync_every: int) -> dict:
    """Proposition: were these rollouts sampled from the CURRENT pi_theta, or from
    a stale weight snapshot? `staleness = step - last_sync_step` -- the number of
    optimizer steps between the weights the inference engine sampled with and the
    current training weights. Makes "how online is online" a reported, ablatable
    quantity rather than a hidden property of the engine.

    Predicted: main online run (resync_every=1) -> staleness==0 at every sample
    (fully on-policy). offline -> staleness == step (frozen, never resynced). A
    resync BUG in online would show staleness>0 here -- distinguishing a real
    poisoning failure from an accidental 'online is secretly offline' artifact."""
    return {"staleness": int(staleness),
            "resync_every": int(resync_every),
            "on_policy": bool(staleness == 0)}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
@dataclass
class DiagnosticRecord:
    step: int
    mode: str
    task: dict
    retrieval: dict
    query: dict
    termination: dict
    kl_importance: dict
    advantage: dict
    boundary: dict
    masking: dict
    policy: dict
    embedding: Optional[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class DiagnosticRunner:
    """Assembles one DiagnosticRecord per step. Strictly grad-free and additive --
    it never touches the training numerics, only reads the StepOutput + trajectory."""

    def __init__(self, ref_scorer, tau: float, embedder=None) -> None:
        self.ref = ref_scorer
        self.tau = tau
        self.embedder = embedder

    def run(self, step_idx: int, mode: str, traj: Trajectory,
            step_out: StepOutput, student_scorer,
            staleness: int = 0, resync_every: int = 1) -> DiagnosticRecord:
        with torch.no_grad():
            return DiagnosticRecord(
                step=step_idx, mode=mode,
                task=answer_em_f1(traj.meta.get("final_answer"), traj.meta.get("gold_answer")),
                retrieval=retrieval_quality(traj.meta),
                query=query_health(traj.meta),
                termination=termination_record(traj),
                kl_importance=kl_importance(student_scorer, self.ref, traj, step_out.score_mask),
                advantage=advantage_stats(step_out, self.tau),
                boundary=boundary_bucketed_advantage(traj, step_out, self.tau),
                masking=masking_guard(step_out.score_mask, traj),
                policy=policy_freshness(staleness, resync_every),
                embedding=embedding_metrics(traj.meta, self.embedder),
            )
