"""delta-OPD loss (opd/delta_loss.py): the seven checks the design fixed, each against a
LONGHAND reference that shares no code with the loss.

  1. V=50: loss and gradient vs a full-softmax, full-vocabulary KL  (< 1e-6)
  2. delta ≡ 0: loss 0 / grad 0 at p_θ = pi_ref; and = coarsened KL(pi_ref || p_θ) elsewhere
  3. y-coordinate projections equal the existing Lightning loss on the same batch
  4. beta = 0.5 target = normalised pi_ref^0.5 q^0.5 on S
  5. y outside top-K': S still holds y, the loss is finite
  6. seq 15360 x V 151936: bytes allocated <= 1.1 x the existing loss, no (seq, V) fp32 tensor
  7. mutations: no bucket -> test 1 red; sg(p_θ) fill -> test 2 red
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opd import delta_loss as dl  # noqa: E402
from opd.loss import compute_opd_loss  # noqa: E402
from opd.masking import build_score_mask, head_aligned_logprobs, head_gathered_logits  # noqa: E402
from opd.trajectory import TokenType, Trajectory  # noqa: E402
from teacher.head import head_from_entry, residual_log_mass  # noqa: E402

V, KP, TAU = 50, 8, 10.0


@pytest.fixture(autouse=True)
def _fp64():
    """The toy references are fp64; scoped per test so the rest of the suite keeps torch's default."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


# ---------------------------------------------------------------- synthetic world
def _world(N=12, seed=0, hard_rows=(3, 7), k_wide=None, y_outside=()):
    """pi_ref rows (some hard = saturated), teacher rows, realised tokens; the cache built
    through the two PRODUCERS' selection code (topk as ref_topk_dump does; teacher.head)."""
    g = torch.Generator().manual_seed(seed)
    lpi = torch.randn(N, V, generator=g) * 1.5
    for r in hard_rows:
        lpi[r, r] += 12.5                                    # saturated (rank-2 below -8) but with a visible tail
    lpi = torch.log_softmax(lpi, -1)
    lq = torch.log_softmax(torch.randn(N, V, generator=g) * 2.5, -1)
    y = torch.tensor([int(torch.multinomial(lpi[i].exp(), 1, generator=g)) for i in range(N)])
    for r in y_outside:
        y[r] = int(lpi[r].argmin())                          # far outside pi_ref's top-K'
    top = torch.topk(lpi, KP, -1)
    ref = dict(pos=np.arange(1, N + 1), y=y.numpy(), top_ids=top.indices.numpy(), top_lp=top.values.numpy(),
               lp_y=lpi[torch.arange(N), y].numpy())
    soft = top.values[:, 1] > dl.SOFT_X2
    head = {}
    for i in range(N):
        if not soft[i]:
            continue
        order = torch.argsort(lq[i], descending=True)
        wide = order if k_wide is None else order[:k_wide]
        entry = {int(t): float(lq[i, t]) for t in wide}
        entry[int(y[i])] = float(lq[i, y[i]])                # the realised token is always in the entry
        h = head_from_entry(entry, int(y[i]), extra_ids=list(top.indices[i].numpy()) + [int(y[i])], k_cap=32, cover=0.99)
        h["log_rest"] = residual_log_mass(h["top_lp"], h["extra_lp"])
        head[i + 1] = h
    cache = dl.build_delta_cache(ref, head, lq[torch.arange(N), y].numpy(), tau=TAU)
    return dict(lpi=lpi, lq=lq, y=y, cache=cache, soft=soft)


def _student(logits):
    """What the student hands the loss: its logits gathered at S and the row log-sum-exp."""
    c = logits  # [N, V] leaf
    return c.gather(1, w_ids(c)), torch.logsumexp(c, -1)


def w_ids(logits):
    raise RuntimeError("bound per test")


def _gather(logits, cache):
    return logits.gather(1, cache.s_ids.clamp_min(0)), torch.logsumexp(logits, -1)


def _cache_consistent_pi_ref(W):
    """pi_ref as the CACHE knows it: on S the stored values (K_q-minus-K' tokens floored at pi_ref's K'-th
    value -- ref_topk keeps top-K' only), off S the true values rescaled so the row still sums to 1.
    At beta = 1 the floors cancel in the target (pi·exp(log q − log pi) = q), so the true pi_ref is
    the right reference there; at beta < 1 they do not, and THIS is what the loss is defined on."""
    lpi, cache = W["lpi"], W["cache"]
    out = lpi.clone()
    for i in range(lpi.shape[0]):
        S = cache.s_ids[i][cache.s_valid[i]]
        out[i, S] = cache.log_pi_ref[i][cache.s_valid[i]].double()
        off = torch.ones(V, dtype=torch.bool); off[S] = False
        # the floors can push Σ_S above 1; the loss clamps its residual at 1e-12 and so does this
        out[i, off] = lpi[i, off] + torch.log(torch.clamp(1 - out[i, S].exp().sum(), min=1e-12) / lpi[i, off].exp().sum())
    return out


def _reference_full_kl(logits, W, beta, lpi=None):
    """LONGHAND: the delta target over the FULL vocabulary, then Σ_V q_hat (log q_hat − log p)."""
    lq, cache = W["lq"], W["cache"]
    lpi = W["lpi"] if lpi is None else lpi
    N = lpi.shape[0]
    logp = torch.log_softmax(logits, -1)
    total = 0.0
    for i in range(N):
        S = cache.s_ids[i][cache.s_valid[i]].tolist()
        w = lpi[i].exp().clone()
        if cache.is_soft[i]:
            for j in S:                                       # J = S: delta_j = log q_j − log pi_j
                w[j] = lpi[i, j].exp() * torch.exp(beta * (lq[i, j] - lpi[i, j]))
            off = torch.ones(V, dtype=torch.bool); off[S] = False
            qB, piB = lq[i].exp()[off].sum(), lpi[i].exp()[off].sum()
            w[off] = lpi[i].exp()[off] * (qB / piB) ** beta   # the bucket's delta, shared off S
        else:
            yy = int(W["y"][i])
            d = torch.clamp(lq[i, yy] - lpi[i, yy], -TAU, TAU)
            w[yy] = lpi[i, yy].exp() * torch.exp(beta * d)
        qh = w / w.sum()
        total = total + (qh * (qh.log() - logp[i])).sum()
    return total / N


def _reference_partition_kl_pi_ref(logits, W):
    """LONGHAND: KL(pi_ref || p_θ) COARSENED to each row's S ∪ {bucket}."""
    lpi, cache = W["lpi"], W["cache"]
    logp = torch.log_softmax(logits, -1)
    total = 0.0
    for i in range(lpi.shape[0]):
        S = cache.s_ids[i][cache.s_valid[i]].tolist()
        a, b = lpi[i].exp()[S], logp[i].exp()[S]
        aB, bB = 1 - a.sum(), 1 - b.sum()
        total = total + (a * (a.log() - b.log())).sum() + aB * (aB.log() - bB.log())
    return total / lpi.shape[0]


# ---------------------------------------------------------------- 1
@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_1_matches_full_vocabulary_kl_in_value_and_gradient(beta):
    W = _world()
    cache = W["cache"]
    # a student whose OFF-S ratio p/pi_ref is constant per row (delta supported on S) -- the
    # condition under which the partition loss and the full-vocab KL coincide exactly
    g = torch.Generator().manual_seed(1)
    delta = torch.zeros(W["lpi"].shape)
    for i in range(cache.n):
        S = cache.s_ids[i][cache.s_valid[i]]
        delta[i, S] = torch.randn(len(S), generator=g) * 0.7
    logits = (W["lpi"] + delta).clone().requires_grad_(True)
    loss, info = dl.delta_kl_loss(*_gather(logits, cache), cache, beta=beta)
    if beta == 1.0:
        ref = _reference_full_kl(logits, W, beta)                # the TRUE pi_ref: the floors cancel at beta = 1
        assert abs(float(_reference_full_kl(logits, W, beta, _cache_consistent_pi_ref(W))) - float(ref)) < 1e-6
    else:
        ref = _reference_full_kl(logits, W, beta, _cache_consistent_pi_ref(W))
        assert cache.pi_floored.any(), "the toy must exercise the K_q-minus-K' floor"
    assert abs(float(loss) - float(ref)) < 1e-6
    g1, = torch.autograd.grad(loss, logits); g2, = torch.autograd.grad(ref, logits)
    assert float((g1 - g2).abs().max()) < 1e-6
    assert info["n_soft"] >= 8 and info["n_hard"] == 2


def test_head_gathered_logits_matches_the_full_logits_in_value_and_gradient():
    torch.manual_seed(0)
    H, N = 16, 30
    head = torch.nn.Linear(H, V, bias=False)
    hidden = torch.randn(N + 1, H, requires_grad=True)
    rows = torch.arange(N)
    ids = torch.randint(0, V, (N, 7)); ids[:, -1] = -1
    g, lse = head_gathered_logits(head, hidden, rows, ids, chunk_rows=8)
    full = head(hidden)[rows]
    assert torch.allclose(g[:, :-1], full.gather(1, ids[:, :-1]), atol=1e-12) and torch.allclose(lse, torch.logsumexp(full, -1), atol=1e-12)
    s = (g[:, :-1] * 0.3 - lse[:, None]).sum(); s.backward(retain_graph=True)
    gh = hidden.grad.clone(); hidden.grad = None
    s2 = (full.gather(1, ids[:, :-1]) * 0.3 - torch.logsumexp(full, -1)[:, None]).sum(); s2.backward()
    assert torch.allclose(gh, hidden.grad, atol=1e-12)


# ---------------------------------------------------------------- 2
def _delta_zero_world():
    """teacher = pi_ref on every row (delta ≡ 0)."""
    W = _world(seed=3)
    W["lq"] = W["lpi"].clone()
    lpi, y = W["lpi"], W["y"]
    top = torch.topk(lpi, KP, -1)
    ref = dict(pos=np.arange(1, lpi.shape[0] + 1), y=y.numpy(), top_ids=top.indices.numpy(), top_lp=top.values.numpy(),
               lp_y=lpi[torch.arange(lpi.shape[0]), y].numpy())
    head = {}
    for i in range(lpi.shape[0]):
        if not W["soft"][i]:
            continue
        entry = {j: float(lpi[i, j]) for j in range(V)}
        h = head_from_entry(entry, int(y[i]), list(top.indices[i].numpy()) + [int(y[i])])
        h["log_rest"] = residual_log_mass(h["top_lp"], h["extra_lp"]); head[i + 1] = h
    W["cache"] = dl.build_delta_cache(ref, head, lpi[torch.arange(lpi.shape[0]), y].numpy(), tau=TAU)
    return W


def _check_2(W):
    cache = W["cache"]
    logits = W["lpi"].clone().requires_grad_(True)
    loss, _ = dl.delta_kl_loss(*_gather(logits, cache), cache, beta=1.0)
    g, = torch.autograd.grad(loss, logits)
    # the cache stores fp32 log-probs (as the real ref_topk / teacher_head parts do): 1e-6 is the design's tolerance
    assert abs(float(loss)) < 1e-6 and float(g.abs().max()) < 1e-6, "delta ≡ 0 and p_θ = pi_ref must give 0 / 0"
    # away from pi_ref the loss IS the coarsened KL(pi_ref || p_θ): the fill is pi_ref, not p_θ
    logits2 = (W["lpi"] + 0.8 * torch.randn(W["lpi"].shape, generator=torch.Generator().manual_seed(9))).clone().requires_grad_(True)
    loss2, _ = dl.delta_kl_loss(*_gather(logits2, cache), cache, beta=1.0)
    ref2 = _reference_partition_kl_pi_ref(logits2, W)
    assert float(loss2) > 1e-3
    assert abs(float(loss2) - float(ref2)) < 1e-6, "with delta ≡ 0 the target must be pi_ref itself"


def test_2_delta_zero_is_the_pi_ref_kl():
    _check_2(_delta_zero_world())


# ---------------------------------------------------------------- 3
def test_3_y_projections_equal_the_existing_lightning_loss():
    W = _world(seed=5)
    cache = W["cache"]; N = cache.n
    logits = W["lpi"].clone().requires_grad_(True)             # the student IS pi_ref, as at step 0
    loss_l, _ = dl.delta_kl_loss(*_gather(logits, cache), cache, beta=1.0, mode="lightning")
    # the existing loss on the same batch, in its own align-to-t vectors (position t scores token t)
    tl = torch.full((N + 1,), float("nan")); sl = torch.full((N + 1,), float("nan"))
    tl[1:] = W["lq"][torch.arange(N), W["y"]]
    sl_leaf = torch.log_softmax(logits, -1)[torch.arange(N), W["y"]]
    sl = torch.cat([torch.tensor([float("nan")]), sl_leaf])
    mask = torch.ones(N + 1, dtype=torch.bool); mask[0] = False
    loss_e, adv = compute_opd_loss(tl, sl, mask, tau=TAU)
    assert abs(float(loss_l) - float(loss_e)) < 1e-6                     # fp32 cache vs fp64 vectors
    ga, = torch.autograd.grad(loss_l, logits, retain_graph=True); gb, = torch.autograd.grad(loss_e, logits)
    assert float((ga - gb).abs().max()) < 1e-6
    # the square projection has the SAME gradient (its value is ½A², a different number by construction)
    loss_s, _ = dl.delta_kl_loss(*_gather(logits, cache), cache, beta=1.0, mode="y_square")
    gs, = torch.autograd.grad(loss_s, logits)
    assert float((gs - gb).abs().max()) < 1e-6


# ---------------------------------------------------------------- 4
def test_4_beta_half_is_the_geometric_interpolation_on_S():
    W = _world(seed=6)
    cache = W["cache"]
    lq_S, lq_B = dl.delta_targets(cache, beta=0.5)
    for i in range(cache.n):
        if not cache.is_soft[i]:
            continue
        S = cache.s_ids[i][cache.s_valid[i]]
        geo = 0.5 * cache.log_pi_ref[i][cache.s_valid[i]].double() + 0.5 * W["lq"][i, S]   # pi_ref as the cache holds it
        geo = geo - torch.logsumexp(geo, 0)
        tgt = lq_S[i][cache.s_valid[i]].double(); tgt = tgt - torch.logsumexp(tgt, 0)
        assert torch.allclose(tgt, geo, atol=1e-6)
    # and beta = 1 on a soft row is the teacher on S ∪ {B}
    lq_S1, lq_B1 = dl.delta_targets(cache, beta=1.0)
    i = int(torch.nonzero(cache.is_soft)[0])
    S = cache.s_ids[i][cache.s_valid[i]]
    assert torch.allclose(lq_S1[i][cache.s_valid[i]].double(), W["lq"][i, S], atol=1e-6)
    assert abs(float(lq_B1[i].exp()) - float(1 - W["lq"][i, S].exp().sum())) < 1e-6


# ---------------------------------------------------------------- 5
def test_5_y_outside_top_k_prime_is_still_in_S_and_the_loss_is_finite():
    W = _world(seed=7, y_outside=(1, 5))
    cache = W["cache"]
    for r in (1, 5):
        yy = int(W["y"][r])
        assert yy not in torch.topk(W["lpi"][r], KP).indices.tolist()
        assert int(cache.s_ids[r, cache.y_col[r]]) == yy
    logits = (W["lpi"] + 0.5).clone().requires_grad_(True)
    loss, info = dl.delta_kl_loss(*_gather(logits, cache), cache, beta=1.0)
    assert torch.isfinite(loss) and torch.isfinite(info["per_row"]).all()
    g, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(g).all()


# ---------------------------------------------------------------- 6
class _Alloc(torch.utils._python_dispatch.TorchDispatchMode):
    """Every op's output tensors: bytes, largest, and any fp32 tensor of the forbidden shape."""

    def __init__(self, forbidden):
        super().__init__(); self.total = 0; self.largest = 0; self.forbidden = forbidden; self.hits = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(t, torch.Tensor):
                self.total += t.numel() * t.element_size(); self.largest = max(self.largest, t.numel() * t.element_size())
                if t.dtype == torch.float32 and tuple(t.shape[-2:]) == self.forbidden:
                    self.hits.append((str(func), tuple(t.shape)))
        return out


@pytest.mark.skipif(os.environ.get("OPD_SKIP_MEM_TEST") == "1", reason="OPD_SKIP_MEM_TEST=1")
def test_6_memory_at_full_length_is_within_1p1x_and_no_seq_by_vocab_fp32():
    torch.set_default_dtype(torch.float32)
    try:
        SEQ, VV, H, M = 15360, 151936, 32, 40
        torch.manual_seed(0)
        head = torch.nn.Linear(H, VV, bias=False)
        ids = torch.randint(1, VV, (SEQ,))
        hidden = torch.randn(SEQ, H)
        N = SEQ - 1
        rows = torch.arange(N)
        s_ids = torch.randint(0, VV, (N, M)); s_ids[:, 0] = ids[1:]
        cache = dl.DeltaCache(pos=torch.arange(1, SEQ), is_soft=torch.ones(N, dtype=torch.bool), s_ids=s_ids,
                              s_valid=torch.ones(N, M, dtype=torch.bool), log_pi_ref=torch.log_softmax(torch.randn(N, M), -1) - 0.01,
                              log_q=torch.log_softmax(torch.randn(N, M), -1) - 0.01, in_J=torch.ones(N, M, dtype=torch.bool),
                              y_col=torch.zeros(N, dtype=torch.long), log_pi_rest=torch.full((N,), float(np.log(0.01))),
                              log_q_rest=torch.full((N,), float(np.log(0.01))), teacher_argmax=s_ids[:, 1], tau=TAU)
        mask = torch.ones(SEQ, dtype=torch.bool); mask[0] = False
        tlp = torch.randn(SEQ)

        def existing():
            h = hidden.clone().requires_grad_(True)
            lp = head_aligned_logprobs(head, h, ids)
            loss, _ = compute_opd_loss(tlp, lp, mask); loss.backward()

        def delta():
            h = hidden.clone().requires_grad_(True)
            g, lse = head_gathered_logits(head, h, rows, s_ids)
            loss, _ = dl.delta_kl_loss(g, lse, cache, beta=1.0); loss.backward()

        a = _Alloc((SEQ, VV)); b = _Alloc((SEQ, VV))
        with a:
            existing()
        with b:
            delta()
        assert b.hits == [] and a.hits == [], (b.hits, a.hits)
        assert b.largest < SEQ * VV * 4 / 8, "a chunk-sized transient is the largest tensor allowed"
        assert b.total <= 1.1 * a.total, f"delta allocates {b.total / a.total:.3f}x the existing loss's bytes"
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            head.to(dev); hidden_d = hidden.to(dev); ids_d = ids.to(dev); rows_d = rows.to(dev); s_ids_d = s_ids.to(dev)
            cache_d = cache.to(dev); mask_d = mask.to(dev); tlp_d = tlp.to(dev)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
            h = hidden_d.clone().requires_grad_(True); loss, _ = compute_opd_loss(tlp_d, head_aligned_logprobs(head, h, ids_d), mask_d); loss.backward()
            pa = torch.cuda.max_memory_allocated(dev); del h, loss; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
            h = hidden_d.clone().requires_grad_(True); g, lse = head_gathered_logits(head, h, rows_d, s_ids_d)
            loss, _ = dl.delta_kl_loss(g, lse, cache_d, beta=1.0); loss.backward()
            pb = torch.cuda.max_memory_allocated(dev)
            assert pb <= 1.1 * pa, f"GPU peak {pb / 2**30:.2f} GiB vs existing {pa / 2**30:.2f} GiB"
    finally:
        torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- 7
def test_7a_dropping_the_bucket_turns_test_1_red(monkeypatch):
    monkeypatch.setattr(dl, "_INCLUDE_BUCKET", False)
    with pytest.raises(AssertionError):
        test_1_matches_full_vocabulary_kl_in_value_and_gradient(1.0)


def test_7b_filling_with_stopgrad_student_turns_test_2_red(monkeypatch):
    monkeypatch.setattr(dl, "_fill_log_pi_ref", lambda cache, log_p_S, log_p_B: (log_p_S.detach(), log_p_B.detach()))
    with pytest.raises(AssertionError):
        _check_2(_delta_zero_world())


# ---------------------------------------------------------------- the step: mask identity and plumbing
def test_delta_train_step_uses_the_same_mask_and_refuses_a_misaligned_cache():
    from teacher.scorer import MockLM, Scorer
    torch.set_default_dtype(torch.float32)
    try:
        lm = MockLM(vocab_size=V, seed=2)
        sc = Scorer(lm, requires_grad=True)
        n_prompt, n_act = 4, 10
        ids = torch.randint(0, V, (n_prompt + n_act,))
        tt = torch.tensor([int(TokenType.PROMPT)] * n_prompt + [int(TokenType.ACTION)] * n_act)
        traj = Trajectory(input_ids=ids, token_type=tt)
        with torch.no_grad():
            full = torch.log_softmax(lm(ids).double(), -1)              # pi_ref := the student at step 0
        pos = torch.arange(n_prompt, n_prompt + n_act)
        lpi = full[pos - 1]; y = ids[pos]
        lq = torch.log_softmax(torch.randn(n_act, V, dtype=torch.float64) * 2, -1)
        top = torch.topk(lpi, KP, -1)
        ref = dict(pos=pos.numpy(), y=y.numpy(), top_ids=top.indices.numpy(), top_lp=top.values.numpy(), lp_y=lpi[torch.arange(n_act), y].numpy())
        head = {}
        for i in range(n_act):
            if top.values[i, 1] > dl.SOFT_X2:
                h = head_from_entry({j: float(lq[i, j]) for j in range(V)}, int(y[i]), list(top.indices[i].numpy()) + [int(y[i])])
                h["log_rest"] = residual_log_mass(h["top_lp"], h["extra_lp"]); head[int(pos[i])] = h
        cache = dl.build_delta_cache(ref, head, lq[torch.arange(n_act), y].numpy(), tau=TAU)
        teacher_logp = torch.full((n_prompt + n_act,), float("nan")); teacher_logp[pos] = lq[torch.arange(n_act), y].float()
        out = dl.delta_train_step(traj, teacher_logp, cache, sc, beta=1.0)
        assert torch.equal(out.score_mask, build_score_mask(tt)) and torch.isfinite(out.loss)
        out.loss.backward()
        assert lm.head.weight.grad is not None and torch.isfinite(lm.head.weight.grad).all()
        assert torch.isfinite(out.student_logp[pos]).all() and torch.isnan(out.student_logp[:n_prompt]).all()
        bad = dl.DeltaCache(**{**cache.__dict__, "pos": cache.pos + 1})
        with pytest.raises(AssertionError):
            dl.delta_train_step(traj, teacher_logp, bad, sc, beta=1.0)
    finally:
        torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- the ablations (MUST_RUN M3 / R21-R23)
def test_ablation_teacher_head_support_is_kq_and_y_only():
    """support="teacher": on soft rows S is exactly the teacher's K_q plus the realised token (no pi_ref
    top-K', no client extras); on hard rows S = {y}; the union support is a superset row by row."""
    from teacher.head import head_from_entry, residual_log_mass
    W = _world(N=12, seed=3)
    # rebuild the same world's inputs to call the producer with support="teacher"
    g = torch.Generator().manual_seed(3)
    lpi = torch.randn(12, V, generator=g) * 1.5
    for r in (3, 7):
        lpi[r, r] += 12.5
    lpi = torch.log_softmax(lpi, -1)
    lq = torch.log_softmax(torch.randn(12, V, generator=g) * 2.5, -1)
    y = torch.tensor([int(torch.multinomial(lpi[i].exp(), 1, generator=g)) for i in range(12)])
    top = torch.topk(lpi, KP, -1)
    ref = dict(pos=np.arange(1, 13), y=y.numpy(), top_ids=top.indices.numpy(), top_lp=top.values.numpy(), lp_y=lpi[torch.arange(12), y].numpy())
    head = {}
    for i in range(12):
        if top.values[i, 1] > dl.SOFT_X2:
            order = torch.argsort(lq[i], descending=True)
            entry = {int(t): float(lq[i, t]) for t in order}; entry[int(y[i])] = float(lq[i, y[i]])
            h = head_from_entry(entry, int(y[i]), extra_ids=list(top.indices[i].numpy()) + [int(y[i])], k_cap=32, cover=0.99)
            h["log_rest"] = residual_log_mass(h["top_lp"], h["extra_lp"]); head[i + 1] = h
    union = dl.build_delta_cache(ref, head, lq[torch.arange(12), y].numpy(), tau=TAU, support="union")
    teach = dl.build_delta_cache(ref, head, lq[torch.arange(12), y].numpy(), tau=TAU, support="teacher")
    assert union.n == teach.n == 12
    for i in range(12):
        S_u = set(union.s_ids[i][union.s_valid[i]].tolist()); S_t = set(teach.s_ids[i][teach.s_valid[i]].tolist())
        assert S_t <= S_u
        if bool(teach.is_soft[i]):
            kq = set(int(t) for t in head[i + 1]["top_ids"])
            assert S_t == kq | {int(y[i])}
            assert bool(union.s_valid[i].sum() >= teach.s_valid[i].sum())
        else:
            assert S_t == {int(y[i])}
        assert int(teach.s_ids[i, teach.y_col[i]]) == int(y[i])
        # the bucket holds more pi_ref mass under the smaller support
        assert float(teach.log_pi_rest[i]) >= float(union.log_pi_rest[i]) - 1e-6
    with pytest.raises(ValueError):
        dl.build_delta_cache(ref, head, lq[torch.arange(12), y].numpy(), tau=TAU, support="bogus")
    # the loss is finite and differentiable on the teacher-only support
    logits = (lpi + 0.3 * torch.randn(lpi.shape, generator=torch.Generator().manual_seed(11))).clone().requires_grad_(True)
    loss, info = dl.delta_kl_loss(*_gather(logits, teach), teach, beta=0.5)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def _teacher_world(N=12, seed=3):
    """The synthetic inputs of the teacher_head test (full-vocab teacher entries, so every extra is known)."""
    from teacher.head import head_from_entry, residual_log_mass
    g = torch.Generator().manual_seed(seed)
    lpi = torch.randn(N, V, generator=g) * 1.5
    for r in (3, 7):
        lpi[r, r] += 12.5
    lpi = torch.log_softmax(lpi, -1)
    lq = torch.log_softmax(torch.randn(N, V, generator=g) * 2.5, -1)
    y = torch.tensor([int(torch.multinomial(lpi[i].exp(), 1, generator=g)) for i in range(N)])
    top = torch.topk(lpi, KP, -1)
    ref = dict(pos=np.arange(1, N + 1), y=y.numpy(), top_ids=top.indices.numpy(), top_lp=top.values.numpy(), lp_y=lpi[torch.arange(N), y].numpy())
    head = {}
    for i in range(N):
        if top.values[i, 1] > dl.SOFT_X2:
            order = torch.argsort(lq[i], descending=True)
            entry = {int(t): float(lq[i, t]) for t in order}; entry[int(y[i])] = float(lq[i, y[i]])
            h = head_from_entry(entry, int(y[i]), extra_ids=list(top.indices[i].numpy()) + [int(y[i])], k_cap=32, cover=0.99)
            h["log_rest"] = residual_log_mass(h["top_lp"], h["extra_lp"]); head[i + 1] = h
    return lpi, lq, y, top, ref, head


def test_ablation_rho_support_is_the_teacher_prefix_at_coverage_rho_plus_y():
    """support="rho:<rho>" (ABLATIONS rho50..rho99): on soft rows S = the shortest teacher-rank prefix reaching mass rho,
    plus y; nested in rho; at rho = 0.99 it is the cached head's own K_q (the 'teacher' support's set); the bucket's
    teacher mass is 1 - the mass on S (every member known); hard rows S = {y}."""
    lpi, lq, y, top, ref, head = _teacher_world(N=12, seed=3)
    tq_y = lq[torch.arange(12), y].numpy()
    teach = dl.build_delta_cache(ref, head, tq_y, tau=TAU, support="teacher")
    caches = {r: dl.build_delta_cache(ref, head, tq_y, tau=TAU, support=f"rho:{r}") for r in (0.5, 0.8, 0.95, 0.99)}
    assert dl.rho_cover("rho:0.9") == 0.9 and dl.rho_cover("union") is None and dl.nested_size("rho:0.5") is None
    for bad in ("rho:0", "rho:0.995", "rho:1.0", "rho:x"):
        with pytest.raises(ValueError):
            dl.build_delta_cache(ref, head, tq_y, tau=TAU, support=bad)
    assert dl.prefix_for_cover([np.log(0.6), np.log(0.3), np.log(0.05)], 0.5) == 1 and dl.prefix_for_cover([np.log(0.6), np.log(0.3), np.log(0.05)], 0.85) == 2
    assert dl.prefix_for_cover([np.log(0.6), np.log(0.3), np.log(0.05)], 0.999) == 3
    for i in range(12):
        yy = int(y[i]); prev = set()
        for r in (0.5, 0.8, 0.95, 0.99):
            c = caches[r]; S = c.s_ids[i][c.s_valid[i]].tolist()
            assert int(c.s_ids[i, c.y_col[i]]) == yy and yy in S and prev <= set(S)
            prev = set(S)
            if bool(c.is_soft[i]):
                h = head[i + 1]; k = dl.prefix_for_cover(h["top_lp"], r)
                assert set(S) == set(int(t) for t in h["top_ids"][:k]) | {yy}
                assert sum(np.exp(h["top_lp"][:k])) >= r or k == len(h["top_lp"])
                assert bool(c.in_J[i][c.s_valid[i]].all())
                assert float(c.log_q_rest[i]) == pytest.approx(float(torch.log(1 - lq[i, S].exp().sum())), abs=1e-6)
                # pi_ref side: members outside pi_ref's top-K' carry the floor (the union support's convention), so the
                # residual is 1 - the cache's OWN pi mass on S; members inside top-K' (and y) carry the exact value
                v = c.s_valid[i]
                exp_rest = float(np.log(max(1.0 - float(c.log_pi_ref[i][v].exp().sum()), 1e-12)))      # the builder's own clamp
                assert float(c.log_pi_rest[i]) == pytest.approx(exp_rest, abs=1e-5)
                for j, tkn in enumerate(S):
                    if not bool(c.pi_floored[i, j]):
                        assert float(c.log_pi_ref[i, j]) == pytest.approx(float(lpi[i, tkn]), abs=1e-6)
            else:
                assert S == [yy] and np.isnan(float(c.log_q_rest[i]))
        # rho = 0.99 reproduces the cached head's own K_q ∪ {y} (the teacher-only support), set for set
        S99 = set(caches[0.99].s_ids[i][caches[0.99].s_valid[i]].tolist()); St = set(teach.s_ids[i][teach.s_valid[i]].tolist())
        assert S99 == St
    logits = (lpi + 0.4 * torch.randn(lpi.shape, generator=torch.Generator().manual_seed(7))).clone().requires_grad_(True)
    loss, info = dl.delta_kl_loss(*_gather(logits, caches[0.8]), caches[0.8], beta=0.5)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert dl.ABLATIONS["rho95"]["support"] == "rho:0.95"


def test_ablation_nested_support_is_y_plus_pi_ref_ranks_and_the_dropped_mass_is_in_the_bucket():
    """support="nested:n" (ABLATIONS supp1/2/4): S = {y} then pi_ref's ranks to |S| = n, nested in n and
    inside the union; BOTH residuals are 1 - the mass on the truncated S (the dropped tokens' pi_ref AND
    teacher mass fold into B, the design choice of batch 3 P0-A); n = 1 is {y} ∪ B and its loss equals the
    two-cell forward KL written longhand."""
    lpi, lq, y, top, ref, head = _teacher_world()
    N = lpi.shape[0]
    tq_y = lq[torch.arange(N), y].numpy()
    union = dl.build_delta_cache(ref, head, tq_y, tau=TAU, support="union")
    nested = {n: dl.build_delta_cache(ref, head, tq_y, tau=TAU, support=f"nested:{n}") for n in (1, 2, 4)}
    assert dl.nested_size("nested:4") == 4 and dl.nested_size("union") is None
    for bad in ("nested:0", "nested", "bogus", "nested:x"):
        with pytest.raises(ValueError):
            dl.build_delta_cache(ref, head, tq_y, tau=TAU, support=bad)
    for i in range(N):
        yy = int(y[i]); S_u = set(union.s_ids[i][union.s_valid[i]].tolist())
        ranks = [int(t) for t in top.indices[i] if int(t) != yy]
        prev = set()
        for n in (1, 2, 4):
            c = nested[n]; S = c.s_ids[i][c.s_valid[i]].tolist()
            assert S == [yy] + ranks[:n - 1]                               # y first, then pi_ref's ranks, K_q never joins
            assert int(c.y_col[i]) == 0 and set(S) <= S_u and prev <= set(S)
            prev = set(S)
            # pi_ref values are exact (no floors: every member is in top-K' or is y)
            assert not bool(c.pi_floored[i].any())
            assert float(c.log_pi_rest[i]) == pytest.approx(float(torch.log(1 - lpi[i, S].exp().sum())), abs=1e-6)
            if bool(c.is_soft[i]):
                # every member has the teacher's value (the entries are full-vocab here), and the bucket
                # is 1 - the teacher mass on S: the union's server residual is NOT reused
                assert bool(c.in_J[i][c.s_valid[i]].all()) and not bool(c.q_floored[i].any())
                assert float(c.log_q_rest[i]) == pytest.approx(float(torch.log(1 - lq[i, S].exp().sum())), abs=1e-6)
                assert torch.allclose(c.log_q[i][c.s_valid[i]], lq[i, S].float(), atol=1e-6)
                assert float(c.log_q_rest[i]) > float(union.log_q_rest[i]) - 1e-9      # a smaller S leaves more in B
            else:
                assert bool(c.in_J[i][0]) and not bool(c.in_J[i][1:].any()) and np.isnan(float(c.log_q_rest[i]))
    # n = 1: the loss is the two-cell forward KL on {y, B}, longhand, with q_hat ∝ (pi^(1-β) q^β, pi_B^(1-β) q_B^β)
    beta = 0.5
    logits = (lpi + 0.4 * torch.randn(lpi.shape, generator=torch.Generator().manual_seed(5))).clone().requires_grad_(True)
    c1 = nested[1]
    loss, info = dl.delta_kl_loss(*_gather(logits, c1), c1, beta=beta)
    logp = torch.log_softmax(logits, -1)
    total = 0.0
    for i in range(N):
        yy = int(y[i]); py, piy, qy = logp[i, yy].exp(), lpi[i, yy].exp(), lq[i, yy].exp()
        if bool(c1.is_soft[i]):
            w_y = piy ** (1 - beta) * qy ** beta; w_B = (1 - piy) ** (1 - beta) * (1 - qy) ** beta
        else:
            d = torch.clamp(lq[i, yy] - lpi[i, yy], -TAU, TAU)
            w_y = piy * torch.exp(beta * d); w_B = 1 - piy
        qh_y, qh_B = w_y / (w_y + w_B), w_B / (w_y + w_B)
        total = total + qh_y * (qh_y.log() - py.log()) + qh_B * (qh_B.log() - (1 - py).log())
    assert abs(float(loss) - float(total / N)) < 1e-6
    g1, = torch.autograd.grad(loss, logits); g2, = torch.autograd.grad(total / N, logits)
    assert float((g1 - g2).abs().max()) < 1e-6
    assert dl.ABLATIONS["supp1"]["support"] == "nested:1" and dl.ABLATIONS["supp4"]["support"] == "nested:4"


def test_ablation_no_bucket_equals_the_test7a_mutation(monkeypatch):
    W = _world(N=12, seed=4); c = W["cache"]
    logits = (W["lpi"] + 0.5 * torch.randn(W["lpi"].shape, generator=torch.Generator().manual_seed(12))).clone()
    l_param, _ = dl.delta_kl_loss(*_gather(logits, c), c, beta=0.5, include_bucket=False)
    monkeypatch.setattr(dl, "_INCLUDE_BUCKET", False)
    l_hook, _ = dl.delta_kl_loss(*_gather(logits, c), c, beta=0.5)
    monkeypatch.setattr(dl, "_INCLUDE_BUCKET", True)
    l_full, _ = dl.delta_kl_loss(*_gather(logits, c), c, beta=0.5)
    assert torch.allclose(l_param, l_hook) and not torch.allclose(l_param, l_full)
    lS, lB = dl.delta_targets(c, 0.5, include_bucket=False)
    assert (lB < -1e20).all() and torch.allclose(torch.where(c.s_valid, lS.exp(), torch.zeros_like(lS)).sum(1), torch.ones(c.n, dtype=lS.dtype))


def test_ablation_no_anchor_equals_the_test7b_mutation(monkeypatch):
    W = _delta_zero_world(); c = W["cache"]
    logits = (W["lpi"] + 0.8 * torch.randn(W["lpi"].shape, generator=torch.Generator().manual_seed(9))).clone()
    l_param, _ = dl.delta_kl_loss(*_gather(logits, c), c, beta=1.0, anchor="student")
    l_ref, _ = dl.delta_kl_loss(*_gather(logits, c), c, beta=1.0)
    monkeypatch.setattr(dl, "_fill_log_pi_ref", lambda cache, log_p_S, log_p_B: (log_p_S.detach(), log_p_B.detach()))
    l_hook, _ = dl.delta_kl_loss(*_gather(logits, c), c, beta=1.0)
    assert torch.allclose(l_param, l_hook) and not torch.allclose(l_param, l_ref)
    with pytest.raises(ValueError):
        dl.delta_kl_loss(*_gather(logits, c), c, beta=1.0, anchor="bogus")
    assert set(dl.ABLATIONS) == {"none", "teacher_head", "no_bucket", "no_anchor", "supp1", "supp2", "supp4", "rho50", "rho80", "rho90", "rho95", "rho99"}
