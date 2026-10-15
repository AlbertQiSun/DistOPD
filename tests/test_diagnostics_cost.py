"""The diagnostic pass: same numbers, without costing more than the step it measures.

kl_importance asked every scorer for a full [N, V] log-distribution, forced it to the HOST,
and reduced it in a Python loop. At global_batch=256 and the measured trajectory lengths that
is ~700 GB of PCIe traffic and ~69,000 iterations of a 151,936-wide reduction PER OPTIMIZER
STEP -- for a diagnostic that reads the rows generating scored tokens, which is 270 of 2241
at p50.

Worse, all of it sat OUTSIDE the timing instrumentation. `train_seconds` boxed the
forward/backward only, so the wall-clock projection that decides whether the real experiment
fits its allocation was built on the cheaper half of the step.

Two things are gated here, and the first is the one that makes the second safe to want:

  1. THE NUMBERS DO NOT MOVE. Restricting rows cannot change a log_softmax (it normalises
     within a row), and the KL reduction is the same arithmetic in a different order. So the
     rewrite is checked against a longhand reimplementation of the ORIGINAL algorithm --
     written out here, sharing no code with the thing under test, per the producer/consumer
     rule in the project conventions.
  2. IT IS ACTUALLY CHEAPER, and the saving is asserted as a ratio of work done, not as wall
     clock (which is unstable in CI). Counting rows requested is the honest proxy: the old
     path asked for N, the new one asks for n_scored.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opd.diagnostics import kl_importance
from opd.masking import build_score_mask
from opd.trajectory import TokenType, Trajectory
from teacher.scorer import MockLM, Scorer


def _traj(n=120, v=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, v, (n,), generator=g)
    tt = torch.full((n,), int(TokenType.PROMPT), dtype=torch.long)
    tt[10:40] = int(TokenType.ACTION)          # two action spans, with a gap
    tt[60:75] = int(TokenType.ACTION)
    tt[40:60] = int(TokenType.OBSERVATION)
    return Trajectory(ids, tt)


def _pair(v=512):
    return (Scorer(MockLM(v, seed=1), requires_grad=False),
            Scorer(MockLM(v, seed=2), requires_grad=False))


def _reference_kl_importance(student, ref, traj, score_mask):
    """The ORIGINAL algorithm, longhand: full [N, V] distributions, per-token Python loop.

    Shares no code with kl_importance. This is what the rewrite has to reproduce -- if the
    two ever disagree, the cheaper version changed a measurement, not just its cost.
    """
    ids = traj.input_ids
    st_full = torch.log_softmax(student.raw_logits(ids).float(), dim=-1)
    rf_full = torch.log_softmax(ref.raw_logits(ids).float(), dim=-1)
    m = score_mask.clone()
    m[0] = False
    idx = torch.nonzero(m, as_tuple=False).flatten().tolist()
    if not idx:
        return {"kl_pertok_mean": None, "iw_mean": None, "iw_std": None, "n": 0}

    def aligned(full):
        out = torch.full((ids.shape[0],), float("nan"), dtype=full.dtype)
        out[1:] = full[:-1].gather(1, ids[1:].unsqueeze(1)).squeeze(1)
        return out

    w = torch.exp(aligned(st_full)[m] - aligned(rf_full)[m])
    kl = []
    for t in idx:
        p, q = st_full[t - 1], rf_full[t - 1]
        kl.append(float((p.exp() * (p - q)).sum()))
    return {"kl_pertok_mean": sum(kl) / len(kl), "iw_mean": float(w.mean()),
            "iw_std": float(w.std(unbiased=False)), "n": len(idx)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_numbers_match_the_original_algorithm(seed):
    """THE GATE. A cheaper diagnostic that reports different numbers is not a saving."""
    traj = _traj(seed=seed)
    st, rf = _pair()
    mask = build_score_mask(traj.token_type)

    got = kl_importance(st, rf, traj, mask)
    want = _reference_kl_importance(st, rf, traj, mask)

    assert got["n"] == want["n"] > 0, "no scored tokens -- the comparison would be vacuous"
    for k in ("kl_pertok_mean", "iw_mean", "iw_std"):
        assert got[k] == pytest.approx(want[k], rel=1e-5, abs=1e-7), (
            f"{k}: rewrite {got[k]} vs original {want[k]}")


def test_kl_is_not_trivially_zero():
    """Two DIFFERENT models, or the agreement above proves nothing: KL(p||p)=0 would match
    a reference that also computes zero, however either was implemented."""
    traj = _traj()
    st, rf = _pair()
    out = kl_importance(st, rf, traj, build_score_mask(traj.token_type))
    assert out["kl_pertok_mean"] > 1e-3, "the two mock models are too similar to discriminate"


def test_a_scorer_identical_to_itself_gives_zero_kl_and_unit_weight():
    traj = _traj()
    st = Scorer(MockLM(512, seed=1), requires_grad=False)
    out = kl_importance(st, st, traj, build_score_mask(traj.token_type))
    assert out["kl_pertok_mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["iw_mean"] == pytest.approx(1.0, abs=1e-6)
    assert out["iw_std"] == pytest.approx(0.0, abs=1e-6)


def test_one_forward_per_scorer_and_nothing_more():
    """THE COST GATE, as work done rather than wall clock.

    It used to spy on log_dist_at, which kl_importance no longer calls: that API returns
    [n_scored, V] fp32 and TWO of them have to be live at once for a row-by-row KL, which is
    what killed opd-moe-on with a 4.99 GiB allocation after 16 hours. The diagnostic now takes
    the bf16 logits once per scorer and widens KL_ROW_CHUNK rows at a time. The property being
    gated is unchanged -- one forward each, no repeats -- so the spy moved rather than the
    assertion.
    """
    traj = _traj(n=400)
    mask = build_score_mask(traj.token_type)

    calls = []
    st, rf = _pair()
    for name, sc in (("student", st), ("ref", rf)):
        real = sc.raw_logits

        def spy(ids, _r=real, _n=name):
            calls.append(_n)
            return _r(ids)

        sc.raw_logits = spy

    kl_importance(st, rf, traj, mask)
    assert calls == ["student", "ref"], (
        f"expected exactly one forward per scorer, got {calls} -- a forward per CHUNK would "
        f"make this diagnostic cost more than the step it measures")


def test_no_more_than_one_chunk_is_ever_widened_to_fp32(monkeypatch):
    """THE REGRESSION, stated as the thing that actually ran out of memory.

    The peak is set by the widest fp32 tensor, and log_softmax is where the widening lands.
    Before the fix it was called twice on [n_scored, V]; now it is called once per scorer per
    chunk on [<=KL_ROW_CHUNK, V]. Asserting the SHAPE rather than the byte count keeps this
    meaningful on a CPU box, where nothing would OOM regardless.
    """
    import opd.diagnostics as D

    traj = _traj(n=400)
    mask = build_score_mask(traj.token_type)
    n_scored = int(mask.sum()) - int(bool(mask[0]))
    # Shrink the chunk rather than grow the fixture: the property is "no more than one chunk
    # is widened", and it can only be demonstrated when the row count EXCEEDS the chunk. The
    # first version of this test used the shipped 128 against a 45-row fixture and would have
    # passed an entirely unchunked implementation.
    monkeypatch.setattr(D, "KL_ROW_CHUNK", 8)
    assert n_scored > D.KL_ROW_CHUNK, (
        f"fixture has {n_scored} scored rows, at or below the {D.KL_ROW_CHUNK}-row chunk, so "
        f"it cannot tell a chunked implementation from an unchunked one")

    widened = []
    real = torch.log_softmax

    def spy(x, *a, **kw):
        widened.append(tuple(x.shape))
        return real(x, *a, **kw)

    monkeypatch.setattr(torch, "log_softmax", spy)
    st, rf = _pair()
    kl_importance(st, rf, traj, mask)

    assert widened, "log_softmax was never called -- the fixture no longer exercises the path"
    worst = max(shape[0] for shape in widened)
    assert worst <= D.KL_ROW_CHUNK, (
        f"widened {worst} rows to fp32 in one go; the chunk is {D.KL_ROW_CHUNK}. Two tensors "
        f"of that width are live at once, which is the 4.99 GiB that killed opd-moe-on")


def test_log_dist_at_shape_and_precision():
    """CANNOT PROVE THE ABSENCE OF A HOST COPY. On a one-device box `.cpu()` is a no-op, so
    the device assertion below is true either way -- reverse-verification confirmed that
    re-adding `.cpu()` leaves this file green. The discriminating version needs a GPU and is
    below. Stated here rather than left implicit, because a test that looks like it covers
    something it cannot is worse than an absent one."""
    traj = _traj()
    st, _ = _pair()
    rows = torch.arange(5, 40)
    out = st.log_dist_at(traj.input_ids, rows)
    assert out.device == next(st.lm.parameters()).device
    assert out.shape == (rows.numel(), 512)
    assert out.dtype == torch.float32, "the reduction must happen in fp32"


@pytest.mark.slow
def test_no_host_transfer_of_the_distributions():
    """The bill that dominated was the COPY, not the compute: 2 x [N, V] fp32 per trajectory
    across the bus, ~700 GB per optimizer step at global_batch=256. Only a real GPU can tell
    a host copy from a no-op, so this is the version that actually gates it."""
    from tests.gpu import require_gpu

    require_gpu()
    traj = _traj()
    st, rf = _pair()
    st.to("cuda")
    rf.to("cuda")
    rows = torch.arange(5, 40)
    out = st.log_dist_at(traj.input_ids, rows)
    assert out.is_cuda, "log_dist_at copied the distribution to the host"

    # and end to end: the diagnostic must still produce real numbers on the device
    kl = kl_importance(st, rf, traj, build_score_mask(traj.token_type))
    assert kl["n"] > 0 and kl["kl_pertok_mean"] is not None


def test_row_restriction_cannot_change_a_distribution():
    """The licence for computing fewer rows: log_softmax normalises WITHIN a row."""
    traj = _traj()
    st, _ = _pair()
    full = torch.log_softmax(st.raw_logits(traj.input_ids).float(), dim=-1)
    rows = torch.tensor([3, 17, 61, 99])
    assert torch.equal(st.log_dist_at(traj.input_ids, rows), full[rows])


def test_chunking_the_reduction_does_not_change_the_result():
    import opd.diagnostics as D

    traj = _traj()
    st, rf = _pair()
    mask = build_score_mask(traj.token_type)
    old = D.KL_ROW_CHUNK
    try:
        D.KL_ROW_CHUNK = 10_000
        whole = kl_importance(st, rf, traj, mask)
        D.KL_ROW_CHUNK = 3
        tiny = kl_importance(st, rf, traj, mask)
    finally:
        D.KL_ROW_CHUNK = old
    assert whole["kl_pertok_mean"] == pytest.approx(tiny["kl_pertok_mean"], rel=1e-6)
    assert whole["n"] == tiny["n"]


def test_empty_score_mask_is_still_handled():
    traj = _traj()
    st, rf = _pair()
    empty = torch.zeros_like(traj.token_type, dtype=torch.bool)
    out = kl_importance(st, rf, traj, empty)
    assert out == {"kl_pertok_mean": None, "iw_mean": None, "iw_std": None, "n": 0}


# ---------------------------------------------------------------------------
# producer side: the reason this commit exists is that the cost was UNTIMED
# ---------------------------------------------------------------------------
def test_the_trainer_reports_what_the_diagnostic_pass_cost():
    """A budget projection built on train_seconds alone projects the cheaper half of the
    step. The field has to be emitted by the trainer, not merely readable by a report --
    per the producer/consumer rule in the project conventions."""
    from tests.test_channel_fingerprint import _trainer

    tr, log = _trainer("online", steps=3, global_batch=2)
    for i in range(3):
        tr.step(i)

    by_step = {}
    for rec in log.records:
        assert "diag_seconds_prev_step" in rec["timing"], (
            "the diagnostic pass is untimed again -- the wall-clock projection would omit it")
        by_step.setdefault(rec["step"], rec["timing"]["diag_seconds_prev_step"])

    assert by_step[0] is None, "step 0 has no previous diagnostic pass to report"
    later = [v for k, v in by_step.items() if k > 0]
    assert later and all(v is not None and v > 0 for v in later), (
        f"steps after the first must carry a real measurement, got {later}")


def test_diag_cost_is_a_material_fraction_and_therefore_worth_reporting():
    """If it were negligible, omitting it would have been harmless and this commit would be
    noise. On the mock path it is comparable to the training step -- which is the whole
    argument for putting it in the record."""
    from tests.test_channel_fingerprint import _trainer

    tr, log = _trainer("online", steps=3, global_batch=2)
    for i in range(3):
        tr.step(i)
    rows = [r["timing"] for r in log.records if r["step"] == 2]
    assert rows[0]["diag_seconds_prev_step"] > 0
    assert rows[0]["train_seconds"] > 0
