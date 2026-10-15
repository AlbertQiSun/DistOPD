"""Gate 3 (@slow): teacher-scoring parity on real weights, split into the two
questions that actually matter -- and NOT the one that never happens.

TEACHER BACKEND = vllm_http (served). Both OPD arms score the teacher with the SAME
served vLLM backend (model.engine = vllm_http, one config value read by online and
offline alike; gate 1 / assert_same_backend rejects a disk cache scored by a different
backend). So "does HF agree with vLLM" is a scenario the EXPERIMENT never enters -- HF
is not a teacher backend at all. What the experiment DOES depend on is decomposed into
two independent questions, tested by two DIFFERENT gates that must not be conflated:

  (A) DETERMINISM -- is vLLM's prompt_logprobs bit-invariant to batch composition?
      This is a THIRD-PARTY, UNDOCUMENTED behavior (vLLM guarantees batch-invariant
      SAMPLING output under VLLM_BATCH_INVARIANT=1; prompt_logprobs is a different code
      path). It is settled by the DETERMINISM PROBE -- score the same ids ALONE vs
      CO-BATCHED, flag on, assert bit-identical -- a phase-2 first-pass HPC gate. It is
      NOT this file's job. If the probe fails, we fall back to a measured epsilon and
      gate 3a is knowingly weakened; only the probe can tell us that.

  (B) SERIALIZATION -- does "score once, save to disk, read back" preserve the numbers?
      This is MY code (the precompute record: input_ids / token_type / teacher_logp /
      backend / meta, plus mask reconstruction from stored fields). It is the HARD GATE
      below.

WHY (A) AND (B) MUST STAY SEPARATE (the mirror of gate 3's original 'protected the
wrong object' mistake). The real invariant is online-live-score == offline-disk-score.
That equality needs BOTH (a) two forwards agree [determinism] AND (b) a saved forward
reads back unchanged [serialization]. If the HARD GATE scored the teacher TWICE (once
live, once in precompute) and compared them, its result would fold batch
non-determinism INTO the serialization check -- even with batch-invariant on, a failure
could not be attributed. So the HARD GATE scores exactly ONCE and round-trips that
single forward through disk. No second forward => batch composition is not a variable
here, by construction. The probe owns (a); this gate owns (b); their composition is
online==offline.

  Structural proof of the separation: the HARD GATE does NOT set VLLM_BATCH_INVARIANT
  and does NOT enforce_eager. It doesn't need to -- a single forward compared to its own
  disk copy is invariant to batch composition trivially. That the determinism flag is
  UNNECESSARY here is exactly what shows serialization and determinism are cleanly split.

So gate 3 is now two parts:

  (a) HARD GATE -- test_vllm_teacher_logp_survives_disk_roundtrip
      The operative serialization invariant: the vLLM teacher logp, scored ONCE, is
      bit-identical after precompute -> save -> read back (float32 store => exact). This
      is milestone-4's through-disk parity re-proven on the REAL vLLM teacher backend.
      ASSUMES determinism has been proven separately by the HPC probe; it verifies only
      that the disk boundary changes no numbers.

  (b) DIAGNOSTIC -- test_vllm_teacher_is_aligned_vs_hf_reference
      An independent HF forward is used ONLY as an alignment reference (never a teacher
      backend): if vLLM's prompt_logprobs were shifted by a BOS/template/dtype
      misalignment, the WHOLE distribution would move to nat scale. We assert the MEDIAN
      HF-vs-vLLM drift is tiny (bf16 kernel noise is sparse; a real misalignment is
      systematic). This is the alignment sanity the pure-id /score_logprobs contract
      relies on -- complementary to the server-side BOS length guard (which catches a
      length shift; this catches a same-length value shift). Evidence, never pass/fail on
      the max.

ONE vLLM ENGINE, SHARED. vLLM V1 does not support two live LLM engines in one process
-- a second engine pollutes the first's CUDA graphs / global state and silently corrupts
its logits. Both tests share a SINGLE module-scoped LLM (built once) and merely wrap it.

RUNBOOK -- do NOT "fix" a HARD-GATE failure by widening a tolerance. float32 save/load is
exact; a failure means the precompute record's store/load dtype or serialization is
broken -- inspect that, never the tolerance. Do NOT "fix" the diagnostic tail by forcing
HF and vLLM to agree in bf16 (a rabbit hole bounded by bf16 itself). And do NOT move the
determinism question into either test -- that is the probe's job.

Requires vllm + real weights + GPU; skipped by default (pytest.ini: -m "not slow"). Run
under VLLM_ENABLE_V1_MULTIPROCESSING=0 to avoid the V1 EngineCore subprocess deadlocking
with pytest capture.
"""
from __future__ import annotations

import os
import tempfile

import pytest

# Default = HF id (works anywhere with hub access). On an OFFLINE HPC node, export
# OPD_TEST_MODEL=/path/to/local/Qwen3-1.7B so from_pretrained reads local weights.
MODEL = os.environ.get("OPD_TEST_MODEL", "Qwen/Qwen3-1.7B")


@pytest.fixture(scope="module")
def vllm_engine():
    """ONE vLLM engine for the whole module. Two live vLLM V1 engines in a single
    process corrupt each other's state, so we build the engine once and share it.

    NOTE: no VLLM_BATCH_INVARIANT / enforce_eager here -- the HARD GATE performs a
    SINGLE forward and compares it to its own disk copy, which is invariant to batch
    composition by construction. Batch invariance is the DETERMINISM PROBE's concern
    (a separate HPC gate), not this module's."""
    pytest.importorskip("vllm")
    from tests.gpu import require_gpu
    require_gpu()
    from vllm import LLM

    from tests.vllm_engine import vllm_engine as _engine
    with _engine(model=MODEL, dtype="bfloat16") as llm:
        yield llm          # torn down for real: see tests/vllm_engine.py


@pytest.mark.slow
def test_vllm_teacher_logp_survives_disk_roundtrip(vllm_engine):
    """HARD GATE (serialization only): the vLLM teacher logp, scored ONCE, is
    bit-identical after precompute -> save -> read back. The teacher backend is vLLM
    (served in real runs; the in-process TeacherVLLMScorer here shares the identical
    backend-agnostic _save_precomputed / load code, so this validates the served
    path's serialization too -- HTTP transport is tested separately).

    Scores EXACTLY ONCE and compares the returned logp to its own disk copy: no second
    forward, so batch non-determinism cannot enter. Determinism is assumed proven by the
    HPC probe; this gate only asserts the disk boundary changes no numbers."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from envs.hf_tokenizer import HFTokenizerAdapter
    from opd.masking import build_score_mask
    from opd.trajectory import Trajectory, TrajectoryBuilder, TokenType
    from teacher.backend import BackendConfig
    from teacher.scorer import TeacherScorer, TeacherVLLMScorer

    tok = HFTokenizerAdapter(transformers.AutoTokenizer.from_pretrained(MODEL))
    # A realistic mixed sequence: prompt (not scored) + an action span (scored).
    prompt_ids = tok.render_prompt_ids("What is the capital of France?")
    action_ids = torch.tensor(tok.encode("The capital of France is Paris."), dtype=torch.long)
    traj = (
        TrajectoryBuilder()
        .add(prompt_ids, TokenType.PROMPT)
        .add(action_ids, TokenType.ACTION)
        .build()
    )

    # The real teacher backend: vLLM (bf16). engine tag is vllm_http (served) in real
    # runs; the in-process wrapper reuses the SAME serialization, which is what we test.
    cfg = BackendConfig(engine="vllm_http", dtype="bfloat16", tokenizer_name=MODEL)
    teacher = TeacherVLLMScorer(vllm_engine, cfg)

    store_mask = build_score_mask(traj.token_type)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vllm_teacher.pt")
        # score ONCE: precompute_and_save returns the exact logp it wrote to disk.
        saved_logp = teacher.precompute_and_save(traj, path, store_dtype="float32")
        rec = TeacherScorer.load(path)

    assert rec["logp_dtype"] == "float32", "stored dtype must be recorded explicitly"
    assert rec["teacher_logp"].dtype == torch.float32
    assert torch.equal(rec["input_ids"].long(), traj.input_ids), "ids must survive disk"

    traj_off = Trajectory(rec["input_ids"].long(), rec["token_type"].long())
    read_mask = build_score_mask(traj_off.token_type)
    assert torch.equal(read_mask, store_mask), "score_mask must be bit-equal through disk"
    assert bool(read_mask.any()), "the fixed trajectory must have scored (action) positions"

    # THE serialization invariant: the SAME single forward, in-memory vs read-from-disk,
    # bit-identical at the scored action positions. float32 store => exact (torch.equal,
    # not allclose). A mismatch is a store/load dtype or serialization bug, never a
    # tolerance to widen -- and never batch non-determinism, which cannot arise from one
    # forward compared to its own disk copy.
    assert torch.equal(saved_logp[read_mask], rec["teacher_logp"][read_mask]), (
        "vLLM teacher logp changed between the in-memory forward and its disk copy -- "
        "serialization/dtype breakage in the precompute record"
    )


@pytest.mark.slow
def test_vllm_teacher_is_aligned_vs_hf_reference(vllm_engine):
    """DIAGNOSTIC (not a parity gate): an independent HF forward is used ONLY as an
    alignment reference (HF is never a teacher backend). If vLLM's prompt_logprobs were
    shifted by a BOS/template/dtype misalignment, the WHOLE distribution would move to
    nat scale. We assert the MEDIAN drift is tiny -- bf16 kernel noise is sparse, a real
    misalignment is systematic. This is the alignment sanity the pure-id /score_logprobs
    contract relies on (complementary to the server-side BOS length guard). The max drift
    is recorded as evidence, never asserted."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from envs.hf_tokenizer import HFTokenizerAdapter
    from teacher.backend import BackendConfig
    from teacher.scorer import HFScorer, VLLMScorer

    tok = HFTokenizerAdapter(transformers.AutoTokenizer.from_pretrained(MODEL))
    # A SINGLE short prompt (~20-30 tokens, mostly template/special tokens) is too small
    # to compute a stable median -- the statistic is then dominated by the template mix,
    # not by whether the two backends condition the CONTENT distribution the same way.
    # Feed several multi-sentence prompts through the REAL chat template so the token
    # population is large and content-heavy: that both exercises BOS/template alignment
    # and gives the median enough support to be meaningful.
    prompts = [
        "What is the capital of France, and what is that city best known for?",
        "Explain in a few sentences why the sky appears blue during the day and red at "
        "sunset, mentioning Rayleigh scattering.",
        "Summarize the plot of a classic fairy tale and describe its central moral lesson.",
        "Describe how a bicycle stays upright while moving, and what happens when it slows "
        "down to a stop.",
    ]

    hf = HFScorer.from_pretrained(MODEL, dtype="bfloat16")
    vl = VLLMScorer(vllm_engine, BackendConfig(engine="vllm_http", dtype="bfloat16",
                                               tokenizer_name=MODEL))

    drifts = []
    for p in prompts:
        ids = tok.render_prompt_ids(p)
        hf_lp = hf.logprobs(ids)
        vl_lp = vl.logprobs(ids)
        finite = torch.isfinite(hf_lp) & torch.isfinite(vl_lp)
        drifts.append((hf_lp[finite] - vl_lp[finite]).abs())
    drift = torch.cat(drifts)

    # ALIGNMENT PROOF via a ROBUST statistic, not a max. bf16 HF-vs-vLLM divergence is
    # not tail-only: at high-entropy positions (many near-tied candidates) even a
    # ~16%-probability token's logprob is kernel-rounding-sensitive and can differ by
    # ~0.2. That is bf16 noise, not misalignment. The discriminator is SYSTEMATIC vs
    # SPARSE -- a real dtype/BOS/template misalignment shifts the WHOLE distribution
    # (median blows up); kernel noise hits only a sparse set of soft/tail positions
    # (median stays tiny). max cannot tell these apart; the median can.
    #
    # THRESHOLD BY SCALE, NOT BY THE OBSERVED NUMBER. bf16 kernel noise puts the median at
    # CENTINAT scale (~0.0x), while a genuine dtype/BOS/template break shifts essentially
    # every token and drives the median to NAT scale (>=0.5, often 1-5). 0.1 sits between
    # them -- ~10x below the misalignment floor and above the bf16-noise band. It is NOT a
    # tolerance tuned to pass: a median of 0.5+ (a real bug) fails it by 5x. If this ever
    # fails, the bulk has moved to nat scale => real misalignment; fix the
    # dtype/BOS/template, do NOT raise this number.
    median_drift = float(drift.median())
    max_drift = float(drift.max())
    frac_within_0p1 = float((drift <= 0.1).float().mean())
    assert median_drift < 0.1, (
        f"median HF-vs-vLLM logprob drift {median_drift:.4f} >= 0.1 => the BULK of "
        "tokens disagree at nat scale, i.e. the vLLM teacher is mis-aligned "
        "(dtype/BOS/template) -- investigate, do NOT loosen this"
    )

    print(
        f"[gate3-diagnostic] vLLM-teacher vs HF-reference abs logprob drift over "
        f"{int(drift.numel())} tokens  median={median_drift:.4f}  max={max_drift:.4f}  "
        f"frac<=0.1={frac_within_0p1:.2%}  (median at centinat scale => vLLM teacher "
        "aligned; sparse max spikes => bf16 kernel noise)"
    )
