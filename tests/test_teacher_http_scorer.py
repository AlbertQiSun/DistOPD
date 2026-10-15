"""Step-1 CPU tests for the SERVED teacher: the pure-id HTTP contract, the two
server-side guards (BOS length + score-span), the cross-service gate-1 fingerprint,
and the determinism-probe HARNESS.

These run WITHOUT vLLM or a GPU by standing up the REAL server handler
(slurm/teacher_server.py::make_handler) over a real socket, driven by a MOCK
scorer_fn that produces vLLM-shaped prompt_logprobs. So the guards exercised here
are the actual production guard code (_aligned_logprobs), not a re-implementation.

HPC-ONLY, NOT covered here (a mock cannot reproduce them -- they need real vLLM):
  * whether vLLM actually injects a BOS on a TokensPrompt (the BOS *guard* is tested;
    vLLM's real behavior is the determinism/BOS probe's job on HPC);
  * whether prompt_logprobs is BIT-invariant under VLLM_BATCH_INVARIANT=1 (the probe
    HARNESS -- score alone vs co-batched, assert bit-identical -- is tested; the real
    bit-identity assertion is a phase-2 first-pass HPC gate).
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import threading
import urllib.error
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataclasses import asdict  # noqa: E402

from opd.masking import build_score_mask, scored_indices  # noqa: E402
from opd.trajectory import TokenType, TrajectoryBuilder  # noqa: E402
from teacher.backend import BackendConfig  # noqa: E402
from teacher.http_scorer import TeacherHTTPError, TeacherHTTPScorer  # noqa: E402


def _load_teacher_server():
    spec = importlib.util.spec_from_file_location(
        "teacher_server", ROOT / "slurm" / "teacher_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TS = _load_teacher_server()


class _LP:
    """Minimal stand-in for vLLM's Logprob (has a `.logprob` attribute)."""

    def __init__(self, v: float) -> None:
        self.logprob = v


def _fake_prompt_logprobs(input_ids, extra_bos: bool = False):
    """vLLM-shaped prompt_logprobs: list len N, [0]=None, [t]={ids[t]: Logprob}.
    Deterministic value -0.01*t so tests can assert exact numbers. extra_bos makes
    it length N+1 to simulate a BOS-injected response (drives the BOS guard)."""
    ids = list(input_ids)
    pl = [None]
    for t in range(1, len(ids)):
        pl.append({ids[t]: _LP(-0.01 * t)})
    if extra_bos:
        pl.append({ids[-1]: _LP(-0.99)})
    return pl


def _make_backend_dict(dtype="bfloat16", tokenizer_name="mock-teacher"):
    return asdict(BackendConfig(engine="vllm_http", dtype=dtype, tokenizer_name=tokenizer_name))


@contextmanager
def serve(scorer_fn, backend_dict):
    """Run the REAL server handler on an ephemeral port; yield the /score_logprobs URL."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), TS.make_handler(scorer_fn, backend_dict))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}/score_logprobs"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def _normal_scorer(input_ids, score_positions):
    """Route through the PRODUCTION guard code (_aligned_logprobs)."""
    pl = _fake_prompt_logprobs(input_ids)
    return TS._aligned_logprobs(pl, input_ids, score_positions), len(pl)


# ---------------------------------------------------------------------------
def test_roundtrip_returns_aligned_tensor():
    ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    with serve(_normal_scorer, _make_backend_dict()) as url:
        scorer = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        logp = scorer.logprobs(ids)
    assert logp.shape == ids.shape
    assert torch.isnan(logp[0]), "position 0 has no prefix -> nan (align-to-t)"
    assert torch.isfinite(logp[1:]).all()
    # values survive the JSON round-trip exactly (float32)
    expected = torch.tensor([float("nan")] + [-0.01 * t for t in range(1, 5)])
    assert torch.allclose(logp[1:], expected[1:], atol=1e-6)


def test_score_positions_match_build_score_mask_across_service():
    """Guard (a): the positions the client SENDS are exactly its build_score_mask
    ACTION indices -- the cross-service form of the M2 masking-parity check."""
    prompt_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    action_ids = torch.tensor([20, 21, 22, 23], dtype=torch.long)
    traj = (TrajectoryBuilder()
            .add(prompt_ids, TokenType.PROMPT)
            .add(action_ids, TokenType.ACTION)
            .build())
    expected_positions = scored_indices(build_score_mask(traj.token_type)).tolist()
    assert expected_positions == [3, 4, 5, 6]

    captured = {}

    def recording_scorer(input_ids, score_positions):
        captured["positions"] = score_positions
        return _normal_scorer(input_ids, score_positions)

    import tempfile
    with serve(recording_scorer, _make_backend_dict()) as url:
        scorer = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "traj.pt")
            logp = scorer.precompute_and_save(traj, path, store_dtype="float32")
            rec = torch.load(path, weights_only=False)

    # the server received EXACTLY the client's ACTION positions
    assert captured["positions"] == expected_positions
    # and the disk round-trip preserves position alignment: build_score_mask over the
    # STORED token_type selects the same set, with finite teacher logp there.
    read_mask = build_score_mask(rec["token_type"].long())
    assert scored_indices(read_mask).tolist() == expected_positions
    assert torch.isfinite(logp[read_mask]).all()
    assert torch.isfinite(rec["teacher_logp"][read_mask]).all()


def test_bos_length_guard_rejects_injected_prefix():
    """Guard (b): a length-N+1 response (as if vLLM prepended BOS) is refused, not
    silently mis-aligned. The GUARD is tested here; whether real vLLM injects a BOS
    is the HPC probe's job (a mock cannot reproduce vLLM tokenization)."""
    ids = [5, 6, 7, 8]
    # direct: the production guard raises on a length mismatch.
    with pytest.raises(TS.ScoreError):
        TS._aligned_logprobs(_fake_prompt_logprobs(ids, extra_bos=True), ids, None)

    # end-to-end: the server turns it into a 400, and the client surfaces it.
    def bos_scorer(input_ids, score_positions):
        pl = _fake_prompt_logprobs(input_ids, extra_bos=True)
        return TS._aligned_logprobs(pl, input_ids, score_positions), len(pl)

    with serve(bos_scorer, _make_backend_dict()) as url:
        scorer = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        # The client must raise TeacherHTTPError, NOT a bare urllib HTTPError: urllib raises
        # before anyone reads the response body, so the server's stated reason was being
        # thrown away (a real precompute died with only "HTTP Error 500" and no diagnosable
        # trace on either side). `.code` is still preserved for status branching.
        with pytest.raises(TeacherHTTPError) as ei:
            scorer.logprobs(torch.tensor(ids, dtype=torch.long))
        assert ei.value.code == 400
        # and the SERVER's reason must survive into the client-visible message -- this is
        # the part whose absence cost a debugging round-trip.
        assert "BOS/prefix" in str(ei.value), str(ei.value)
        assert "length" in ei.value.detail


def test_span_guard_rejects_drifted_position():
    """Guard (a) server-side: a score position at index 0 or >= N is refused --
    it means the client's span drifted from a valid ACTION set."""
    ids = [5, 6, 7, 8]
    with pytest.raises(TS.ScoreError):
        TS._aligned_logprobs(_fake_prompt_logprobs(ids), ids, [0])       # position 0 has no prefix
    with pytest.raises(TS.ScoreError):
        TS._aligned_logprobs(_fake_prompt_logprobs(ids), ids, [len(ids)])  # out of range


def test_determinism_probe_harness_mock():
    """The determinism-probe HARNESS: score the same ids ALONE vs CO-BATCHED (here,
    two independent calls) and assert bit-identical. The mock is deterministic so it
    passes trivially -- it exercises the harness + the bit-identity comparison, NOT
    vLLM's real batch-invariance, which is the phase-2 first-pass HPC gate."""
    ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    with serve(_normal_scorer, _make_backend_dict()) as url:
        scorer = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        alone = scorer.logprobs(ids)
        cobatched = scorer.logprobs(ids)
    finite = torch.isfinite(alone) & torch.isfinite(cobatched)
    assert torch.equal(alone[finite], cobatched[finite]), "probe harness: must be bit-identical"


def test_fingerprint_gate_across_service():
    """Cross-service gate 1: the client refuses to bind to a served teacher whose
    backend fingerprint differs from its own expectation (template/dtype/tokenizer)."""
    served = _make_backend_dict(dtype="bfloat16", tokenizer_name="miro-teacher")
    with serve(_normal_scorer, served) as url:
        # matching client: constructs cleanly
        TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="miro-teacher")
        # mismatched tokenizer: construction must fail fast
        with pytest.raises(AssertionError):
            TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="different-teacher")
        # mismatched dtype: also fails
        with pytest.raises(AssertionError):
            TeacherHTTPScorer(url, dtype="float32", tokenizer_name="miro-teacher")


def test_config_engine_is_vllm_http():
    """The served client tags its backend vllm_http (not a generic vllm), so gate 1's
    fingerprint distinguishes it from any in-process vLLM logp."""
    served = _make_backend_dict()
    with serve(_normal_scorer, served) as url:
        scorer = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
    assert scorer.config.engine == "vllm_http"


def test_engine_runtime_is_adopted_from_health_not_asserted():
    """The client cannot know the server's max_model_len, so gate 1 must ADOPT it rather
    than assert on it -- asserting would fail every startup. Adoption is what lets the
    DISK boundary catch a teacher restarted under different settings."""
    backend = _make_backend_dict()
    backend.update(max_model_len=17408, max_num_batched_tokens=8192,
                   gpu_memory_utilization=0.72)
    with serve(lambda ids, sp: (_fake_prompt_logprobs(ids), len(ids)), backend) as url:
        scorer = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
    assert scorer.config.max_model_len == 17408
    assert scorer.config.max_num_batched_tokens == 8192
    assert scorer.config.gpu_memory_utilization == 0.72


def test_engine_runtime_drift_is_caught_at_the_disk_boundary():
    """The point of recording it: a cache scored under one chunk budget must not be read
    by a run using another. max_num_batched_tokens decides chunk boundaries, and
    VLLM_BATCH_INVARIANT pins batch composition, NOT chunking -- so this can move numbers
    with nothing else to notice."""
    from teacher.backend import BackendConfig, assert_same_backend

    at_precompute = BackendConfig(engine="vllm_http", dtype="bfloat16",
                                  tokenizer_name="miro", max_model_len=17408,
                                  max_num_batched_tokens=8192)
    at_read = BackendConfig(engine="vllm_http", dtype="bfloat16",
                            tokenizer_name="miro", max_model_len=17408,
                            max_num_batched_tokens=4096)          # teacher restarted
    with pytest.raises(AssertionError) as ei:
        assert_same_backend(at_read, at_precompute)
    assert "max_num_batched_tokens" in str(ei.value)


# --- Task-2 step 3: token-budgeted batch scoring ---------------------------------------
def _batch_scorer(items):
    return [(_normal_scorer(ids, sp)) for ids, sp in items]


def _serve_batching(record=None):
    """Real handler wired to a scorer_fn that also exposes .batch, as production does."""
    def scorer_fn(input_ids, score_positions):
        return _normal_scorer(input_ids, score_positions)
    def batch(items):
        if record is not None:
            record.append([len(ids) for ids, _ in items])
        return _batch_scorer(items)
    scorer_fn.batch = batch
    return scorer_fn


def test_batch_scoring_matches_one_at_a_time():
    seqs = [torch.arange(5, 5 + n, dtype=torch.long) for n in (40, 60, 30)]
    with serve(_serve_batching(), _make_backend_dict()) as url:
        sc = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        singles = [sc.logprobs(s) for s in seqs]
        batched = sc.logprobs_batch([(s, None) for s in seqs])
    assert len(batched) == len(singles)
    for i, (a, b) in enumerate(zip(singles, batched)):
        assert torch.equal(torch.nan_to_num(a, nan=-1.0), torch.nan_to_num(b, nan=-1.0)), i


def test_requests_are_grouped_by_TOKENS_not_by_count():
    """Grouping must follow token counts: lengths here span 10x, so a fixed
    trajectories-per-request rule would put wildly different work in each request."""
    calls = []
    seqs = [torch.arange(1, 1 + n, dtype=torch.long) for n in (30, 30, 30, 200)]
    with serve(_serve_batching(calls), _make_backend_dict()) as url:
        sc = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        out = sc.logprobs_batch([(s, None) for s in seqs], token_budget=100)
    assert len(out) == 4
    # 30+30+30=90 fits; 200 exceeds the budget alone and is sent by itself.
    assert calls == [[30, 30, 30], [200]], calls


def test_oversized_trajectory_is_sent_alone_not_dropped():
    calls = []
    big = torch.arange(1, 1 + 500, dtype=torch.long)
    with serve(_serve_batching(calls), _make_backend_dict()) as url:
        sc = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        out = sc.logprobs_batch([(big, None)], token_budget=100)
    assert len(out) == 1 and out[0].shape[0] == 500
    assert calls == [[500]]


def test_short_batch_response_is_refused():
    """A dropped result would shift every later trajectory onto another's logprobs."""
    def dropping(input_ids, score_positions):
        return _normal_scorer(input_ids, score_positions)
    dropping.batch = lambda items: _batch_scorer(items)[:-1]
    seqs = [torch.arange(1, 1 + n, dtype=torch.long) for n in (20, 25)]
    with serve(dropping, _make_backend_dict()) as url:
        sc = TeacherHTTPScorer(url, dtype="bfloat16", tokenizer_name="mock-teacher")
        with pytest.raises(TeacherHTTPError) as ei:
            sc.logprobs_batch([(s, None) for s in seqs], token_budget=1000)
    assert "1 results for 2" in str(ei.value)


def test_published_fingerprint_matches_the_engine_that_was_built():
    """/health must advertise what the engine ACTUALLY runs.

    It once advertised max_num_batched_tokens=null while the engine ran 8192, because the
    CLI arg (default None) and the engine builder (shared default 8192) resolved the value
    independently. A fingerprint that misreports its configuration is worse than none: it
    would later flag a false drift (null -> 8192) and could never say what a cache was
    really scored under. Exercising the module's own symbols also catches the NameError a
    py_compile pass cannot -- the resolver has to actually be imported where it is used.
    """
    ek = TS.teacher_engine_kwargs(dtype="bfloat16", max_model_len=17408,
                                  gpu_memory_utilization=0.72, tensor_parallel_size=1)
    cfg = BackendConfig(engine="vllm_http", dtype=ek["dtype"], tokenizer_name="miro",
                        max_model_len=ek["max_model_len"],
                        max_num_batched_tokens=ek["max_num_batched_tokens"],
                        gpu_memory_utilization=ek["gpu_memory_utilization"])
    published = asdict(cfg)
    for field in ("max_model_len", "max_num_batched_tokens", "gpu_memory_utilization"):
        assert published[field] == ek[field], field
    assert published["max_num_batched_tokens"] == 8192, "the certified budget must be published"
    assert TS.config_summary(**ek)          # the log line the job records must render


def test_build_scorer_signature_matches_how_main_calls_it():
    """_build_scorer needs vLLM, so no CPU test ever CALLS it -- which is how a signature
    change landed on the call site but not the definition and only failed on the node, at
    server startup, after a 27-second model load. py_compile and the rest of this suite
    both passed. Binding the signature is the cheapest check that the two agree."""
    import inspect
    ek = TS.teacher_engine_kwargs(dtype="bfloat16")
    # exactly the call main() makes: (model, resolved_engine_kwargs)
    inspect.signature(TS._build_scorer).bind("/path/to/model", ek)
    # and it must NOT still accept the old (model, dtype, max_model_len, **kw) shape
    with pytest.raises(TypeError):
        inspect.signature(TS._build_scorer).bind("/m", "bfloat16", 17408)
