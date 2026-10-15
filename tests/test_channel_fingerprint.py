"""Gate on the hardware/stack half of the backend fingerprint.

A second execution channel (different GPU type, different CUDA/vLLM build) is planned so
work can go wherever the queue is shorter. The rule attached to it is that one experiment's
precompute and both its training arms stay inside ONE channel -- cross-channel is fine,
cross-arm is not, because teacher logprobs computed on different hardware are different
numbers and that difference would sit exactly where the effect is supposed to be.

Before these fields, nothing machine-checked that rule. engine, dtype, tokenizer, template
and the vLLM settings are all IDENTICAL across channels by construction, so an A800-scored
cache and an H20-scored cache carried the same fingerprint and gate 1 accepted either. The
constraint lived only in whoever remembered it -- and every failure this project has hit has
been that shape.

So: the fingerprint must SEPARATE two channels, the manifest must record who SAMPLED as well
as who SCORED, and the analysis-time check must refuse a mixed pair.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.trainer import MANIFEST_NAME, OPDConfig, precompute_offline
from rollout.policy import MockPolicy
from teacher.backend import (
    ENGINE_RUNTIME_FIELDS,
    BackendConfig,
    assert_same_backend,
    runtime_fingerprint,
)
from teacher.scorer import MockLM, TeacherScorer

HW = ("gpu_name", "gpu_sm", "torch_version", "vllm_version")

H20 = dict(gpu_name="NVIDIA H20", gpu_sm="sm_90",
           torch_version="2.11.0+cu130", vllm_version="0.25.1")
A800 = dict(gpu_name="NVIDIA A800-SXM4-80GB", gpu_sm="sm_80",
            torch_version="2.8.0+cu126", vllm_version="0.11.0")


def _cfg(**kw):
    return BackendConfig(engine="vllm_http", dtype="bfloat16",
                         tokenizer_name="/models/MiroThinker-1.7-mini",
                         max_model_len=17408, max_num_batched_tokens=8192,
                         gpu_memory_utilization=0.72, **kw)


def test_runtime_fingerprint_reports_the_four_axes():
    rt = runtime_fingerprint()
    assert set(rt) == set(HW)
    assert rt["torch_version"], "torch version must always be knowable"


@pytest.mark.parametrize("field", HW)
def test_every_hardware_axis_is_part_of_the_fingerprint(field):
    """Each one alone must be enough to reject. gpu_sm in particular: the batch-invariant
    kernels are parameterised by SM count, so two cards of the same name but different
    capability are still different channels."""
    assert field in ENGINE_RUNTIME_FIELDS, f"{field} is not adopted from /health"
    with pytest.raises(AssertionError, match=field):
        assert_same_backend(_cfg(**H20), _cfg(**{**H20, field: A800[field]}))


def test_the_exact_case_that_used_to_slip_through():
    """Same model, same template, same dtype, same vLLM settings -- different machine. This
    combination was ACCEPTED before, and it is the one the two-channel plan produces."""
    with pytest.raises(AssertionError):
        assert_same_backend(_cfg(**H20), _cfg(**A800))


def test_same_channel_still_passes():
    assert_same_backend(_cfg(**H20), _cfg(**H20))


def test_unrecorded_is_distinguishable_from_matching():
    """An old artifact carries None. That must NOT read as 'matches whatever you compare it
    to' -- an unstamped cache is unknown provenance, not agreement."""
    with pytest.raises(AssertionError):
        assert_same_backend(_cfg(**H20), _cfg())


def test_manifest_records_who_SAMPLED_not_only_who_scored():
    """The teacher's machine arrives via /health. The SAMPLER's is this process's own, and
    nothing else in the artifact records it -- so a cache rolled out on one channel and
    consumed by an arm on another would look perfectly consistent."""
    with tempfile.TemporaryDirectory() as d:
        tok = MockTokenizer(vocab_size=2048)
        cfg = OPDConfig(mode="offline", precompute_dir=d, global_batch=2)
        precompute_offline(cfg, ["q1?", "q2?"], MockPolicy(tok, ["Paris"], "Paris"),
                           MockRetriever(["doc"], 0), tok,
                           TeacherScorer(MockLM(tok.vocab_size, seed=1)))
        man = json.load(open(os.path.join(d, MANIFEST_NAME), encoding="utf-8"))

    assert "sampler" in man, "the manifest does not say which machine rolled out this cache"
    assert set(man["sampler"]) == set(HW)
    assert man["sampler"]["torch_version"]
    assert "backend" in man and "gpu_sm" in man["backend"], (
        "the teacher's machine must be in the backend block too")


# ---------------------------------------------------------------------------
# the analysis-time check: the only place both arms are visible at once
# ---------------------------------------------------------------------------
def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "chan", pathlib.Path(__file__).resolve().parents[1] / "scripts"
        / "check_channel_parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHK = _load_checker()


def _diag(path, runtime, n=3, cache_sampler=None, code_version="abc1234"):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            rec = {"step": i, "micro": 0, "runtime": runtime,
                   "code_version": code_version}
            if cache_sampler is not None:
                rec["cache_sampler_runtime"] = cache_sampler
            f.write(json.dumps(rec) + "\n")


def _run(argv):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = CHK.main()
    return code, buf.getvalue()


def _with_argv(argv):
    old = sys.argv
    sys.argv = ["check_channel_parity.py", *argv]
    try:
        return _run(argv)
    finally:
        sys.argv = old


def test_matching_arms_pass():
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20)
        _diag(off, H20, cache_sampler=H20)
        man = os.path.join(d, "manifest.json")
        json.dump({"sampler": H20, "backend": {**H20, "engine": "vllm_http"}},
                  open(man, "w", encoding="utf-8"))
        code, out = _with_argv([on, off, "--manifest", man])
    assert code == 0, out
    assert "OK: one channel throughout" in out


def test_arms_on_different_hardware_are_REFUSED():
    """THE GATE. This is the mistake the two-channel plan makes easy."""
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20)
        _diag(off, A800, cache_sampler=A800)
        code, out = _with_argv([on, off])
    assert code == 1
    assert "more than one execution channel" in out
    assert "Do NOT compare these runs" in out


def test_a_cache_sampled_on_the_other_channel_is_caught():
    """Subtler: both arms TRAIN on H20, but the offline cache was rolled out on A800. The
    per-arm runtimes agree; only the cache's sampler betrays it."""
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20)
        _diag(off, H20, cache_sampler=A800)
        code, out = _with_argv([on, off])
    assert code == 1, out
    assert "cache sampler" in out


def test_arms_on_different_CODE_VERSIONS_are_REFUSED():
    """Same hardware, same stack -- different COMMITS. Numerics-relevant since the
    chunked-lm_head change (2026-08-19, ~1 ulp vs the full-logits era): two commits are
    two kernel paths, the same confound as a hardware split, one level up. The concrete
    pairing this catches: the online smoke finished under the full-logits code the day
    before the fused path landed."""
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20, code_version="040a99c")            # full-logits era
        _diag(off, H20, cache_sampler=H20, code_version="deadbee")  # fused era
        code, out = _with_argv([on, off])
    assert code == 1, out
    assert "code_version differs across arms" in out
    assert "DIFFERENT numerics" in out


def test_equated_code_versions_pass_when_the_diff_avoids_the_numerics_surface(monkeypatch):
    """The earned special-case green (2026-08-19: offline ran from 0c7c41a, the online
    launcher fix landed as a later commit touching only slurm/ + analysis files). The
    gate itself runs the diff and accepts ONLY when nothing the training process
    imports or reads changed -- launcher-level differences are covered by the per-row
    stamps the gate already compares. No human adjudication note involved."""
    monkeypatch.setattr(CHK, "_git_diff_files",
                        lambda a, b: ["slurm/train_online.sbatch",
                                      "scripts/check_channel_parity.py",
                                      "tests/test_channel_fingerprint.py"])
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20, code_version="deadbee")
        _diag(off, H20, cache_sampler=H20, code_version="0c7c41a")
        code, out = _with_argv([on, off, "--equate-code-versions", "0c7c41a,deadbee"])
    assert code == 0, out
    assert "ACCEPTED" in out and "OK: one channel throughout" in out


def test_equated_code_versions_REFUSED_when_the_diff_moves_numerics(monkeypatch):
    """The flag must not be a bypass: a diff touching the numerics surface is exactly
    what the code_version stamp exists to catch, and no operator convenience overrides
    it."""
    monkeypatch.setattr(CHK, "_git_diff_files",
                        lambda a, b: ["slurm/train_online.sbatch", "opd/loss.py"])
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20, code_version="deadbee")
        _diag(off, H20, cache_sampler=H20, code_version="0c7c41a")
        code, out = _with_argv([on, off, "--equate-code-versions", "0c7c41a,deadbee"])
    assert code == 1, out
    assert "opd/loss.py" in out and "REFUSED" in out


def test_equate_without_git_is_an_unverifiable_claim(monkeypatch):
    """Same doctrine as an unstamped artifact: if the equivalence cannot be verified,
    it is refused -- not trusted because someone typed the flag."""
    def _boom(a, b):
        raise RuntimeError("no git here")
    monkeypatch.setattr(CHK, "_git_diff_files", _boom)
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20, code_version="deadbee")
        _diag(off, H20, cache_sampler=H20, code_version="0c7c41a")
        code, out = _with_argv([on, off, "--equate-code-versions", "0c7c41a,deadbee"])
    assert code == 1, out
    assert "REFUSED" in out and "unverifiable" in out


def test_rows_without_code_version_are_holes_not_agreement():
    """Pre-stamp artifacts must read as INCOMPLETE, exactly like unstamped runtimes: an
    absent stamp is unknown provenance, not a match."""
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20, code_version=None)
        _diag(off, H20, cache_sampler=H20, code_version=None)
        code, out = _with_argv([on, off])
    assert code == 1, out
    assert "code_version" in out


def test_an_unstamped_file_is_not_treated_as_agreement():
    with tempfile.TemporaryDirectory() as d:
        on, off = os.path.join(d, "on.jsonl"), os.path.join(d, "off.jsonl")
        _diag(on, H20)
        with open(off, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 0, "micro": 0}) + "\n")
        code, out = _with_argv([on, off])
    assert code == 1
    assert "NO runtime stamp" in out


def test_a_split_INSIDE_one_arm_is_caught():
    """A resumed run whose second half landed on another node is a split inside one arm --
    which the arms-agree check alone would miss if both arms happened to be mixed the same
    way."""
    with tempfile.TemporaryDirectory() as d:
        on = os.path.join(d, "on.jsonl")
        with open(on, "w", encoding="utf-8") as f:
            for rt in (H20, H20, A800):
                f.write(json.dumps({"step": 0, "micro": 0, "runtime": rt}) + "\n")
        code, out = _with_argv([on])
    assert code == 1
    assert "more than one execution channel" in out


# ---------------------------------------------------------------------------
# the PRODUCER side: the checker above is fed synthetic rows, which proves nothing about
# whether the trainer actually writes them. Deleting the stamp left all 14 tests green.
# ---------------------------------------------------------------------------
def _trainer(mode, precompute_dir=None, steps=1, global_batch=2):
    from opd.diagnostics import DiagnosticRunner
    from opd.trainer import OfflineSource, OnlineSource, OPDTrainer, snapshot_ref
    from teacher.scorer import Scorer

    tok = MockTokenizer(vocab_size=8192)
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = Scorer(MockLM(tok.vocab_size, seed=7), requires_grad=True)
    cfg = OPDConfig(mode=mode, steps=steps, global_batch=global_batch, seed=3,
                    precompute_dir=precompute_dir)
    qs, golds = [f"q{i}?" for i in range(4)], [["Paris"]] * 4
    policy, retr = MockPolicy(tok, ["Paris"], "Paris"), MockRetriever(["a doc"], seed=0)
    if mode == "online":
        src = OnlineSource(qs, policy, retr, tok, teacher, cfg, golds=golds)
    else:
        src = OfflineSource(qs, policy, retr, tok, teacher, cfg,
                            expected_backend=teacher.config, golds=golds)
    diag = DiagnosticRunner(ref_scorer=snapshot_ref(student), tau=cfg.tau, embedder=None)

    class _Log:
        def __init__(self): self.records = []
        def log(self, rec): self.records.append(rec.to_dict())
        def close(self): pass

    log = _Log()
    return OPDTrainer(student, src, cfg, diagnostics=diag, logger=log), log


def test_the_trainer_stamps_every_row_it_writes():
    """Without this the parity checker is a well-tested reader of a field nobody emits."""
    trainer, log = _trainer("online")
    trainer.step(0)
    assert log.records, "no diagnostic records were produced"
    for rec in log.records:
        assert "runtime" in rec, "a diagnostic row cannot say which machine produced it"
        assert set(rec["runtime"]) == set(HW)


def test_offline_rows_carry_the_CACHE_sampler_too():
    """An offline row involves two machines -- the one training now and the one that rolled
    out its data -- and only the second can reveal a cache imported from another channel."""
    with tempfile.TemporaryDirectory() as d:
        trainer, log = _trainer("offline", precompute_dir=d)
        trainer.step(0)
        assert log.records
        for rec in log.records:
            assert "runtime" in rec
            assert "cache_sampler_runtime" in rec, (
                "an offline row does not record where its trajectories were sampled")
            assert set(rec["cache_sampler_runtime"]) == set(HW)


def test_online_rows_do_NOT_claim_a_cache_sampler():
    """Online samples and trains in one process; inventing the field there would assert a
    provenance split that does not exist."""
    trainer, log = _trainer("online")
    trainer.step(0)
    assert all("cache_sampler_runtime" not in r for r in log.records)
