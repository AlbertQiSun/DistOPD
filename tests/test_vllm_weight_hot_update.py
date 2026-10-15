"""Confound #1, real-machine half (@slow, HPC): does vLLM's weight hot-update
actually LAND in the live sampler?

WHY THIS TEST EXISTS
--------------------
vLLM keeps its OWN copy of the weights; the trainer updates a separate training-side
copy (student.lm). `VLLMPolicy.sync_from` is the one bridge -- it calls
`model.load_weights(...)` to push the updated student weights into the running engine
so the next rollouts are truly on-policy. If that push silently no-ops (wrong param
names, a device/dtype mismatch swallowed by copy_, prefix-cache masking, a vLLM
internal-format change), the engine keeps SAMPLING from stale weights and the online
arm degrades into a frozen, offline-like run. Worse: a resync BUG produces an
"online worse than offline" curve that is INDISTINGUISHABLE from real OPD poisoning --
the exact confound this whole phase exists to eliminate.

The mock (tests/test_online_resync.py) proves the resync SEMANTICS without a GPU
(bit-equality after sync; staleness == step - last_sync_step; snapshot-not-alias).
It CANNOT prove the real vLLM path, because it has no live inference engine. This
test proves the real half, and only a GPU can:

  1. LANDS + CAUSAL -- perturb the student's weights, sync_from, and the live
     sampler's OWN greedy prompt-logprobs move far beyond same-engine determinism.
     (If load_weights no-oped, the bulk change would be ~0.)
  2. CLEAN + REVERSIBLE -- push the ORIGINAL weights back through the same path and
     the engine reproduces baseline logprobs bit-close. Only possible if load_weights
     truly OVERWRITES the live sampler weights (not appends, not partially writes).
  3. SELF-CALIBRATED -- the revert is orders of magnitude tighter than the
     perturbation, so the proof does not hinge on guessing the perturbation size.

PREFIX CACHING IS DISABLED for this engine on purpose: with it on, vLLM keys its KV
cache on tokens, not weights, so re-scoring the SAME prompt after a weight swap could
return STALE cached logprobs and mask the very thing we are testing. temperature=0
greedy prompt_logprobs is the deterministic probe: on a fixed prompt it is a pure
function of the sampler's current weights.

Requires vllm + real weights + GPU; skipped by default (pytest.ini: -m "not slow").
Run under VLLM_ENABLE_V1_MULTIPROCESSING=0 (V1 EngineCore subprocess deadlocks with
pytest capture).
"""
from __future__ import annotations

import os

import pytest

# Offline HPC: export OPD_TEST_MODEL=/path/to/local/Qwen3-1.7B (default is the HF id).
MODEL = os.environ.get("OPD_TEST_MODEL", "Qwen/Qwen3-1.7B")


@pytest.fixture(scope="module")
def vllm_engine_no_cache():
    """ONE vLLM engine, prefix caching OFF so every score recomputes with the CURRENT
    weights (a token-keyed KV cache would otherwise return stale logprobs after a
    weight swap and mask the update)."""
    pytest.importorskip("vllm")
    from tests.gpu import require_gpu
    require_gpu()
    from vllm import LLM

    from tests.vllm_engine import vllm_engine as _engine
    with _engine(model=MODEL, dtype="bfloat16", enable_prefix_caching=False) as llm:
        yield llm          # torn down for real: see tests/vllm_engine.py


def _perturb_(lm, scale: float = 0.1, seed: int = 0) -> None:
    """Move the student's weights like an optimizer step would: add Gaussian noise at
    `scale` * (per-tensor std) to every float parameter. Big enough to shift logprobs
    well past bf16 same-engine determinism, small enough to stay finite (RMSNorm damps
    multiplicative blow-up), deterministic via a fixed CPU generator."""
    import torch

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in lm.parameters():
            if not p.dtype.is_floating_point:
                continue
            std = p.detach().float().std().clamp_min(1e-6)
            noise = torch.randn(p.shape, generator=g, dtype=torch.float32) * (scale * std)
            p.add_(noise.to(p.dtype))


@pytest.mark.slow
def test_vllm_sync_from_rewrites_live_sampler_weights(vllm_engine_no_cache):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from envs.hf_tokenizer import HFTokenizerAdapter
    from rollout.vllm_policy import VLLMPolicy
    from teacher.backend import BackendConfig
    from teacher.scorer import HFScorer, VLLMScorer

    tok = HFTokenizerAdapter(transformers.AutoTokenizer.from_pretrained(MODEL))
    ids = tok.render_prompt_ids("What is the capital of France?")

    # Probe = the SAMPLER's own greedy prompt-logprobs (deterministic, temp=0): on a
    # fixed prompt this is a pure function of the engine's current weights.
    vl = VLLMScorer(vllm_engine_no_cache,
                    BackendConfig(engine="vllm", dtype="bfloat16", tokenizer_name=MODEL))
    policy = VLLMPolicy(vllm_engine_no_cache, tok, max_action_tokens=2560)

    # BASELINE -- engine still holds the checkpoint weights it loaded at startup.
    before = vl.logprobs(ids)

    # Student = a SECOND copy of the same weights (HF, CPU). Snapshot the pristine
    # state, then perturb the live copy the way a training step would move them.
    student = HFScorer.from_pretrained(MODEL, dtype="bfloat16")
    orig_state = {k: v.clone() for k, v in student.lm.state_dict().items()}
    _perturb_(student.lm)

    # HOT-UPDATE: push the perturbed student weights into the running engine.
    policy.sync_from(student)
    after = vl.logprobs(ids)

    # REVERT: reload the pristine weights into the student, push them back through the
    # SAME code path. The engine must return to baseline.
    student.lm.load_state_dict(orig_state)
    policy.sync_from(student)
    restored = vl.logprobs(ids)

    finite = torch.isfinite(before) & torch.isfinite(after) & torch.isfinite(restored)
    d_after = (after[finite] - before[finite]).abs()
    d_restored = (restored[finite] - before[finite]).abs()
    med_after = float(d_after.median())
    med_restored = float(d_restored.median())
    max_restored = float(d_restored.max())

    # (1) LANDS + CAUSAL: the perturbed student weights measurably moved the live
    # sampler's OWN logprobs, far beyond same-engine determinism (~0 for identical
    # weights). A silent no-op in load_weights would leave the bulk unchanged.
    assert med_after > 0.1, (
        f"vLLM sync_from did NOT change the sampler's logprobs (median move {med_after:.4f}) "
        "-- the weight hot-update is a silent no-op; online would sample from STALE weights "
        "and degrade into a frozen offline run. Inspect load_weights param-name mapping / "
        "device / dtype, do NOT relax this."
    )

    # (2) CLEAN + REVERSIBLE: pushing the ORIGINAL weights back reproduces baseline
    # bit-close. Only possible if load_weights truly OVERWRITES the live sampler
    # weights (same engine + identical weights => deterministic identical logprobs).
    # Compare on FINITE positions only: position 0 is nan by the align-to-t convention
    # (no preceding context), and nan != nan would spuriously fail a raw allclose.
    assert max_restored < 2e-2, (
        f"vLLM sync_from is not a clean reversible overwrite (median restore drift "
        f"{med_restored:.4f}, max {max_restored:.4f}) -- pushing the original weights back "
        "did not restore baseline, so the write is partial or path-dependent."
    )

    # (3) SELF-CALIBRATED: the revert is orders tighter than the perturbation, so the
    # proof does not depend on guessing the perturbation magnitude.
    assert med_restored < 0.1 * med_after, (
        f"revert drift {med_restored:.4f} not << perturbation move {med_after:.4f}"
    )

    print(
        f"[hot-update] vLLM sync_from  perturb median move={med_after:.4f}  "
        f"revert median drift={med_restored:.4f}  revert max drift={max_restored:.4f}  "
        "(large perturb move + tiny revert drift => load_weights rewrites the LIVE sampler "
        "=> online rollouts are genuinely on-policy)"
    )
