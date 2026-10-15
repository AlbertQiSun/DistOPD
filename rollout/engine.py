"""THE sampling policy's vLLM engine configuration -- one definition, every importer.

Mirrors teacher/engine.py, and exists for the same reason: this project has twice shipped
a gate that certified a configuration production did not run (the weight hot-update test
built its engine with prefix caching OFF while the real arm ran with it ON; the
determinism probe built its engine at gpu_memory_utilization 0.90 while the server ran
0.72). Both were caught late and by accident. Keeping the kwargs in one place makes that
class of drift something you have to write code to cause.

TWO SETTINGS HERE ARE LOAD-BEARING, NOT PERFORMANCE KNOBS:

  enable_prefix_caching=False -- vLLM keys its KV cache on TOKENS, not on weights, and
    sync_from's load_weights does NOT invalidate it. With caching on and resync_every=1,
    the SYSTEM_PROMPT prefix every request shares would be computed once under step-0
    weights and reused for the whole run, so rollouts would sample NEW weights while
    attending to STALE KV -- "online" secretly part-stale, which is the confound this
    phase exists to eliminate and which looks exactly like real OPD poisoning.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 -- vLLM v0.25.1 runs the V1 EngineCore in a SUBPROCESS
    by default, which puts the sampled model in another process and leaves sync_from with
    no in-process handle to hot-update. It is set here rather than only in the sbatch so
    that anything building this engine (probe included) gets it.

VLLM_BATCH_INVARIANT IS NOW PINNED ON (POLICY_BATCH_INVARIANT below), on measured grounds --
scripts/policy_invariance_probe.py, recorded in runs/gate_log.jsonl:

  * WITHOUT it, batch composition changes the SAMPLED TOKENS. Greedy (temperature 0, so no
    RNG involved at all) flipped in 6 of 6 comparisons -- a 100% rate, not a marginal
    effect -- and the sampled-temperature cases flipped 12 of 12. So "same settings, same
    seed, reproducible" is simply FALSE without it: a trajectory's content depends on which
    other trajectories happened to share its batch, and a real rollout puts each trajectory
    through ~7 different batch compositions as the batch desynchronises.
  * WITH it, every case is bit-identical and the sampled-token logprob residual is exactly
    0.000e+00.
  * COST: 2.10x on policy-only wall clock (18.2s -> 38.2s for 32 trajectories), which lands
    in the pre-agreed "2-4x, check it fits the budget" band; at scale that projects to
    ~15.8h per arm against a 48h window, so the flag stays on.
    RE-CHECKED against the later end-to-end measurements and it still holds with room:
    rollout measures 1.55 s/trajectory WITH the flag, so 38,400 trajectories is 16.5h, and
    the ONLINE arm (rollout + training in one job) totals 30.9h -- 17h inside the window.
    Turning the flag off would save ~8.6h of that and cost reproducibility, which was never
    the trade on offer.

The probe can still override it -- comparing both settings is its job -- but production
must not: an override here silently reintroduces a reproducibility failure that no gate
downstream would catch.
"""
from __future__ import annotations

import os

POLICY_DTYPE = "bfloat16"

# See the module docstring: measured, not assumed. Off => 100% greedy flip rate.
POLICY_BATCH_INVARIANT = True


def pin_policy_env(batch_invariant: bool = POLICY_BATCH_INVARIANT) -> None:
    """Set the policy engine's env BEFORE vllm is imported.

    The default is the PINNED production value. The probe passes it explicitly because
    comparing both settings is its entire job; nothing else should, since an override
    reintroduces a reproducibility failure no downstream gate would catch.
    """
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"     # see module docstring
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def student_vanilla_math():
    """Context manager: the student's math runs on VANILLA torch kernels, even in the
    process that hosts the batch-invariant policy engine.

    THE MEASUREMENT THAT DICTATES THIS SHAPE. vLLM's batch-invariant mode registers a direct
    CUDA kernel for aten::linear (among mm/addmm/matmul/bmm/softmax). Vanilla `linear` is a
    COMPOSITE op -- autograd differentiates it through the matmul decomposition -- but with a
    direct kernel registered, autograd records `linear_backward` as the derivative, and that
    op has NO CUDA kernel. First backward:

        NotImplementedError: Could not run 'aten::linear_backward' ... 'CUDA' backend
        (bench --pin-batch-invariant, <node>, 2026-08-18)

    So training under the overrides is not slow, it is IMPOSSIBLE on this stack -- which
    kills the symmetric option (pin both arms) by measurement. What remains: the student
    trains VANILLA in both arms. Offline is vanilla because its process never enables the
    mode; online gets vanilla through this bracket, entered for the training and diagnostic
    sections of every step. Where the mode was never enabled the bracket is a no-op, so both
    arms still execute the same code path.

    ~300 destroy/re-enable cycles per online run is exactly what an earlier design note
    rejected as "intermediate states no gate has seen". Two gates now see them:
    tests/test_student_kernel_parity.py (bit-parity of student logprobs, engine present vs
    absent) and a 3b-probe re-run after bracket cycles. Rejected-then-adopted is fine when
    the alternative was measured to be impossible; adopted WITHOUT the gates would not be.

    On exit the mode is re-enabled and VERIFIED -- returning with the overrides half-restored
    would hand the next rollout a sampling configuration gate 3b never certified.
    """
    import sys as _sys

    return _StudentVanillaMath(_sys.modules.get(
        "vllm.model_executor.layers.batch_invariant"))


class _StudentVanillaMath:
    def __init__(self, bi) -> None:
        self._bi = bi
        self._entered_with_mode_on = False

    def __enter__(self):
        bi = self._bi
        if bi is None or not getattr(bi, "_batch_invariant_MODE", False):
            return self                      # vanilla already -- the offline arm's path
        self._entered_with_mode_on = True
        lib = bi._batch_invariant_LIB
        lib._destroy()                       # de-registers every impl this Library added
        bi._batch_invariant_LIB = None
        bi._batch_invariant_MODE = False
        import torch as _torch

        # enable() also monkeypatches the PYTHON attribute torch.bmm; _destroy cannot see
        # that. The original is the C binding, recoverable without a stash.
        _torch.bmm = _torch._C._VariableFunctions.bmm
        if _torch.cuda.is_available():
            # match the never-enabled process (torch defaults) -- what the offline arm and
            # the gate-6 bench run under. Reduced-precision reductions default to True.
            _torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            _torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            _torch.backends.cuda.preferred_blas_library(backend="cublas")
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._entered_with_mode_on:
            return False
        self._bi.enable_batch_invariant_mode()
        if not self._bi._batch_invariant_MODE:
            raise RuntimeError(
                "could not re-enable vllm batch-invariant mode after the student's vanilla "
                "bracket -- the next rollout would sample under a configuration gate 3b "
                "never certified. Refusing to continue.")
        return False


def policy_engine_kwargs(*, dtype: str = POLICY_DTYPE, **extra) -> dict:
    """vLLM LLM(...) kwargs for the sampling policy."""
    kwargs = dict(dtype=dtype, enable_prefix_caching=False)
    kwargs.update(extra)
    return kwargs


def policy_config_summary(**overrides) -> str:
    kw = policy_engine_kwargs(**overrides)
    return (f"dtype={kw['dtype']} prefix_caching={kw['enable_prefix_caching']} "
            f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')} "
            f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')} "
            + " ".join(f"{k}={v}" for k, v in sorted(kw.items())
                       if k not in ("dtype", "enable_prefix_caching")))
