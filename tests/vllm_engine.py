"""ONE vLLM engine per pytest process, torn down for real.

`pytest -m slow` runs several GPU tests in one process, and each used to build its own
`LLM(...)`. vLLM sizes its KV cache as a FRACTION OF FREE MEMORY at startup, so the first
engine took 83 of 95 GiB and every later engine died with

    ValueError: Free memory on device cuda:0 (5.26/95.07 GiB) on startup is less than
    desired GPU memory utilization (0.92, 87.46 GiB)

-- an error about the SECOND engine that is really about the FIRST one never letting go.
Running the files one at a time hides it; that is a workaround, not a fix, and it leaves
the next person to rediscover it.

Two things here, doing different jobs:

  * `gpu_memory_utilization` is capped LOW. These tests score a 1.7B model over a couple of
    short sequences; they have no use for an 83 GiB KV cache, and taking it is what starves
    everything downstream (including the plain HF scorer some of them load afterwards).
  * the one-live-engine guard fails LOUDLY. Two live V1 engines in one process interfere
    through CUDA graphs -- the documented reason the teacher is served as a separate
    PROCESS rather than just on a separate card (configs/opd_qwen3.yaml). Capping memory
    would let two coexist and quietly reintroduce exactly that.

TEARDOWN IS HARDER THAN IT WAS, and the reason is worth stating because it is a direct cost
of getting the configuration right. The engine is now built through production_engine_kwargs,
which pins VLLM_ENABLE_V1_MULTIPROCESSING=0 so the EngineCore stays in THIS process -- what
sync_from needs, and what the online arm runs. The old teardown assumed the opposite: a child
process exiting reclaimed everything, so `del llm` plus a wait was enough. In-process, the KV
pool and the distributed state are held by objects and module-level globals here, and vLLM's
own destroy_* helpers are the only thing that drops them.

An earlier version of this docstring claimed the wait "is true either way". It was not, and
that claim was itself the unverified kind this repo forbids: the first cluster run after the
parity change accumulated two engines' worth of memory in one pytest process and refused the
third build at 35% free.
"""
from __future__ import annotations

import contextlib
import gc
import time

# Small enough that several engines COULD fit -- the guard below is what stops them.
TEST_GPU_MEMORY_UTILIZATION = 0.30
_MIN_FREE_TO_START = 0.60          # fraction of the device that must be free to build one

_LIVE: list = []


def _free_fraction() -> float:
    import torch
    free, total = torch.cuda.mem_get_info()
    return free / total


def _shutdown(llm, *, want_free: float = _MIN_FREE_TO_START, timeout_s: int = 30) -> None:
    import torch

    # ALL OF THIS IS REQUIRED IN-PROCESS, and none of it was needed before. With
    # VLLM_ENABLE_V1_MULTIPROCESSING=0 the EngineCore lives in THIS process, so nothing is
    # reclaimed by a child exiting -- the KV pool, the model runner and the NCCL/dist state are
    # all held by objects here, and `del llm; empty_cache()` releases none of them. Observed
    # after the engine-parity change: two engines' worth of memory accumulated across one
    # pytest process and the third build was refused at 35% free.
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
    if _LIVE:
        _LIVE.clear()
    del llm

    # vLLM's own teardown. Wrapped because the import path is version-specific and a missing
    # symbol must not mask the memory error below -- which states the real problem either way.
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as exc:                        # noqa: BLE001
        print(f"[vllm-engine] vLLM teardown helpers unavailable ({exc!r}); relying on gc")

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ADVISORY, NOT FATAL -- deliberately, and this is the second correction to this function.
    # An in-process EngineCore does not give the memory back: measured twice on <node> and
    # <node>, both times exactly 35% free after a 180s wait, with and without the destroy_*
    # calls above. So raising here only converts a known limitation into a gate failure, and
    # the wait itself just burns 3 minutes per test.
    #
    # The check that MATTERS is the one at BUILD time in vllm_engine(): it refuses to start an
    # engine on a starved device and says why. This one is a warning that helps read the log.
    # What actually solves it is a process boundary -- scripts/run_slow_isolated.sh -- because
    # the state being held is vLLM's module-level globals, not ours.
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _free_fraction() >= want_free:
            return
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1.0)
    print(f"[vllm-engine] WARNING: {_free_fraction():.0%} of the GPU free after {timeout_s}s -- "
          "an in-process EngineCore does not release its KV pool. Harmless if this process "
          "builds no further engine (scripts/run_slow_isolated.sh gives each test its own); "
          "the next build in THIS process would refuse to start.", flush=True)


def production_engine_kwargs(**overrides) -> dict:
    """Pin the env and return the kwargs the REAL policy engine is built with.

    THE POINT OF THIS FUNCTION. rollout/engine.py exists, in its own words, because "this
    project has twice shipped a gate that certified a configuration production did not run"
    -- and it says the env is set there "so that anything building this engine (probe
    included) gets it". This module was the one thing building a policy engine that never
    called it, so the slow gates ran a configuration the online arm does not.

    Certify gate 5 is where that surfaced. VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the V1
    EngineCore IN THIS PROCESS; without it v0.25.1 puts the sampled model in a subprocess and
    sync_from has no handle to hot-update, which is exactly what the gate reported. So the
    hot-update gate had been failing on a layout the arm never uses -- the same test that
    caused the FIRST instance of this class (its engine had prefix caching off while the arm
    ran it on).

    Separated from vllm_engine so tests/test_policy_engine_parity.py can check what gets
    pinned without a GPU or a model.
    """
    from rollout.engine import pin_policy_env, policy_engine_kwargs

    pin_policy_env()                       # MUST precede `import vllm`
    kwargs = policy_engine_kwargs()
    kwargs.update(overrides)
    kwargs.setdefault("gpu_memory_utilization", TEST_GPU_MEMORY_UTILIZATION)
    return kwargs


@contextlib.contextmanager
def vllm_engine(**kwargs):
    """Build the process's single vLLM engine, and tear it down for real afterwards.

    Built through production_engine_kwargs, so the engine under test IS the engine the
    online arm runs -- see that function for why this stopped being optional.
    """
    kwargs = production_engine_kwargs(**kwargs)
    from vllm import LLM

    if _LIVE:
        raise RuntimeError(
            "a vLLM V1 engine is already live in this process. Two live engines interfere "
            "through CUDA graphs -- that is why the real teacher is a separate PROCESS, not "
            "just a separate card. Tear the first one down before building another."
        )
    free = _free_fraction()
    if free < _MIN_FREE_TO_START:
        raise RuntimeError(
            f"only {free:.0%} of the GPU is free before building an engine. A previous "
            "engine has not released its KV cache -- building now would fail with vLLM's "
            "confusing 'free memory < desired utilization' error, which blames the new "
            "engine for the old one's leak."
        )
    llm = LLM(**kwargs)
    _LIVE.append(llm)
    try:
        yield llm
    finally:
        _shutdown(llm)
