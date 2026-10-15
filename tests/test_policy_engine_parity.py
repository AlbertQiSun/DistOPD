"""The engine the GPU gates build must be the engine the online arm runs.

rollout/engine.py opens by naming this exact failure twice over: "this project has twice
shipped a gate that certified a configuration production did not run (the weight hot-update
test built its engine with prefix caching OFF while the real arm ran with it ON; the
determinism probe built its engine at gpu_memory_utilization 0.90 while the server ran
0.72)". It centralises the settings so that, in its words, "anything building this engine
(probe included) gets it".

tests/vllm_engine.py was the one thing building a policy engine that never called it. It went
straight to `LLM(**kwargs)`, so every @slow policy gate ran vLLM's DEFAULT layout -- V1
EngineCore in a subprocess -- while the arm runs it in-process. Certify gate 5 is where that
landed: sync_from can only rewrite weights it has an in-process handle to, so the hot-update
gate failed on a layout the online arm never uses. Third instance of the same class, and the
second time on that same test.

These checks need no GPU and no model: they compare what gets PINNED against
rollout/engine.py, which is the single source of truth for both.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rollout.engine import POLICY_BATCH_INVARIANT, policy_engine_kwargs
from tests.vllm_engine import TEST_GPU_MEMORY_UTILIZATION, production_engine_kwargs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Establish the condition rather than inherit it -- these variables are set for real in
    any process that has already built a policy engine."""
    for k in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_BATCH_INVARIANT"):
        monkeypatch.delenv(k, raising=False)


def test_the_test_helper_pins_the_in_process_engine_core():
    """THE GATE-5 BUG. Without this, v0.25.1 puts the sampled model in a subprocess and
    sync_from has nothing to hot-update -- so the hot-update gate reports a failure of the
    online arm's on-policy guarantee when what actually differed was the engine layout."""
    production_engine_kwargs()
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", (
        "tests/vllm_engine.py does not pin the in-process EngineCore, so the @slow policy "
        "gates run a layout the online arm never uses")


def test_the_test_helper_pins_batch_invariance_to_the_production_value():
    """POLICY_BATCH_INVARIANT is pinned ON on measured grounds -- without it batch composition
    changed the SAMPLED TOKENS in 6 of 6 greedy comparisons. A gate running with it off is
    certifying reproducibility the arm does not have."""
    production_engine_kwargs()
    want = "1" if POLICY_BATCH_INVARIANT else "0"
    assert os.environ.get("VLLM_BATCH_INVARIANT") == want


def test_prefix_caching_matches_production():
    """The FIRST instance of this class, kept as a live check rather than a war story: the
    hot-update test once ran with prefix caching off while the arm ran it on, so rollouts
    sampled new weights while attending to KV computed under step-0 weights."""
    kw = production_engine_kwargs()
    assert kw["enable_prefix_caching"] == policy_engine_kwargs()["enable_prefix_caching"]
    assert kw["enable_prefix_caching"] is False


def test_every_production_kwarg_survives_into_the_test_engine():
    """Not just the two named above. Whatever rollout/engine.py decides is part of the policy
    engine has to reach the engine these gates build.

    CANNOT FAIL AS CURRENTLY IMPLEMENTED, and that is stated rather than left to be assumed:
    production_engine_kwargs CALLS policy_engine_kwargs, so adding a kwarg there propagates by
    construction -- reverse-verification confirmed that injecting one leaves this green. What
    it does guard is the refactor where the helper stops deriving and starts listing settings
    itself, which is precisely how the two historical divergences began."""
    prod, test = policy_engine_kwargs(), production_engine_kwargs()
    for k, v in prod.items():
        assert test[k] == v, f"{k}: production {v!r} but the test engine builds {test[k]!r}"


def test_only_the_memory_cap_differs():
    """The ONE legitimate difference, stated so it cannot grow. These tests score a 1.7B model
    over short sequences; taking production's KV share would starve everything else in the
    process (tests/vllm_engine.py's docstring has the error it used to produce)."""
    extra = set(production_engine_kwargs()) - set(policy_engine_kwargs())
    assert extra == {"gpu_memory_utilization"}, (
        f"the test engine adds {sorted(extra)} beyond production. Only the memory cap is "
        "allowed to differ -- everything else is the configuration under certification.")
    assert production_engine_kwargs()["gpu_memory_utilization"] == TEST_GPU_MEMORY_UTILIZATION


def test_a_caller_can_still_override():
    """The hot-update test builds with its own settings; overrides must win over the
    production defaults, or the helper cannot serve the cases that need to differ."""
    assert production_engine_kwargs(dtype="float32")["dtype"] == "float32"
    assert production_engine_kwargs(gpu_memory_utilization=0.5)["gpu_memory_utilization"] == 0.5
